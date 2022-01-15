#include <cstdint>
#include "cnn_common.h"
#include "data.h"
#include "platform.h"
#include "my_debug.h"
#include "op_utils.h"
#include "my_dsplib.h"
#include "intermittent-cnn.h"

void alloc_gemm(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    const ParameterInfo *A = input[0], *B = input[1];

    MY_ASSERT(A->dims[0] == 1);

    output->dims[0] = A->dims[0];
#if JAPARI
    output->dims[1] = B->dims[1] / BATCH_SIZE * (BATCH_SIZE + 1) + B->dims[1] % BATCH_SIZE;
#else
    output->dims[1] = B->dims[1];
#endif
    output->bitwidth = 16;
    output->slot = get_next_slot(model, A);
    output->scale = A->scale * B->scale;

    uint16_t output_len = output->dims[0] * output->dims[1];

    output->params_len = output_len * upper_gauss(B->dims[0], node->flags.extra.gemm.tile_channel) * sizeof(int16_t);
}

struct GemmInputChunkHandlerParams {
    int16_t* buffer;
    uint16_t buffer_offset;
};

void GemmInputChunkHandler(uint32_t offset, uint16_t real_chunk_len, int8_t state_bit, void* _params) {
    GemmInputChunkHandlerParams* params = reinterpret_cast<GemmInputChunkHandlerParams*>(_params);
    my_printf_debug("GemmInputChunkHandler offset=%d real_chunk_len=%d state_bit=%d" NEWLINE, offset, real_chunk_len, state_bit);
    int16_t* to_offset = params->buffer + offset - params->buffer_offset;
    my_offset_q15_batched(to_offset, -state_bit*0x4000, to_offset, real_chunk_len);
}

// https://tjsw.medium.com/86f06ac768da
template<uint8_t move_from, uint8_t batch_offset, std::enable_if_t<move_from < BATCH_SIZE>* = nullptr>
static inline void move_filter(int16_t*) {}

template<uint8_t move_from, uint8_t batch_offset, std::enable_if_t<move_from >= BATCH_SIZE>* = nullptr>
static inline void move_filter(int16_t* filter) {
    const uint8_t move_to = move_from/BATCH_SIZE*(BATCH_SIZE+1)+batch_offset;
    filter[move_to] = filter[move_from];
    move_filter<move_from-1, (batch_offset >= 1) ? (batch_offset-1) : (BATCH_SIZE-1)>(filter);
}

template<uint8_t offset>
static inline void clear_filter(int16_t* filter) {
    filter[offset] = 0;
    if (offset >= BATCH_SIZE+1) {
        clear_filter<offset-(BATCH_SIZE+1)>(filter);
    }
}

template<>
inline void clear_filter<BATCH_SIZE>(int16_t* filter) {
    filter[BATCH_SIZE] = 0;
}

void handle_gemm(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    const ParameterInfo *A = input[0], *B = input[1], *C = input[2];
    const NodeFlags* flags = &node->flags;

    my_printf_debug("Gemm! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    int16_t A_len = A->dims[0] * A->dims[1] + 2,
            output_len = output->dims[0] * output->dims[1];

    int16_t *buffer_a = lea_buffer,
            *buffer_temp = buffer_a + A_len;
#if JAPARI
            buffer_temp += 2;
    int16_t* buffer_b = buffer_temp + extend_for_footprints(OP_FILTERS);
#else
    int16_t* buffer_b = buffer_temp + OP_FILTERS;
#endif
    make_buffer_aligned(&buffer_b);

    uint16_t i = 0, tile = 0, j = 0, j_with_footprints = 0;

#if INTERMITTENT
    uint32_t first_unfinished_value_offset = job_index_to_offset(output, run_recovery(model, output));

#if INDIRECT_RECOVERY
    int16_t offset;
    uint16_t next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, first_unfinished_value_offset, model, output);
    offset = -offset;
#endif

    first_unfinished_value_offset = batch_start(first_unfinished_value_offset);

    fix_first_unfinished_value_offset(model, &first_unfinished_value_offset);

    tile = first_unfinished_value_offset / output_len;
    i = tile * flags->extra.gemm.tile_channel;
    j_with_footprints = first_unfinished_value_offset % output_len;

#if JAPARI
    j = j_with_footprints / (BATCH_SIZE + 1) * BATCH_SIZE;
#else
    j = j_with_footprints;
#endif

#endif

    for (; i < B->dims[0]; i += flags->extra.gemm.tile_channel, tile++) {
        const uint16_t tile_channels = MIN_VAL(flags->extra.gemm.tile_channel, B->dims[0] - i);
        const uint16_t extended_tile_channels = tile_channels + 2;

#if JAPARI
        if (has_footprints(A)) {
            // somehow loading many pieces is faster than loading a chunk and moving values around to remove footprints, even with external FRAM
            uint16_t input_offset = extend_for_footprints(i);
            for (uint16_t idx = 0, output_idx = 0; output_idx < tile_channels; idx += BATCH_SIZE + 1, output_idx += BATCH_SIZE) {
                my_memcpy_from_param(model, buffer_a + output_idx, A, input_offset + idx, BATCH_SIZE * sizeof(uint16_t));
            }
        } else
#endif
        {
            my_memcpy_from_param(model, buffer_a, A, i, tile_channels * sizeof(uint16_t));
        }

#if STATEFUL
        start_cpu_counter();
        GemmInputChunkHandlerParams params{buffer_a, i};
        iterate_chunks(model, A, i, tile_channels, GemmInputChunkHandler, &params);
        stop_cpu_counter(&Counters::stripping);
#endif
        buffer_a[tile_channels] = -0x8000;
        buffer_a[tile_channels + 1] = 0;

        my_printf_debug("Tile for A" NEWLINE);
        dump_matrix_debug(buffer_a, 1, extended_tile_channels, ValueInfo(A, model));

        int16_t output_offset = tile * output_len + j_with_footprints;

        for (; j < B->dims[1]; j += OP_FILTERS) {
            int16_t tile_width;
            uint8_t incomplete_tile = 0;
            if (OP_FILTERS > B->dims[1] - j) {
                tile_width = B->dims[1] - j;
                incomplete_tile = 1;
            } else {
                tile_width = OP_FILTERS;
            }
            int16_t values_to_preserve = tile_width,
                    full_tile_width = tile_width;
#if JAPARI
            values_to_preserve = extend_for_footprints(tile_width);
            full_tile_width = (values_to_preserve + 1) / 2 * 2;
#endif
            int16_t *filter_ptr = buffer_b;
            my_fill_q15(0, filter_ptr, extended_tile_channels * full_tile_width);
            for (uint16_t row = 0; row < tile_channels; row++) {
                my_memcpy_from_param(model, filter_ptr,
                          B, (i + row) * B->dims[1] + j,
                          tile_width * sizeof(uint16_t));
#if JAPARI
                // move loaded filters around to create zeros for footprint kernels
                start_cpu_counter();
                if (incomplete_tile) {
                    int8_t move_offset = values_to_preserve - tile_width;
                    int8_t cur_remaining = values_to_preserve % (BATCH_SIZE + 1);
                    for (int8_t move_dest = values_to_preserve - 1; move_dest >= 0; move_dest--) {
                        if (cur_remaining == 0) {
                            filter_ptr[move_dest] = 0;
                            move_offset--;
                            cur_remaining = BATCH_SIZE;
                            continue;
                        }
                        filter_ptr[move_dest] = filter_ptr[move_dest - move_offset];
                        cur_remaining--;
                    }
                } else {
                    const uint8_t last_elem = OP_FILTERS-1;
                    move_filter<last_elem, last_elem % BATCH_SIZE>(filter_ptr);
                    clear_filter<last_elem/(BATCH_SIZE+1)*(BATCH_SIZE+1)+BATCH_SIZE>(filter_ptr);
                }
                stop_cpu_counter(&Counters::embedding);
#endif
                filter_ptr += full_tile_width;
            }
#if JAPARI
            my_fill_q15(0, filter_ptr, 2 * full_tile_width);
            uint8_t processed_biases = 0, bias_offset = 0;
            for (uint16_t idx = 0; idx < values_to_preserve; idx++) {
                if (processed_biases == BATCH_SIZE) {
                    processed_biases = 0;
                    filter_ptr[idx] = param_state_bit(model, output, output_offset);
                } else {
                    if (tile == 0) {
                        filter_ptr[idx] = -static_cast<int32_t>(get_q15_param(model, C, bias_offset + j)) / A->scale;
                    }
                    bias_offset++;
                    processed_biases++;
                }
            }
#else
            if (tile == 0) {
                for (uint16_t idx = 0; idx < values_to_preserve; idx++) {
                    filter_ptr[idx] = -static_cast<int32_t>(get_q15_param(model, C, idx + j)) / A->scale;
                }
            }
#endif

#if INDIRECT_RECOVERY
            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif

#if STATEFUL
            start_cpu_counter();
            uint16_t tile_width_first = tile_width;
            if (next_output_turning_point != INVALID_TURNING_POINT) {
                my_printf_debug("next_output_turning_point=%d output_offset=%d" NEWLINE, next_output_turning_point, output_offset);
                tile_width_first = MIN_VAL(next_output_turning_point - output_offset, tile_width);
            }
            my_printf_debug("tile_width_first=%d" NEWLINE, tile_width_first);
            MY_ASSERT(tile_width_first <= tile_width);
            my_offset_q15_batched(filter_ptr, -offset, filter_ptr, tile_width_first);
            if (tile_width_first != tile_width) {
                my_offset_q15_batched(filter_ptr + tile_width_first, offset, filter_ptr + tile_width_first, tile_width - tile_width_first);
            }
            stop_cpu_counter(&Counters::embedding);
#endif

            my_printf_debug("Tile for B" NEWLINE);
            dump_matrix_debug(buffer_b, extended_tile_channels, full_tile_width, ValueInfo(B, model));

#if STATEFUL
            my_matrix_mpy_q15(1, extended_tile_channels, extended_tile_channels, full_tile_width, buffer_a, buffer_b, buffer_temp,
                              output, output_offset, values_to_preserve, offset, tile_width_first);
#else
            my_matrix_mpy_q15(1, extended_tile_channels, extended_tile_channels, full_tile_width, buffer_a, buffer_b, buffer_temp,
                              output, output_offset, values_to_preserve, 0, 0);
#endif

            my_printf_debug("matrix_mpy_results" NEWLINE);
            dump_matrix_debug(buffer_temp, full_tile_width, ValueInfo(output, model));
            my_printf_debug(NEWLINE);

            compare_vm_nvm(buffer_temp, model, output, output_offset, values_to_preserve);

            my_printf_debug("output_offset=%d" NEWLINE, output_offset);
#if HAWAII
            hawaii_record_footprints(model, values_to_preserve);
#endif
            output_offset += values_to_preserve;
        }
        j = j_with_footprints = 0;
    }

    flip_state_bit(model, output);

    my_printf_debug("handle_gemm output" NEWLINE);
    dump_params_debug(model, output, node->output_name);
}

void alloc_gemmmerge(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    output->slot = get_next_slot(model, input[0]);
    int16_t output_len = output->dims[0] * output->dims[1];
    output->params_len = output_len * sizeof(int16_t);
}

void handle_gemmmerge(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    const ParameterInfo *X = input[0];

    my_printf_debug("GemmMerge!" NEWLINE);

    int16_t output_len = X->dims[0] * X->dims[1];

    int16_t output_tile_size = node->flags.extra.gemmmerge.tile_length;
    if (!output_tile_size) {
        output_tile_size = output_len;
    }
#if JAPARI
    output_tile_size = extend_for_footprints(output_tile_size);
#endif

    uint16_t merge_offset = 0;
#if INTERMITTENT
    merge_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
#endif

    int16_t *buffer_temp = lea_buffer,
            *buffer_gemm = buffer_temp + output_tile_size;
    make_buffer_aligned(&buffer_gemm);

    int16_t n_tiles = X->params_len / output_len / sizeof(int16_t);
    my_printf_debug("n_tiles=%d" NEWLINE, n_tiles);
    MY_ASSERT(n_tiles);

    for (; merge_offset < output_len; merge_offset += output_tile_size) {
        int16_t cur_tile_size = MIN_VAL(output_tile_size, output_len - merge_offset);
        my_fill_q15(0, buffer_gemm, cur_tile_size);

        for (uint16_t tile = 0; tile < n_tiles; tile++) {
            my_memcpy_from_param(model, buffer_temp, input[0], tile * output_len + merge_offset, cur_tile_size * sizeof(int16_t));
#if STATEFUL
            start_cpu_counter();
            for (uint16_t idx = BATCH_SIZE - 1; idx < cur_tile_size; idx += BATCH_SIZE) {
                strip_state(buffer_temp + idx);
            }
            stop_cpu_counter(&Counters::stripping);
#endif
            my_add_q15(buffer_gemm, buffer_temp, buffer_gemm, cur_tile_size);
            my_printf_debug("accumulated buffer_gemm" NEWLINE);
            dump_matrix_debug(buffer_gemm, cur_tile_size, ValueInfo(output, model));
        }

#if INDIRECT_RECOVERY
        start_cpu_counter();
        OutputChunkHandlerParams params;
        params.buffer = buffer_gemm;
        params.buffer_offset = merge_offset;
        iterate_chunks(model, output, merge_offset, cur_tile_size, OutputChunkHandler, &params);
        stop_cpu_counter(&Counters::embedding);
#endif
        my_printf_debug("buffer_gemm after adjusting states; merge_offset=%d" NEWLINE, merge_offset);
        dump_matrix_debug(buffer_gemm, cur_tile_size, ValueInfo(output, model));

        my_memcpy_to_param(output, merge_offset, buffer_gemm, cur_tile_size * sizeof(int16_t), 0);
#if HAWAII
        hawaii_record_footprints(model, cur_tile_size);
#endif
    }

    flip_state_bit(model, output);

    my_printf_debug("handle_gemmmerge output" NEWLINE);
    dump_params_debug(model, output, node->output_name);
}
