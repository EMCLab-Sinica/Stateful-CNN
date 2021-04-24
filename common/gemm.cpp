#include <stdint.h>

#include "cnn_common.h"
#include "platform.h"
#include "my_debug.h"
#include "op_utils.h"
#include "my_dsplib.h"
#include "intermittent-cnn.h"

void alloc_gemm(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    const ParameterInfo *A = input[0], *B = input[1];

    MY_ASSERT(A->dims[0] == 1);

    output->dims[0] = A->dims[0];
#if JAPARI
    output->dims[1] = upper_gauss(B->dims[1], BATCH_SIZE) * (BATCH_SIZE + 1);
#elif STATEFUL
    output->dims[1] = upper_gauss(B->dims[1], BATCH_SIZE) * BATCH_SIZE;
#else
    output->dims[1] = B->dims[1];
#endif
    output->bitwidth = 16;
    output->slot = get_next_slot(model, A);
    output->scale = A->scale * B->scale;

    uint16_t output_len = output->dims[0] * output->dims[1];

    output->params_len = output_len * upper_gauss(B->dims[0], flags->extra.gemm.tile_channel) * sizeof(int16_t);
}

void GemmInputChunkHandler(uint32_t offset, uint16_t real_chunk_len, uint8_t state_bit, void* _params) {
    int16_t* buffer_a = reinterpret_cast<int16_t*>(_params);
    my_printf_debug("GemmInputChunkHandler offset=%d real_chunk_len=%d state_bit=%d" NEWLINE, offset, real_chunk_len, state_bit);
    if (state_bit) {
        int16_t* to_offset = buffer_a + offset;
        my_offset_q15_batched(to_offset, -0x4000, to_offset, real_chunk_len);
    }
}

void handle_gemm(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    const ParameterInfo *A = input[0], *B = input[1], *C = input[2];

    my_printf_debug("Gemm! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    int16_t A_len = A->dims[0] * A->dims[1] + 2,
            output_len = output->dims[0] * output->dims[1];

    int16_t *buffer_a = lea_buffer,
            *buffer_temp = buffer_a + A_len;
#if JAPARI
            buffer_temp += 2;
#endif
    int16_t *buffer_b = buffer_temp + output->params_len / sizeof(int16_t);
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
    offset ^= 0x4000;
#endif

#if JAPARI
    first_unfinished_value_offset -= BATCH_SIZE;
#else
    first_unfinished_value_offset -= (BATCH_SIZE - 1);
#endif

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

#if INDIRECT_RECOVERY
    MY_ASSERT(flags->extra.gemm.tile_width >= BATCH_SIZE);
#endif

    for (; i < B->dims[0]; i += flags->extra.gemm.tile_channel, tile++) {
        uint16_t tile_channels = MIN_VAL(flags->extra.gemm.tile_channel, B->dims[0] - i);
        uint16_t extended_tile_channels = tile_channels + 2;
        buffer_a[tile_channels] = -0x8000;
        buffer_a[tile_channels + 1] = 0;

#if JAPARI
        if (has_footprints(A)) {
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
        iterate_chunks(model, A, i, tile_channels, GemmInputChunkHandler, buffer_a);
#endif

        my_printf_debug("Tile for A" NEWLINE);
        dump_matrix2_debug(buffer_a, 1, extended_tile_channels, ValueInfo(A, model));

        int16_t output_offset = tile * output_len + j_with_footprints;

        for (; j < B->dims[1]; j += flags->extra.gemm.tile_width) {
            int16_t tile_width = MIN_VAL(flags->extra.gemm.tile_width, B->dims[1] - j);
            int16_t values_to_preserve = tile_width,
                    full_tile_width = tile_width;
#if JAPARI
            values_to_preserve = extend_for_footprints(tile_width);
            full_tile_width = (values_to_preserve + 1) / 2 * 2;
#endif
            int16_t *filter_ptr = buffer_b;
            my_fill_q15(0, filter_ptr, extended_tile_channels * full_tile_width);
            for (uint16_t row = 0; row < tile_channels; row++) {
#if JAPARI
                int16_t* cur_filter_start = filter_ptr;
                uint8_t copy_size = MIN_VAL(BATCH_SIZE, tile_width);
                for (uint16_t col = 0; filter_ptr < cur_filter_start + values_to_preserve; col += BATCH_SIZE) {
                    my_memcpy_from_param(model, filter_ptr,
                              B, (i + row) * B->dims[1] + j + col,
                              copy_size * sizeof(uint16_t));
                    filter_ptr += copy_size;
                    if (tile_width >= BATCH_SIZE) {
                        filter_ptr++;
                    }
                    if (values_to_preserve != full_tile_width) {
                        filter_ptr++;
                    }
                    my_printf_debug("filter_ptr = lea_buffer + %ld" NEWLINE, filter_ptr - lea_buffer);
                }
#else
                my_memcpy_from_param(model, filter_ptr,
                          B, (i + row) * B->dims[1] + j,
                          tile_width * sizeof(uint16_t));
                filter_ptr += full_tile_width;
#endif
            }
#if JAPARI
            my_fill_q15(0, filter_ptr, 2 * full_tile_width);
            uint8_t processed_biases = 0, bias_offset = 0;
            for (uint16_t idx = 0; idx < values_to_preserve; idx++) {
                if (processed_biases == BATCH_SIZE) {
                    processed_biases = 0;
                    filter_ptr[idx] = (param_state_bit(model, output, output_offset) ? 1 : -1);
                } else {
                    filter_ptr[idx] = -static_cast<int32_t>(get_q15_param(model, C, bias_offset + j)) / A->scale;
                    bias_offset++;
                    processed_biases++;
                }
            }
#else
            for (uint16_t idx = 0; idx < values_to_preserve; idx++) {
                filter_ptr[idx] = -static_cast<int32_t>(get_q15_param(model, C, idx + j)) / A->scale;
            }
#endif

#if INDIRECT_RECOVERY
            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif

#if STATEFUL
            uint16_t tile_width_first = tile_width;
            if (next_output_turning_point != INVALID_TURNING_POINT) {
                my_printf_debug("next_output_turning_point=%d output_offset=%d" NEWLINE, next_output_turning_point, output_offset);
                tile_width_first = MIN_VAL(next_output_turning_point - output_offset, tile_width);
            }
            my_printf_debug("tile_width_first=%d" NEWLINE, tile_width_first);
            MY_ASSERT(tile_width_first <= tile_width);
            my_offset_q15_batched(filter_ptr, -offset, filter_ptr, tile_width_first);
            if (tile_width_first != tile_width) {
                my_offset_q15_batched(filter_ptr + tile_width_first, -(offset ^ 0x4000), filter_ptr + tile_width_first, tile_width - tile_width_first);
            }
#endif

            my_printf_debug("Tile for B" NEWLINE);
            dump_matrix2_debug(buffer_b, extended_tile_channels, full_tile_width, ValueInfo(B, model));

#if HAWAII
            my_matrix_mpy_q15(1, extended_tile_channels, extended_tile_channels, full_tile_width, buffer_a, buffer_b, buffer_temp, nullptr, 0, 0);
#else
            my_matrix_mpy_q15(1, extended_tile_channels, extended_tile_channels, full_tile_width, buffer_a, buffer_b, buffer_temp,
                              output, output_offset, values_to_preserve);
#endif

            my_printf_debug("matrix_mpy_results" NEWLINE);
            dump_matrix_debug(buffer_temp, full_tile_width, ValueInfo(output, model));
            my_printf_debug(NEWLINE);

            compare_vm_nvm(buffer_temp, model, output, output_offset, values_to_preserve);

            my_printf_debug("output_offset=%d" NEWLINE, output_offset);
#if HAWAII
            hawaii_preserve_vector(model, output, output_offset, buffer_temp, values_to_preserve);
#endif
            output_offset += values_to_preserve;
        }
        j = j_with_footprints = 0;
    }

    flip_state_bit(model, output);

    my_printf_debug("handle_gemm output" NEWLINE);
    dump_params_debug(model, output);
}

void alloc_gemmmerge(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    output->slot = get_next_slot(model, input[0]);
    int16_t output_len = output->dims[0] * output->dims[1];
    output->params_len = output_len * sizeof(int16_t);
}

void handle_gemmmerge(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    const ParameterInfo *X = input[0];

    my_printf_debug("GemmMerge!" NEWLINE);

    int16_t output_len = X->dims[0] * X->dims[1];

    int16_t *buffer_temp = lea_buffer,
            *buffer_gemm = buffer_temp + output_len;
    make_buffer_aligned(&buffer_gemm);

    my_fill_q15(0, buffer_gemm, output_len);

    int16_t n_tiles = X->params_len / output_len / sizeof(int16_t);
    my_printf_debug("n_tiles=%d" NEWLINE, n_tiles);
    MY_ASSERT(n_tiles);

    for (uint16_t tile = 0; tile < n_tiles; tile++) {
        my_memcpy_from_param(model, buffer_temp, input[0], tile * output_len, output_len * sizeof(int16_t));
#if STATEFUL
        // XXX: use LEA?
        for (uint16_t idx = 0; idx < output_len; idx++) {
            if (get_value_state_bit(buffer_temp[idx])) {
                buffer_temp[idx] -= 0x4000;
            }
        }
#endif
        my_add_q15(buffer_gemm, buffer_temp, buffer_gemm, output_len);
        my_printf_debug("accumulated buffer_gemm" NEWLINE);
        dump_matrix_debug(buffer_gemm, output_len, ValueInfo(output, model));
    }

    my_printf_debug("Find max_multiplier for buffer_gemm" NEWLINE);
    uint16_t max_multiplier = find_max_multiplier(model, output, buffer_gemm, output_len);

    MY_ASSERT(max_multiplier != 0);

    int16_t scaleFract;
    uint8_t shift;

    // XXX: reduce calls to find_max_multiplier?
    float_to_scale_params(&scaleFract, &shift, 1.0f * max_multiplier);
    my_scale_q15(buffer_gemm, scaleFract, shift, buffer_gemm, output_len);

    output->scale /= max_multiplier;

    my_printf_debug("buffer_gemm with bias" NEWLINE);
    dump_matrix_debug(buffer_gemm, output_len, ValueInfo(output, model));

    max_multiplier = find_max_multiplier(model, output, buffer_gemm, output_len);
    float_to_scale_params(&scaleFract, &shift, 1.0f * max_multiplier);
    my_scale_q15(buffer_gemm, scaleFract, shift, buffer_gemm, output_len);
    output->scale /= max_multiplier;

    my_printf_debug("buffer_gemm after scaling up" NEWLINE);
    dump_matrix_debug(buffer_gemm, output_len, ValueInfo(output, model));

#if INDIRECT_RECOVERY
    iterate_chunks(model, output, 0, 0, OutputChunkHandler, buffer_gemm);
#endif

    my_memcpy_to_param(output, 0, buffer_gemm, output->params_len, 0);

    flip_state_bit(model, output);

    my_printf_debug("handle_gemmmerge output" NEWLINE);
    dump_params_debug(model, output);
}
