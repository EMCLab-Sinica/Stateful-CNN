#include <stdint.h>

#include "cnn_common.h"
#include "platform.h"
#include "my_debug.h"
#include "op_utils.h"
#include "my_dsplib.h"
#include "intermittent-cnn.h"

static struct {
    int16_t tile_channel;
    int16_t tile_width;
} gemm_params;

static inline int16_t upper_gauss(int16_t a, int16_t b) {
    return (a + b - 1) / b;
}

void alloc_gemm(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *A = input[0], *B = input[1];

    MY_ASSERT(A->dims[0] == 1);

    uint16_t output_len = A->dims[0] * B->dims[1];

    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->bitwidth = 16;
    output->slot = get_next_slot(model, A);
    output->scale = A->scale * B->scale;

    int16_t total_buffer_size = LEA_BUFFER_SIZE - A->dims[0] * A->dims[1];
    gemm_params.tile_width = B->dims[1];
    while (1) {
        my_printf_debug("tile_width=%d" NEWLINE, gemm_params.tile_width);
        /* LEA wants addresses to be 4 byte-aligned, or 2 Q15-aligned */
        gemm_params.tile_channel = (ARM_PSTATE_LEN / gemm_params.tile_width) / 2 * 2;
        for (; gemm_params.tile_channel > 0; gemm_params.tile_channel -= 2) {
            int16_t tmp = upper_gauss(B->dims[0], gemm_params.tile_channel);
            my_printf_debug("tile_channel=%d, tmp=%d" NEWLINE, gemm_params.tile_channel, tmp);
            if (total_buffer_size - gemm_params.tile_channel * gemm_params.tile_width >= output_len * tmp) {
                break;
            }
        }
        my_printf_debug("tile_channel = %d" NEWLINE, gemm_params.tile_channel);
        if (gemm_params.tile_channel > 0) {
            break;
        }
        MY_ASSERT(gemm_params.tile_width % 2 == 0);
        gemm_params.tile_width /= 2;
    }

    while (gemm_params.tile_width * gemm_params.tile_channel >= ARM_PSTATE_LEN) {
        MY_ASSERT(gemm_params.tile_width % 2 == 0);
        gemm_params.tile_width /= 2;
    }

    output->params_len = output_len * upper_gauss(B->dims[0], gemm_params.tile_channel) * sizeof(int16_t);
}

class GemmInputChunkHandler : public ChunkHandler {
public:
    GemmInputChunkHandler(int16_t *_buffer_a) : buffer_a(_buffer_a) {}

    void handle_chunk(uint32_t offset, uint16_t real_chunk_len, uint8_t state_bit) const override {
        if (state_bit) {
            int16_t* to_offset = buffer_a + offset;
            my_offset_q15(to_offset, -0x4000, to_offset, real_chunk_len);
        }
    }

private:
    int16_t *buffer_a;
};

void handle_gemm(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *A = input[0], *B = input[1];

    my_printf_debug("Gemm! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    // my_printf_debug("A" NEWLINE);
    // dump_params_debug(model, A);
    my_printf_debug("B" NEWLINE);
    dump_params_debug(model, B);

    int16_t A_len = A->dims[0] * A->dims[1],
            output_len = A->dims[0] * B->dims[1];

    int16_t *buffer_a = lea_buffer,
            *buffer_temp = buffer_a + A_len,
            *buffer_b = buffer_temp + output_len * upper_gauss(B->dims[0], gemm_params.tile_channel);

    my_memcpy_from_param(model, buffer_a, A, 0, A_len * sizeof(uint16_t));

#if STATEFUL
    iterate_chunks(model, A, 0, 0, GemmInputChunkHandler(buffer_a));

    int16_t offset, next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, 0 /*TODO: first_unfinished_value_offset*/, model, output);
    offset ^= 0x4000;
#endif

    for (uint16_t i = 0, tile = 0; i < B->dims[0]; i += gemm_params.tile_channel, tile++) {
        uint16_t tile_channels = MIN_VAL(gemm_params.tile_channel, B->dims[0] - i);

        uint8_t segmented_copy = 1;
        if (gemm_params.tile_width == B->dims[1]) {
            my_memcpy_from_param(model, buffer_b,
                      B, i * B->dims[1],
                      tile_channels * B->dims[1] * sizeof(uint16_t));
            segmented_copy = 0;
        }

        my_printf_debug("Tile for A" NEWLINE);
        dump_matrix_debug(buffer_a + i, tile_channels, ValueInfo(A, model));
        for (uint16_t j = 0; j < B->dims[1]; j += gemm_params.tile_width) {
            int16_t tile_width = MIN_VAL(gemm_params.tile_width, B->dims[0] - j);
            if (segmented_copy) {
                for (uint16_t row = 0; row < tile_channels; row++) {
                    my_memcpy_from_param(model, buffer_b + row * tile_width,
                              B, (i + row) * B->dims[1] + j,
                              tile_width * sizeof(uint16_t));
                }
            }
            my_printf_debug("Tile for B" NEWLINE);
            dump_matrix_debug(buffer_b, tile_channels * tile_width, ValueInfo(B, model));

            my_matrix_mpy_q15(1, tile_channels, tile_channels, tile_width, buffer_a + i, buffer_b, buffer_temp, 0);

            int16_t output_offset = tile * output_len + j;
#if STATEFUL
            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
            uint16_t tile_width_first = tile_width;
            if (next_output_turning_point > 0) {
                tile_width_first = MIN_VAL(next_output_turning_point - output_offset, tile_width);
            }
            my_printf_debug("tile_width_first=%d" NEWLINE, tile_width_first);
            MY_ASSERT(tile_width_first <= tile_width);
            my_offset_q15(buffer_temp, offset, buffer_temp, tile_width_first);
            if (tile_width_first != tile_width) {
                my_offset_q15(buffer_temp + tile_width_first, offset ^ 0x4000, buffer_temp + tile_width_first, tile_width - tile_width_first);
            }
#endif

            my_printf_debug("temp with states" NEWLINE);
            dump_matrix_debug(buffer_temp, tile_width, ValueInfo(output, model));
            my_printf_debug(NEWLINE);

            my_memcpy_to_param(output, output_offset, buffer_temp, tile_width * sizeof(int16_t));
        }
    }

#if STATEFUL
    flip_state_bit(model, output);
#endif

    my_printf_debug("handle_gemm output" NEWLINE);
    dump_params_debug(model, output);
}

void alloc_gemmmerge(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    const ParameterInfo *X = input[0];

    output->slot = get_next_slot(model, input[0]);
    output->params_len = X->dims[0] * X->dims[1] * sizeof(int16_t);
}

void handle_gemmmerge(struct Model *model, const struct ParameterInfo **input, struct ParameterInfo *output, const struct NodeFlags *flags) {
    const ParameterInfo *X = input[0], *C = input[1];

    my_printf_debug("GemmMerge!" NEWLINE);

    int16_t output_len = X->dims[0] * X->dims[1];

    int16_t *buffer_temp = lea_buffer,
            *buffer_gemm = buffer_temp + output_len,
            *buffer_c = buffer_gemm + output_len;

    my_fill_q15(0, buffer_gemm, output_len);

    int16_t n_tiles = X->params_len / output_len / sizeof(int16_t);
    my_printf_debug("n_tiles=%d" NEWLINE, n_tiles);

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

    my_memcpy_from_param(model, buffer_c, C, 0, output_len * sizeof(int16_t));

    my_printf_debug("C" NEWLINE);
    dump_params_debug(model, C);

    my_printf_debug("Find max_multiplier for buffer_gemm" NEWLINE);
    uint16_t max_multiplier_gemm = find_max_multiplier(model, output, buffer_gemm, output_len);
    my_printf_debug("Find max_multiplier for C" NEWLINE);
    uint16_t max_multiplier_c = find_max_multiplier(model, C, buffer_c);
    uint16_t max_multiplier = MIN_VAL(max_multiplier_gemm, max_multiplier_c);

    int16_t scaleFract;
    uint8_t shift;

    // XXX: reduce calls to find_max_multiplier?
    float_to_scale_params(&scaleFract, &shift, 1.0f * max_multiplier);
    my_scale_q15(buffer_gemm, scaleFract, shift, buffer_gemm, output_len);

    float_to_scale_params(&scaleFract, &shift, 1.0f * C->scale / X->scale * max_multiplier);
    my_scale_q15(buffer_c, scaleFract, shift, buffer_c, output_len);
    my_add_q15(buffer_gemm, buffer_c, buffer_gemm, output_len);

    output->scale /= max_multiplier;

    my_printf_debug("buffer_gemm with bias" NEWLINE);
    dump_matrix_debug(buffer_gemm, output_len, ValueInfo(output, model));

    max_multiplier = find_max_multiplier(model, output, buffer_gemm, output_len);
    float_to_scale_params(&scaleFract, &shift, 1.0f * max_multiplier);
    my_scale_q15(buffer_gemm, scaleFract, shift, buffer_gemm, output_len);

    my_printf_debug("buffer_gemm after scaling up" NEWLINE);
    dump_matrix_debug(buffer_gemm, output_len, ValueInfo(output, model));

    output->scale /= max_multiplier;

#if STATEFUL
    iterate_chunks(model, output, 0, 0, OutputChunkHandler(buffer_gemm));
#endif

    my_memcpy_to_param(output, 0, buffer_gemm, output->params_len);

#if STATEFUL
    flip_state_bit(model, output);
#endif

    my_printf_debug("handle_gemmmerge output" NEWLINE);
    dump_params_debug(model, output);
}
