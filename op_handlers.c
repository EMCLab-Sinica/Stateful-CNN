#include <DSPLib.h>

#include "cnn_common.h"
#include "op_handlers.h"
#include "debug.h"
#include "platform.h"
#include "conv.h"
#include "intermittent-cnn.h"

DSPLIB_DATA(lea_buffer, 4)
int16_t lea_buffer[LEA_BUFFER_SIZE];

#define RESHAPE_AUTO_DIM (uint16_t)(-1)

void alloc_maxpool(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    uint16_t stride = flags & 0x0f;

    ParameterInfo *data = input[0];

    const uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t new_H = H / stride;
    uint16_t new_W = W / stride;

    output->params_len = new_H * new_W * CHANNEL * sizeof(int16_t);
    output->bitwidth = data->bitwidth;
    output->slot = get_next_slot(data);
    output->dims[0] = 1;
    output->dims[1] = CHANNEL;
    output->dims[2] = new_H;
    output->dims[3] = new_W;
}

void handle_maxpool(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
#ifndef WITH_PROGRESS_EMBEDDING
    UNUSED(model);
#endif

    my_printf_debug("MaxPool!" NEWLINE);

    uint16_t stride = flags & 0x0f;
    uint16_t kernel_size = (flags & 0xf0) >> 4;
    uint8_t need_nhwc2nchw = ((flags & 0xff00) >> 8 == NHWC2NCHW);

    /* XXX: add flags; assume no padding for now */
    ParameterInfo *data = input[0];

    my_printf_debug("handle_maxpool input" NEWLINE);
    dump_params(data);

    const uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t new_H = H / stride;
    uint16_t new_W = W / stride;

    uint16_t tile_c = get_tile_c(output);
    my_printf_debug("tile_c = %d" NEWLINE, tile_c);

    int16_t *data_baseptr = get_q15_param(data, 0);

#ifdef WITH_PROGRESS_EMBEDDING
    uint8_t do_flip_state = get_state_bit(model, data->slot) == get_state_bit(model, output->slot);
    if (do_flip_state && get_state_bit(model, data->slot)) {
        int16_t *data_ptr = data_baseptr;
        uint32_t len = data->params_len / sizeof(int16_t);
        for (uint32_t idx = 0; idx < len; idx++) {
            *data_ptr -= 0x4000;
            data_ptr++;
        }
    }
#endif

    int16_t offset_h, offset_w;
    if (data->flags & TRANSPOSED) {
        offset_h = CHANNEL;
        offset_w = H * CHANNEL;
    } else {
        offset_h = W * CHANNEL;
        offset_w = CHANNEL;
    }
    int16_t *output_baseptr = get_q15_param(output, 0);
    for (uint16_t tile_c_offset = 0; tile_c_offset < CHANNEL; tile_c_offset += tile_c) {
        uint16_t real_tile_c = MIN_VAL(tile_c, CHANNEL - tile_c_offset);
        int16_t *output_ptr;
        if (need_nhwc2nchw) {
            output_ptr = output_baseptr;
        } else {
            output_ptr = output_baseptr + tile_c_offset * new_H * new_W;
        }
        for (uint16_t h = 0; h + stride <= H; h += stride) {
            for (uint16_t w = 0; w + stride <= W; w += stride) {
                for (uint16_t c = 0; c < real_tile_c; c++) {
                    my_printf_debug("h=%d ", h);
                    my_printf_debug("w=%d ", w);
                    my_printf_debug("c=%d" NEWLINE, tile_c_offset + c);

                    int16_t max_val = INT16_MIN;
                    for (uint16_t sH = 0; sH < kernel_size; sH++) {
                        for (uint16_t sW = 0; sW < kernel_size; sW++) {
                            int16_t val;
                            // XXX: use a moving pointer instead of data_baseptr makes it slower. Why!?
                            // Output from handle_conv uses NHWC
                            val = data_baseptr[(h+sH) * offset_h + (w+sW) * offset_w + tile_c_offset + c];
                            print_q15_debug(val);
                            // XXX: use LEA?
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    // need a space as print_q15_debug does not append spaces when DUMP_INTEGERS is not defined
                    my_printf_debug(" max=");
                    print_q15_debug(max_val);
                    my_printf_debug(NEWLINE "offset=%d" NEWLINE, (uint16_t)(output_ptr - output_baseptr));
#ifdef WITH_PROGRESS_EMBEDDING
                    if (do_flip_state && !get_state_bit(model, data->slot)) {
                        max_val += 0x4000;
                    }
#endif
                    if (!need_nhwc2nchw) {
                        *output_ptr = max_val;
                        output_ptr++;
                    } else {
                        *(output_ptr + (tile_c_offset + c) * new_H * new_W + h / stride * new_W + w / stride) = max_val;
                    }
                }
            }
        }
    }

    my_printf_debug("handle_maxpool output" NEWLINE);
    if (!need_nhwc2nchw) {
        for (uint16_t c = 0; c < CHANNEL; c += tile_c) {
            output->dims[1] = MIN_VAL(tile_c, CHANNEL - c);
            dump_params_nhwc(output, c * new_H * new_W);
        }
        output->dims[1] = CHANNEL;
    } else if (tile_c == CHANNEL) {
        dump_params(output);
    }

    flip_state_bit(model, output);
}

void alloc_add(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    ParameterInfo *A = input[0], *B = input[1];
    MY_ASSERT(A->bitwidth == 16 && B->bitwidth == 16);

    output->params_len = A->params_len;
    output->bitwidth = A->bitwidth;
    output->slot = get_next_slot(A);
    output->dims[0] = 1;
    output->dims[1] = A->dims[1];
}

void handle_add(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(model);
    UNUSED(flags);

    /* Add: Y = X + W */
    my_printf_debug("Add!" NEWLINE);

    ParameterInfo *A = input[0], *B = input[1];

    msp_add_q15_params params = { .length = A->dims[1] };

    int16_t *buffer_a = lea_buffer,
            *buffer_b = lea_buffer + output->params_len / sizeof(int16_t);
    my_memcpy(buffer_a, get_q15_param(A, 0), output->params_len);
    my_memcpy(buffer_b, get_q15_param(B, 0), output->params_len);
    msp_status status = msp_add_q15(&params, buffer_a, buffer_b, buffer_a);
    msp_checkStatus(status);

    my_memcpy(get_q15_param(output, 0), buffer_a, output->params_len);
}

void alloc_matmul(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    ParameterInfo *A = input[0], *B = input[1];

    uint16_t output_len = A->dims[0] * B->dims[1];

    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->params_len = output_len * sizeof(int16_t);
    output->bitwidth = 16;
    output->slot = get_next_slot(A);
}

void handle_matmul(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
#ifndef WITH_PROGRESS_EMBEDDING
    UNUSED(model);
#endif

    UNUSED(flags);

    ParameterInfo *A = input[0], *B = input[1];

    my_printf_debug("handle_matmul inputs" NEWLINE);
    // dump_params(A);
    my_printf_debug("B" NEWLINE);
    dump_params(B);
    my_printf_debug("MatMul! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    MY_ASSERT(A->dims[0] * A->dims[1] <= 256);

    int16_t A_len = A->dims[0] * A->dims[1];

    int16_t *buffer_a = lea_buffer,
            *buffer_temp = buffer_a + A_len,
            *buffer_matmul = buffer_temp + A->dims[0] * B->dims[1],
            *buffer_b = buffer_matmul + A->dims[0] * B->dims[1];

    msp_fill_q15_params fill_params = {
        .length = 256,
        .value = 0,
    };
    msp_status status = msp_fill_q15(&fill_params, buffer_matmul);
    msp_checkStatus(status);

    my_memcpy(buffer_a, get_q15_param(A, 0), A->dims[0] * A->dims[1] * sizeof(uint16_t));

#ifdef WITH_PROGRESS_EMBEDDING
    if (get_state_bit(model, A->slot)) {
        for (uint16_t idx = 0; idx < A_len; idx++) {
            buffer_a[idx] -= 0x4000;
        }
    }
#endif

    /* LEA wants addresses to be 4-aligned */
    uint16_t step = (uint16_t)((256 / B->dims[1]) / 4 * 4);
    for (uint16_t i = 0; i < B->dims[0]; i = (uint16_t)(i + step)) {
        msp_matrix_mpy_q15_params params;
        uint16_t current_width = (uint16_t)MIN_VAL(step, B->dims[0] - i);
        params.srcARows = A->dims[0];
        params.srcACols = current_width;
        params.srcBRows = current_width;
        params.srcBCols = B->dims[1];

        my_memcpy(buffer_b,
                  get_q15_param(B, i * B->dims[1]),
                  current_width * B->dims[1] * sizeof(uint16_t));

        my_printf_debug("strip for A" NEWLINE);
        dump_matrix(buffer_a + A->dims[0] * i, (size_t)(A->dims[0] * current_width));
        my_printf_debug("B" NEWLINE);
        dump_matrix(buffer_b, (size_t)(current_width * B->dims[1]));

        status = msp_matrix_mpy_q15(
            &params,
            buffer_a + A->dims[0] * i,
            buffer_b,
            buffer_temp);
        msp_checkStatus(status);

        my_printf_debug("temp" NEWLINE);
        dump_matrix(buffer_temp, (size_t)(A->dims[0] * B->dims[1]));

        msp_add_q15_params params2 = { .length = output->params_len / sizeof(int16_t) };
        status = msp_add_q15(&params2, buffer_matmul, buffer_temp, buffer_matmul);
        msp_checkStatus(status);
    }
    my_memcpy(get_q15_param(output, 0), buffer_matmul, output->params_len);

    my_printf_debug("handle_matmul output" NEWLINE);
    dump_params(output);

    flip_state_bit(model, output);
}

void handle_relu(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    my_printf_debug("ReLu!" NEWLINE);

    uint32_t first_unfinished_value_offset = recovery_from_state_bits(model, output);

    ParameterInfo *X = input[0];
    my_printf_debug("handle_relu input" NEWLINE);
    dump_params_nhwc(X, 0);

    /* XXX: use LEA? */
    uint16_t bitwidth = X->bitwidth;
    MY_ASSERT(bitwidth == 16);
    int16_t *data = get_q15_param(X, 0);
    int16_t data_len = X->params_len / (bitwidth / 8);

    int16_t threshold, offset;
#ifdef WITH_PROGRESS_EMBEDDING
    if (get_state_bit(model, X->slot)) {
        threshold = 0x4000;
        offset = -0x4000;
    } else {
        threshold = 0;
        offset = 0x4000;
    }
#else
    threshold = offset = 0;
#endif

    my_printf_debug("threshold = %d" NEWLINE, threshold);
    my_printf_debug("offset = %d" NEWLINE, offset);

    int16_t *data_ptr = data + first_unfinished_value_offset;
    for (uint16_t i = first_unfinished_value_offset; i < data_len; i++) {
        if (*data_ptr < threshold) {
            *data_ptr = threshold;
        }
        *data_ptr += offset;
        data_ptr++;
    }
    my_printf_debug("handle_relu output" NEWLINE);
    dump_params_nhwc(output, 0);

    flip_state_bit(model, output);
}

void handle_reshape(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(model);
    UNUSED(flags);

    my_printf_debug("Reshape!" NEWLINE);

    ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth = data->bitwidth;
    output->slot = data->slot;
    MY_ASSERT(shape->bitwidth == 64);
    /*
     * At most one dimension of the new shape can be -1. In this case, the
     * value is inferred from the size of the tensor and the remaining
     * dimensions.
     *
     * A dimension could also be 0, in which case the actual dimension value
     * is unchanged (i.e. taken from the input tensor).
     * */
    uint32_t new_len = 1;
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
        if (!output->dims[i]) {
            output->dims[i] = data->dims[i];
        }
        if (output->dims[i] != RESHAPE_AUTO_DIM) {
            new_len *= output->dims[i];
        }
    }
    for (uint8_t i = shape->dims[0]; i < 4; i++) {
        output->dims[i] = 0;
    }
    uint16_t inferred_dim = output->params_len / sizeof(int16_t);
    int8_t auto_idx = -1;
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i] != RESHAPE_AUTO_DIM && output->dims[i] != 0) {
            inferred_dim /= output->dims[i];
        } else if (output->dims[i] == RESHAPE_AUTO_DIM) {
            auto_idx = i;
        }
    }
    if (auto_idx != -1) {
        output->dims[auto_idx] = inferred_dim;
        new_len *= inferred_dim;
    }
    MY_ASSERT(new_len * sizeof(int16_t) == output->params_len)
}

void handle_squeeze(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(model);
    UNUSED(flags);

    my_printf_debug("Squeeze!" NEWLINE);

    ParameterInfo *data = input[0];
    /* XXX: add flags; assume squeeze all one-size axes */
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth = data->bitwidth;
    output->slot = data->slot;
    for (uint8_t i = 0, j = 0; i < 4; i++) {
        if (input[0]->dims[i] != 1) {
            output->dims[j] = input[0]->dims[i];
            j++;
        }
    }
}

void handle_concat(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(model);
    UNUSED(flags);

    my_printf_debug("Concat!" NEWLINE);

    ParameterInfo *A = input[0], *B = input[1];
    // XXX: assume concatenating 2 tensors at the CHANNEL dimension and they
    // have the same number of channels.
    MY_ASSERT(A->dims[1] == B->dims[1]);
    output->tile_c = A->dims[1];
    output->dims[1] *= 2;
}

void handle_dropout(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(model);
    UNUSED(input);
    UNUSED(output);
    UNUSED(flags);

    ERROR_OCCURRED();
}

void alloc_globalaveragepool(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    ParameterInfo *data = input[0];

    MY_ASSERT(data->dims[0] == 1);
    uint16_t output_len = data->dims[1];

    output->dims[0] = output->dims[2] = output->dims[3] = 1;
    output->dims[1] = data->dims[1];
    output->params_len = output_len * sizeof(int16_t);
    output->bitwidth = 16;
    output->slot = SLOT_INTERMEDIATE_VALUES;
}

void handle_globalaveragepool(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(model);
    UNUSED(output);
    UNUSED(flags);

    my_printf_debug("GlobalAveragePool!" NEWLINE);

    ParameterInfo *data = input[0];
    uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    int16_t *input_baseptr = get_q15_param(data, 0),
            *output_baseptr = get_q15_param(output, 0);
    uint16_t len = H * W;
    for (uint16_t c = 0; c < CHANNEL; c++) {
        uint32_t total = 0;
        for (uint16_t h = 0; h < H; h++) {
            for (uint16_t w = 0; w < W; w++) {
                // Input is from Conv, which uses NHWC
                total += input_baseptr[h * W * CHANNEL + w * CHANNEL + c];
            }
        }
        output_baseptr[c] = total / len;
    }

    dump_params(output);
}

void handle_softmax(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(model);
    UNUSED(input);
    UNUSED(output);
    UNUSED(flags);

    // Do nothing - softmax does not change the relative order of values.
    // Just let run_model determine the max value
}

void handle_transpose(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(model);
    UNUSED(flags);

    my_printf_debug("Transpose!" NEWLINE);

    ParameterInfo *X = input[0];
    // not actually transpose data as we happen to need NHWC
    // XXX: assume NHWC -> NCHW
    output->dims[1] = X->dims[3];
    output->dims[2] = X->dims[1];
    output->dims[3] = X->dims[2];
}
