#include <DSPLib.h>

#include "ops.h"
#include "op_handlers.h"
#include "debug.h"
#include "platform.h"

DSPLIB_DATA(lea_buffer, 4)
int16_t lea_buffer[LEA_BUFFER_SIZE];

void handle_maxpool(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    my_printf_debug("MaxPool!" NEWLINE);

    uint16_t stride = flags;

    /* XXX: add flags; assume no padding for now */
    ParameterInfo *data = input[0];

    my_printf_debug("handle_maxpool input" NEWLINE);
    dump_params(data);

    const uint16_t channel = data->dims[3], H = data->dims[1], W = data->dims[2];
    output->params_len = data->params_len / (uint16_t)(stride * stride);
    output->bitwidth_and_flags = data->bitwidth_and_flags | get_next_slot(data);
    output->dims[0] = 1;
    output->dims[1] = H / stride;
    output->dims[2] = W / stride;
    output->dims[3] = channel;

#ifdef WITH_PROGRESS_EMBEDDING
    int16_t state_bit = model->state_bit;
    if (state_bit) {
        model->state_bit = 0;
    } else {
        model->state_bit = 1;
    }
#endif

    int16_t *data_baseptr = get_q15_param(data, 0);
    for (uint16_t c = 0; c < channel; c++) {
        int16_t *output_ptr = get_q15_param(output, c);
        for (uint16_t h = 0; h +stride <= H; h += stride) {
            for (uint16_t w = 0; w + stride <= W; w += stride) {
                my_printf_debug("h=%d ", h);
                my_printf_debug("w=%d ", w);
                my_printf_debug("c=%d" NEWLINE, c);

                int16_t max_val = INT16_MIN;
                for (uint16_t sH = 0; sH < stride; sH++) {
                    for (uint16_t sW = 0; sW < stride; sW++) {
                        int16_t val = data_baseptr[(h+sH) * W * channel + (w+sW) * channel + c];
#ifdef WITH_PROGRESS_EMBEDDING
                        if (state_bit) {
                            val += 0x8000;
                        }
#endif
                        print_q15_debug(val);
                        // XXX: use LEA?
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                my_printf_debug("max=");
                print_q15_debug(max_val);
                my_printf_debug(NEWLINE "offset=%d" NEWLINE, (uint16_t)(output_ptr - get_q15_param(output, 0)));
#ifdef WITH_PROGRESS_EMBEDDING
                if (!state_bit) {
                    max_val += 0x8000;
                }
#endif
                *output_ptr = max_val;
                output_ptr += channel;
            }
        }
    }

    my_printf_debug("handle_maxpool output" NEWLINE);
    dump_params(output);
}

void handle_add(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    /* Add: Y = X + W */
    my_printf_debug("Add!" NEWLINE);

    if (get_param_bitwidth(input[0]) != 16 || get_param_bitwidth(input[1]) != 16) {
        // unsupported bitwidth
        ERROR_OCCURRED();
    }
    ParameterInfo *A = input[0], *B = input[1];
    output->params_len = input[0]->params_len;
    output->bitwidth_and_flags = input[0]->bitwidth_and_flags | get_next_slot(A);
    output->dims[0] = 1;
    output->dims[1] = A->dims[1];

    msp_add_q15_params params = { .length = A->dims[1] };

    int16_t *buffer_a = lea_buffer,
            *buffer_b = lea_buffer + output->params_len / sizeof(int16_t);
    my_memcpy(buffer_a, get_q15_param(A, 0), output->params_len);
    my_memcpy(buffer_b, get_q15_param(B, 0), output->params_len);
    msp_status status = msp_add_q15(&params, buffer_a, buffer_b, buffer_a);
    msp_checkStatus(status);

    my_memcpy(get_q15_param(output, 0), buffer_a, output->params_len);
}

void handle_matmul(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    ParameterInfo *A = input[0], *B = input[1];

    my_printf_debug("handle_matmul inputs" NEWLINE);
    // dump_params(A);
    my_printf_debug("B" NEWLINE);
    dump_params(B);
    my_printf_debug("MatMul! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    uint16_t output_len = (uint16_t)(A->dims[0] * B->dims[1]);
    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->params_len = (uint16_t)(output_len * 2);
    output->bitwidth_and_flags = 16 << FLAG_SLOTS_WIDTH | get_next_slot(A);

    if (A->dims[0] * A->dims[1] > 256) {
        // Matrix A too large!
        ERROR_OCCURRED();
    }

    int16_t *buffer_a = lea_buffer,
            *buffer_temp = buffer_a + A->dims[0] * A->dims[1],
            *buffer_matmul = buffer_temp + A->dims[0] * B->dims[1],
            *buffer_b = buffer_matmul + A->dims[0] * B->dims[1];

    msp_fill_q15_params fill_params = {
        .length = 256,
        .value = 0,
    };
    msp_status status = msp_fill_q15(&fill_params, buffer_matmul);
    msp_checkStatus(status);

    my_memcpy(buffer_a, get_q15_param(A, 0), (uint16_t)(A->dims[0] * A->dims[1] * sizeof(uint16_t)));

    /* LEA wants addresses to be 4-aligned */
    uint16_t step = (uint16_t)((256 / B->dims[1]) / 4 * 4);
    for (uint16_t i = 0; i < B->dims[0]; i = (uint16_t)(i + step)) {
        msp_matrix_mpy_q15_params params;
        uint16_t current_width = (uint16_t)MIN_VAL(step, B->dims[0] - i);
        params.srcARows = A->dims[0];
        params.srcACols = current_width;
        params.srcBRows = current_width;
        params.srcBCols = B->dims[1];

        my_memcpy(buffer_b, get_q15_param(B, (uint16_t)(i * B->dims[1])), (uint16_t)(current_width * B->dims[1] * sizeof(uint16_t)));

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

        msp_add_q15_params params2 = { .length = output_len };
        status = msp_add_q15(&params2, buffer_matmul, buffer_temp, buffer_matmul);
        msp_checkStatus(status);
    }
    my_memcpy(get_q15_param(output, 0), buffer_matmul, output->params_len);

    my_printf_debug("handle_matmul output" NEWLINE);
    dump_params(output);
}

void handle_relu(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    my_printf_debug("ReLu!" NEWLINE);

    ParameterInfo *X = input[0];
    my_memcpy(output, X, sizeof(ParameterInfo));

    /* XXX: use LEA? */
    uint16_t bitwidth = get_param_bitwidth(X);
    if (bitwidth != 16) {
        // unsupported bitwidth for ReLu
        ERROR_OCCURRED();
    }
    int16_t *data = get_q15_param(X, 0);
    int16_t data_len = X->params_len / (bitwidth / 8);

#ifdef WITH_PROGRESS_EMBEDDING
    uint16_t state_bit = model->state_bit;
    if (state_bit) {
        model->state_bit = 0;
    } else {
        model->state_bit = 1;
    }
#endif
    for (uint16_t i = 0; i < data_len; i++) {
#ifdef WITH_PROGRESS_EMBEDDING
        if (state_bit) {
            data[i] += 0x8000;
        }
#endif
        if (data[i] < 0) {
            data[i] = 0;
        }
#ifdef WITH_PROGRESS_EMBEDDING
        if (!state_bit) {
            data[i] += 0x8000;
        }
#endif
    }
    dump_params(output);
}

void handle_reshape(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    my_printf_debug("Reshape!" NEWLINE);

    ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    if (get_param_bitwidth(shape) != 64) {
        // unsupported shape format
        ERROR_OCCURRED();
    }
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
    }
    /*
     * XXX: Here is an heuristic - no conv nodes after reshape, so remapping
     * NHWC back to NCHW.
     * */
    uint8_t do_nhwc2nchw = get_param_slot_id(data) != FLAG_SLOTS;
    if (do_nhwc2nchw) {
        // data are intermediate values
        int16_t *output_addr = get_q15_param(output, 0);
        my_memcpy(lea_buffer, output_addr, output->params_len);
        uint16_t NUM = data->dims[0], H = data->dims[1],
                 W = data->dims[2], CHANNEL = data->dims[3];
        for (uint16_t n = 0; n < NUM; n++) {
            for (uint16_t c = 0; c < CHANNEL; c++) {
                for (uint16_t h = 0; h < H; h++) {
                    for (uint16_t w = 0; w < W; w++) {
                        uint16_t old_idx = n * CHANNEL * H * W + c * H * W       + h * W       + w,
                                 new_idx = n * H * W * CHANNEL + h * W * CHANNEL + w * CHANNEL + c;
                        output_addr[new_idx] = lea_buffer[old_idx];
                    }
                }
            }
        }
    }

    if (do_nhwc2nchw) {
        my_printf_debug("handle_reshape output" NEWLINE);
        dump_params(output);
    }
}

void handle_squeeze(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    my_printf_debug("Squeeze!" NEWLINE);

    ParameterInfo *data = input[0];
    /* XXX: add flags; assume squeeze all one-size axes */
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    for (uint8_t i = 0, j = 0; i < 4; i++) {
        if (input[0]->dims[i] != 1) {
            output->dims[j] = input[0]->dims[i];
            j++;
        }
    }
}
