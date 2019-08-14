#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <DSPLib.h>

#include "ops.h"

uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output) {
    printf("Conv!" NEWLINE);

    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
    /* input: N x C x H x W, filter: M x C x kH x kW */
    if (conv_input->bitwidth_and_flags >> 1 != 16 || conv_filter->bitwidth_and_flags >> 1 != 16) {
        printf("Error: incorrect bitwidth." NEWLINE);
        return 1;
    }
    uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
             output_C = conv_filter->dims[0], // output_C = input_N
             kH = conv_filter->dims[2], kW = conv_filter->dims[3],
             C = conv_filter->dims[1];
    /* TODO: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
    output->params_len = (uint16_t)(output_C * H * W * 2);
    output->bitwidth_and_flags = 16 << 1 | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = output_C;
    output->dims[2] = H;
    output->dims[3] = W;

    /* MSP430 LEA requires length to be even */
    msp_mac_q15_params params = { .length = (uint16_t)(kH * kW / 2 * 2) };
    uint8_t truncated = (params.length != kH * kW);
    uint16_t buffer_size = (uint16_t)(sizeof(uint16_t) * params.length);
    int16_t *lea_buffer_input = malloc(buffer_size),
            *lea_buffer_filter = malloc(buffer_size);
    int16_t *output_data = get_q15_param(output, 0);
    for (uint16_t conv_idx = 0; conv_idx < conv_filter->dims[0]; conv_idx++) {
        for (uint16_t channel = 0; channel < C; channel++) {
            /* copy filter data */
            memcpy(lea_buffer_filter,
                   get_q15_param(conv_filter, (size_t)((conv_idx * C + channel) * params.length)),
                   buffer_size);
            for (uint16_t output_h = 0; output_h < H; output_h++) {
                for (uint16_t output_w = 0; output_w < W; output_w++) {
                    /* copy input data, row by row */
                    for (uint16_t h = 0; h < kH; h++) {
                        size_t size = kW;
                        if (truncated && h == kH - 1) {
                            size--;
                        }
                        memcpy(lea_buffer_input + h * kW,  /* dest */
                               get_q15_param(conv_input, (size_t)(output_h * W + output_w)),  /* src */
                               size * sizeof(uint16_t));  /* size */
                    }
                    int32_t iq31_mac_result;
                    msp_status status = msp_mac_q15(&params, lea_buffer_input, lea_buffer_filter, &iq31_mac_result);
                    msp_checkStatus(status);
                    if (truncated) {
#ifndef NDEBUG
                        // printf("Adding truncated product back" NEWLINE);
#endif
                        uint16_t last_idx = (uint16_t)(kH * kW - 1);
                        iq31_mac_result += (*get_q15_param(conv_input, last_idx)) * (*get_q15_param(conv_filter, last_idx)) * 2;
                    }
#ifndef NDEBUG
                    printf("%f ", (float)iq31_mac_result / 2147483648.0f);
#endif
                    int16_t q15_mac_result = iq31_to_q15(&iq31_mac_result);
                    q15_mac_result = (int16_t)(q15_mac_result + *get_q15_param(bias, conv_idx));
                    output_data[conv_idx * H * W + output_h * W + output_w] = q15_mac_result;
                }
            }
#ifndef NDEBUG
            printf(NEWLINE);
#endif
        }
    }

    free(lea_buffer_input);
    free(lea_buffer_filter);
    return 0;
}

uint8_t handle_maxpool(ParameterInfo *input[], ParameterInfo *output) {
    printf("MaxPool!" NEWLINE);
    /* TODO: add flags; assume stripe=2, no padding for now */
    const uint16_t stride = 2; // for less type conversions
    ParameterInfo *data = input[0];
    output->params_len = data->params_len / (uint16_t)(stride * stride);
    output->bitwidth_and_flags = data->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = data->dims[1];
    output->dims[2] = data->dims[2] / stride;
    output->dims[3] = data->dims[3] / stride;
    int16_t lea_buffer[4];
    uint16_t C = data->dims[1], H = data->dims[2], W = data->dims[3];
    msp_max_q15_params params = { .length = 4 };
    int16_t max_val;
    uint16_t index;
    for (uint16_t i = 0; i < C; i++) {
        for (uint16_t j = 0; j < H; j = (uint16_t)(j + stride)) {
            for (uint16_t k = 0; k < W; k = (uint16_t)(k + stride)) {
                lea_buffer[0] = *get_q15_param(data, (size_t)(i * H * W + j     * W + k    ));
                lea_buffer[1] = *get_q15_param(data, (size_t)(i * H * W + j     * W + (k+1)));
                lea_buffer[2] = *get_q15_param(data, (size_t)(i * H * W + (j+1) * W + k    ));
                lea_buffer[3] = *get_q15_param(data, (size_t)(i * H * W + (j+1) * W + (k+1)));
                msp_status status = msp_max_q15(&params, lea_buffer, &max_val, &index);
                msp_checkStatus(status);
                *get_q15_param(output, (size_t)(i * H * W + j * W + k)) = max_val;
            }
        }
    }
    return 0;
}

uint8_t handle_add(ParameterInfo *input[], ParameterInfo *output) {
    /* Add: Y = X + W */
    printf("Add!" NEWLINE);
    if (input[0]->bitwidth_and_flags >> 1 != input[1]->bitwidth_and_flags >> 1) {
        printf("Error: mismatched bitwidth" NEWLINE);
        return 1;
    }
    output->params_len = input[0]->params_len;
    output->bitwidth_and_flags = input[0]->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
    /* TODO */
    return 1;
}

uint8_t handle_matmul(ParameterInfo *input[], ParameterInfo *output) {
    printf("MatMul!" NEWLINE);
    /* TODO */
    (void)input;
    (void)output;
    return 1;
}

uint8_t handle_relu(ParameterInfo *input[], ParameterInfo *output) {
    printf("ReLu!" NEWLINE);
    ParameterInfo *X = input[0];
    memcpy(output, X, sizeof(ParameterInfo));
    /* TODO: use LEA? */
    uint16_t bitwidth = X->bitwidth_and_flags >> 1;
    for (uint32_t i = 0; i < X->params_len / (bitwidth / 8); i++) {
        if (bitwidth == 16) {
            int16_t *ptr = get_q15_param(X, i);
            if (*ptr < 0) {
                *ptr = 0;
            }
        } else {
            printf("Error: unsupported bitwidth for ReLu." NEWLINE);
        }
    }
    return 0;
}

uint8_t handle_reshape(ParameterInfo *input[], ParameterInfo *output) {
    (void)input;
    (void)output;
    printf("Reshape!" NEWLINE);
    ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    if (shape->bitwidth_and_flags >> 1 != 64) {
        printf("Error: unsupported shape format." NEWLINE);
        return 1;
    }
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
    }
    return 0;
}

uint8_t handle_squeeze(ParameterInfo *input[], ParameterInfo *output) {
    (void)input;
    (void)output;
    printf("Squeeze!" NEWLINE);
    ParameterInfo *data = input[0];
    /* TODO: add flags; assume squeeze all one-size axes */
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    for (uint8_t i = 0, j = 0; i < 4; i++) {
        if (input[0]->dims[i] != 1) {
            output->dims[j] = input[0]->dims[i];
            j++;
        }
    }
    return 0;
}
