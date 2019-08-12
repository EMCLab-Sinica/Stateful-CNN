#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <DSPLib.h>

#include "ops.h"

uint8_t handle_conv(ParameterInfo *input[2], ParameterInfo *output) {
    printf("Conv!" NEWLINE);

    ParameterInfo *conv_input = input[0], *conv_filter = input[1];
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
    output->params_len = (uint16_t)(output_C * H * W * 4); /* 4 bytes as IQ31 values are stored */
    output->bitwidth_and_flags = 32 << 1 | FLAG_INTERMEDIATE_VALUES;

    /* MSP430 LEA requires length to be even */
    msp_mac_q15_params params = { .length = (uint16_t)(kH * kW / 2 * 2) };
    uint8_t truncated = (params.length != kH * kW);
    uint16_t buffer_size = (uint16_t)(sizeof(uint16_t) * params.length);
    int16_t *lea_buffer_input = malloc(buffer_size),
            *lea_buffer_filter = malloc(buffer_size);
    int32_t *output_data = get_iq31_param(output, 0);
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
                    int32_t mac_result;
                    msp_status status = msp_mac_q15(&params, lea_buffer_input, lea_buffer_filter, &mac_result);
                    msp_checkStatus(status);
                    if (truncated) {
#ifndef NDEBUG
                        // printf("Adding truncated product back" NEWLINE);
#endif
                        uint16_t last_idx = (uint16_t)(kH * kW - 1);
                        mac_result += (*get_q15_param(conv_input, last_idx)) * (*get_q15_param(conv_filter, last_idx)) * 2;
                    }
#ifndef NDEBUG
                    printf("%f ", (float)mac_result / 2147483648.0f);
#endif
                    output_data[conv_idx * H * W + output_h * W + output_w] = mac_result;
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
    /* TODO: add flags; assume stripe=2 for now */
    output->params_len = input[0]->params_len / (2 * 2);
    output->bitwidth_and_flags = input[0]->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
    /* TODO */
    return 1;
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
    /* TODO */
    (void)input;
    (void)output;
    return 1;
}

uint8_t handle_relu(ParameterInfo *input[], ParameterInfo *output) {
    /* TODO */
    (void)input;
    (void)output;
    return 1;
}

uint8_t handle_reshape(ParameterInfo *input[], ParameterInfo *output) {
    (void)input;
    (void)output;
    /* actual reshaping is done by connecting nodes */
    return 0;
}

