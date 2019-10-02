#include <string.h>

#include <DSPLib.h>

#ifdef __MSP430__
#include <driverlib.h>
#define USE_DMA 1
#else
#define USE_DMA 0
#endif

#include "ops.h"

#ifdef __MSP430__
#pragma DATA_SECTION(lea_buffer_input, ".leaRAM")
#pragma DATA_SECTION(lea_buffer_filter, ".leaRAM")
#pragma DATA_SECTION(lea_buffer_another, ".leaRAM")
#pragma DATA_SECTION(lea_buffer_temp, ".leaRAM")
#pragma DATA_SECTION(iq31_mac_result, ".leaRAM")
#endif
int16_t lea_buffer_input[256], lea_buffer_filter[256], lea_buffer_another[256], lea_buffer_temp[64];
int32_t iq31_mac_result;

static inline void my_memcpy(void *dest, const void *src, size_t n) {
#if !USE_DMA
    memcpy(dest, src, n);
#else
    DMA0SA = (void (*)( )) src;
    DMA0DA = (void (*)( )) dest;

    DMA0SZ = n >> 1;  /* DMAxSZ is in words (2 bytes) */

    DMA0CTL = DMA_TRANSFER_BLOCK + DMASRCINCR_3 + DMADSTINCR_3 + DMAEN;

    DMA0CTL |= DMAREQ;
#endif
}

uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output) {
    my_printf("Conv!" NEWLINE);

    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
    /* input: N x H x W x C, filter: M x kH x kW x C*/
    if (conv_input->bitwidth_and_flags >> 1 != 16 || conv_filter->bitwidth_and_flags >> 1 != 16) {
        my_printf("Error: incorrect bitwidth." NEWLINE);
        return 1;
    }
    /* Cannot use C as a variable name here as C is a macro on MSP430 :( */
    const uint16_t H = conv_input->dims[1], W = conv_input->dims[2],
                   input_N = conv_filter->dims[0],
                   kH = conv_filter->dims[1], kW = conv_filter->dims[2],
                   CHANNEL = conv_filter->dims[3];
    /* TODO: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
    output->params_len = (uint16_t)(input_N * H * W * 2);
    output->bitwidth_and_flags = 16 << 1 | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = H;
    output->dims[2] = W;
    output->dims[3] = input_N;

    /* MSP430 LEA requires length to be even */
    msp_mac_q15_params params = { .length = (uint16_t)(CHANNEL * kH * kW / 2 * 2) };
    uint8_t truncated = (params.length != CHANNEL * kH * kW);
    uint16_t buffer_size = (uint16_t)(sizeof(uint16_t) * params.length);
    if (buffer_size > sizeof(lea_buffer_filter)) {
        my_printf("Error: buffer too small." NEWLINE);
        return 1;
    }
    int16_t *output_data = get_q15_param(output, 0);
    for (uint16_t conv_idx = 0; conv_idx < input_N; conv_idx++) {
        //my_printf("conv_idx = %d" NEWLINE, conv_idx);
        /* copy filter data */
        my_memcpy(lea_buffer_filter,
                  get_q15_param(conv_filter, (size_t)(conv_idx * CHANNEL * kH * kW)),
                  buffer_size);
        for (uint16_t output_h = 0; output_h < H; output_h++) {
            for (uint16_t output_w = 0; output_w < W; output_w++) {
                /* copy input data, row by row */
                int16_t *input_addr = get_q15_param(conv_input, (size_t)((output_h * W + output_w) * CHANNEL));
                for (uint16_t h = 0; h < kH; h++) {
                    size_t size = (size_t)(kW * CHANNEL);
                    if (truncated && h == kH - 1) {
                        size--;
                    }
                    /* TODO: handle padding */
                    my_memcpy(lea_buffer_input + h * kW * CHANNEL,  // dest
                              input_addr + h * W * CHANNEL,  // src
                              size * sizeof(uint16_t));  // size
                }
                msp_status status = msp_mac_q15(&params, lea_buffer_input, lea_buffer_filter, &iq31_mac_result);
                msp_checkStatus(status);
                if (truncated) {
#ifndef NDEBUG
                    // my_printf("Adding truncated product back" NEWLINE);
#endif
                    uint16_t last_idx = (uint16_t)(kH * kW - 1);
                    iq31_mac_result += (*get_q15_param(conv_input, last_idx)) * (*get_q15_param(conv_filter, last_idx)) * 2;
                }
#ifndef NDEBUG
                my_printf("%f ", (float)iq31_mac_result / 2147483648.0f);
#endif
                int16_t q15_mac_result = iq31_to_q15(&iq31_mac_result);
                q15_mac_result = (int16_t)(q15_mac_result + *get_q15_param(bias, conv_idx));
                output_data[conv_idx * H * W + output_h * W + output_w] = q15_mac_result;
            }
#ifndef NDEBUG
            my_printf(NEWLINE);
#endif
        }
    }

    return 0;
}

uint8_t handle_maxpool(ParameterInfo *input[], ParameterInfo *output) {
    my_printf("MaxPool!" NEWLINE);
    /* TODO: add flags; assume stripe=2, no padding for now */
    const uint16_t stride = 2; // for less type conversions
    ParameterInfo *data = input[0];
    output->params_len = data->params_len / (uint16_t)(stride * stride);
    output->bitwidth_and_flags = data->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = data->dims[1];
    output->dims[2] = data->dims[2] / stride;
    output->dims[3] = data->dims[3] / stride;
    const uint16_t channel = data->dims[1], H = data->dims[2], W = data->dims[3];
    msp_max_q15_params params = { .length = 4 };
    int16_t max_val;
    uint16_t index;
    int16_t *lea_buffer_maxpool = lea_buffer_input;
    for (uint16_t i = 0; i < channel; i++) {
        for (uint16_t j = 0; j < H; j = (uint16_t)(j + stride)) {
            for (uint16_t k = 0; k < W; k = (uint16_t)(k + stride)) {
                lea_buffer_maxpool[0] = *get_q15_param(data, (size_t)(i * H * W + j     * W + k    ));
                lea_buffer_maxpool[1] = *get_q15_param(data, (size_t)(i * H * W + j     * W + (k+1)));
                lea_buffer_maxpool[2] = *get_q15_param(data, (size_t)(i * H * W + (j+1) * W + k    ));
                lea_buffer_maxpool[3] = *get_q15_param(data, (size_t)(i * H * W + (j+1) * W + (k+1)));
                msp_status status = msp_max_q15(&params, lea_buffer_maxpool, &max_val, &index);
                msp_checkStatus(status);
                *get_q15_param(output, (size_t)(i * H * W + j * W + k)) = max_val;
            }
        }
    }
    return 0;
}

uint8_t handle_add(ParameterInfo *input[], ParameterInfo *output) {
    /* Add: Y = X + W */
    my_printf("Add!" NEWLINE);
    if (input[0]->bitwidth_and_flags >> 1 != 16 || input[1]->bitwidth_and_flags >> 1 != 16) {
        my_printf("Error: unsupported bitwidth" NEWLINE);
        return 1;
    }
    ParameterInfo *A = input[0], *B = input[1];
    output->params_len = input[0]->params_len;
    output->bitwidth_and_flags = input[0]->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = A->dims[1];

    msp_add_q15_params params = { .length = A->dims[1] };

    int16_t *lea_buffer_A = lea_buffer_input,
            *lea_buffer_B = lea_buffer_another;
    my_memcpy(lea_buffer_A, get_q15_param(A, 0), output->params_len);
    my_memcpy(lea_buffer_B, get_q15_param(B, 0), output->params_len);
    msp_status status = msp_add_q15(&params, lea_buffer_A, lea_buffer_B, lea_buffer_A);
    msp_checkStatus(status);

    my_memcpy(get_q15_param(output, 0), lea_buffer_A, output->params_len);

    return 0;
}

uint8_t handle_matmul(ParameterInfo *input[], ParameterInfo *output) {
    ParameterInfo *A = input[0], *B = input[1];

    my_printf("MatMul! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    uint16_t output_len = (uint16_t)(A->dims[0] * B->dims[1]);
    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->params_len = (uint16_t)(output_len * 2);
    output->bitwidth_and_flags = 16 << 1 | FLAG_INTERMEDIATE_VALUES;

    if (A->dims[0] * A->dims[1] > 256) {
        my_printf("Matrix A too large!" NEWLINE);
        return 1;
    }

    int16_t *lea_buffer_A = lea_buffer_filter,
            *lea_buffer_B = lea_buffer_another,
            *lea_buffer_matmul = lea_buffer_input;
    my_memcpy(lea_buffer_A, get_q15_param(A, 0), (uint16_t)(A->dims[0] * A->dims[1]));

    /* LEA wants addresses to be 4-aligned */
    uint16_t step = (uint16_t)((256 / B->dims[1]) / 4 * 4);
    for (uint16_t i = 0; i < B->dims[0]; i = (uint16_t)(i + step)) {
        msp_matrix_mpy_q15_params params;
        uint16_t current_width = (uint16_t)MIN_VAL(step, B->dims[0] - i);
        params.srcARows = A->dims[0];
        params.srcACols = current_width;
        params.srcBRows = current_width;
        params.srcBCols = B->dims[1];

        my_memcpy(lea_buffer_B, get_q15_param(B, (uint16_t)(i * B->dims[1])), (uint16_t)(current_width * B->dims[1]));
        msp_status status = msp_matrix_mpy_q15(
            &params,
            lea_buffer_A + A->dims[0] * i,
            lea_buffer_B,
            lea_buffer_temp);
        msp_checkStatus(status);

        msp_add_q15_params params2 = { .length = output_len };
        status = msp_add_q15(&params2, lea_buffer_matmul, lea_buffer_temp, lea_buffer_matmul);
        msp_checkStatus(status);
    }
    my_memcpy(get_q15_param(output, 0), lea_buffer_matmul, output->params_len);

    return 0;
}

uint8_t handle_relu(ParameterInfo *input[], ParameterInfo *output) {
    my_printf("ReLu!" NEWLINE);
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
            my_printf("Error: unsupported bitwidth for ReLu." NEWLINE);
        }
    }
    return 0;
}

uint8_t handle_reshape(ParameterInfo *input[], ParameterInfo *output) {
    my_printf("Reshape!" NEWLINE);
    ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    if (shape->bitwidth_and_flags >> 1 != 64) {
        my_printf("Error: unsupported shape format." NEWLINE);
        return 1;
    }
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
    }
    return 0;
}

uint8_t handle_squeeze(ParameterInfo *input[], ParameterInfo *output) {
    my_printf("Squeeze!" NEWLINE);
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
