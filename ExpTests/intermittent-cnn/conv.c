// disable debug code in DSPLib
//#define MSP_DISABLE_DIAGNOSTICS

#include <DSPLib.h>
#ifdef USE_ARM_CMSIS
#include <arm_math.h>
#endif
#include "cnn_common.h"
#include "debug.h"
#include "op_handlers.h"
#include "ops.h"

#ifdef __MSP430__
#include <FreeRTOS.h>
#endif

#define configCONV_STACK_SIZE 100

// TODO: make these adjustable on runtime
#define TILE_W 1
#define OUTPUT_LEN 50

// to make the code clearer
#ifndef USE_ARM_CMSIS
#define TEMP_FILTER_WIDTH 1
#else
#define TEMP_FILTER_WIDTH 0
#endif

#define CONV_TASK_FLAG_PROCESSED_FILTERS_BASE 2
typedef struct ConvTaskParams {
    ParameterInfo *conv_input;
    ParameterInfo *conv_filter;
    ParameterInfo *bias;
    ParameterInfo *output;
    uint16_t conv_idx;
    uint16_t output_h;
    uint16_t output_w;
    uint16_t flags;
    uint8_t tile_h;
    uint16_t dest_offset;
    uint8_t filter_limit;
    uint8_t truncated;
    uint16_t OUTPUT_CHANNEL;
    uint16_t W_by_OUTPUT_CHANNEL;
    uint16_t H_by_OUTPUT_CHANNEL;
    uint16_t state_bit;
    msp_matrix_mpy_q15_params matrix_mpy_params;
    int16_t *filter_buffer_addr;
    int16_t cached_filter_idx;
    uint8_t pending_filters[NUM_FILTERS];
    uint8_t pending_filter_idx;
} ConvTaskParams;

int16_t * const matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN;

static void convTask(uint8_t offset_h, uint8_t tile_h, ConvTaskParams *conv_params) {
    /* put var declarations first to make the compiler happy */
    int16_t *filter_addr;
    /* Cannot use C as a variable name here as C is a macro on MSP430 :( */
    uint16_t kH = conv_params->conv_filter->dims[1];
    int16_t filter_offset = kH * conv_params->dest_offset;
    msp_matrix_mpy_q15_params *p_matrix_mpy_params = &(conv_params->matrix_mpy_params);

    /* copy filter data */
    if (conv_params->cached_filter_idx != conv_params->conv_idx) {
        conv_params->filter_buffer_addr = matrix_mpy_results - filter_offset * (conv_params->filter_limit + TEMP_FILTER_WIDTH);
#ifndef USE_ARM_CMSIS
        int16_t *filter_tmp = matrix_mpy_results - filter_offset; // before transpose

        for (uint8_t idx = 0; idx < conv_params->filter_limit; idx++) {
            filter_addr = get_q15_param(
                conv_params->conv_filter,
                (conv_params->conv_idx + idx) * filter_offset);
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->conv_idx + idx);
            uint16_t buffer_size = sizeof(int16_t) * filter_offset;
            my_memcpy(filter_tmp,
                      filter_addr,
                      buffer_size);

            msp_interleave_q15_params params;
            params.length = p_matrix_mpy_params->srcBRows;
            params.numChannels = conv_params->filter_limit;
            params.channel = idx;
            msp_status status = msp_interleave_q15(
                &params,
                filter_tmp, /* src */
                conv_params->filter_buffer_addr /* dst */
            );
            msp_checkStatus(status);
        }
#else
        filter_addr = get_q15_param(
            conv_params->conv_filter,
            conv_params->conv_idx * filter_offset);
        uint16_t buffer_size = sizeof(int16_t) * filter_offset * conv_params->filter_limit;
        my_memcpy(conv_params->filter_buffer_addr, filter_addr, buffer_size);
#endif
        conv_params->cached_filter_idx = conv_params->conv_idx;
    }

    int16_t *filter_buffer_addr = conv_params->filter_buffer_addr;

    my_printf_debug("conv_params->output_h = %d" NEWLINE, conv_params->output_h + offset_h);

    int16_t *input_buffer_addr = lea_buffer + offset_h * conv_params->dest_offset;

    /* XXX: assume stride=1 */

    // XXX: LEA doc requires all matrix dimensions to be even, while LEA
    // appears to still give correct results when srcARows is odd
    // srcBCols should really be even, though
    // http://e2e.ti.com/support/microcontrollers/msp430/f/166/t/716353?MSP430FR5992-MSP-DSPLib-msp-matrix-mpy-q15
    p_matrix_mpy_params->srcARows = (tile_h - offset_h + kH - 1) / kH;

#ifdef WITH_PROGRESS_EMBEDDING
    if (!conv_params->state_bit) {
        int16_t *indicator_addr = input_buffer_addr + filter_offset - (conv_params->truncated ? 2 : 1);
        for (uint8_t i = 0; i < p_matrix_mpy_params->srcARows; i++) {
            *indicator_addr = -32768;
            indicator_addr += filter_offset;
        }
    }
#endif

#ifndef USE_ARM_CMSIS
    msp_status status = msp_matrix_mpy_q15(
        &matrix_mpy_params,
        input_buffer_addr,
        filter_buffer_addr,
        matrix_mpy_results
    );
    msp_checkStatus(status);
#else
    arm_matrix_instance_q15 A, B, C;
    arm_mat_init_q15(&A, p_matrix_mpy_params->srcARows, p_matrix_mpy_params->srcACols, input_buffer_addr);
    arm_mat_init_q15(&B, p_matrix_mpy_params->srcBRows, p_matrix_mpy_params->srcBCols, filter_buffer_addr);
    arm_mat_init_q15(&C, p_matrix_mpy_params->srcARows, p_matrix_mpy_params->srcBCols, matrix_mpy_results);
    arm_status status = arm_mat_mult_fast_q15(&A, &B, &C, NULL);
    if (status != ARM_MATH_SUCCESS) {
        ERROR_OCCURRED();
    }
#endif

    /* START dump data */
    my_printf_debug("conv_idx=%d ", conv_params->conv_idx);
    my_printf_debug("output_h=%d ", conv_params->output_h + offset_h);
    my_printf_debug("output_w=%d" NEWLINE, conv_params->output_w);

    my_printf_debug("input_buffer_addr = lea_buffer + %d" NEWLINE, (int)(input_buffer_addr - lea_buffer));
    my_printf_debug("input" NEWLINE);
    dump_matrix2(input_buffer_addr, p_matrix_mpy_params->srcARows, p_matrix_mpy_params->srcACols);
    my_printf_debug("filter_buffer_addr = lea_buffer + LEA_BUFFER_SIZE - %d" NEWLINE, (int)(lea_buffer + LEA_BUFFER_SIZE - filter_buffer_addr));
    my_printf_debug("filter" NEWLINE);
#ifndef USE_ARM_CMSIS
    dump_matrix2(filter_buffer_addr, p_matrix_mpy_params->srcBRows, p_matrix_mpy_params->srcBCols);
#else
    dump_matrix2(filter_buffer_addr, p_matrix_mpy_params->srcBCols, p_matrix_mpy_params->srcBRows);
#endif

    my_printf_debug("matrix_mpy_results" NEWLINE);
    dump_matrix2(matrix_mpy_results, p_matrix_mpy_params->srcARows, p_matrix_mpy_params->srcBCols);
    my_printf_debug(NEWLINE);
    /* END dump data */

#ifdef WITH_PROGRESS_EMBEDDING
    if (!conv_params->state_bit) {
        int16_t *indicator_addr = input_buffer_addr + filter_offset - (conv_params->truncated ? 2 : 1);
        for (uint8_t i = 0; i < p_matrix_mpy_params->srcARows; i++) {
            *indicator_addr = 0;
            indicator_addr += filter_offset;
        }
    }
#endif

#if NVM_BYTE_ADDRESSABLE
    int16_t *output_data = get_q15_param(conv_params->output,
            (conv_params->output_h + offset_h) * conv_params->W_by_OUTPUT_CHANNEL +
            conv_params->output_w * conv_params->OUTPUT_CHANNEL +
            conv_params->conv_idx);
#else
    int16_t *output_data = get_q15_param(conv_params->output,
            conv_params->output_w * conv_params->H_by_OUTPUT_CHANNEL +
            (conv_params->output_h + offset_h) * conv_params->OUTPUT_CHANNEL +
            conv_params->conv_idx);
#endif
    int16_t *result_addr = matrix_mpy_results;
    // XXX: use DMA makes the whole loop slower as calling DMA for a few numbers brings more overhead than benefits
    uint8_t n_filters = MIN_VAL(conv_params->filter_limit, conv_params->OUTPUT_CHANNEL - conv_params->conv_idx);
    for (uint8_t idx = 0; idx < p_matrix_mpy_params->srcARows; idx++) {
        my_printf_debug("output_data offset = %d" NEWLINE, (uint16_t)(output_data - get_q15_param(conv_params->output, 0)));
        for (uint8_t idx2 = 0; idx2 < n_filters; idx2++) {
#if !defined(MY_NDEBUG) && defined(WITH_PROGRESS_EMBEDDING)
            if (!conv_params->state_bit && *result_addr < 0x4000 && *result_addr >= -0x4000) {
                ERROR_OCCURRED();
            }
#endif
            output_data[idx2] = *result_addr;
            result_addr++;
        }
#if NVM_BYTE_ADDRESSABLE
        output_data += kH * conv_params->W_by_OUTPUT_CHANNEL;
#else
        output_data += kH * conv_params->OUTPUT_CHANNEL;
#endif
    }
}

static inline void schedule_tile(uint16_t idx, uint16_t n_conv, uint16_t output_h, uint16_t output_w, uint8_t tile_h, uint8_t tile_w, uint16_t W, ConvTaskParams *conv_params) {
    conv_params->conv_idx = idx;
    conv_params->tile_h = tile_h;
    uint16_t kH = conv_params->conv_filter->dims[1];
    msp_matrix_mpy_q15_params *p_matrix_mpy_params = &(conv_params->matrix_mpy_params);

    p_matrix_mpy_params->srcACols = p_matrix_mpy_params->srcBRows = kH * conv_params->dest_offset;
    p_matrix_mpy_params->srcBCols = MIN_VAL(conv_params->filter_limit, n_conv - idx);
    if ((p_matrix_mpy_params->srcACols & 1) || (p_matrix_mpy_params->srcBCols & 1)) {
        ERROR_OCCURRED();
    }
    for (uint8_t i = 0; i < MIN_VAL(tile_w, W - output_w); i++) {
        for (uint8_t j = 0; j < kH; j++) {
            conv_params->output_h = output_h;
            conv_params->output_w = output_w + i;
            convTask(j, tile_h, conv_params);
        }
    }
}

static inline void handle_conv_inner_loop(uint16_t n_conv, uint16_t output_h, uint16_t output_w, uint8_t tile_h, uint8_t tile_w, uint16_t H, uint16_t W, ConvTaskParams *conv_params) {
    uint16_t kH = conv_params->conv_filter->dims[1],
             CHANNEL = conv_params->conv_filter->dims[3];
    int8_t field_size = (int8_t)((kH - 1) / 2);

    /* copy input data, row by row */

    int16_t *input_addr = get_q15_param(
        conv_params->conv_input,
        CHANNEL * (output_h * W + output_w)
    );

    /* int32_t instead of int16_t as TI's compiler cannot handle negative
     * offsets correctly. The expression `input_addr + (int16_t)(-2)` is
     * compiled as:
     * 1. -2 is represented as 0x00FFFE (general registers are 24-bit long).
     *    Assume this value is stored in R11.
     * 2. RLAM.A #1,R11  # multiply by 2 to transform the offset for int16_t
     *    to the difference of addresses.
     * In step 2, R11 becomes 0x01FFFC, while it should be -4, or 0x00FFFC,
     * and thus the resultant address is offset by 0x10000.
     */
    int32_t w_start = int16_max(-field_size,    -output_w),
            w_end   = int16_min( field_size, W-1-output_w);
    int16_t *src = NULL,
            *dest;
    int16_t src_offset = W * CHANNEL;
    // two additional filters for values before transpose
    uint16_t inputs_len = MIN_VAL(
        LEA_BUFFER_SIZE - OUTPUT_LEN - (conv_params->filter_limit + TEMP_FILTER_WIDTH) * kH * conv_params->dest_offset,
        (H + kH - 1) * conv_params->dest_offset
    );

    dest = lea_buffer;

    H = conv_params->conv_input->dims[1];
    int32_t h_start = int16_max(        -field_size,    -output_h),
            h_end =   int16_min(tile_h-1+field_size, H-1-output_h);

    my_printf_debug("Reinitialize input buffer" NEWLINE "inputs_len = %d" NEWLINE, inputs_len);

#ifndef USE_ARM_CMSIS
    msp_fill_q15_params fill_params = {
        .length = inputs_len,
        .value = 0,
    };
    msp_status status = msp_fill_q15(&fill_params, lea_buffer);
    msp_checkStatus(status);
#else
    arm_fill_q15(0, lea_buffer, inputs_len);
#endif

    dest += (h_start + field_size) * conv_params->dest_offset + (w_start + field_size) * CHANNEL;

    my_printf_debug("h_start=%d ", h_start);
    my_printf_debug("h_end=%d" NEWLINE, h_end);

    size_t size = (w_end-w_start+1) * CHANNEL;
    src = input_addr + (h_start * W + w_start) * CHANNEL;
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    (int)(dest - lea_buffer));
    for (int32_t h = h_start; h <= h_end; h++) {
        my_memcpy(dest, src, size * sizeof(int16_t));
#ifdef WITH_PROGRESS_EMBEDDING
        if (conv_params->state_bit) {
            for (uint16_t idx = 0; idx < size; idx++) {
                dest[idx] += 0x8000;
            }
        }
#endif
        src += src_offset;
        dest += conv_params->dest_offset;
    }
    if (conv_params->flags & CONV_BIAS_MERGED) {
        for (uint8_t idx = 0; idx <= h_end - h_start + 2 * field_size; idx++) {
            uint16_t offset = (idx + 1) * conv_params->dest_offset - (conv_params->truncated ? 2 : 1);
#ifdef WITH_PROGRESS_EMBEDDING
            offset--;
#endif
            lea_buffer[offset] = 32767; // 32767 is _Q15(1.0)
        }
    }

    for (uint8_t idx = 0; idx < n_conv; idx += conv_params->filter_limit) {
        if (conv_params->cached_filter_idx == idx) {
            schedule_tile(idx, n_conv, output_h, output_w, tile_h, tile_w, W, conv_params);
        } else {
            my_printf_debug("Filters starting from %d are not cached, append them to the pending list" NEWLINE, idx);
            conv_params->pending_filters[conv_params->pending_filter_idx] = idx;
            conv_params->pending_filter_idx++;
        }
    }
    for (uint8_t idx = 0; idx < conv_params->pending_filter_idx; idx++) {
        uint8_t filter_idx = conv_params->pending_filters[idx];
        schedule_tile(filter_idx, n_conv, output_h, output_w, tile_h, tile_w, W, conv_params);
        my_printf_debug("Mark filter %d as processed" NEWLINE, filter_idx);
    }
    conv_params->pending_filter_idx = 0;
}

void handle_conv(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
    my_printf_debug("Conv!" NEWLINE);

    setOutputValue(1);

    if (conv_input->bitwidth != 16 || conv_filter->bitwidth != 16) {
        // incorrect bitwidth
        ERROR_OCCURRED();
    }
    /* original: input: N x C x H x W, filter: M x C x kW x kW
     * remapped: input: N x H x W x C, filter: M x kH x kW x C */
    const uint16_t H = conv_input->dims[1], W = conv_input->dims[2],
                   input_N = conv_filter->dims[0];
    /* XXX: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
    output->params_len = (uint16_t)(input_N * H * W * 2);
    output->bitwidth = 16;
    output->slot = get_next_slot(conv_input);
    output->dims[0] = 1;
    output->dims[1] = H;
    output->dims[2] = W;
    output->dims[3] = input_N;
#if !NVM_BYTE_ADDRESSABLE
    output->flags |= TRANSPOSED;
#endif

    ConvTaskParams conv_params_obj;
    ConvTaskParams *conv_params = &conv_params_obj;
    conv_params->conv_input = conv_input;
    conv_params->conv_filter = conv_filter;
    conv_params->bias = bias;
    conv_params->output = output;
    conv_params->flags = flags;
    conv_params->filter_buffer_addr = NULL;
    conv_params->cached_filter_idx = -1;

    // TODO: determine these values automatically
    uint8_t tile_h = 1; // fallback value
    if (H == 14) {
#ifdef CYPRESS
        tile_h = 14;
#else
        tile_h = 7;
#endif
    } else if (H == 28) {
        tile_h = 28;
    }

    uint16_t kH = conv_filter->dims[1],
             kW = conv_filter->dims[2],
             CHANNEL = conv_filter->dims[3];
    conv_params->pending_filter_idx = 0;
    conv_params->dest_offset = kW * CHANNEL;
    conv_params->OUTPUT_CHANNEL = conv_filter->dims[0];
    conv_params->W_by_OUTPUT_CHANNEL = W * conv_params->OUTPUT_CHANNEL;
    conv_params->H_by_OUTPUT_CHANNEL = H * conv_params->OUTPUT_CHANNEL;
#ifdef WITH_PROGRESS_EMBEDDING
    conv_params->state_bit = model->state_bit;
    if (conv_params->state_bit) {
        model->state_bit = 0;
    } else {
        model->state_bit = 1;
    }
#endif

    if (conv_params->flags & CONV_BIAS_MERGED) {
        conv_params->dest_offset++;
    }
#ifdef WITH_PROGRESS_EMBEDDING
    conv_params->dest_offset++;
#endif
    /* MSP430 LEA requires length to be even */
    conv_params->truncated = (conv_params->dest_offset / 2 * 2 != conv_params->dest_offset);
    if (conv_params->truncated) {
        // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
        // also odd, so dummy values are needed between slices to make
        // addresses even.
        // a dummy value for each slice (kW * CHANNEL q15 values)
        conv_params->dest_offset++;
    }

    conv_params->filter_limit = MIN_VAL(
        conv_filter->dims[0],
        // `/ 2 * 2` as LEA requires matrix dimensions to be even
        ((LEA_BUFFER_SIZE - OUTPUT_LEN - conv_params->dest_offset * (kH + tile_h - 1)) / (conv_params->dest_offset * kH) - TEMP_FILTER_WIDTH) / 2 * 2
    );

    my_printf_debug("filter_limit: %d" NEWLINE, conv_params->filter_limit);

    for (uint16_t output_w = 0; output_w < W; output_w += TILE_W) {
        for (uint16_t output_h = 0; output_h < H; output_h += tile_h) {
            handle_conv_inner_loop(input_N, output_h, output_w, tile_h, TILE_W, H, W, conv_params);
        }
    }

    my_printf_debug("handle_conv output" NEWLINE);
    dump_params(output);

    setOutputValue(0);
}
