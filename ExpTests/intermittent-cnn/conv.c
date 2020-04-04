// disable debug code in DSPLib
//#define MSP_DISABLE_DIAGNOSTICS

#include <DSPLib.h>
#ifdef USE_ARM_CMSIS
#include <arm_math.h>
#endif
#include "common.h"
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

int16_t *input_buffer_addr;

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
} ConvTaskParams;

static ConvTaskParams conv_params;
static struct {
    uint16_t dest_offset;
    uint8_t filter_limit;
    uint8_t truncated;
    uint16_t OUTPUT_CHANNEL;
    uint16_t W_by_OUTPUT_CHANNEL;
} global_conv_params;

static msp_matrix_mpy_q15_params matrix_mpy_params;
static int16_t *filter_buffer_addr;
static int16_t cached_filter_idx;
static uint8_t pending_filters[NUM_FILTERS];
static uint8_t pending_filter_idx = 0;

int16_t *matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN;

static void convTask(uint8_t offset_h, uint8_t tile_h) {
    /* put var declarations first to make the compiler happy */
    int16_t *filter_addr;
    /* Cannot use C as a variable name here as C is a macro on MSP430 :( */
    uint16_t kH = conv_params.conv_filter->dims[1];

    /* copy filter data */
    if (cached_filter_idx != conv_params.conv_idx) {
        int16_t filter_offset = kH * global_conv_params.dest_offset;
        filter_buffer_addr = matrix_mpy_results - filter_offset * (global_conv_params.filter_limit + TEMP_FILTER_WIDTH);
#ifndef USE_ARM_CMSIS
        int16_t *filter_tmp = matrix_mpy_results - filter_offset; // before transpose

        for (uint8_t idx = 0; idx < global_conv_params.filter_limit; idx++) {
            filter_addr = get_q15_param(
                conv_params.conv_filter,
                (conv_params.conv_idx + idx) * filter_offset);
            my_printf_debug("Copying filter %d" NEWLINE, conv_params.conv_idx + idx);
            uint16_t buffer_size = sizeof(int16_t) * filter_offset;
            my_memcpy(filter_tmp,
                      filter_addr,
                      buffer_size);

            msp_interleave_q15_params params;
            params.length = matrix_mpy_params.srcBRows;
            params.numChannels = global_conv_params.filter_limit;
            params.channel = idx;
            msp_status status = msp_interleave_q15(
                &params,
                filter_tmp, /* src */
                filter_buffer_addr /* dst */
            );
            msp_checkStatus(status);
        }
#else
        filter_addr = get_q15_param(
            conv_params.conv_filter,
            conv_params.conv_idx * filter_offset);
        uint16_t buffer_size = sizeof(int16_t) * filter_offset * global_conv_params.filter_limit;
        my_memcpy(filter_buffer_addr, filter_addr, buffer_size);
#endif
        cached_filter_idx = conv_params.conv_idx;
    }

    my_printf_debug("conv_params.output_h = %d" NEWLINE, conv_params.output_h + offset_h);
    input_buffer_addr = lea_buffer + offset_h * global_conv_params.dest_offset;

    /* XXX: assume stride=1 */

    // XXX: LEA doc requires all matrix dimensions to be even, while LEA
    // appears to still give correct results when srcARows is odd
    // srcBCols should really be even, though
    // http://e2e.ti.com/support/microcontrollers/msp430/f/166/t/716353?MSP430FR5992-MSP-DSPLib-msp-matrix-mpy-q15
    matrix_mpy_params.srcARows = (tile_h - offset_h + kH - 1) / kH;

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
    arm_mat_init_q15(&A, matrix_mpy_params.srcARows, matrix_mpy_params.srcACols, input_buffer_addr);
    arm_mat_init_q15(&B, matrix_mpy_params.srcBRows, matrix_mpy_params.srcBCols, filter_buffer_addr);
    arm_mat_init_q15(&C, matrix_mpy_params.srcARows, matrix_mpy_params.srcBCols, matrix_mpy_results);
    arm_status status = arm_mat_mult_fast_q15(&A, &B, &C, NULL);
    if (status != ARM_MATH_SUCCESS) {
        ERROR_OCCURRED();
    }
#endif

    /* START dump data */
    my_printf_debug("conv_idx=%d ", conv_params.conv_idx);
    my_printf_debug("output_h=%d ", conv_params.output_h + offset_h);
    my_printf_debug("output_w=%d" NEWLINE, conv_params.output_w);

    my_printf_debug("input_buffer_addr = lea_buffer + %d" NEWLINE, (int)(input_buffer_addr - lea_buffer));
    my_printf_debug("input" NEWLINE);
    dump_matrix2(input_buffer_addr, matrix_mpy_params.srcARows, matrix_mpy_params.srcACols);
    my_printf_debug("filter_buffer_addr = lea_buffer + LEA_BUFFER_SIZE - %d" NEWLINE, (int)(lea_buffer + LEA_BUFFER_SIZE - filter_buffer_addr));
    my_printf_debug("filter" NEWLINE);
    dump_matrix2(filter_buffer_addr, matrix_mpy_params.srcBRows, matrix_mpy_params.srcBCols);

    my_printf_debug("matrix_mpy_results" NEWLINE);
    dump_matrix2(matrix_mpy_results, matrix_mpy_params.srcARows, matrix_mpy_params.srcBCols);
    my_printf_debug(NEWLINE);
    /* END dump data */

    int16_t *output_data = get_q15_param(conv_params.output, (conv_params.output_h + offset_h) * global_conv_params.W_by_OUTPUT_CHANNEL + conv_params.output_w * global_conv_params.OUTPUT_CHANNEL + conv_params.conv_idx);
    int16_t *result_addr = matrix_mpy_results;
    // XXX: use DMA makes the whole loop slower as calling DMA for a few numbers brings more overhead than benefits
    uint8_t n_filters = MIN_VAL(global_conv_params.filter_limit, global_conv_params.OUTPUT_CHANNEL - conv_params.conv_idx);
    for (uint8_t idx = 0; idx < matrix_mpy_params.srcARows; idx++) {
        for (uint8_t idx2 = 0; idx2 < n_filters; idx2++) {
            output_data[idx2] = *result_addr;
            result_addr++;
        }
        output_data += kH * global_conv_params.W_by_OUTPUT_CHANNEL;
    }
}

static inline void schedule_tile(uint16_t idx, uint16_t n_conv, uint16_t output_h, uint16_t output_w, uint8_t tile_h, uint8_t tile_w, uint16_t W) {
    conv_params.conv_idx = idx;
    conv_params.tile_h = tile_h;
    uint16_t kH = conv_params.conv_filter->dims[1];
    matrix_mpy_params.srcACols = matrix_mpy_params.srcBRows = kH * global_conv_params.dest_offset;
    matrix_mpy_params.srcBCols = MIN_VAL(global_conv_params.filter_limit, n_conv - idx);
    if ((matrix_mpy_params.srcACols & 1) || (matrix_mpy_params.srcBCols & 1)) {
        ERROR_OCCURRED();
    }
    for (uint8_t i = 0; i < MIN_VAL(tile_w, W - output_w); i++) {
        for (uint8_t j = 0; j < kH; j++) {
            conv_params.output_h = output_h;
            conv_params.output_w = output_w + i;
            convTask(j, tile_h);
        }
    }
}

static inline void handle_conv_inner_loop(uint16_t n_conv, uint16_t output_h, uint16_t output_w, uint8_t tile_h, uint8_t tile_w, uint16_t H, uint16_t W) {
    uint16_t kH = conv_params.conv_filter->dims[1],
             CHANNEL = conv_params.conv_filter->dims[3];
    int8_t field_size = (int8_t)((kH - 1) / 2);

    /* copy input data, row by row */

    int16_t *input_addr = get_q15_param(
        conv_params.conv_input,
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
        LEA_BUFFER_SIZE - OUTPUT_LEN - (global_conv_params.filter_limit + TEMP_FILTER_WIDTH) * kH * global_conv_params.dest_offset,
        (H + kH - 1) * global_conv_params.dest_offset
    );

    dest = lea_buffer;

    H = conv_params.conv_input->dims[1];
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

    dest += (h_start + field_size) * global_conv_params.dest_offset + (w_start + field_size) * CHANNEL;

    my_printf_debug("h_start=%d ", h_start);
    my_printf_debug("h_end=%d" NEWLINE, h_end);

    size_t size = (size_t)((w_end-w_start+1) * CHANNEL * sizeof(uint16_t)); // in bytes
    src = input_addr + (h_start * W + w_start) * CHANNEL;
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    (int)(dest - lea_buffer));
    for (int32_t h = h_start; h <= h_end; h++) {
        my_memcpy(dest, src, size);
        src += src_offset;
        dest += global_conv_params.dest_offset;
    }
    if (conv_params.flags & CONV_BIAS_MERGED) {
        for (uint8_t idx = 0; idx <= h_end - h_start + 2 * field_size; idx++) {
            lea_buffer[(idx + 1) * global_conv_params.dest_offset - (global_conv_params.truncated ? 2 : 1)] = 32767; // 32767 is _Q15(1.0)
        }
    }

    for (uint8_t idx = 0; idx < n_conv; idx += global_conv_params.filter_limit) {
        if (cached_filter_idx == idx) {
            schedule_tile(idx, n_conv, output_h, output_w, tile_h, tile_w, W);
        } else {
            my_printf_debug("Filters starting from %d are not cached, append them to the pending list" NEWLINE, idx);
            pending_filters[pending_filter_idx] = idx;
            pending_filter_idx++;
        }
    }
    for (uint8_t idx = 0; idx < pending_filter_idx; idx++) {
        uint8_t filter_idx = pending_filters[idx];
        schedule_tile(filter_idx, n_conv, output_h, output_w, tile_h, tile_w, W);
        my_printf_debug("Mark filter %d as processed" NEWLINE, filter_idx);
    }
    pending_filter_idx = 0;
}

uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
    my_printf_debug("Conv!" NEWLINE);

    setOutputValue(1);

    if (get_param_bitwidth(conv_input) != 16 || get_param_bitwidth(conv_filter) != 16) {
        // incorrect bitwidth
        ERROR_OCCURRED();
    }
    /* original: input: N x C x H x W, filter: M x C x kW x kW
     * remapped: input: N x H x W x C, filter: M x kH x kW x C */
    const uint16_t H = conv_input->dims[1], W = conv_input->dims[2],
                   input_N = conv_filter->dims[0];
    /* XXX: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
    output->params_len = (uint16_t)(input_N * H * W * 2);
    output->bitwidth_and_flags = 16 << FLAG_SLOTS_WIDTH | get_next_slot(conv_input);
    output->dims[0] = 1;
    output->dims[1] = H;
    output->dims[2] = W;
    output->dims[3] = input_N;

    uint8_t ret = 0;

    conv_params.conv_input = conv_input;
    conv_params.conv_filter = conv_filter;
    conv_params.bias = bias;
    conv_params.output = output;
    conv_params.flags = flags;
    input_buffer_addr = NULL;

    filter_buffer_addr = NULL;
    cached_filter_idx = -1;

    uint8_t tile_h = 1; // fallback value
    if (H == 14) {
        tile_h = 7;
    } else if (H == 28) {
        tile_h = 28;
    }

    uint16_t kH = conv_filter->dims[1],
             kW = conv_filter->dims[2],
             CHANNEL = conv_filter->dims[3];
    global_conv_params.dest_offset = kW * CHANNEL;
    global_conv_params.OUTPUT_CHANNEL = conv_filter->dims[0];
    global_conv_params.W_by_OUTPUT_CHANNEL = W * global_conv_params.OUTPUT_CHANNEL;

    if (conv_params.flags & CONV_BIAS_MERGED) {
        global_conv_params.dest_offset++;
    }
    /* MSP430 LEA requires length to be even */
    global_conv_params.truncated = (global_conv_params.dest_offset / 2 * 2 != global_conv_params.dest_offset);
    if (global_conv_params.truncated) {
        // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
        // also odd, so dummy values are needed between slices to make
        // addresses even.
        // a dummy value for each slice (kW * CHANNEL q15 values)
        global_conv_params.dest_offset++;
    }

    global_conv_params.filter_limit = MIN_VAL(
        conv_filter->dims[0],
        // `/ 2 * 2` as LEA requires matrix dimensions to be even
        ((LEA_BUFFER_SIZE - OUTPUT_LEN - global_conv_params.dest_offset * (kH + tile_h - 1)) / (global_conv_params.dest_offset * kH) - TEMP_FILTER_WIDTH) / 2 * 2
    );

    my_printf_debug("filter_limit: %d" NEWLINE, global_conv_params.filter_limit);

    for (uint16_t output_w = 0; output_w < W; output_w += TILE_W) {
        for (uint16_t output_h = 0; output_h < H; output_h += tile_h) {
            handle_conv_inner_loop(input_N, output_h, output_w, tile_h, TILE_W, H, W);
        }
    }

    my_printf_debug("handle_conv output" NEWLINE);
    dump_params(output);

    setOutputValue(0);

    return ret;
}
