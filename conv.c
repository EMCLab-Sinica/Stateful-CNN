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

#define configCONV_STACK_SIZE 100

// TODO: make these adjustable on runtime
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
    ParameterInfo *conv_bias;
    ParameterInfo *output;

    /* aux vars remaining constant for a conv layer */
    uint16_t H;
    uint16_t W;
    uint16_t kH;
    uint16_t kW;
    uint16_t CHANNEL; // Cannot use C as a variable name here as C is a macro on MSP430 :(
    uint16_t OUTPUT_CHANNEL;
    uint16_t flags;
    uint8_t tile_c_offset;
    uint8_t tile_c_index;
    uint8_t tile_h;
    uint8_t tile_c;
    uint8_t n_tiles_c;
    uint16_t dest_offset;
    uint16_t filter_offset;
    uint16_t filter_limit;
    uint8_t truncated;
    uint16_t state_bit;

    uint16_t conv_idx;
    uint16_t output_h;
    uint16_t output_w;
    msp_matrix_mpy_q15_params matrix_mpy_params;
    int16_t *filter_buffer_addr;
    int16_t cached_filter_idx;
    uint8_t cached_tile_c_offset;
    uint8_t pending_filters[NUM_FILTERS];
    uint8_t pending_filter_idx;
} ConvTaskParams;

int16_t * const matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN;

static void convTask(uint8_t offset_h, ConvTaskParams *conv_params) {
    /* put var declarations first to make the compiler happy */
    int16_t *filter_addr;
    msp_matrix_mpy_q15_params *p_matrix_mpy_params = &(conv_params->matrix_mpy_params);

    /* copy filter data */
    if (conv_params->cached_filter_idx != conv_params->conv_idx || conv_params->cached_tile_c_offset != conv_params->tile_c_offset) {
        conv_params->filter_buffer_addr = matrix_mpy_results - conv_params->filter_offset * (conv_params->filter_limit + TEMP_FILTER_WIDTH);
#ifndef USE_ARM_CMSIS
        int16_t *filter_tmp = matrix_mpy_results - conv_params->filter_offset; // before transpose

        msp_fill_q15_params fill_params = {
            .length = conv_params->filter_offset,
            .value = 0,
        };
        msp_status status = msp_fill_q15(&fill_params, filter_tmp);
        msp_checkStatus(status);

        uint16_t buffer_size = sizeof(int16_t) * conv_params->kW;
        for (uint8_t idx = 0; idx < conv_params->filter_limit; idx++) {
            filter_addr = get_q15_param(
                conv_params->conv_filter,
                (conv_params->conv_idx + idx) * conv_params->kH * conv_params->kW * conv_params->CHANNEL,
                WILL_NOT_WRITE
            );
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->conv_idx + idx);
            *(filter_tmp + conv_params->dest_offset - 1) = *get_q15_param(conv_params->conv_bias, conv_params->conv_idx + idx, WILL_NOT_WRITE) / -conv_params->n_tiles_c;
            for (uint8_t idx2 = 0; idx2 < conv_params->kH; idx2++) {
                for (uint8_t c = 0; c < conv_params->tile_c; c++) {
                    my_memcpy(filter_tmp + idx2 * conv_params->dest_offset + c * conv_params->kW,
                              filter_addr + (c + conv_params->tile_c_offset) * conv_params->kH * conv_params->kW + idx2 * conv_params->kW,
                              buffer_size);
                }
            }
#ifdef WITH_PROGRESS_EMBEDDING
            filter_tmp[conv_params->filter_offset - (conv_params->truncated?2:1)] -= 0x4000;
#endif

            msp_interleave_q15_params params;
            params.length = p_matrix_mpy_params->srcBRows;
            params.numChannels = conv_params->filter_limit;
            params.channel = idx;
            status = msp_interleave_q15(
                &params,
                filter_tmp, /* src */
                conv_params->filter_buffer_addr /* dst */
            );
            msp_checkStatus(status);
        }
#else
        filter_addr = get_q15_param(
            conv_params->conv_filter,
            conv_params->conv_idx * conv_params->filter_offset,
            WILL_NOT_WRITE
        );
        uint16_t buffer_size = sizeof(int16_t) * conv_params->filter_offset * conv_params->filter_limit;
        my_memcpy(conv_params->filter_buffer_addr, filter_addr, buffer_size);
#ifdef WITH_PROGRESS_EMBEDDING
        for (uint8_t idx = 0; idx < conv_params->filter_limit; idx++) {
            uint16_t offset = (idx + 1) * conv_params->filter_offset - (conv_params->truncated?2:1);
            my_printf_debug("offset for the bias with embedded progress = %d" NEWLINE, offset);
            conv_params->filter_buffer_addr[offset] -= 0x4000;
        }
#endif // WITH_PROGRESS_EMBEDDING

#endif // USE_ARM_CMSIS
        conv_params->cached_filter_idx = conv_params->conv_idx;
        conv_params->cached_tile_c_offset = conv_params->tile_c_offset;
    }

    int16_t *filter_buffer_addr = conv_params->filter_buffer_addr;

    my_printf_debug("conv_params->output_h = %d" NEWLINE, conv_params->output_h + offset_h);

    int16_t *input_buffer_addr = lea_buffer + offset_h * conv_params->dest_offset;

    /* XXX: assume stride=1 */

    // XXX: LEA doc requires all matrix dimensions to be even, while LEA
    // appears to still give correct results when srcARows is odd
    // srcBCols should really be even, though
    // http://e2e.ti.com/support/microcontrollers/msp430/f/166/t/716353?MSP430FR5992-MSP-DSPLib-msp-matrix-mpy-q15
    p_matrix_mpy_params->srcARows = (conv_params->tile_h - offset_h + conv_params->kH - 1) / conv_params->kH;

#ifndef USE_ARM_CMSIS
    msp_status status = msp_matrix_mpy_q15(
        p_matrix_mpy_params,
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
#ifndef MY_NDEBUG
    my_printf_debug("conv_idx=");
    for (uint8_t idx = 0; idx < conv_params->filter_limit; idx++) {
        my_printf_debug("%d ", conv_params->conv_idx + idx);
    }
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
#endif
    /* END dump data */

    int16_t *output_baseptr = get_q15_param(conv_params->output, conv_params->tile_c_index * conv_params->OUTPUT_CHANNEL * conv_params->H * conv_params->W, WILL_WRITE);
    int16_t *output_data = output_baseptr +
            conv_params->conv_idx * conv_params->H * conv_params->W +
            (conv_params->output_h + offset_h) * conv_params->W +
            conv_params->output_w;
    int16_t *result_addr = matrix_mpy_results;
    // XXX: use DMA makes the whole loop slower as calling DMA for a few numbers brings more overhead than benefits
    uint8_t n_filters = MIN_VAL(conv_params->filter_limit, conv_params->OUTPUT_CHANNEL - conv_params->conv_idx);
    for (uint8_t idx = 0; idx < p_matrix_mpy_params->srcARows; idx++) {
        my_printf_debug("output_data offset = %d" NEWLINE, (uint16_t)(output_data - output_baseptr));
        for (uint8_t idx2 = 0; idx2 < n_filters; idx2++) {
#if !defined(MY_NDEBUG) && defined(WITH_PROGRESS_EMBEDDING)
            if (!conv_params->state_bit && *result_addr < 0x2000 && *result_addr >= -0x2000) {
                ERROR_OCCURRED();
            }
#endif
            *(output_data + idx2 * conv_params->H * conv_params->W) = *result_addr;
            result_addr++;
        }
        output_data += conv_params->kH * conv_params->W;
    }
}

static inline void schedule_tile(uint16_t idx, ConvTaskParams *conv_params) {
    conv_params->conv_idx = idx;
    msp_matrix_mpy_q15_params *p_matrix_mpy_params = &(conv_params->matrix_mpy_params);

    p_matrix_mpy_params->srcACols = p_matrix_mpy_params->srcBRows = conv_params->filter_offset;
    p_matrix_mpy_params->srcBCols = MIN_VAL(conv_params->filter_limit, conv_params->OUTPUT_CHANNEL - idx);
    if ((p_matrix_mpy_params->srcACols & 1) || (p_matrix_mpy_params->srcBCols & 1)) {
        ERROR_OCCURRED();
    }
    for (uint8_t j = 0; j < conv_params->kH; j++) {
        convTask(j, conv_params);
    }
}

static inline void handle_conv_inner_loop(void *pvParameters) {
    ConvTaskParams *conv_params = (ConvTaskParams*)pvParameters;

    int8_t field_size = (conv_params->kH - 1) / 2;

    /* copy input data, row by row */

    int16_t *input_addr = get_q15_param(
        conv_params->conv_input,
        0,
        WILL_NOT_WRITE
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
    int32_t w_start = int16_max(-field_size,                 -conv_params->output_w),
            w_end   = int16_min( field_size, conv_params->W-1-conv_params->output_w);
    int16_t *dest;
    // TEMP_FILTER_WIDTH additional filters for values before transpose
    uint16_t inputs_len = MIN_VAL(
        LEA_BUFFER_SIZE - OUTPUT_LEN - (conv_params->filter_limit + TEMP_FILTER_WIDTH) * conv_params->kH * conv_params->dest_offset,
        (conv_params->H + conv_params->kH - 1) * conv_params->dest_offset
    );

    dest = lea_buffer;

    int32_t h_start = int16_max(                     -field_size,                 -conv_params->output_h),
            h_end =   int16_min(conv_params->tile_h-1+field_size, conv_params->H-1-conv_params->output_h);

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

    dest += (h_start + field_size) * conv_params->dest_offset;

    my_printf_debug("h_start=%" PRId32 " ", h_start);
    my_printf_debug("h_end=%" PRId32 NEWLINE, h_end);

    size_t size = w_end - w_start + 1;
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    (int)(dest - lea_buffer));
    for (int32_t h = h_start; h <= h_end; h++) {
        for (uint16_t c = 0; c < conv_params->tile_c; c++) {
            my_memcpy(
                dest + c * conv_params->kW + (w_start + field_size),
                input_addr + (c + conv_params->tile_c_offset) * conv_params->H * conv_params->W + (conv_params->output_h + h) * conv_params->W + (conv_params->output_w + w_start),
                size * sizeof(int16_t));
        }
        dest += conv_params->dest_offset;
    }
    uint16_t offset = conv_params->dest_offset - 1;
    while (offset < inputs_len) {
        lea_buffer[offset] = -0x8000; // _Q15(-1.0)
        offset += conv_params->dest_offset;
    }

    for (uint8_t idx = 0; idx < conv_params->OUTPUT_CHANNEL; idx += conv_params->filter_limit) {
        if (conv_params->cached_filter_idx == idx) {
            schedule_tile(idx, conv_params);
        } else {
            my_printf_debug("Filters starting from %d are not cached, append them to the pending list" NEWLINE, idx);
            conv_params->pending_filters[conv_params->pending_filter_idx] = idx;
            conv_params->pending_filter_idx++;
        }
    }
    for (uint8_t idx = 0; idx < conv_params->pending_filter_idx; idx++) {
        uint8_t filter_idx = conv_params->pending_filters[idx];
        schedule_tile(filter_idx, conv_params);
        my_printf_debug("Mark filter %d as processed" NEWLINE, filter_idx);
    }
    conv_params->pending_filter_idx = 0;

    TASK_FINISHED();
}

void handle_conv(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *conv_bias = input[2];
    my_printf_debug("Conv!" NEWLINE);

    setOutputValue(1);

    if (conv_input->bitwidth != 16 || conv_filter->bitwidth != 16) {
        // incorrect bitwidth
        ERROR_OCCURRED();
    }
    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   CHANNEL = conv_filter->dims[1],
                   OUTPUT_CHANNEL = conv_filter->dims[0];

    ConvTaskParams conv_params_obj;
    ConvTaskParams *conv_params = &conv_params_obj;

    // TODO: determine these values automatically
    uint8_t tile_c;
    conv_params->tile_h = 1; // fallback value
    if (H == 14) {
        conv_params->tile_h = 7;
        tile_c = 4;
    } else if (H == 28) {
        conv_params->tile_h = 28;
        tile_c = 1;
    }
    conv_params->n_tiles_c = (CHANNEL + tile_c - 1) / tile_c;
    my_printf_debug("n_tiles_c = %d" NEWLINE, conv_params->n_tiles_c);

    /* XXX: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
    output->bitwidth = 16;
    output->slot = get_next_slot(conv_input);
    output->dims[0] = 1;
    // Although handle_conv requires more memory than params_len, only the first OUTPUT_CHANNEL
    // channels are useful after merging results from tiling
    output->dims[1] = OUTPUT_CHANNEL;
    output->dims[2] = H;
    output->dims[3] = W;
    output->params_len = OUTPUT_CHANNEL * H * W * sizeof(int16_t);

    conv_params->conv_input = conv_input;
    conv_params->conv_filter = conv_filter;
    conv_params->conv_bias = conv_bias;
    conv_params->output = output;
    conv_params->flags = flags;
    conv_params->filter_buffer_addr = NULL;
    conv_params->cached_filter_idx = -1;
    conv_params->H = H;
    conv_params->W = W;

    conv_params->kH = conv_filter->dims[2];
    conv_params->kW = conv_filter->dims[3];
    conv_params->CHANNEL = CHANNEL;
    conv_params->pending_filter_idx = 0;
    conv_params->OUTPUT_CHANNEL = OUTPUT_CHANNEL;
#ifdef WITH_PROGRESS_EMBEDDING
    conv_params->state_bit = model->state_bit;
    // XXX
    model->state_bit = 1;
    if (conv_params->state_bit) {
        int16_t *input_ptr = get_q15_param(conv_input, 0, WILL_WRITE);
        uint32_t len = conv_input->params_len / sizeof(int16_t);
        for (uint16_t idx = 0; idx < len; idx++) {
            *input_ptr -= 0x4000;
            input_ptr++;
        }
    }
#endif

    for (uint8_t tile_c_offset = 0, tile_c_index = 0; tile_c_offset < CHANNEL ; tile_c_offset += tile_c, tile_c_index++) {
        conv_params->tile_c = MIN_VAL(tile_c, CHANNEL - tile_c_offset);
        // +1 for bias
        conv_params->dest_offset = conv_params->kH * conv_params->tile_c + 1;
        /* MSP430 LEA requires length to be even */
        conv_params->truncated = (conv_params->dest_offset / 2 * 2 != conv_params->dest_offset);
        if (conv_params->truncated) {
            // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
            // also odd, so dummy values are needed between slices to make
            // addresses even.
            // a dummy value for each slice (kW * CHANNEL q15 values)
            conv_params->dest_offset++;
        }
        conv_params->filter_offset = conv_params->kH * conv_params->dest_offset;

        conv_params->filter_limit = MIN_VAL(
            conv_params->OUTPUT_CHANNEL,
            // `/ 2 * 2` as LEA requires matrix dimensions to be even
            ((LEA_BUFFER_SIZE - OUTPUT_LEN - conv_params->dest_offset * (conv_params->kH + conv_params->tile_h - 1)) / (conv_params->dest_offset * conv_params->kH) - TEMP_FILTER_WIDTH) / 2 * 2
        );

        my_printf_debug("filter_limit: %d" NEWLINE, conv_params->filter_limit);

        conv_params->tile_c_offset = tile_c_offset;
        conv_params->tile_c_index = tile_c_index;
        for (uint16_t output_w = 0; output_w < W; output_w++) {
            for (uint16_t output_h = 0; output_h < H; output_h += conv_params->tile_h) {
                conv_params->output_h = output_h;
                conv_params->output_w = output_w;
                handle_conv_inner_loop(conv_params);
            }
        }
    }

    my_printf_debug("handle_conv output" NEWLINE);
    output->dims[1] *= conv_params->n_tiles_c;
    output->params_len *= conv_params->n_tiles_c;
    dump_params(output);
    output->dims[1] /= conv_params->n_tiles_c;
    output->params_len /= conv_params->n_tiles_c;

    int16_t *output_baseptr = get_q15_param(conv_params->output, 0, WILL_WRITE);
    for (uint16_t c = 0; c < OUTPUT_CHANNEL; c++) {
        for (uint16_t output_h = 0; output_h < H; output_h++) {
            for (uint16_t output_w = 0; output_w < H; output_w++) {
                // XXX: handle state bits
                for (uint8_t tile_c_index = 1; tile_c_index * tile_c < CHANNEL; tile_c_index++) {
                    output_baseptr[c * H * W + output_h * W + output_w] += output_baseptr[(c + tile_c_index * OUTPUT_CHANNEL) * H * W + output_h * W + output_w];
                }
            }
        }
    }

    dump_params(output);

    setOutputValue(0);
}
