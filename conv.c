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
#define OUTPUT_LEN 100

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
    // OUTPUT_H and OUTPUT_W to handle stride != 1
    uint16_t OUTPUT_H;
    uint16_t OUTPUT_W;
    uint16_t kH;
    uint16_t kW;
    uint16_t CHANNEL; // Cannot use C as a variable name here as C is a macro on MSP430 :(
    uint16_t OUTPUT_CHANNEL;
    uint16_t stride;
    // offset_h and offset_w to handle auto_pad=VALID
    uint8_t offset_h;
    uint8_t offset_w;
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

static ConvTaskParams conv_params_obj;

int16_t * const matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN;

uint8_t get_tile_c(ParameterInfo *param) {
    // TODO: determine these values automatically
    uint16_t CHANNEL = param->dims[1], H = param->dims[2];
    if (H == 14) {
        return 3;
    } else if (H == 28) {
        return 1;
    } else {
        return CHANNEL;
    }
}

static void convTask(uint8_t offset_h, ConvTaskParams *conv_params) {
    /* put var declarations first to make the compiler happy */
    int16_t *filter_addr;
    uint8_t n_filters = MIN_VAL(conv_params->filter_limit, conv_params->OUTPUT_CHANNEL - conv_params->conv_idx);
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

        uint16_t buffer_size = sizeof(int16_t) * conv_params->tile_c;
        uint16_t filter_src_offset = conv_params->kH * conv_params->kW * conv_params->CHANNEL;
        for (uint8_t idx = 0; idx < n_filters; idx++) {
            // TODO: cache reordered filters on NVM
            filter_addr = get_q15_param(
                conv_params->conv_filter,
                (conv_params->conv_idx + idx) * filter_src_offset
            );
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->conv_idx + idx);
            if (conv_params->tile_c_index == 0) {
                *(filter_tmp + conv_params->dest_offset - 1) = -*get_q15_param(conv_params->conv_bias, conv_params->conv_idx + idx);
            }
            for (uint16_t h = 0; h < conv_params->kH; h++) {
                int16_t *filter_dest_ptr = filter_tmp + h * conv_params->dest_offset;
                int16_t *filter_src_ptr = filter_addr + h * conv_params->kW * conv_params->CHANNEL + conv_params->tile_c_offset;
                for (uint16_t w = 0; w < conv_params->kW; w++) {
                    my_memcpy(filter_dest_ptr,
                              filter_src_ptr,
                              buffer_size);
                    filter_dest_ptr += conv_params->tile_c;
                    filter_src_ptr += conv_params->CHANNEL;
                }
            }
#ifdef WITH_PROGRESS_EMBEDDING
            filter_tmp[conv_params->filter_offset - (conv_params->truncated?2:1)] -= 0x4000;
#endif

            msp_interleave_q15_params params;
            params.length = p_matrix_mpy_params->srcBRows;
            params.numChannels = n_filters;
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
            conv_params->conv_idx * conv_params->filter_offset);
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

    int16_t offset = conv_params->OUTPUT_W * conv_params->OUTPUT_CHANNEL;
    int16_t *output_baseptr = get_q15_param(conv_params->output, conv_params->tile_c_index * conv_params->OUTPUT_H * offset);
    int16_t *output_data = output_baseptr +
            conv_params->conv_idx +
            (conv_params->output_h + offset_h) / conv_params->stride * conv_params->OUTPUT_W * conv_params->OUTPUT_CHANNEL +
            conv_params->output_w / conv_params->stride * conv_params->OUTPUT_CHANNEL;
    int16_t *result_addr = matrix_mpy_results;
    for (uint8_t idx = 0; idx < p_matrix_mpy_params->srcARows; idx++) {
        my_printf_debug("output_data offset = %d" NEWLINE, (uint16_t)(output_data - output_baseptr));
        MY_ASSERT((uint8_t*)(output_data + n_filters) < intermediate_values() + INTERMEDIATE_VALUES_SIZE);
#if !defined(MY_NDEBUG) && defined(WITH_PROGRESS_EMBEDDING)
        for (uint8_t idx2 = 0; idx2 < n_filters; idx2++) {
            if (!conv_params->state_bit && *result_addr < 0x2000 && *result_addr >= -0x2000) {
                ERROR_OCCURRED();
            }
        }
#endif
        my_memcpy(output_data, result_addr, n_filters * sizeof(int16_t));
        result_addr += n_filters;
        output_data += conv_params->kH * offset;
    }
}

static inline void schedule_tile(uint16_t idx, ConvTaskParams *conv_params) {
    conv_params->conv_idx = idx;
    msp_matrix_mpy_q15_params *p_matrix_mpy_params = &(conv_params->matrix_mpy_params);

    // XXX: LEA doc requires all matrix dimensions to be even, while LEA
    // appears to still give correct results when srcARows is odd
    // srcBCols should really be even, though
    // http://e2e.ti.com/support/microcontrollers/msp430/f/166/t/716353?MSP430FR5992-MSP-DSPLib-msp-matrix-mpy-q15
    p_matrix_mpy_params->srcARows = 1;
    p_matrix_mpy_params->srcACols = p_matrix_mpy_params->srcBRows = conv_params->filter_offset;
    p_matrix_mpy_params->srcBCols = MIN_VAL(conv_params->filter_limit, conv_params->OUTPUT_CHANNEL - idx);
    MY_ASSERT(p_matrix_mpy_params->srcARows * p_matrix_mpy_params->srcBCols <= OUTPUT_LEN);
    if ((p_matrix_mpy_params->srcACols & 1) || (p_matrix_mpy_params->srcBCols & 1)) {
        ERROR_OCCURRED();
    }
    for (uint8_t j = 0; j < conv_params->H - conv_params->offset_h - conv_params->output_h; j += conv_params->stride) {
        convTask(j, conv_params);
    }
}

static inline void handle_conv_inner_loop(void *pvParameters) {
    ConvTaskParams *conv_params = (ConvTaskParams*)pvParameters;

    int8_t field_size = (conv_params->kH - 1) / 2;

    /* copy input data, row by row */

    int16_t *input_addr = get_q15_param(
        conv_params->conv_input,
        conv_params->tile_c_offset * conv_params->H * conv_params->W);

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

    size_t size = (w_end - w_start + 1) * conv_params->tile_c;
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    (int)(dest - lea_buffer));
    for (int32_t h = h_start; h <= h_end; h++) {
        int16_t *input_src_addr = input_addr + (conv_params->output_h + h) * conv_params->W * conv_params->tile_c + (conv_params->output_w + w_start) * conv_params->tile_c;
        my_memcpy(
            dest + (w_start + field_size) * conv_params->tile_c,
            input_src_addr,
            size * sizeof(int16_t));
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
            MY_ASSERT(conv_params->pending_filter_idx < NUM_FILTERS);
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

uint32_t alloc_conv(ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    ParameterInfo *conv_input = input[0], *conv_filter = input[1];

    if (conv_input->bitwidth != 16 || conv_filter->bitwidth != 16) {
        // incorrect bitwidth
        ERROR_OCCURRED();
    }

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   CHANNEL = conv_filter->dims[1],
                   OUTPUT_CHANNEL = conv_filter->dims[0];

    ConvTaskParams *conv_params = &conv_params_obj;

    conv_params->kH = conv_filter->dims[2];
    conv_params->kW = conv_filter->dims[3];

    conv_params->stride = flags & 0x0f;
    if ((flags & 0xff00) >> 8 == AUTO_PAD_VALID) {
        conv_params->offset_h = conv_params->kH / 2;
        conv_params->offset_w = conv_params->kW / 2;
    } else {
        conv_params->offset_h = conv_params->offset_w = 0;
    }

    /* XXX: extend flags; assume dilation=(1, 1) for now */
    output->bitwidth = 16;
    output->slot = SLOT_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    // Although handle_conv requires more memory than params_len, only the first OUTPUT_CHANNEL
    // channels are useful after merging results from tiling
    output->dims[1] = OUTPUT_CHANNEL;
    output->dims[2] = conv_params->OUTPUT_H = (H - conv_params->offset_h * 2) / conv_params->stride;
    output->dims[3] = conv_params->OUTPUT_W = (W - conv_params->offset_w * 2) / conv_params->stride;
    output->params_len = OUTPUT_CHANNEL * conv_params->OUTPUT_H * conv_params->OUTPUT_W * sizeof(int16_t);

    uint8_t tile_c = get_tile_c(conv_input);
    conv_params->n_tiles_c = (CHANNEL + tile_c - 1) / tile_c;

    return output->params_len * conv_params->n_tiles_c;
}

void handle_conv(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
#ifndef WITH_PROGRESS_EMBEDDING
    UNUSED(model);
#endif
    // flags already handled in alloc_conv
    UNUSED(flags);

    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *conv_bias = input[2];
    my_printf_debug("Conv!" NEWLINE);

    setOutputValue(1);

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   CHANNEL = conv_filter->dims[1],
                   OUTPUT_CHANNEL = conv_filter->dims[0];

    ConvTaskParams *conv_params = &conv_params_obj;

    uint8_t tile_c = get_tile_c(conv_input);
    conv_params->tile_h = H; // fallback value
    if (H == 14) {
        conv_params->tile_h = 14;
    } else if (H == 28) {
        conv_params->tile_h = 28;
    }

    MY_ASSERT(conv_params->tile_h / conv_params->stride * conv_params->stride == conv_params->tile_h);

    my_printf_debug("n_tiles_c = %d" NEWLINE, conv_params->n_tiles_c);

    conv_params->conv_input = conv_input;
    conv_params->conv_filter = conv_filter;
    conv_params->conv_bias = conv_bias;
    conv_params->output = output;
    conv_params->filter_buffer_addr = NULL;
    conv_params->cached_filter_idx = -1;
    conv_params->H = H;
    conv_params->W = W;

    conv_params->CHANNEL = CHANNEL;
    conv_params->pending_filter_idx = 0;
    conv_params->OUTPUT_CHANNEL = OUTPUT_CHANNEL;
#ifdef WITH_PROGRESS_EMBEDDING
    conv_params->state_bit = model->state_bit;
    // XXX
    model->state_bit = 1;
    if (conv_params->state_bit) {
        int16_t *input_ptr = get_q15_param(conv_input, 0);
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
        for (uint16_t output_w = conv_params->offset_w; output_w < W - conv_params->offset_w; output_w += conv_params->stride) {
            for (uint16_t output_h = conv_params->offset_h; output_h < H - conv_params->offset_h; output_h += conv_params->tile_h) {
                conv_params->output_h = output_h;
                conv_params->output_w = output_w;
                handle_conv_inner_loop(conv_params);
            }
        }
    }

    my_printf_debug("handle_conv output" NEWLINE);

    // XXX: handle state bits
    int16_t *output_baseptr = get_q15_param(conv_params->output, 0);
    uint16_t chunk_len = (LEA_BUFFER_SIZE - 1) / conv_params->n_tiles_c / 2 * 2;
    uint32_t tiling_results_len = OUTPUT_CHANNEL * conv_params->OUTPUT_H * conv_params->OUTPUT_W;
    float scale_q15 = SCALE;
    uint8_t scale_shift = 0;
    while (scale_q15 >= 1) {
        scale_q15 /= 2.0f;
        scale_shift++;
    }

    for (uint8_t tile_c_index = 0; tile_c_index * tile_c < CHANNEL; tile_c_index++) {
        dump_params_nhwc(output, tile_c_index * tiling_results_len);
    }

    for (uint32_t tiling_results_offset = 0; tiling_results_offset < tiling_results_len; tiling_results_offset += chunk_len) {
        uint32_t real_chunk_len = MIN_VAL(chunk_len, tiling_results_len - tiling_results_offset);
        my_printf_debug("real_chunk_len = %d" NEWLINE, real_chunk_len);
        for (uint8_t tile_c_index = 0; tile_c_index * tile_c < CHANNEL; tile_c_index++) {
            int16_t *to_add = lea_buffer + tile_c_index * chunk_len;
            my_memcpy(to_add,
                      output_baseptr + tile_c_index * tiling_results_len + tiling_results_offset,
                      real_chunk_len * sizeof(int16_t));
            msp_scale_q15_params params2 = {
                .length = real_chunk_len,
                .scale = _Q15(scale_q15),
                .shift = scale_shift,
            };
            // scale up results as in convolution values are scaled down twice (input & weights)
            msp_status status = msp_scale_q15(&params2, to_add, to_add);
            msp_checkStatus(status);
            if (tile_c_index != 0) {
                msp_add_q15_params params3 = { .length = real_chunk_len };
                status = msp_add_q15(&params3, lea_buffer, to_add, lea_buffer);
                msp_checkStatus(status);
            }
        }
        my_memcpy(output_baseptr + tiling_results_offset, lea_buffer, real_chunk_len * sizeof(int16_t));
    }

    my_printf_debug("After scaling up back" NEWLINE);

    dump_params_nhwc(output, 0);

    setOutputValue(0);
}
