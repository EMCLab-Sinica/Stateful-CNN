// disable debug code in DSPLib
//#define MSP_DISABLE_DIAGNOSTICS

#include <DSPLib.h>
#ifdef USE_ARM_CMSIS
#include <arm_math.h>
#endif
#include "cnn_common.h"
#include "debug.h"
#include "op_handlers.h"
#include "intermittent-cnn.h"

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
    uint16_t input_tile_c_offset;
    uint16_t input_tile_c_index;
    uint16_t tile_h;
    uint16_t cur_input_tile_c;
    uint16_t n_tiles_c;
    uint16_t dest_offset;
    uint16_t filter_offset;
    uint16_t filter_limit;
    uint8_t truncated;
    uint8_t input_state_bit;
    uint8_t old_output_state_bit;

    uint16_t conv_idx;
    uint16_t conv_idx_base;
    uint16_t input_h;
    uint16_t input_w;
    int16_t *filter_buffer_addr;
    int16_t cached_filter_idx;
    uint16_t cached_input_tile_c_offset;
} ConvTaskParams;

static ConvTaskParams conv_params_obj;

int16_t * const matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN;

void determine_tile_c(ParameterInfo *param) {
    // TODO: determine these values automatically
    uint16_t CHANNEL = param->dims[1], H = param->dims[2];
    if (H == 14 && CHANNEL == 8) {
        param->tile_c = 3;
    } else if (H == 15 && CHANNEL == 64) {
        param->tile_c = 32;
    }
}

static void convTask(uint16_t offset_h, ConvTaskParams *conv_params) {
    /* put var declarations first to make the compiler happy */
    int16_t *filter_addr;
    uint16_t output_tile_c = conv_params->output->tile_c;
    uint16_t n_filters = MIN_VAL(conv_params->filter_limit, output_tile_c - (conv_params->conv_idx - conv_params->conv_idx_base));

    /* copy filter data */
    if (conv_params->cached_filter_idx != conv_params->conv_idx || conv_params->cached_input_tile_c_offset != conv_params->input_tile_c_offset) {
        conv_params->filter_buffer_addr = matrix_mpy_results - conv_params->filter_offset * (conv_params->filter_limit + TEMP_FILTER_WIDTH);
#ifndef USE_ARM_CMSIS
        int16_t *filter_tmp = matrix_mpy_results - conv_params->filter_offset; // before transpose

        msp_fill_q15_params fill_params;
        fill_params.length = conv_params->filter_offset;
        fill_params.value = 0;
        msp_status status = msp_fill_q15(&fill_params, filter_tmp);
        msp_checkStatus(status);
#else
        int16_t *filter_tmp = conv_params->filter_buffer_addr;

        arm_fill_q15(0, filter_tmp, n_filters * conv_params->filter_offset);
#endif


        uint16_t buffer_size = sizeof(int16_t) * conv_params->cur_input_tile_c;
        uint16_t filter_src_offset = conv_params->kH * conv_params->kW * conv_params->CHANNEL;
        for (uint16_t idx = 0; idx < n_filters; idx++) {
            // TODO: cache reordered filters on NVM
            filter_addr = get_q15_param(
                conv_params->conv_filter,
                (conv_params->conv_idx + idx) * filter_src_offset
            );
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->conv_idx + idx);
            for (uint16_t h = 0; h < conv_params->kH; h++) {
                int16_t *filter_dest_ptr = filter_tmp + h * conv_params->dest_offset;
                int16_t *filter_src_ptr = filter_addr + h * conv_params->kW * conv_params->CHANNEL + conv_params->input_tile_c_offset;
                for (uint16_t w = 0; w < conv_params->kW; w++) {
                    my_memcpy(filter_dest_ptr,
                              filter_src_ptr,
                              buffer_size);
                    filter_dest_ptr += conv_params->cur_input_tile_c;
                    filter_src_ptr += conv_params->CHANNEL;
                }
            }
#ifdef WITH_PROGRESS_EMBEDDING
            if (!conv_params->old_output_state_bit) {
                filter_tmp[conv_params->filter_offset - 1] = -0x4000;
            } else {
#endif
                // XXX: why is this needed? Should already be zero with msp_fill_q15 above
                filter_tmp[conv_params->filter_offset - 1] = 0;
#ifdef WITH_PROGRESS_EMBEDDING
            }
#endif
            if (conv_params->input_tile_c_index == 0) {
                filter_tmp[conv_params->filter_offset - 1] += -*get_q15_param(conv_params->conv_bias, conv_params->conv_idx + idx) / conv_params->conv_input->scale;
            }

#ifndef USE_ARM_CMSIS
            msp_interleave_q15_params params;
            params.length = conv_params->filter_offset;
            params.numChannels = n_filters;
            params.channel = idx;
            status = msp_interleave_q15(
                &params,
                filter_tmp, /* src */
                conv_params->filter_buffer_addr /* dst */
            );
            msp_checkStatus(status);
#else
            filter_tmp += conv_params->filter_offset;
#endif
        }

        conv_params->cached_filter_idx = conv_params->conv_idx;
        conv_params->cached_input_tile_c_offset = conv_params->input_tile_c_offset;
    }

    int16_t *filter_buffer_addr = conv_params->filter_buffer_addr;

    my_printf_debug("input_h=%d" NEWLINE, conv_params->input_h + offset_h);

    int16_t *input_buffer_addr = lea_buffer + offset_h * conv_params->dest_offset;

    // XXX: LEA doc requires all matrix dimensions to be even, while LEA
    // appears to still give correct results when srcARows is odd
    // srcBCols should really be even, though
    // http://e2e.ti.com/support/microcontrollers/msp430/f/166/t/716353?MSP430FR5992-MSP-DSPLib-msp-matrix-mpy-q15
    uint16_t A_rows, A_cols, B_rows, B_cols;
    A_rows = 1;
    A_cols = B_rows = conv_params->filter_offset;
    B_cols = n_filters;
    MY_ASSERT(A_rows * B_cols <= OUTPUT_LEN);
    MY_ASSERT((A_cols & 1) || (B_cols & 1) == 0);
#ifndef USE_ARM_CMSIS
    msp_matrix_mpy_q15_params matrix_mpy_params;
    matrix_mpy_params.srcARows = A_rows;
    matrix_mpy_params.srcACols = A_cols;
    matrix_mpy_params.srcBRows = B_rows;
    matrix_mpy_params.srcBCols = B_cols;
    msp_status status = msp_matrix_mpy_q15(
        &matrix_mpy_params,
        input_buffer_addr,
        filter_buffer_addr,
        matrix_mpy_results
    );
    msp_checkStatus(status);
#else
    arm_matrix_instance_q15 A, B, C;
    arm_mat_init_q15(&A, A_rows, A_cols, input_buffer_addr);
    arm_mat_init_q15(&B, B_rows, B_cols, filter_buffer_addr);
    arm_mat_init_q15(&C, A_rows, B_cols, matrix_mpy_results);
    arm_status status = arm_mat_mult_fast_q15(&A, &B, &C, NULL);
    MY_ASSERT(status == ARM_MATH_SUCCESS);
#endif

    /* START dump data */
#ifndef MY_NDEBUG
    my_printf_debug("conv_idx=");
    for (uint16_t idx = 0; idx < n_filters; idx++) {
        my_printf_debug("%d ", conv_params->conv_idx + idx);
    }
    my_printf_debug("output_h=%d ", (conv_params->input_h + offset_h) / conv_params->stride);
    my_printf_debug("output_w=%d" NEWLINE, conv_params->input_w / conv_params->stride);

    my_printf_debug("input_buffer_addr = lea_buffer + %d" NEWLINE, (int)(input_buffer_addr - lea_buffer));
    my_printf_debug("input" NEWLINE);
    dump_matrix2(input_buffer_addr, A_rows, A_cols, conv_params->conv_input->scale, 0);
    my_printf_debug("filter_buffer_addr = lea_buffer + LEA_BUFFER_SIZE - %d" NEWLINE, (int)(lea_buffer + LEA_BUFFER_SIZE - filter_buffer_addr));
    my_printf_debug("filter" NEWLINE);
#ifndef USE_ARM_CMSIS
    dump_matrix2(filter_buffer_addr, B_rows, B_cols, conv_params->conv_filter->scale, 0);
#else
    dump_matrix2(filter_buffer_addr, B_cols, B_rows, conv_params->conv_filter->scale, 0);
#endif

    my_printf_debug("matrix_mpy_results" NEWLINE);
    dump_matrix2(
        matrix_mpy_results,
        A_rows,
        B_cols,
        conv_params->conv_input->scale * conv_params->conv_filter->scale,
#ifdef WITH_PROGRESS_EMBEDDING
        !conv_params->old_output_state_bit
#else
        0
#endif
    );
    my_printf_debug(NEWLINE);
#endif
    /* END dump data */

    // use NWHC so that output is written continuously on the address space
    int16_t *output_baseptr = get_q15_param(conv_params->output, 0);
    int16_t *output_data = output_baseptr +
            conv_params->OUTPUT_W * conv_params->OUTPUT_H * (conv_params->input_tile_c_index * conv_params->OUTPUT_CHANNEL + conv_params->conv_idx_base) +   // n
            conv_params->input_w / conv_params->stride * conv_params->OUTPUT_H * output_tile_c +         // w
            (conv_params->input_h + offset_h) / conv_params->stride * output_tile_c +                    // h
            conv_params->conv_idx - conv_params->conv_idx_base;                                          // c
    my_printf_debug("output_data offset = %d" NEWLINE, (uint16_t)(output_data - output_baseptr));
    MY_ASSERT((uint8_t*)(output_data + n_filters) < intermediate_values(NUM_SLOTS));
#if !defined(MY_NDEBUG) && defined(WITH_PROGRESS_EMBEDDING)
    for (uint16_t idx2 = 0; idx2 < n_filters; idx2++) {
        if (!conv_params->old_output_state_bit && !get_value_state_bit(matrix_mpy_results[idx2])) {
            ERROR_OCCURRED();
        }
    }
#endif
    my_memcpy(output_data, matrix_mpy_results, n_filters * sizeof(int16_t));
}

static void handle_conv_inner_loop(ConvTaskParams *conv_params) {
    int8_t field_size = (conv_params->kH - 1) / 2;

    /* copy input data, row by row */

    ParameterInfo *conv_input = conv_params->conv_input;
    int16_t *input_addr = get_q15_param(conv_input, 0);
    if (conv_input->flags & SEPARATE_TILING) {
        int8_t slot_difference = conv_params->conv_input->extra_info[conv_params->input_tile_c_index] - conv_params->conv_input->extra_info[0];
        input_addr += slot_difference * INTERMEDIATE_VALUES_SIZE / sizeof(int16_t);
    } else {
        input_addr += conv_params->input_tile_c_offset * conv_params->H * conv_params->W;
    }

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
    int32_t w_start = int16_max(-field_size,                 -conv_params->input_w),
            w_end   = int16_min( field_size, conv_params->W-1-conv_params->input_w);
    int16_t *dest;
    // TEMP_FILTER_WIDTH additional filters for values before transpose
    uint16_t inputs_len = MIN_VAL(
        LEA_BUFFER_SIZE - OUTPUT_LEN - (conv_params->filter_limit + TEMP_FILTER_WIDTH) * conv_params->kH * conv_params->dest_offset,
        (conv_params->H + conv_params->kH - 1) * conv_params->dest_offset
    );

    dest = lea_buffer;

    int32_t h_start = int16_max(                     -field_size,                 -conv_params->input_h),
            h_end =   int16_min(conv_params->tile_h-1+field_size, conv_params->H-1-conv_params->input_h);

    my_printf_debug("Reinitialize input buffer" NEWLINE "inputs_len = %d" NEWLINE, inputs_len);

#ifndef USE_ARM_CMSIS
    msp_fill_q15_params fill_params;
    fill_params.length = inputs_len;
    fill_params.value = 0;
    msp_status status = msp_fill_q15(&fill_params, lea_buffer);
    msp_checkStatus(status);
#else
    arm_fill_q15(0, lea_buffer, inputs_len);
#endif

    dest += (h_start + field_size) * conv_params->dest_offset;

    my_printf_debug("h_start=%" PRId32 " ", h_start);
    my_printf_debug("h_end=%" PRId32 NEWLINE, h_end);

    size_t size = (w_end - w_start + 1) * conv_params->cur_input_tile_c;
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    (int)(dest - lea_buffer));
    for (int32_t h = h_start; h <= h_end; h++) {
        int16_t *input_src_addr = input_addr + (conv_params->input_h + h) * conv_params->W * conv_params->cur_input_tile_c + (conv_params->input_w + w_start) * conv_params->cur_input_tile_c;
        int16_t *dest_addr = dest + (w_start + field_size) * conv_params->cur_input_tile_c;
        my_memcpy(
            dest_addr,
            input_src_addr,
            size * sizeof(int16_t));
#ifdef WITH_PROGRESS_EMBEDDING
        if (conv_params->input_state_bit) {
            msp_offset_q15_params offset_params;
            offset_params.length = size / 2 * 2;
            offset_params.offset = -0x4000;
            status = msp_offset_q15(&offset_params, dest_addr, dest_addr);
            msp_checkStatus(status);
            if (size % 2) {
                dest_addr[size - 1] -= 0x4000;
            }
        }
#endif
        dest += conv_params->dest_offset;
    }
    uint16_t offset = conv_params->dest_offset - 1;
    while (offset < inputs_len) {
        lea_buffer[offset] = -0x8000; // _Q15(-1.0)
        offset += conv_params->dest_offset;
    }

    my_printf_debug("Loaded inputs" NEWLINE);
    // state = 0 as state bits are already removed by msp_offset_q15 above
    dump_matrix(lea_buffer, inputs_len, conv_params->conv_input->scale, 0);

    for (uint16_t j = 0; j < conv_params->H - conv_params->offset_h - conv_params->input_h; j += conv_params->stride) {
        // conv_idx is set to initial_c in handle_conv
        for (; conv_params->conv_idx - conv_params->conv_idx_base < conv_params->output->tile_c; conv_params->conv_idx += conv_params->filter_limit) {
            convTask(j, conv_params);
        }
        // reset here for further processing
        conv_params->conv_idx = conv_params->conv_idx_base;
    }
}

void alloc_conv(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    ParameterInfo *conv_input = input[0], *conv_filter = input[1];

    MY_ASSERT(conv_input->bitwidth == 16 && conv_filter->bitwidth == 16);

    MY_ASSERT(conv_input->dims[1] == conv_filter->dims[1]);

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
    uint16_t input_tile_c = conv_input->tile_c;
    conv_params->n_tiles_c = (CHANNEL + input_tile_c - 1) / input_tile_c;

    /* XXX: extend flags; assume dilation=(1, 1) for now */
    output->bitwidth = 16;
    output->slot = get_next_slot(model, conv_input);
    output->dims[0] = 1;
    // Although handle_conv requires more memory than params_len, only the first OUTPUT_CHANNEL
    // channels are useful after merging results from tiling
    output->dims[1] = OUTPUT_CHANNEL;
    output->dims[2] = conv_params->OUTPUT_H = (H - conv_params->offset_h * 2) / conv_params->stride;
    output->dims[3] = conv_params->OUTPUT_W = (W - conv_params->offset_w * 2) / conv_params->stride;
    output->params_len = conv_params->n_tiles_c * OUTPUT_CHANNEL * conv_params->OUTPUT_H * conv_params->OUTPUT_W * sizeof(int16_t);
    output->flags = TRANSPOSED;
    output->flags |= conv_params->n_tiles_c << 4;
    output->scale = conv_input->scale * conv_filter->scale;
}

void handle_conv(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
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
    conv_params->OUTPUT_CHANNEL = OUTPUT_CHANNEL;
#ifdef WITH_PROGRESS_EMBEDDING
    // Needs to handle multiple input state bits as IFM may be from different slots (separate tiling)
    uint8_t input_state_bits[3]; // should accomodate n_tiles_c
    if (conv_input->flags & SEPARATE_TILING) {
        input_state_bits[0] = get_state_bit(model, conv_input->extra_info[0]);
        input_state_bits[1] = get_state_bit(model, conv_input->extra_info[1]);
    } else {
        input_state_bits[0] = get_state_bit(model, conv_input->slot);
    }
    conv_params->old_output_state_bit = get_state_bit(model, output->slot);
#endif

    uint32_t first_unfinished_value_offset = recovery_from_state_bits(model, output);
    // Dimensions: NWHC
    uint16_t initial_c = first_unfinished_value_offset % OUTPUT_CHANNEL;
    first_unfinished_value_offset /= OUTPUT_CHANNEL;
    uint16_t initial_h = first_unfinished_value_offset % conv_params->OUTPUT_H;
    first_unfinished_value_offset /= conv_params->OUTPUT_H;
    uint16_t initial_w = first_unfinished_value_offset % conv_params->OUTPUT_W;
    uint16_t initial_n = first_unfinished_value_offset / conv_params->OUTPUT_W;
    my_printf_debug("initial_n = %d" NEWLINE, initial_n);
    my_printf_debug("initial_h = %d" NEWLINE, initial_h);
    my_printf_debug("initial_w = %d" NEWLINE, initial_w);
    my_printf_debug("initial_c = %d" NEWLINE, initial_c);

    uint16_t initial_input_w = conv_params->offset_w + initial_w * conv_params->stride;
    uint16_t initial_input_h = conv_params->offset_h + initial_h * conv_params->stride;

    // TODO: state recovery with partially done MM

    uint16_t input_tile_c = conv_input->tile_c;
    output->tile_c = OUTPUT_CHANNEL;
    determine_tile_c(output);
    uint16_t output_tile_c = output->tile_c;

    for (uint16_t input_tile_c_offset = initial_n * input_tile_c, input_tile_c_index = initial_n; input_tile_c_offset < CHANNEL ; input_tile_c_offset += input_tile_c, input_tile_c_index++) {
#ifdef WITH_PROGRESS_EMBEDDING
        if (conv_params->conv_input->flags & SEPARATE_TILING) {
            conv_params->input_state_bit = input_state_bits[input_tile_c_index];
        } else {
            conv_params->input_state_bit = input_state_bits[0];
        }
#endif
        conv_params->cur_input_tile_c = MIN_VAL(input_tile_c, CHANNEL - input_tile_c_offset);
        my_printf_debug("cur_input_tile_c = %d" NEWLINE, conv_params->cur_input_tile_c);
        // +1 for bias
        conv_params->dest_offset = conv_params->kH * conv_params->cur_input_tile_c + 1;
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
            output_tile_c,
            // `/ 2 * 2` as LEA requires matrix dimensions to be even
            ((LEA_BUFFER_SIZE - OUTPUT_LEN - conv_params->dest_offset * (conv_params->kH + conv_params->tile_h - 1)) / (conv_params->dest_offset * conv_params->kH) - TEMP_FILTER_WIDTH) / 2 * 2
        );

        my_printf_debug("filter_limit: %d" NEWLINE, conv_params->filter_limit);

        conv_params->input_tile_c_offset = input_tile_c_offset;
        conv_params->input_tile_c_index = input_tile_c_index;
        for (uint16_t conv_idx_base = 0; conv_idx_base < OUTPUT_CHANNEL; conv_idx_base += output_tile_c) {
            uint16_t input_w = conv_params->offset_w;
            if (input_tile_c_index == initial_n) {
                input_w = initial_input_w;
            }
            conv_params->conv_idx_base = conv_idx_base;
            for (; input_w < W - conv_params->offset_w; input_w += conv_params->stride) {
                uint16_t input_h = conv_params->offset_h;
                if (input_tile_c_index == initial_n && input_w == initial_input_w) {
                    input_h = initial_input_h;
                }
                for (; input_h < H - conv_params->offset_h; input_h += conv_params->tile_h) {
                    conv_params->input_h = input_h;
                    conv_params->input_w = input_w;
                    conv_params->conv_idx = conv_idx_base;
                    if (input_tile_c_index == initial_n && input_w == initial_input_w && input_h == initial_input_h) {
                        conv_params->conv_idx = initial_c;
                    }
                    handle_conv_inner_loop(conv_params);
                }
            }
        }
    }

    flip_state_bit(model, output);

#ifndef MY_NDEBUG
    uint32_t tiling_results_len = OUTPUT_CHANNEL * conv_params->OUTPUT_H * conv_params->OUTPUT_W;

    my_printf_debug("handle_conv output" NEWLINE);
    for (uint16_t input_tile_c_index = 0; input_tile_c_index * input_tile_c < CHANNEL; input_tile_c_index++) {
        dump_params_nhwc(model, output, input_tile_c_index * tiling_results_len);
    }
#endif
}

void alloc_convmerge(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    ParameterInfo *data = input[0];

    my_memcpy(output, data, sizeof(struct ParameterInfo));

    uint16_t OUTPUT_CHANNEL = data->dims[1],
             OUTPUT_H = data->dims[2],
             OUTPUT_W = data->dims[3];

    output->slot = get_next_slot(model, data);
    output->params_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W * sizeof(int16_t);
}

void handle_convmerge(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags) {
    UNUSED(flags);

    // XXX: make this function idempotent

    // Do not use conv_params here as its intialization in alloc_conv and
    // handle_conv might be skipped if the Conv node has finished.
    ParameterInfo *data = input[0];
    uint16_t OUTPUT_CHANNEL = data->dims[1],
             OUTPUT_H = data->dims[2],
             OUTPUT_W = data->dims[3];

    my_printf_debug("ConvMerge!" NEWLINE);

    uint8_t n_tiles_c = data->flags >> 4;

    MY_ASSERT(n_tiles_c);

    uint32_t tiling_results_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;

    int16_t *data_baseptr = get_q15_param(data, 0);
    int16_t *output_baseptr = get_q15_param(output, 0);
    uint16_t chunk_len = (LEA_BUFFER_SIZE - 1) / n_tiles_c / 2 * 2;

    uint16_t overflow_factor = find_overflow_factor(model, data);
    msp_scale_q15_params scale_params;
    float_to_scale_params(&scale_params, 1.0f * SCALE / overflow_factor);
#ifdef WITH_PROGRESS_EMBEDDING
    int16_t input_offset = get_state_bit(model, data->slot) ? -0x4000 : 0;
    int16_t output_offset = get_state_bit(model, output->slot) ? 0 : 0x4000;
    my_printf_debug("input_offset = %d, ", input_offset);
    my_printf_debug("output_offset = %d" NEWLINE, output_offset);
    msp_offset_q15_params offset_params;
#endif
    msp_status status;

    // XXX: use iterate_chunks() ?
    for (uint32_t tiling_results_offset = 0; tiling_results_offset < tiling_results_len; tiling_results_offset += chunk_len) {
        uint32_t real_chunk_len = MIN_VAL(chunk_len, tiling_results_len - tiling_results_offset);
#ifdef WITH_PROGRESS_EMBEDDING
        offset_params.length = real_chunk_len;
#endif
        my_printf_debug("real_chunk_len = %d" NEWLINE, real_chunk_len);
        for (uint16_t input_tile_c_index = 0; input_tile_c_index < n_tiles_c; input_tile_c_index++) {
            int16_t *to_add = lea_buffer + input_tile_c_index * chunk_len;
            my_memcpy(to_add,
                      data_baseptr + input_tile_c_index * tiling_results_len + tiling_results_offset,
                      real_chunk_len * sizeof(int16_t));
            // scale up results as in convolution values are scaled down twice (input & weights)
#ifdef WITH_PROGRESS_EMBEDDING
            offset_params.offset = input_offset;
            status = msp_offset_q15(&offset_params, to_add, to_add);
            msp_checkStatus(status);
#endif
            scale_params.length = real_chunk_len;
            status = msp_scale_q15(&scale_params, to_add, to_add);
            msp_checkStatus(status);
            if (input_tile_c_index != 0) {
                msp_add_q15_params params3;
                params3.length = real_chunk_len;
                status = msp_add_q15(&params3, lea_buffer, to_add, lea_buffer);
                msp_checkStatus(status);
            }
#ifdef WITH_PROGRESS_EMBEDDING
            offset_params.offset = output_offset;
            status = msp_offset_q15(&offset_params, to_add, to_add);
            msp_checkStatus(status);
#endif
        }
        my_memcpy(output_baseptr + tiling_results_offset, lea_buffer, real_chunk_len * sizeof(int16_t));
    }

    my_printf_debug("After scaling up back and merging tiling results" NEWLINE);

    output->scale = output->scale * overflow_factor / SCALE;

    setOutputValue(0);

    flip_state_bit(model, output);

    dump_params_nhwc(model, output, 0);
}
