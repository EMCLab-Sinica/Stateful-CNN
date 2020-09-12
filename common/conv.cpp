// disable debug code in DSPLib
//#define MSP_DISABLE_DIAGNOSTICS

#ifndef USE_ARM_CMSIS
#include <DSPLib.h>
#endif
#include <inttypes.h> // for PRId32
#include "cnn_common.h"
#include "debug.h"
#include "op_handlers.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"

#define configCONV_STACK_SIZE 100

// TODO: make these adjustable on runtime
#define OUTPUT_LEN 100

// to make the code clearer
#ifndef USE_ARM_CMSIS
#define TEMP_FILTER_WIDTH 1
#else
#define TEMP_FILTER_WIDTH 0
#endif

static int16_t last_output_data_offset;

#define CONV_TASK_FLAG_PROCESSED_FILTERS_BASE 2
typedef struct ConvTaskParams {
    ParameterInfo *conv_input;
    ParameterInfo *real_conv_input; // for separate channel tiling
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
    uint8_t truncated;
#ifdef WITH_PROGRESS_EMBEDDING
    uint8_t old_output_state_bit;
    uint8_t turning_point_idx;
    int16_t next_turning_point;
    SlotInfo* cur_slot_info;
#endif

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

void determine_tile_c(ParameterInfo *param, ParameterInfo *filter) {
    // TODO: determine these values automatically
    uint16_t CHANNEL = param->dims[1], H = param->dims[2];
    uint16_t kH = 0, INPUT_CHANNEL = 0;
    if (filter) {
        INPUT_CHANNEL = filter->dims[1];
        kH = filter->dims[2];
    }
    if (H == 14 && CHANNEL == 8) {
        param->tile_c = 3;
    } else if (H == 15 && CHANNEL == 64) {
        param->tile_c = 32;
    } else if (H == 7 && CHANNEL == 64 && kH == 3) {
        param->tile_c = 6;
    } else if (H == 7 && CHANNEL == 32 && INPUT_CHANNEL == 128 && kH == 1) {
        param->tile_c = 16;
    } else if (H == 7 && CHANNEL == 128 && INPUT_CHANNEL == 32 && kH == 1) {
        param->tile_c = 44;
    } else if (H == 7 && CHANNEL == 128 && INPUT_CHANNEL == 32 && kH == 3) {
        param->tile_c = 2;
    } else if (INPUT_CHANNEL == 256 && kH == 1) {
        param->tile_c = 4;
    }
}

#ifdef WITH_PROGRESS_EMBEDDING
static void flip_filter_state_bits(ConvTaskParams *conv_params, uint16_t cur_output_tile_c, uint16_t len, uint8_t first_round) {
    MY_ASSERT(len < OUTPUT_LEN);
    my_printf_debug("Flipping %d state bits in filters" NEWLINE, len);
    // need negating filter value here as it will be multiplied with _Q15(-1.0), or -32768
#ifndef USE_ARM_CMSIS
    int16_t *to_flip_state_bits = conv_params->filter_buffer_addr + cur_output_tile_c * conv_params->filter_offset;
    if (first_round) {
        to_flip_state_bits -= len;
    } else {
        to_flip_state_bits -= cur_output_tile_c;
    }
    int16_t offset = get_value_state_bit(-*to_flip_state_bits) ? 0x4000 : -0x4000;
    my_offset_q15(to_flip_state_bits, offset, to_flip_state_bits, len);
#else
    int16_t *to_flip_state_bits = conv_params->filter_buffer_addr + conv_params->filter_offset - 1;
    if (first_round) {
        to_flip_state_bits += (cur_output_tile_c - len) * conv_params->filter_offset;
    }
    int16_t offset = get_value_state_bit(-*to_flip_state_bits) ? 0x4000 : -0x4000;
    for (uint16_t idx = 0; idx < len; idx++) {
        *to_flip_state_bits += offset;
        to_flip_state_bits += conv_params->filter_offset;
    }
#endif
}
#endif

static void convTask(uint16_t offset_h, ConvTaskParams *conv_params) {
    // cur_output_tile_c should be signed, or MAX_VAL below is broken with TI's compiler
    int16_t cur_output_tile_c = MIN_VAL(conv_params->output->tile_c, conv_params->OUTPUT_CHANNEL - conv_params->conv_idx);
    MY_ASSERT(cur_output_tile_c);

    // use NWHC so that output is written continuously on the address space
    int16_t cur_output_data_offset =
            conv_params->OUTPUT_W * conv_params->OUTPUT_H * (conv_params->input_tile_c_index * conv_params->OUTPUT_CHANNEL + conv_params->conv_idx_base) +   // n
            conv_params->input_w / conv_params->stride * conv_params->OUTPUT_H * cur_output_tile_c +     // w
            (conv_params->input_h + offset_h) / conv_params->stride * cur_output_tile_c +                // h
            conv_params->conv_idx - conv_params->conv_idx_base;                                          // c

#ifdef WITH_PROGRESS_EMBEDDING
    SlotInfo *cur_slot_info = conv_params->cur_slot_info;
    int16_t n_keep_state_bits = cur_output_tile_c;
    uint8_t need_cleanup_state_bits = 0;
    if (conv_params->turning_point_idx < cur_slot_info->n_turning_points && conv_params->next_turning_point > 0) {
        my_printf_debug("next_turning_point = %d" NEWLINE, conv_params->next_turning_point);
        n_keep_state_bits -= MAX_VAL(0, cur_output_data_offset + cur_output_tile_c - conv_params->next_turning_point);
    }
    my_printf_debug("n_keep_state_bits = %d" NEWLINE, n_keep_state_bits);
    MY_ASSERT(n_keep_state_bits >= 0);
#endif

    /* copy filter data */
    if (conv_params->cached_filter_idx != conv_params->conv_idx || conv_params->cached_input_tile_c_offset != conv_params->input_tile_c_offset) {
        conv_params->filter_buffer_addr = matrix_mpy_results - conv_params->filter_offset * (cur_output_tile_c + TEMP_FILTER_WIDTH);
#ifndef USE_ARM_CMSIS
        int16_t *filter_tmp = matrix_mpy_results - conv_params->filter_offset; // before transpose
        uint16_t fill_length = conv_params->filter_offset;
#else
        int16_t *filter_tmp = conv_params->filter_buffer_addr;
        uint16_t fill_length = cur_output_tile_c * conv_params->filter_offset;
#endif
        my_fill_q15(0, filter_tmp, fill_length);

        uint16_t buffer_size = sizeof(int16_t) * conv_params->cur_input_tile_c;
        uint16_t filter_len = conv_params->kH * conv_params->kW * conv_params->CHANNEL;
        for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
            uint16_t filter_src_offset = (conv_params->conv_idx + idx) * filter_len;
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->conv_idx + idx);
            for (uint16_t h = 0; h < conv_params->kH; h++) {
                int16_t *filter_dest_ptr = filter_tmp + h * conv_params->dest_offset;
                uint16_t cur_filter_src_offset = filter_src_offset + h * conv_params->kW * conv_params->CHANNEL + conv_params->input_tile_c_offset;
                for (uint16_t w = 0; w < conv_params->kW; w++) {
                    my_memcpy_from_param(filter_dest_ptr, conv_params->conv_filter, cur_filter_src_offset, buffer_size);
                    filter_dest_ptr += conv_params->cur_input_tile_c;
                    cur_filter_src_offset += conv_params->CHANNEL;
                }
            }
#ifdef WITH_PROGRESS_EMBEDDING
            if ((!conv_params->old_output_state_bit && idx < n_keep_state_bits) || (conv_params->old_output_state_bit && idx >= n_keep_state_bits)) {
                my_printf_debug("Adding state bit for newly loaded filter idx=%d" NEWLINE, idx);
                filter_tmp[conv_params->filter_offset - 1] = -0x4000;
                need_cleanup_state_bits = 1;
            } else
#endif
            {
                // XXX: why is this needed? Should already be zero with my_fill_q15 above
                filter_tmp[conv_params->filter_offset - 1] = 0;
            }
            if (conv_params->input_tile_c_index == 0) {
                filter_tmp[conv_params->filter_offset - 1] += -get_q15_param(conv_params->conv_bias, conv_params->conv_idx + idx) / conv_params->conv_input->scale;
            }

#ifndef USE_ARM_CMSIS
            msp_interleave_q15_params params;
            params.length = conv_params->filter_offset;
            params.numChannels = cur_output_tile_c;
            params.channel = idx;
            msp_status status = msp_interleave_q15(
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
    } else {
#ifdef WITH_PROGRESS_EMBEDDING
        if (n_keep_state_bits != cur_output_tile_c) {
            need_cleanup_state_bits = 1;
            int16_t n_flip_state_bits = cur_output_tile_c - n_keep_state_bits;
            flip_filter_state_bits(conv_params, cur_output_tile_c, n_flip_state_bits, 1);
        }
#endif
    }

    int16_t *filter_buffer_addr = conv_params->filter_buffer_addr;

    int16_t *input_buffer_addr = lea_buffer + offset_h * conv_params->dest_offset;

    uint16_t A_rows, A_cols, B_rows, B_cols;
    A_rows = 1;
    A_cols = B_rows = conv_params->filter_offset;
    B_cols = cur_output_tile_c;
    MY_ASSERT(A_rows * B_cols <= OUTPUT_LEN);
    my_matrix_mpy_q15(A_rows, A_cols, B_rows, B_cols, input_buffer_addr, filter_buffer_addr, matrix_mpy_results, 1);

    /* START dump data */
#if MY_DEBUG >= 2
    my_printf_debug("input_h=%d" NEWLINE, conv_params->input_h + offset_h);
    my_printf_debug("conv_idx=");
    for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
        my_printf_debug("%d ", conv_params->conv_idx + idx);
        MY_ASSERT(conv_params->conv_idx + idx < conv_params->OUTPUT_CHANNEL);
    }
    my_printf_debug("output_h=%d ", (conv_params->input_h + offset_h) / conv_params->stride);
    my_printf_debug("output_w=%d" NEWLINE, conv_params->input_w / conv_params->stride);

    my_printf_debug("input_buffer_addr = lea_buffer + %d" NEWLINE, (int)(input_buffer_addr - lea_buffer));
    my_printf_debug("input" NEWLINE);
    dump_matrix2_debug(input_buffer_addr, A_rows, A_cols, ValueInfo(conv_params->conv_input));
    my_printf_debug("filter_buffer_addr = lea_buffer + LEA_BUFFER_SIZE - %d" NEWLINE, (int)(lea_buffer + LEA_BUFFER_SIZE - filter_buffer_addr));
    my_printf_debug("filter" NEWLINE);
#ifndef USE_ARM_CMSIS
    dump_matrix2_debug(filter_buffer_addr, B_rows, B_cols, ValueInfo(conv_params->conv_filter));
#else
    dump_matrix2_debug(filter_buffer_addr, B_cols, B_rows, ValueInfo(conv_params->conv_filter));
#endif

    my_printf_debug("matrix_mpy_results" NEWLINE);
    ValueInfo val_info;
    val_info.scale = conv_params->conv_input->scale * conv_params->conv_filter->scale;
#ifdef WITH_PROGRESS_EMBEDDING
    val_info.state = !conv_params->old_output_state_bit;
#endif
    dump_matrix2_debug(matrix_mpy_results, A_rows, B_cols, val_info);
    my_printf_debug(NEWLINE);
#endif
    /* END dump data */

    my_printf_debug("output_data offset = %d" NEWLINE, cur_output_data_offset);
    MY_ASSERT(cur_output_data_offset > last_output_data_offset);
    last_output_data_offset = cur_output_data_offset;

    MY_ASSERT(cur_output_data_offset + cur_output_tile_c < INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);
    my_memcpy_to_param(conv_params->output, cur_output_data_offset, matrix_mpy_results, cur_output_tile_c * sizeof(int16_t));

#ifdef WITH_PROGRESS_EMBEDDING
    if (n_keep_state_bits != cur_output_tile_c) {
        conv_params->turning_point_idx++;
        conv_params->old_output_state_bit ^= 1;
        my_printf_debug("old_output_state_bit flipped to %d" NEWLINE, conv_params->old_output_state_bit);
        if (conv_params->turning_point_idx < cur_slot_info->n_turning_points) {
            conv_params->next_turning_point = cur_slot_info->turning_points[conv_params->turning_point_idx];
        }

        if (need_cleanup_state_bits) {
            flip_filter_state_bits(conv_params, cur_output_tile_c, n_keep_state_bits, 0);
        }
    }
#endif
}

static void handle_conv_inner_loop(Model *model, ConvTaskParams *conv_params) {
    int8_t field_size = (conv_params->kH - 1) / 2;

    /* copy input data, row by row */

    int16_t input_offset = 0;
    if (conv_params->conv_input->flags & SEPARATE_TILING) {
        conv_params->real_conv_input = get_parameter_info(conv_params->conv_input->extra_info[conv_params->input_tile_c_index]);
        // Not touching input_offset as IFM for different input tiles are in different slots and each starts address 0 in
        // corresponding slots
    } else {
        conv_params->real_conv_input = conv_params->conv_input;
        input_offset += conv_params->input_tile_c_offset * conv_params->H * conv_params->W;
    }

    /* int32_t instead of int16_t as TI's compiler cannot handle negative
     * offsets correctly. The expression `input_offset + (int16_t)(-2)` is
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
        LEA_BUFFER_SIZE - OUTPUT_LEN - (conv_params->output->tile_c + TEMP_FILTER_WIDTH) * conv_params->kH * conv_params->dest_offset,
        (conv_params->H + conv_params->kH - 1) * conv_params->dest_offset
    );

    dest = lea_buffer;

    int32_t h_start = int16_max(                     -field_size,                 -conv_params->input_h),
            h_end =   int16_min(conv_params->tile_h-1+field_size, conv_params->H-1-conv_params->input_h);

    my_printf_debug("Reinitialize input buffer" NEWLINE "inputs_len = %d" NEWLINE, inputs_len);

    my_fill_q15(0, lea_buffer, inputs_len);

    dest += (h_start + field_size) * conv_params->dest_offset;

    my_printf_debug("h_start=%" PRId32 " ", h_start);
    my_printf_debug("h_end=%" PRId32 NEWLINE, h_end);

    size_t size = (w_end - w_start + 1) * conv_params->cur_input_tile_c;
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    (int)(dest - lea_buffer));
    int16_t input_src_offset;
    dump_turning_points_debug(conv_params->output);
    for (int32_t h = h_start; h <= h_end; h++) {
        input_src_offset = input_offset + (conv_params->input_h + h) * conv_params->W * conv_params->cur_input_tile_c + (conv_params->input_w + w_start) * conv_params->cur_input_tile_c;
        int16_t *dest_addr = dest + (w_start + field_size) * conv_params->cur_input_tile_c;
        my_printf_debug("Load input from range [%d, %ld)" NEWLINE, input_src_offset, input_src_offset + size);
        my_memcpy_from_param(
            dest_addr,
            conv_params->real_conv_input, input_src_offset,
            size * sizeof(int16_t));
        dest += conv_params->dest_offset;
    }
    my_printf_debug("Loaded inputs before removing state bits" NEWLINE);
    dump_matrix_debug(lea_buffer, inputs_len, ValueInfo());
#ifdef WITH_PROGRESS_EMBEDDING
    // Not using iterate_chunks here as it is too slow
    // TODO: use LEA
    if (conv_params->cur_slot_info->n_turning_points) {
        int16_t *ptr = lea_buffer;
        for (size_t idx = 0; idx < inputs_len; idx++) {
            if (get_value_state_bit(*ptr)) {
                *ptr -= 0x4000;
            }
            ptr++;
        }
    }
#endif
    uint16_t offset = conv_params->dest_offset - 1;
    while (offset < inputs_len) {
        lea_buffer[offset] = -0x8000; // _Q15(-1.0)
        offset += conv_params->dest_offset;
    }

    my_printf_debug("Loaded inputs" NEWLINE);
    // state = 0 as state bits are already removed by my_offset_q15 above
    dump_matrix_debug(lea_buffer, inputs_len, ValueInfo(conv_params->real_conv_input));

    for (uint16_t j = 0; j < conv_params->H - conv_params->offset_h - conv_params->input_h; j += conv_params->stride) {
        // conv_idx is set to initial_c in handle_conv
        convTask(j, conv_params);
        // reset here for further processing
        conv_params->conv_idx = conv_params->conv_idx_base;
    }
}

void alloc_conv(Model *model, ParameterInfo *input[], ParameterInfo *output, NodeFlags* flags) {
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

    conv_params->stride = flags->stride;
    if (flags->generic == AUTO_PAD_VALID) {
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

void handle_conv(Model *model, ParameterInfo *input[], ParameterInfo *output, NodeFlags*) {
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

    // TODO: state recovery with partially done MM

    uint16_t input_tile_c = conv_input->tile_c;
    output->tile_c = OUTPUT_CHANNEL;
    determine_tile_c(output, conv_filter);
    uint16_t output_tile_c = output->tile_c;
    my_printf_debug("output_tile_c = %d" NEWLINE, output_tile_c);

    last_output_data_offset = -1;

    conv_params->input_tile_c_offset = 0;
    conv_params->input_tile_c_index = 0;
    conv_params->input_w = conv_params->offset_w;
    conv_params->input_h = conv_params->offset_h;
    conv_params->conv_idx_base = 0;
    conv_params->conv_idx = 0;
#ifdef WITH_PROGRESS_EMBEDDING
    SlotInfo *cur_slot_info = conv_params->cur_slot_info = get_slot_info(output->slot);
    conv_params->turning_point_idx = 0;
    conv_params->next_turning_point = -1;
    if (conv_params->turning_point_idx < cur_slot_info->n_turning_points) {
        conv_params->next_turning_point = cur_slot_info->turning_points[conv_params->turning_point_idx];
    }

    conv_params->old_output_state_bit = get_state_bit(model, output->slot);
    my_printf_debug("old_output_state_bit = %d" NEWLINE, conv_params->old_output_state_bit);

    uint32_t first_unfinished_value_offset = recovery_from_state_bits(model, output);
    // Dimensions: channel-tiled NWHC
    uint16_t slice_size_input_channel_tiling = conv_params->OUTPUT_W * conv_params->OUTPUT_H * conv_params->OUTPUT_CHANNEL;
    conv_params->input_tile_c_index = first_unfinished_value_offset / slice_size_input_channel_tiling;
    conv_params->input_tile_c_offset = conv_params->input_tile_c_index * input_tile_c;
    first_unfinished_value_offset %= slice_size_input_channel_tiling;

    uint16_t slice_size_output_channel_tiling = conv_params->OUTPUT_W * conv_params->OUTPUT_H * output_tile_c;
    conv_params->conv_idx_base = first_unfinished_value_offset / slice_size_output_channel_tiling * output_tile_c;
    conv_params->conv_idx = conv_params->conv_idx_base;
    first_unfinished_value_offset %= slice_size_output_channel_tiling;

    uint16_t cur_output_tile_c = MIN_VAL(output_tile_c, OUTPUT_CHANNEL - conv_params->conv_idx_base);
    uint16_t slice_size_column = cur_output_tile_c * conv_params->OUTPUT_H;
    conv_params->input_w += first_unfinished_value_offset / slice_size_column * conv_params->stride;
    first_unfinished_value_offset %= slice_size_column;

    conv_params->input_h += first_unfinished_value_offset / cur_output_tile_c * conv_params->stride;
    first_unfinished_value_offset %= cur_output_tile_c;

    conv_params->conv_idx += first_unfinished_value_offset;

    my_printf_debug("initial output N = %d" NEWLINE, conv_params->input_tile_c_index);
    my_printf_debug("initial output H = %d" NEWLINE, conv_params->input_h / conv_params->stride);
    my_printf_debug("initial output W = %d" NEWLINE, conv_params->input_w / conv_params->stride);
    my_printf_debug("initial output C = %d" NEWLINE, conv_params->conv_idx);
#endif

    for (; conv_params->input_tile_c_offset < CHANNEL; conv_params->input_tile_c_offset += input_tile_c, conv_params->input_tile_c_index++) {
        conv_params->cur_input_tile_c = MIN_VAL(input_tile_c, CHANNEL - conv_params->input_tile_c_offset);
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

        while (conv_params->conv_idx_base < OUTPUT_CHANNEL) {
            for (; conv_params->input_w < W - conv_params->offset_w; conv_params->input_w += conv_params->stride) {
                for (; conv_params->input_h < H - conv_params->offset_h; conv_params->input_h += conv_params->tile_h) {
                    handle_conv_inner_loop(model, conv_params);
                }
                conv_params->input_h = conv_params->offset_h;
            }
            conv_params->input_w = conv_params->offset_w;
            conv_params->conv_idx_base += output_tile_c;
            conv_params->conv_idx = conv_params->conv_idx_base;
        }
        conv_params->conv_idx = conv_params->conv_idx_base = 0;
    }

#ifdef WITH_PROGRESS_EMBEDDING
    flip_state_bit(model, output);
#endif

#if MY_DEBUG >= 2
    uint32_t tiling_results_len = OUTPUT_CHANNEL * conv_params->OUTPUT_H * conv_params->OUTPUT_W;

    my_printf_debug("handle_conv output" NEWLINE);
    for (uint16_t input_tile_c_index = 0; input_tile_c_index * input_tile_c < CHANNEL; input_tile_c_index++) {
        dump_params_nhwc_debug(model, output, input_tile_c_index * tiling_results_len);
    }
#endif
}

void alloc_convmerge(Model *model, ParameterInfo *input[], ParameterInfo *output, NodeFlags*) {
    ParameterInfo *data = input[0];

    my_memcpy(output, data, sizeof(struct ParameterInfo));

    uint16_t OUTPUT_CHANNEL = data->dims[1],
             OUTPUT_H = data->dims[2],
             OUTPUT_W = data->dims[3];

    output->slot = get_next_slot(model, data);
    output->params_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W * sizeof(int16_t);
}

class ConvMergeInputChunkHandler : public ChunkHandler {
public:
    ConvMergeInputChunkHandler(int16_t *_to_add, uint16_t _data_offset)
        : to_add(_to_add), data_offset(_data_offset) {}

    void operator () (uint32_t range_offset, uint16_t range_len, uint8_t state_bit) const override {
        my_printf_debug("input range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
        int16_t *to_offset = to_add + range_offset - data_offset;
        if (state_bit) {
            my_offset_q15(to_offset, -0x4000, to_offset, range_len);
        }
    }

private:
    int16_t *to_add;
    uint16_t data_offset;
};

class ConvMergeOutputChunkHandler : public ChunkHandler {
public:
    ConvMergeOutputChunkHandler(uint32_t _tiling_results_offset)
        : tiling_results_offset(_tiling_results_offset) {}

    void operator () (uint32_t range_offset, uint16_t range_len, uint8_t state_bit) const override {
        my_printf_debug("output range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
        int16_t *to_offset = lea_buffer + range_offset - tiling_results_offset;
        // output state bit has not been flipped yet
        if (!state_bit) {
            my_offset_q15(to_offset, 0x4000, to_offset, range_len);
        }
    }

private:
    uint32_t tiling_results_offset;
};

void handle_convmerge(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, NodeFlags*) {
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

    uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / n_tiles_c / 2 * 2);

    uint16_t overflow_factor = find_overflow_factor(model, data) * n_tiles_c;
    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, 1.0f * SCALE / overflow_factor);

    // XXX: use iterate_chunks() for the outer loop?
    for (uint32_t tiling_results_offset = 0; tiling_results_offset < tiling_results_len; tiling_results_offset += chunk_len) {
        uint32_t real_chunk_len = MIN_VAL(chunk_len, tiling_results_len - tiling_results_offset);
        my_printf_debug("real_chunk_len = %d" NEWLINE, real_chunk_len);
        for (uint16_t input_tile_c_index = 0; input_tile_c_index < n_tiles_c; input_tile_c_index++) {
            int16_t *to_add = lea_buffer + input_tile_c_index * chunk_len;
            uint16_t data_offset = input_tile_c_index * tiling_results_len + tiling_results_offset;
            my_memcpy_from_param(to_add, data, data_offset, real_chunk_len * sizeof(int16_t));
#ifdef WITH_PROGRESS_EMBEDDING
            iterate_chunks(model, data, data_offset, real_chunk_len, ConvMergeInputChunkHandler(to_add, data_offset));
#endif // WITH_PROGRESS_EMBEDDING
            // scale up results as in convolution values are scaled down twice (input & weights)
            my_printf_debug("Before my_scale_q15" NEWLINE);
            dump_matrix_debug(to_add, real_chunk_len, ValueInfo(data));
            my_scale_q15(to_add, scaleFract, shift, to_add, real_chunk_len);
            my_printf_debug("After my_scale_q15" NEWLINE);
            dump_matrix_debug(to_add, real_chunk_len, ValueInfo(data));
            if (input_tile_c_index != 0) {
                my_add_q15(lea_buffer, to_add, lea_buffer, real_chunk_len);
            }
        }
#ifdef WITH_PROGRESS_EMBEDDING
        iterate_chunks(model, output, tiling_results_offset, real_chunk_len, ConvMergeOutputChunkHandler(tiling_results_offset));
#endif
        my_memcpy_to_param(output, tiling_results_offset, lea_buffer, real_chunk_len * sizeof(int16_t));
    }

    my_printf_debug("After scaling up back and merging tiling results" NEWLINE);

    output->scale = output->scale * overflow_factor / SCALE;

    setOutputValue(0);

#ifdef WITH_PROGRESS_EMBEDDING
    flip_state_bit(model, output);
#endif

    dump_params_nhwc_debug(model, output, 0);
}
