#include <inttypes.h> // for PRId32
#include "cnn_common.h"
#include "my_debug.h"
#include "op_utils.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"

// TODO: make these adjustable on runtime
#if !USE_ARM_CMSIS
#define OUTPUT_LEN 100
#else
#define OUTPUT_LEN 256
#endif

/* Better to not use macros
 * https://stackoverflow.com/a/3437484/3786245
 */
static inline int16_t int16_min(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t int16_max(int16_t a, int16_t b) {
    return a > b ? a : b;
}

#define CONV_TASK_FLAG_PROCESSED_FILTERS_BASE 2
typedef struct ConvTaskParams {
    Model* model;
    const ParameterInfo *conv_input;
    const ParameterInfo *real_conv_input; // for separate channel tiling
    const ParameterInfo *conv_filter;
    const ParameterInfo *conv_bias;
    ParameterInfo *output;
    const NodeFlags* flags;

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
    uint16_t N_FILTERS;
    uint16_t stride;
    // offset_h and offset_w to handle auto_pad=VALID
    uint8_t offset_h;
    uint8_t offset_w;
    uint16_t input_tile_c_offset;
    uint16_t input_tile_c_index;
    uint16_t tile_h;
    uint8_t cur_input_tile_c;
    uint16_t cur_filter_tile_c;
    uint16_t n_tiles_c;
    uint16_t dest_offset;
    uint16_t filter_offset;
    uint8_t truncated;
#if INDIRECT_RECOVERY
    int16_t old_output_offset ;
    uint8_t turning_point_idx;
    uint16_t next_turning_point;
    SlotInfo* cur_slot_info;
#endif
#if JAPARI
    uint8_t conv_input_has_footprints;
    uint16_t input_tile_c_offset_with_footprints;
    uint8_t force_align_footprints;
#endif
#if STATEFUL
    uint8_t output_padding;
#endif

    uint16_t filter_idx;
    uint16_t filter_tile_index;
    uint16_t input_h;
    uint16_t input_w;
    int16_t *filter_buffer_addr;
    int16_t cached_filter_idx;
    uint16_t cached_input_tile_c_offset;
} ConvTaskParams;

static ConvTaskParams conv_params_obj;

int16_t * const matrix_mpy_results = lea_buffer + LEA_BUFFER_SIZE - OUTPUT_LEN;

#if INDIRECT_RECOVERY
static void flip_filter_state_bits(ConvTaskParams *conv_params, uint16_t n_filters, uint16_t len, uint8_t first_round) {
    MY_ASSERT(len < OUTPUT_LEN);
    my_printf_debug("Flipping %d state bits in filters" NEWLINE, len);
#if STATEFUL
    int16_t *to_flip_state_bits = conv_params->filter_buffer_addr + n_filters * conv_params->filter_offset;
    if (first_round) {
        to_flip_state_bits -= len;
    } else {
        to_flip_state_bits -= n_filters;
    }
    // need negating filter value here as it will be multiplied with _Q15(-1.0), or -32768
    int16_t offset = get_value_state_bit(-*(to_flip_state_bits + BATCH_SIZE - 1)) ? 0x4000 : -0x4000;
    my_offset_q15_batched(to_flip_state_bits, offset, to_flip_state_bits, len);
#endif
#if JAPARI
    int16_t *to_flip_state_bits = conv_params->filter_buffer_addr + n_filters * (conv_params->filter_offset - 1);
    if (first_round) {
        for (uint16_t idx = BATCH_SIZE; idx < n_filters; idx += BATCH_SIZE + 1) {
            if (idx < n_filters - len) {
                continue;
            }
            to_flip_state_bits[idx] = -to_flip_state_bits[idx];
        }
    } else {
        for (uint16_t idx = BATCH_SIZE; idx < len; idx += BATCH_SIZE + 1) {
            to_flip_state_bits[idx] = -to_flip_state_bits[idx];
        }
    }
#endif
}
#endif

static void convTask(uint16_t offset_h, ConvTaskParams *conv_params) {
    // cur_output_tile_c should be signed, or MAX_VAL below is broken with TI's compiler
    int16_t output_tile_c = conv_params->flags->extra.conv.output_tile_c;
    int16_t cur_output_tile_c = output_tile_c - conv_params->filter_idx % output_tile_c;
    my_printf_debug("cur_output_tile_c = %d" NEWLINE, cur_output_tile_c);
    MY_ASSERT(cur_output_tile_c > 0);

    int16_t n_filters = cur_output_tile_c;
#if !HAWAII
    int16_t values_to_preserve = n_filters;
#endif
    int16_t channel_offset_c = conv_params->filter_idx;
#if JAPARI
    values_to_preserve = extend_for_footprints(n_filters, conv_params->force_align_footprints);
    n_filters = padding_for_lea(values_to_preserve);
    channel_offset_c = extend_for_footprints(channel_offset_c);
#endif
#if STATEFUL
    if (conv_params->output_padding) {
        values_to_preserve += conv_params->output_padding;
        n_filters = padding_for_lea(values_to_preserve);
    }
#endif
    // use NWHC so that output is written continuously on the address space
    uint16_t cur_output_data_offset =
             conv_params->OUTPUT_W * conv_params->OUTPUT_H * (conv_params->input_tile_c_index * conv_params->OUTPUT_CHANNEL) +   // n
             conv_params->input_w / conv_params->stride * conv_params->OUTPUT_H * conv_params->OUTPUT_CHANNEL +       // w
             (conv_params->input_h + offset_h) / conv_params->stride * conv_params->OUTPUT_CHANNEL +                  // h
             channel_offset_c;                                                                                   // c

#if INDIRECT_RECOVERY
    SlotInfo *cur_slot_info = conv_params->cur_slot_info;
    int16_t n_keep_state_bits = n_filters;
    if (conv_params->turning_point_idx <= cur_slot_info->n_turning_points && conv_params->next_turning_point != INVALID_TURNING_POINT) {
        my_printf_debug("next_turning_point = %d" NEWLINE, conv_params->next_turning_point);
        uint16_t ending_offset = MAX_VAL(conv_params->next_turning_point, cur_output_data_offset);
        if (ending_offset < cur_output_data_offset + n_filters) {
            n_keep_state_bits -= cur_output_data_offset + n_filters - ending_offset;
        }
    }
    my_printf_debug("n_keep_state_bits = %d" NEWLINE, n_keep_state_bits);
    MY_ASSERT(n_keep_state_bits >= 0);
#endif

    /* copy filter data */
    if (conv_params->cached_filter_idx != conv_params->filter_idx || conv_params->cached_input_tile_c_offset != conv_params->input_tile_c_offset) {
        conv_params->filter_buffer_addr = matrix_mpy_results - conv_params->filter_offset * (n_filters + TEMP_FILTER_WIDTH);
        my_fill_q15(0, conv_params->filter_buffer_addr, conv_params->filter_offset * n_filters);

        int16_t *filter_tmp = matrix_mpy_results - conv_params->filter_offset; // before transpose
        uint16_t fill_length = conv_params->filter_offset;
        my_fill_q15(0, filter_tmp, fill_length);
        uint16_t buffer_size = sizeof(int16_t) * conv_params->cur_filter_tile_c;
        uint16_t filter_len = conv_params->kH * conv_params->kW * conv_params->CHANNEL;
        for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
            uint16_t filter_src_offset = (conv_params->filter_idx + idx) * filter_len;
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->filter_idx + idx);
            for (uint16_t h = 0; h < conv_params->kH; h++) {
                int16_t *filter_dest_ptr = filter_tmp + h * conv_params->dest_offset;
                uint16_t cur_filter_src_offset = filter_src_offset + h * conv_params->kW * conv_params->CHANNEL + conv_params->input_tile_c_offset;
                for (uint16_t w = 0; w < conv_params->kW; w++) {
                    my_memcpy_from_param(conv_params->model, filter_dest_ptr, conv_params->conv_filter, cur_filter_src_offset, buffer_size);
                    filter_dest_ptr += conv_params->cur_filter_tile_c;
                    cur_filter_src_offset += conv_params->CHANNEL;
                }
            }
#if STATEFUL
            if (((!conv_params->old_output_offset && idx < n_keep_state_bits) || (conv_params->old_output_offset && idx >= n_keep_state_bits)) &&
                    ((BATCH_SIZE == 1 || ((cur_output_data_offset + idx) % BATCH_SIZE == BATCH_SIZE - 1)))) {
                my_printf_debug("Adding state bit for newly loaded filter idx=%d" NEWLINE, idx);
                filter_tmp[conv_params->filter_offset - 1] = -0x4000;
            } else
#endif
            {
                // XXX: why is this needed? Should already be zero with my_fill_q15 above
                filter_tmp[conv_params->filter_offset - 1] = 0;
            }
            if (conv_params->input_tile_c_index == 0) {
                // convert int16_t to int32_t first as on MSP430, registers are 20 bit while there are only 16 bits when int16_t is converted to uint16_t
                // If the dividend is negative, the quotient is wrong
                int16_t bias_val = -static_cast<int32_t>(get_q15_param(conv_params->model, conv_params->conv_bias, conv_params->filter_idx + idx)) / conv_params->conv_input->scale;
#if !STATEFUL
                filter_tmp[conv_params->filter_offset - 1] = bias_val;
#else
                filter_tmp[conv_params->filter_offset - 1] += bias_val;
#endif
            }

            uint16_t channel = idx;
#if JAPARI
            channel += channel / BATCH_SIZE;
#endif
            my_interleave_q15(filter_tmp, channel, n_filters, conv_params->filter_buffer_addr, conv_params->filter_offset);
        }

#if JAPARI
        int16_t* footprint_channels_ptr = conv_params->filter_buffer_addr + n_filters * (conv_params->filter_offset - 1);
        for (int16_t idx = BATCH_SIZE; idx < n_filters; idx += BATCH_SIZE + 1) {
            if (idx < n_keep_state_bits) {
                *(footprint_channels_ptr + idx) = (conv_params->old_output_offset ? 1 : -1);
            } else {
                *(footprint_channels_ptr + idx) = (conv_params->old_output_offset ? -1 : 1);
            }
        }
#endif

#if STATEFUL
        if (conv_params->output_padding &&
            ((!conv_params->old_output_offset && n_filters - 1 < n_keep_state_bits) ||
              (conv_params->old_output_offset && n_filters - 1 >= n_keep_state_bits))) {
            conv_params->filter_buffer_addr[n_filters * conv_params->filter_offset - 1] = -0x4000;
        }
#endif

        conv_params->cached_filter_idx = conv_params->filter_idx;
        conv_params->cached_input_tile_c_offset = conv_params->input_tile_c_offset;
    } else {
#if INDIRECT_RECOVERY
        if (n_keep_state_bits != n_filters) {
            int16_t n_flip_state_bits = n_filters - n_keep_state_bits;
            flip_filter_state_bits(conv_params, n_filters, n_flip_state_bits, 1);
        }
#endif
    }

    int16_t *filter_buffer_addr = conv_params->filter_buffer_addr;

    int16_t *input_buffer_addr = lea_buffer + offset_h * conv_params->dest_offset;

    uint16_t A_rows, A_cols, B_rows, B_cols;
    A_rows = 1;
    A_cols = B_rows = conv_params->filter_offset;
    B_cols = n_filters;
    MY_ASSERT(A_rows * B_cols <= OUTPUT_LEN);
    MY_ASSERT(input_buffer_addr + A_rows * A_cols <= filter_buffer_addr);
#if HAWAII
    my_matrix_mpy_q15(A_rows, A_cols, B_rows, B_cols, input_buffer_addr, filter_buffer_addr, matrix_mpy_results, nullptr, 0, 0);
#else
    my_matrix_mpy_q15(A_rows, A_cols, B_rows, B_cols, input_buffer_addr, filter_buffer_addr, matrix_mpy_results,
                      conv_params->output, cur_output_data_offset, values_to_preserve);
#endif

    /* START dump data */
    my_printf_debug("input_h=%d" NEWLINE, conv_params->input_h + offset_h);
    my_printf_debug("filter_idx=");
#if MY_DEBUG >= 1
    for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
        my_printf_debug("%d ", conv_params->filter_idx + idx);
        MY_ASSERT(conv_params->filter_idx + idx < conv_params->N_FILTERS);
    }
#endif
    my_printf_debug("output_h=%d ", (conv_params->input_h + offset_h) / conv_params->stride);
    my_printf_debug("output_w=%d" NEWLINE, conv_params->input_w / conv_params->stride);

    my_printf_debug("input_buffer_addr = lea_buffer + %d" NEWLINE, static_cast<int>(input_buffer_addr - lea_buffer));
    my_printf_debug("input" NEWLINE);
    dump_matrix2_debug(input_buffer_addr, A_rows, A_cols, ValueInfo(conv_params->conv_input));
    my_printf_debug("filter_buffer_addr = lea_buffer + LEA_BUFFER_SIZE - %d" NEWLINE, static_cast<int>(lea_buffer + LEA_BUFFER_SIZE - filter_buffer_addr));
    my_printf_debug("filter" NEWLINE);
    dump_matrix2_debug(filter_buffer_addr, B_rows, B_cols, ValueInfo(conv_params->conv_filter));

    my_printf_debug("matrix_mpy_results" NEWLINE);
    dump_matrix2_debug(matrix_mpy_results, A_rows, B_cols, ValueInfo(conv_params->output));
    my_printf_debug(NEWLINE);

    compare_vm_nvm(matrix_mpy_results, conv_params->model, conv_params->output, cur_output_data_offset, values_to_preserve);
    /* END dump data */

    my_printf_debug("output_data offset = %d" NEWLINE, cur_output_data_offset);

    MY_ASSERT(cur_output_data_offset + n_filters < INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);

#if HAWAII
    uint16_t batch_offset = 0;
    for (uint16_t row = 0; row < A_rows; row++) {
        batch_offset += hawaii_preserve_vector(conv_params->model, conv_params->output, cur_output_data_offset + batch_offset, matrix_mpy_results + batch_offset, B_cols);
    }
#endif

#if INDIRECT_RECOVERY
    if (n_keep_state_bits != n_filters) {
        check_next_turning_point(conv_params->old_output_offset, conv_params->turning_point_idx,
                                 conv_params->next_turning_point, conv_params->cur_slot_info, cur_output_data_offset + conv_params->OUTPUT_CHANNEL);
        my_printf_debug("old_output_offset flipped to %d" NEWLINE, conv_params->old_output_offset);

        flip_filter_state_bits(conv_params, n_filters, n_keep_state_bits, 0);
    }
#endif
}

static inline uint16_t load_input_vector(uint32_t src_addr, int16_t* dest_addr, uint16_t len, const ConvTaskParams* conv_params) {
    my_printf_debug("Load %d IFM values from range [%d, %d) ",
                    len, src_addr, static_cast<int>(src_addr + len));
    int16_t* memcpy_dest_addr = nullptr;
    uint16_t loaded_len = 0;

    MY_ASSERT(len != 0);

#if JAPARI
    if (conv_params->conv_input_has_footprints) {
        memcpy_dest_addr = input_buffer_with_footprints;
    } else
#endif
    {
        memcpy_dest_addr = dest_addr;
        loaded_len = len;
    }
    my_memcpy_from_param(
        conv_params->model, memcpy_dest_addr,
        conv_params->real_conv_input, src_addr,
        len * sizeof(int16_t));
#if JAPARI
    if (conv_params->conv_input_has_footprints) {
        // Use nested loops as skipping footprints by `% (BATCH_SIZE)` is quite slow on boards
        int16_t *dest_ptr = dest_addr,
                *src_ptr = input_buffer_with_footprints;
        for (uint16_t src_idx = 0; src_idx < len; src_idx += (BATCH_SIZE + 1)) {
            for (uint8_t batch_offset = 0; batch_offset < BATCH_SIZE; batch_offset++) {
                *dest_ptr = *src_ptr;
                dest_ptr++;
                src_ptr++;
            }
            src_ptr++; // skipping footprints
        }
        loaded_len = dest_ptr - dest_addr;
    }
#endif

#if MY_DEBUG >= 1
    for (uint16_t idx = 0; idx < loaded_len; idx++) {
        my_printf_debug("%d ", dest_addr[idx]);
    }
    my_printf_debug(NEWLINE);
#endif
    return loaded_len;
}

static void handle_conv_inner_loop(Model *model, ConvTaskParams *conv_params) {
    int8_t field_size = (conv_params->kH - 1) / 2;

    /* copy input data, row by row */

    int8_t real_input_index = -1;
    if (conv_params->conv_input->flags & SEPARATE_TILING) {
        real_input_index = (2 * conv_params->input_tile_c_index >= conv_params->n_tiles_c) ? 1 : 0;
        conv_params->real_conv_input = get_parameter_info(conv_params->conv_input->extra_info[real_input_index]);
    } else {
        conv_params->real_conv_input = conv_params->conv_input;
    }

    /* int32_t instead of int16_t as TI's compiler cannot handle negative
     * offsets correctly. The expression `ptr + (int16_t)(-2)` is
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
    int16_t max_n_filters = conv_params->flags->extra.conv.output_tile_c;
#if JAPARI
    max_n_filters *= 2;
#endif
    // TEMP_FILTER_WIDTH additional filters for values before transpose
    uint16_t inputs_len = MIN_VAL(
        LEA_BUFFER_SIZE - OUTPUT_LEN - (max_n_filters + TEMP_FILTER_WIDTH) * conv_params->filter_offset,
        (conv_params->tile_h + 2 * field_size) * conv_params->dest_offset
    );
    MY_ASSERT(inputs_len < LEA_BUFFER_SIZE); // make sure no overflow occurs in the previous line

    dest = lea_buffer;

    int32_t h_start = int16_max(                                       -field_size, -conv_params->input_h),
            h_end =   int16_min(conv_params->tile_h-conv_params->stride+field_size,      conv_params->H-1);

    my_printf_debug("Reinitialize input buffer" NEWLINE "inputs_len = %d" NEWLINE, inputs_len);

    my_fill_q15(0, lea_buffer, inputs_len);

    dest += (h_start + field_size) * conv_params->dest_offset;

    my_printf_debug("h_start=%" PRId32 " ", h_start);
    my_printf_debug("h_end=%" PRId32 NEWLINE, h_end);

    uint16_t cur_input_tile_c = conv_params->cur_input_tile_c;
    uint8_t im2col_channel_offset = cur_input_tile_c;
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    static_cast<int>(dest - lea_buffer));
    uint16_t cur_input_channel = conv_params->CHANNEL;
    if (conv_params->conv_input->flags & SEPARATE_TILING) {
        cur_input_channel /= 2;
    }
#if JAPARI
    if (conv_params->conv_input_has_footprints) {
        cur_input_tile_c = extend_for_footprints(cur_input_tile_c);
        cur_input_channel = extend_for_footprints(cur_input_channel);
    }
#endif
    int16_t input_src_offset = (conv_params->input_h + h_start) * conv_params->W * cur_input_channel + (conv_params->input_w + w_start) * cur_input_channel;
#if JAPARI
    input_src_offset += conv_params->input_tile_c_offset_with_footprints;
#else
    input_src_offset += conv_params->input_tile_c_offset;
#endif
    if (real_input_index == 1) {
        input_src_offset -= cur_input_channel;
    }
#if INDIRECT_RECOVERY
    dump_turning_points_debug(model, conv_params->real_conv_input);
#endif
    for (int32_t h = h_start; h <= h_end; h++) {
        int16_t *dest_addr = dest + (w_start + field_size) * im2col_channel_offset;
#if STATEFUL
        int16_t *orig_dest_addr = dest_addr;
#endif
        uint16_t input_row_len = (w_end - w_start + 1) * cur_input_tile_c;
        uint32_t src_addr = input_src_offset;
        if (cur_input_tile_c == cur_input_channel) {
            load_input_vector(src_addr, dest_addr, input_row_len, conv_params);
        } else {
            for (int32_t w = w_start; w <= w_end; w++) {
                load_input_vector(src_addr, dest_addr, cur_input_tile_c, conv_params);
                dest_addr += im2col_channel_offset;
                src_addr += cur_input_channel;
            }
        }

#if STATEFUL
        // stripping states inside the h loop is faster as biases multipliers can be skipped
        int16_t *input_row_end = orig_dest_addr + input_row_len;
        uint8_t start_state = get_value_state_bit(*(orig_dest_addr + BATCH_SIZE - 1));
        // if input_tile_c is smaller than BATCH_SIZE, state bits are not always at offset BATCH_SIZE - 1
        if (conv_params->flags->extra.conv.input_tile_c >= BATCH_SIZE && start_state == get_value_state_bit(*(input_row_end - 1))) {
            // XXX: a heuristic - assume there is at most one turning points in a row
            my_printf_debug("Using my_offset_q15 for stripping state bits" NEWLINE);
            if (start_state) {
                my_offset_q15_batched(orig_dest_addr, -0x4000, orig_dest_addr, input_row_len);
            }
        } else {
            my_printf_debug("Using a loop for stripping state bits" NEWLINE);
            for (int16_t *dest_ptr = orig_dest_addr; dest_ptr < input_row_end; dest_ptr++) {
                int16_t val = *dest_ptr;
                if (get_value_state_bit(val)) {
                    *dest_ptr = val - 0x4000;
                }
            }
        }
#endif
        dest += conv_params->dest_offset;
        input_src_offset += conv_params->W * cur_input_channel;
    }
    if (conv_params->real_conv_input->scale != conv_params->conv_input->scale) {
        int16_t scaleFract;
        uint8_t shift;
        float_to_scale_params(&scaleFract, &shift, 1.0f * conv_params->real_conv_input->scale / conv_params->conv_input->scale);
        my_scale_q15(lea_buffer, scaleFract, shift, lea_buffer, inputs_len);
    }
#if STATEFUL && MY_DEBUG >= 1
    int16_t *ptr = lea_buffer;
    for (size_t idx = 0; idx < inputs_len; idx++) {
        MY_ASSERT(!get_value_state_bit(*ptr), "Input index %d has value with unexpected state: %d" NEWLINE, idx, *ptr);
        ptr++;
    }
#endif
    uint16_t bias_multipler_offset = conv_params->dest_offset - 1;
    while (bias_multipler_offset < inputs_len) {
        lea_buffer[bias_multipler_offset] = -0x8000; // _Q15(-1.0)
        bias_multipler_offset += conv_params->dest_offset;
    }

    my_printf_debug("Loaded inputs" NEWLINE);
    // state = 0 as state bits are already removed by my_offset_q15 above
    dump_matrix_debug(lea_buffer, inputs_len, ValueInfo(conv_params->real_conv_input));

    uint16_t cur_tile_h = MIN_VAL(conv_params->H - conv_params->offset_h - conv_params->input_h, conv_params->tile_h);
    for (uint16_t j = 0; j < cur_tile_h; j += conv_params->stride) {
        // filter_idx is set to initial_c in handle_conv
        convTask(j, conv_params);
        // reset here for further processing
        conv_params->filter_idx = conv_params->filter_tile_index * conv_params->flags->extra.conv.output_tile_c;
    }
}

void alloc_conv(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    const ParameterInfo *conv_input = input[0], *conv_filter = input[1];

    MY_ASSERT(conv_input->bitwidth == 16 && conv_filter->bitwidth == 16);

#if !JAPARI
    // skip the check for JAPARI as it is too complex
    MY_ASSERT(conv_input->dims[1] == conv_filter->dims[1]);
#endif

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t CHANNEL = conv_input->dims[1], H = conv_input->dims[2], W = conv_input->dims[3];
    uint16_t OUTPUT_CHANNEL = conv_filter->dims[0];

    ConvTaskParams *conv_params = &conv_params_obj;

    conv_params->model = model;
    conv_params->flags = flags;

    conv_params->kH = conv_filter->dims[2];
    conv_params->kW = conv_filter->dims[3];
    // XXX: many places in this file assume odd kernel sizes...
    MY_ASSERT(conv_params->kH % 2 ==1);
    MY_ASSERT(conv_params->kW % 2 ==1);

    conv_params->stride = flags->stride;
    if (flags->generic == AUTO_PAD_VALID) {
        conv_params->offset_h = conv_params->kH / 2;
        conv_params->offset_w = conv_params->kW / 2;
        conv_params->OUTPUT_H = (H - conv_params->kH) / conv_params->stride + 1;
        conv_params->OUTPUT_W = (W - conv_params->kW) / conv_params->stride + 1;
    } else {
        conv_params->offset_h = conv_params->offset_w = 0;
        // By definition, output_shape[i] = ceil(input_shape[i] / strides[i]) for SAME
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv
        conv_params->OUTPUT_H = (H + conv_params->stride - 1) / conv_params->stride;
        conv_params->OUTPUT_W = (W + conv_params->stride - 1) / conv_params->stride;
    }

#if JAPARI
    conv_params->force_align_footprints = (OUTPUT_CHANNEL % BATCH_SIZE != 0);
    OUTPUT_CHANNEL = extend_for_footprints(OUTPUT_CHANNEL, conv_params->force_align_footprints);
    if (has_footprints(conv_input)) {
        conv_params->n_tiles_c = CHANNEL / (BATCH_SIZE + 1) * BATCH_SIZE / flags->extra.conv.input_tile_c;
    } else
#endif
    {
        conv_params->n_tiles_c = CHANNEL / flags->extra.conv.input_tile_c;
    }
#if STATEFUL
    if (flags->extra.conv.output_tile_c % BATCH_SIZE) {
        conv_params->output_padding = BATCH_SIZE - flags->extra.conv.output_tile_c % BATCH_SIZE;
    } else {
        conv_params->output_padding = 0;
    }
    OUTPUT_CHANNEL += conv_params->output_padding;
#endif
    my_printf_debug("input_tile_c=%d, output_tile_c=%d" NEWLINE, flags->extra.conv.input_tile_c, flags->extra.conv.output_tile_c);

    /* XXX: extend flags; assume dilation=(1, 1) for now */
    output->bitwidth = 16;
    output->slot = get_next_slot(model, conv_input);
    output->params_len = conv_params->n_tiles_c * conv_params->OUTPUT_H * conv_params->OUTPUT_W * OUTPUT_CHANNEL * sizeof(int16_t);
    output->dims[0] = 1;
    output->dims[1] = OUTPUT_CHANNEL;
    output->dims[2] = conv_params->OUTPUT_H;
    output->dims[3] = conv_params->OUTPUT_W;
    output->flags |= TRANSPOSED;
    output->flags &= ~SEPARATE_TILING;
    output->scale = conv_input->scale * conv_filter->scale;
}

void handle_conv(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    const ParameterInfo *conv_input = input[0], *conv_filter = input[1], *conv_bias = input[2];
    my_printf_debug("Conv!" NEWLINE);

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   CHANNEL = conv_filter->dims[1];

    ConvTaskParams *conv_params = &conv_params_obj;

    conv_params->tile_h = MIN_VAL(H, DEFAULT_TILE_H * conv_params->stride);

    my_printf_debug("n_tiles_c = %d" NEWLINE, conv_params->n_tiles_c);

    conv_params->conv_input = conv_input;
    conv_params->conv_filter = conv_filter;
    conv_params->conv_bias = conv_bias;
    conv_params->output = output;
    conv_params->filter_buffer_addr = NULL;
    conv_params->cached_filter_idx = -1;
    conv_params->H = H;
    conv_params->W = W;
#if JAPARI
    conv_params->conv_input_has_footprints = has_footprints(conv_input);
#endif

    conv_params->CHANNEL = CHANNEL;
    conv_params->OUTPUT_CHANNEL = output->dims[1];
    conv_params->N_FILTERS = conv_filter->dims[0];

    conv_params->input_tile_c_offset = 0;
    conv_params->input_tile_c_index = 0;
    conv_params->input_w = conv_params->offset_w;
    conv_params->input_h = conv_params->offset_h;
    conv_params->filter_tile_index = 0;
    conv_params->filter_idx = 0;
#if INTERMITTENT

    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    uint32_t first_unfinished_value_offset = job_index_to_offset(output, first_unfinished_job_idx);
#if JAPARI
    first_unfinished_value_offset -= BATCH_SIZE;
#else
    first_unfinished_value_offset -= (BATCH_SIZE - 1);
#endif

    fix_first_unfinished_value_offset(model, &first_unfinished_value_offset);

#if INDIRECT_RECOVERY
    find_initial_state_bit(&conv_params->old_output_offset, &conv_params->turning_point_idx, &conv_params->next_turning_point,
                           &conv_params->cur_slot_info, job_index_to_offset(output, first_unfinished_job_idx), model, output);

    my_printf_debug("old_output_offset = %d" NEWLINE, conv_params->old_output_offset);
#endif

    uint16_t cur_output_tile_c = flags->extra.conv.output_tile_c;
#if JAPARI
    cur_output_tile_c = extend_for_footprints(cur_output_tile_c, conv_params->force_align_footprints);
#endif
    uint16_t slice_size_input_channel_tiling = conv_params->OUTPUT_W * conv_params->OUTPUT_H * conv_params->OUTPUT_CHANNEL;

    conv_params->input_tile_c_index = first_unfinished_value_offset / slice_size_input_channel_tiling;
    // Not extending for JAPARI footprints here as input_tile_c_offset will be extended later
    conv_params->input_tile_c_offset = conv_params->input_tile_c_index * flags->extra.conv.input_tile_c;
    first_unfinished_value_offset %= slice_size_input_channel_tiling;

    conv_params->filter_tile_index = (first_unfinished_value_offset % conv_params->OUTPUT_CHANNEL) / cur_output_tile_c;
    conv_params->filter_idx = conv_params->filter_tile_index * flags->extra.conv.output_tile_c;

#if STATEFUL
    uint8_t filter_offset_in_tile = first_unfinished_value_offset % (cur_output_tile_c + conv_params->output_padding);
#else
    uint8_t filter_offset_in_tile = first_unfinished_value_offset % cur_output_tile_c;
#endif

#if JAPARI
    filter_offset_in_tile = filter_offset_in_tile / (BATCH_SIZE + 1) * BATCH_SIZE;
#endif
    conv_params->filter_idx += filter_offset_in_tile;
    first_unfinished_value_offset /= conv_params->OUTPUT_CHANNEL;

    conv_params->input_w += first_unfinished_value_offset / conv_params->OUTPUT_H * conv_params->stride;
    first_unfinished_value_offset %= conv_params->OUTPUT_H;

    conv_params->input_h += first_unfinished_value_offset * conv_params->stride;

    my_printf_debug("initial output N = %d" NEWLINE, conv_params->input_tile_c_index);
    my_printf_debug("initial output H = %d" NEWLINE, conv_params->input_h / conv_params->stride);
    my_printf_debug("initial output W = %d" NEWLINE, conv_params->input_w / conv_params->stride);
    my_printf_debug("initial output C = %d" NEWLINE, conv_params->filter_idx);
    // = happens when all values are finished
    MY_ASSERT(conv_params->input_tile_c_index <= conv_params->n_tiles_c);
#endif

    int16_t input_channels = conv_input->dims[1];
#if JAPARI
    if (conv_params->conv_input_has_footprints) {
        input_channels = input_channels / (BATCH_SIZE + 1) * BATCH_SIZE;
    }
#endif
    for (; conv_params->input_tile_c_offset < input_channels; conv_params->input_tile_c_offset += flags->extra.conv.input_tile_c) {
        conv_params->cur_input_tile_c = MIN_VAL(flags->extra.conv.input_tile_c, input_channels - conv_params->input_tile_c_offset);
        conv_params->cur_filter_tile_c = conv_params->cur_input_tile_c;
#if JAPARI
        conv_params->input_tile_c_offset_with_footprints = extend_for_footprints(conv_params->input_tile_c_offset);
#endif
        my_printf_debug("cur_input_tile_c = %d" NEWLINE, conv_params->cur_input_tile_c);
        conv_params->dest_offset = conv_params->kH * conv_params->cur_input_tile_c;
        // +1 for bias
        conv_params->dest_offset++;
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

        while (true) {
            for (; conv_params->input_w < W - conv_params->offset_w; conv_params->input_w += conv_params->stride) {
                for (; conv_params->input_h < H - conv_params->offset_h; conv_params->input_h += conv_params->tile_h) {
                    handle_conv_inner_loop(model, conv_params);
                }
                conv_params->input_h = conv_params->offset_h;
            }
            conv_params->input_w = conv_params->offset_w;
            conv_params->filter_tile_index++;
            if (conv_params->filter_tile_index * flags->extra.conv.output_tile_c >= conv_params->N_FILTERS) {
                break;
            }
            conv_params->filter_idx = conv_params->filter_tile_index * flags->extra.conv.output_tile_c;
#if INDIRECT_RECOVERY
            uint32_t new_output_offset = conv_params->input_tile_c_index * conv_params->OUTPUT_CHANNEL * conv_params->OUTPUT_H * conv_params->OUTPUT_W;
#if JAPARI
            new_output_offset += extend_for_footprints(conv_params->filter_idx);
#else
            new_output_offset += conv_params->filter_idx;
#endif
            find_initial_state_bit(&conv_params->old_output_offset, &conv_params->turning_point_idx, &conv_params->next_turning_point, &conv_params->cur_slot_info,
                                   new_output_offset, model, output);
#endif
        }
        conv_params->filter_idx = conv_params->filter_tile_index = 0;

        conv_params->input_tile_c_index++;
#if INDIRECT_RECOVERY
        find_initial_state_bit(&conv_params->old_output_offset, &conv_params->turning_point_idx, &conv_params->next_turning_point, &conv_params->cur_slot_info,
                               conv_params->input_tile_c_index * conv_params->OUTPUT_CHANNEL * conv_params->OUTPUT_H * conv_params->OUTPUT_W, model, output);
#endif
    }

    flip_state_bit(model, output);

    my_printf_debug("handle_conv output" NEWLINE);
    dump_params_nhwc_debug(model, output);
}

void alloc_convmerge(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *data = input[0];

    uint16_t OUTPUT_CHANNEL = data->dims[1],
             OUTPUT_H = data->dims[2],
             OUTPUT_W = data->dims[3];

    output->slot = get_next_slot(model, data);
    output->params_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W * sizeof(int16_t);
}

#if STATEFUL
struct ConvMergeInputChunkHandlerParams {
    int16_t *to_add;
    uint16_t data_offset;
};

void ConvMergeInputChunkHandler(uint32_t range_offset, uint16_t range_len, uint8_t state_bit, void* _params) {
    ConvMergeInputChunkHandlerParams* params = reinterpret_cast<ConvMergeInputChunkHandlerParams*>(_params);
    my_printf_debug("input range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
    int16_t *to_offset = params->to_add + range_offset - params->data_offset;
    if (state_bit) {
        my_offset_q15_batched(to_offset, -0x4000, to_offset, range_len);
    }
}
#endif

#if JAPARI
struct ConvMergeOutputChunkHandlerParams {
    uint32_t tiling_results_offset;
};

void ConvMergeOutputChunkHandler(uint32_t range_offset, uint16_t range_len, uint8_t state_bit, void* _params) {
    ConvMergeOutputChunkHandlerParams* params = reinterpret_cast<ConvMergeOutputChunkHandlerParams*>(_params);
    my_printf_debug("output range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
    int16_t *to_offset = lea_buffer + range_offset - params->tiling_results_offset;
    uint16_t n_footprints = (range_len + BATCH_SIZE) / (BATCH_SIZE + 1);
    int16_t* footprint_buffer = lea_buffer + (LEA_BUFFER_SIZE - n_footprints) / 2 * 2;
    my_fill_q15((state_bit ? -1 : 1), footprint_buffer, n_footprints);
    my_interleave_q15(footprint_buffer, BATCH_SIZE - (range_offset % (BATCH_SIZE + 1)), BATCH_SIZE + 1, to_offset, n_footprints);
}
#endif

void handle_convmerge(struct Model *model, const ParameterInfo *input[], struct ParameterInfo *output, const NodeFlags*) {
    // XXX: make this function idempotent

    // Do not use conv_params here as its intialization in alloc_conv and
    // handle_conv might be skipped if the Conv node has finished.
    const ParameterInfo *data = input[0];
    uint16_t OUTPUT_CHANNEL = data->dims[1],
             OUTPUT_H = data->dims[2],
             OUTPUT_W = data->dims[3];

    my_printf_debug("ConvMerge!" NEWLINE);

    uint8_t n_tiles_c = data->params_len / sizeof(int16_t) / (OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W);

    MY_ASSERT(n_tiles_c);

    uint32_t tiling_results_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;

    // uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / n_tiles_c / 2 * 2);
    uint16_t chunk_len = OUTPUT_CHANNEL;
    uint32_t tiling_results_offset = 0;
#if INTERMITTENT
    uint32_t first_unfinished_job_index = run_recovery(model, output);

#if JAPARI
    uint16_t n_chunks = chunk_len / (BATCH_SIZE + 1) / 2 * 2;
    chunk_len = n_chunks * (BATCH_SIZE + 1);
#endif
    MY_ASSERT(chunk_len % 2 == 0);
    MY_ASSERT(chunk_len * n_tiles_c < LEA_BUFFER_SIZE);

    tiling_results_offset = first_unfinished_job_index;
#if JAPARI
    tiling_results_offset *= (BATCH_SIZE + 1);
#else
    tiling_results_offset *= BATCH_SIZE;
#endif

#endif

#if INDIRECT_RECOVERY
    int16_t old_output_offset;
    uint8_t output_turning_point_idx;
    uint16_t next_output_turning_point;
    SlotInfo *cur_output_slot_info;

    find_initial_state_bit(&old_output_offset, &output_turning_point_idx, &next_output_turning_point,
                           &cur_output_slot_info, job_index_to_offset(output, first_unfinished_job_index), model, output);

    my_printf_debug("old_output_offset = %d" NEWLINE, old_output_offset);
#endif

    float scale_f = 1.0 * find_max_multiplier(model, data) / n_tiles_c;
    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, scale_f);

    // XXX: use iterate_chunks() for the outer loop?
    for (; tiling_results_offset < tiling_results_len; tiling_results_offset += chunk_len) {
        uint16_t real_chunk_len = MIN_VAL(chunk_len, tiling_results_len - tiling_results_offset);
        my_printf_debug("real_chunk_len = %d" NEWLINE, real_chunk_len);
        for (uint16_t input_tile_c_index = 0; input_tile_c_index < n_tiles_c; input_tile_c_index++) {
            int16_t *to_add = lea_buffer + input_tile_c_index * chunk_len;
            uint16_t data_offset = input_tile_c_index * tiling_results_len + tiling_results_offset;
            my_memcpy_from_param(model, to_add, data, data_offset, real_chunk_len * sizeof(int16_t));
#if STATEFUL
            ConvMergeInputChunkHandlerParams params({to_add, data_offset});
            iterate_chunks(model, data, data_offset, real_chunk_len, ConvMergeInputChunkHandler, &params);
#endif
            // scale up results as in convolution values are scaled down twice (input & weights)
            my_printf_debug("Chunk offset %d, input tile %d" NEWLINE, tiling_results_offset, input_tile_c_index);
            my_printf_debug("Before my_scale_q15" NEWLINE);
            ValueInfo val_info_data(data);
            dump_matrix_debug(to_add, real_chunk_len, val_info_data);
            my_scale_q15(to_add, scaleFract, shift, to_add, real_chunk_len);
            my_printf_debug("After my_scale_q15" NEWLINE);
            val_info_data.scale /= scale_f;
            dump_matrix_debug(to_add, real_chunk_len, val_info_data);
            if (input_tile_c_index != 0) {
                my_add_q15(lea_buffer, to_add, lea_buffer, real_chunk_len);
            }
        }
#if INDIRECT_RECOVERY

#if STATEFUL
        if (!old_output_offset) {
            my_offset_q15_batched(lea_buffer, 0x4000, lea_buffer, MIN_VAL(next_output_turning_point - tiling_results_offset, real_chunk_len));
        } else if (next_output_turning_point < tiling_results_offset + real_chunk_len) {
            int16_t* to_offset = lea_buffer + next_output_turning_point - tiling_results_offset;
            my_offset_q15_batched(to_offset, 0x4000, to_offset, real_chunk_len - (next_output_turning_point - tiling_results_offset));
        }
        check_next_turning_point(old_output_offset, output_turning_point_idx,
                                 next_output_turning_point, cur_output_slot_info, tiling_results_offset + real_chunk_len);
#endif

#if JAPARI
        ConvMergeOutputChunkHandlerParams params({tiling_results_offset});
        iterate_chunks(model, output, tiling_results_offset, real_chunk_len, ConvMergeOutputChunkHandler, &params);
#endif

        my_printf_debug("After writing state bits in [%d, %d)" NEWLINE, tiling_results_offset, tiling_results_offset + real_chunk_len);
        dump_matrix_debug(lea_buffer, real_chunk_len, ValueInfo(output));
#endif

#if JAPARI
#endif
#if !HAWAII
        my_memcpy_to_param(output, tiling_results_offset, lea_buffer, real_chunk_len * sizeof(int16_t), 0);
#else
        hawaii_preserve_vector(model, output, tiling_results_offset, lea_buffer, real_chunk_len);
#endif
    }

    my_printf_debug("After scaling up back and merging tiling results" NEWLINE);

    output->scale /= scale_f;

    flip_state_bit(model, output);

    dump_params_nhwc_debug(model, output);
}
