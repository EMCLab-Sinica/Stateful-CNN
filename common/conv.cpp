#include <inttypes.h> // for PRId32
#include "cnn_common.h"
#include "my_debug.h"
#include "op_utils.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"

// TODO: make these adjustable on runtime
#define OUTPUT_LEN 100

// to make the code clearer
#define TEMP_FILTER_WIDTH 1

#if MY_DEBUG >= 1
static int16_t last_output_data_offset;
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
    uint16_t cur_input_tile_c;
    uint16_t cur_filter_tile_c;
    uint16_t n_tiles_c;
    uint16_t dest_offset;
    uint16_t filter_offset;
    uint8_t truncated;
#if STATEFUL
    int16_t old_output_offset ;
    uint8_t turning_point_idx;
    int16_t next_turning_point;
    SlotInfo* cur_slot_info;
#endif
#if JAPARI
    uint8_t conv_input_is_intermediate_data;
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

#if STATEFUL
static void flip_filter_state_bits(ConvTaskParams *conv_params, uint16_t n_filters, uint16_t len, uint8_t first_round) {
    MY_ASSERT(len < OUTPUT_LEN);
    my_printf_debug("Flipping %d state bits in filters" NEWLINE, len);
    // need negating filter value here as it will be multiplied with _Q15(-1.0), or -32768
    int16_t *to_flip_state_bits = conv_params->filter_buffer_addr + n_filters * conv_params->filter_offset;
    if (first_round) {
        to_flip_state_bits -= len;
    } else {
        to_flip_state_bits -= n_filters;
    }
    int16_t offset = get_value_state_bit(-*to_flip_state_bits) ? 0x4000 : -0x4000;
    my_offset_q15(to_flip_state_bits, offset, to_flip_state_bits, len);
}
#endif

static void convTask(uint16_t offset_h, ConvTaskParams *conv_params) {
    // cur_output_tile_c should be signed, or MAX_VAL below is broken with TI's compiler
    int16_t cur_output_tile_c = MIN_VAL(conv_params->output->tile_c, conv_params->N_FILTERS - conv_params->conv_idx);
    int16_t cur_output_tile_c_full = MIN_VAL(conv_params->output->tile_c, conv_params->N_FILTERS - conv_params->conv_idx_base);
    my_printf_debug("cur_output_tile_c = %d, cur_output_tile_c_full = %d" NEWLINE, cur_output_tile_c, cur_output_tile_c_full);
    MY_ASSERT(cur_output_tile_c > 0);
    MY_ASSERT(cur_output_tile_c_full > 0);

    int16_t n_filters = cur_output_tile_c;
    int16_t channel_offset_n = conv_params->conv_idx_base;
    int16_t channel_offset_c = conv_params->conv_idx - conv_params->conv_idx_base;
#if JAPARI
    n_filters *= 2;
    cur_output_tile_c_full *= 2;
    channel_offset_n *= 2;
    channel_offset_c *= 2;
#endif
    // use NWHC so that output is written continuously on the address space
    int16_t cur_output_data_offset =
            conv_params->OUTPUT_W * conv_params->OUTPUT_H * (conv_params->input_tile_c_index * conv_params->OUTPUT_CHANNEL + channel_offset_n) +   // n
            conv_params->input_w / conv_params->stride * conv_params->OUTPUT_H * cur_output_tile_c_full +       // w
            (conv_params->input_h + offset_h) / conv_params->stride * cur_output_tile_c_full +                  // h
            channel_offset_c;                                                                                   // c

#if STATEFUL
    SlotInfo *cur_slot_info = conv_params->cur_slot_info;
    int16_t n_keep_state_bits = n_filters;
    uint8_t need_cleanup_state_bits = 0;
    if (conv_params->turning_point_idx <= cur_slot_info->n_turning_points && conv_params->next_turning_point > 0) {
        my_printf_debug("next_turning_point = %d" NEWLINE, conv_params->next_turning_point);
        n_keep_state_bits -= MAX_VAL(0, cur_output_data_offset + n_filters - conv_params->next_turning_point);
    }
    my_printf_debug("n_keep_state_bits = %d" NEWLINE, n_keep_state_bits);
    MY_ASSERT(n_keep_state_bits >= 0);
#endif

    /* copy filter data */
    if (conv_params->cached_filter_idx != conv_params->conv_idx || conv_params->cached_input_tile_c_offset != conv_params->input_tile_c_offset) {
        conv_params->filter_buffer_addr = matrix_mpy_results - conv_params->filter_offset * (n_filters + TEMP_FILTER_WIDTH);
        int16_t *filter_tmp = matrix_mpy_results - conv_params->filter_offset; // before transpose
        uint16_t fill_length = conv_params->filter_offset;
        my_fill_q15(0, filter_tmp, fill_length);

        uint16_t buffer_size = sizeof(int16_t) * conv_params->cur_filter_tile_c;
        uint16_t filter_len = conv_params->kH * conv_params->kW * conv_params->CHANNEL;
        for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
            uint16_t filter_src_offset = (conv_params->conv_idx + idx) * filter_len;
            my_printf_debug("Copying filter %d" NEWLINE, conv_params->conv_idx + idx);
            for (uint16_t h = 0; h < conv_params->kH; h++) {
                int16_t *filter_dest_ptr = filter_tmp + h * conv_params->dest_offset;
                uint16_t cur_filter_src_offset = filter_src_offset + h * conv_params->kW * conv_params->CHANNEL;
#if JAPARI
                if (conv_params->conv_input_is_intermediate_data) {
                    cur_filter_src_offset += conv_params->input_tile_c_offset / 2;
                } else
#endif
                {
                    cur_filter_src_offset += conv_params->input_tile_c_offset;
                }
                for (uint16_t w = 0; w < conv_params->kW; w++) {
                    my_memcpy_from_param(conv_params->model, filter_dest_ptr, conv_params->conv_filter, cur_filter_src_offset, buffer_size);
                    filter_dest_ptr += conv_params->cur_filter_tile_c;
                    cur_filter_src_offset += conv_params->CHANNEL;
                }
            }
#if STATEFUL
            if ((!conv_params->old_output_offset && idx < n_keep_state_bits) || (conv_params->old_output_offset && idx >= n_keep_state_bits)) {
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
                filter_tmp[conv_params->filter_offset - 1] += -get_q15_param(conv_params->model, conv_params->conv_bias, conv_params->conv_idx + idx) / conv_params->conv_input->scale;
            }

            uint16_t channel = idx;
#if JAPARI
            channel *= 2;
#endif
            my_interleave_q15(filter_tmp, channel, n_filters, conv_params->filter_buffer_addr, conv_params->filter_offset);
        }
#if JAPARI
        for (int16_t idx = 0; idx < cur_output_tile_c; idx++) {
            conv_params->filter_buffer_addr[n_filters * conv_params->filter_offset - 2 * idx - 1] = -get_layer_sign(conv_params->model);
        }
#endif

        conv_params->cached_filter_idx = conv_params->conv_idx;
        conv_params->cached_input_tile_c_offset = conv_params->input_tile_c_offset;
    } else {
#if STATEFUL
        if (n_keep_state_bits != n_filters) {
            need_cleanup_state_bits = 1;
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
    my_matrix_mpy_q15(A_rows, A_cols, B_rows, B_cols, input_buffer_addr, filter_buffer_addr, matrix_mpy_results);

    /* START dump data */
#if MY_DEBUG >= 2
    my_printf_debug("input_h=%d" NEWLINE, conv_params->input_h + offset_h);
    my_printf_debug("conv_idx=");
    for (uint16_t idx = 0; idx < cur_output_tile_c; idx++) {
        my_printf_debug("%d ", conv_params->conv_idx + idx);
        MY_ASSERT(conv_params->conv_idx + idx < conv_params->N_FILTERS);
    }
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
#endif
    /* END dump data */

#if MY_DEBUG >= 1
    my_printf_debug("output_data offset = %d" NEWLINE, cur_output_data_offset);
    MY_ASSERT(cur_output_data_offset > last_output_data_offset);
    last_output_data_offset = cur_output_data_offset;
#endif

    MY_ASSERT(cur_output_data_offset + n_filters < INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);
    my_memcpy_to_param(conv_params->output, cur_output_data_offset, matrix_mpy_results, n_filters * sizeof(int16_t));

#if HAWAII
    write_hawaii_layer_footprint(conv_params->model->layer_idx, cur_output_data_offset);
#endif

#if STATEFUL
    if (n_keep_state_bits != n_filters) {
        check_next_turning_point(conv_params->old_output_offset, conv_params->turning_point_idx,
                                 conv_params->next_turning_point, conv_params->cur_slot_info, cur_output_data_offset + n_filters);
        my_printf_debug("old_output_offset flipped to %d" NEWLINE, conv_params->old_output_offset);

        if (need_cleanup_state_bits) {
            flip_filter_state_bits(conv_params, n_filters, n_keep_state_bits, 0);
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
    int16_t max_n_filters = conv_params->output->tile_c;
#if JAPARI
    max_n_filters *= 2;
#endif
    // TEMP_FILTER_WIDTH additional filters for values before transpose
    uint16_t inputs_len = MIN_VAL(
        LEA_BUFFER_SIZE - OUTPUT_LEN - (max_n_filters + TEMP_FILTER_WIDTH) * conv_params->filter_offset,
        (conv_params->H + conv_params->kH - 1) * conv_params->dest_offset
    );
    MY_ASSERT(inputs_len < LEA_BUFFER_SIZE); // make sure no overflow occurs in the previous line

    dest = lea_buffer;

    int32_t h_start = int16_max(                     -field_size,                 -conv_params->input_h),
            h_end =   int16_min(conv_params->tile_h-1+field_size, conv_params->H-1-conv_params->input_h);

    my_printf_debug("Reinitialize input buffer" NEWLINE "inputs_len = %d" NEWLINE, inputs_len);

    my_fill_q15(0, lea_buffer, inputs_len);

    dest += (h_start + field_size) * conv_params->dest_offset;

    my_printf_debug("h_start=%" PRId32 " ", h_start);
    my_printf_debug("h_end=%" PRId32 NEWLINE, h_end);

    size_t input_row_len = (w_end - w_start + 1) * conv_params->cur_input_tile_c;
#if JAPARI
    MY_ASSERT(input_row_len <= INPUT_BUFFER_WITH_FOOTPRINTS_LEN);
#endif
    my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                    static_cast<int>(dest - lea_buffer));
    int16_t input_src_offset;
    input_src_offset = input_offset + (conv_params->input_h + h_start) * conv_params->W * conv_params->cur_input_tile_c + (conv_params->input_w + w_start) * conv_params->cur_input_tile_c;
#if STATEFUL
    dump_turning_points_debug(model, conv_params->real_conv_input);

    int16_t offset, next_turning_point;
    uint8_t turning_point_idx;
    SlotInfo *input_slot_info;
    find_initial_state_bit(&offset, &turning_point_idx, &next_turning_point, &input_slot_info, input_src_offset, model, conv_params->real_conv_input);
#endif
    for (int32_t h = h_start; h <= h_end; h++) {
        int16_t *orig_dest_addr;
        orig_dest_addr = dest + (w_start + field_size) * conv_params->cur_filter_tile_c;
        int16_t *dest_addr;
#if JAPARI
        if (conv_params->conv_input_is_intermediate_data) {
            dest_addr = input_buffer_with_footprints;
        } else
#endif
        {
            dest_addr = orig_dest_addr;
        }
        my_printf_debug("Load %ld IFM values from range [%d, %d)" NEWLINE, input_row_len, input_src_offset, static_cast<int>(input_src_offset + input_row_len));
        my_memcpy_from_param(
            model, dest_addr,
            conv_params->real_conv_input, input_src_offset,
            input_row_len * sizeof(int16_t));

#if JAPARI
        if (conv_params->conv_input_is_intermediate_data) {
            my_deinterleave_q15(dest_addr, 0, 2, orig_dest_addr, input_row_len / 2);
        }
#endif

#if STATEFUL
        int16_t input_src_offset_end = input_src_offset + input_row_len;
        if (offset) {
            MY_ASSERT(static_cast<uint16_t>(next_turning_point) > input_src_offset);
            my_offset_q15(dest_addr, -offset, dest_addr, MIN_VAL(static_cast<int16_t>(input_row_len), static_cast<uint16_t>(next_turning_point) - input_src_offset));
        } else if (static_cast<uint16_t>(next_turning_point) < input_src_offset_end) {
            MY_ASSERT(next_turning_point >= input_src_offset);
            int16_t *to_offset  = dest_addr + next_turning_point - input_src_offset;
            my_offset_q15(to_offset, -0x4000, to_offset, input_src_offset_end - next_turning_point);
        }
#endif
        dest += conv_params->dest_offset;
        input_src_offset += conv_params->W * conv_params->cur_input_tile_c;
#if STATEFUL
        check_next_turning_point(offset, turning_point_idx, next_turning_point, input_slot_info, input_src_offset);
#endif
    }
    if (conv_params->real_conv_input->scale != conv_params->conv_input->scale) {
        int16_t scaleFract;
        uint8_t shift;
        float_to_scale_params(&scaleFract, &shift, 1.0f * conv_params->real_conv_input->scale / conv_params->conv_input->scale);
        my_scale_q15(lea_buffer, scaleFract, shift, lea_buffer, inputs_len);
    }
#if STATEFUL && MY_DEBUG >= 2
    int16_t *ptr = lea_buffer;
    for (size_t idx = 0; idx < inputs_len; idx++) {
        MY_ASSERT(!get_value_state_bit(*ptr));
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
        // conv_idx is set to initial_c in handle_conv
        convTask(j, conv_params);
        // reset here for further processing
        conv_params->conv_idx = conv_params->conv_idx_base;
    }
}

void alloc_conv(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    const ParameterInfo *conv_input = input[0], *conv_filter = input[1];

    MY_ASSERT(conv_input->bitwidth == 16 && conv_filter->bitwidth == 16);

#if JAPARI
    if (is_intermediate_data(conv_input)) {
        MY_ASSERT(conv_input->dims[1] == conv_filter->dims[1] * 2);
    } else
#endif
    {
        MY_ASSERT(conv_input->dims[1] == conv_filter->dims[1]);
    }

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   CHANNEL = conv_filter->dims[1];
    uint16_t OUTPUT_CHANNEL = conv_filter->dims[0];
#if JAPARI
    OUTPUT_CHANNEL *= 2;
#endif

    ConvTaskParams *conv_params = &conv_params_obj;

    conv_params->model = model;

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
    output->scale = conv_input->scale * conv_filter->scale;
}

void handle_conv(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *conv_input = input[0], *conv_filter = input[1], *conv_bias = input[2];
    my_printf_debug("Conv!" NEWLINE);

    /* input: N x C x H x W, filter: M x C x kH x kW */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   CHANNEL = conv_filter->dims[1];

    ConvTaskParams *conv_params = &conv_params_obj;

    conv_params->tile_h = MIN_VAL(H, DEFAULT_TILE_H);

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
#if JAPARI
    conv_params->conv_input_is_intermediate_data = is_intermediate_data(conv_input);
#endif

    conv_params->CHANNEL = CHANNEL;
    conv_params->OUTPUT_CHANNEL = output->dims[1];
    conv_params->N_FILTERS = conv_params->OUTPUT_CHANNEL;
#if JAPARI
    conv_params->N_FILTERS /= 2;
#endif

    uint16_t input_tile_c = conv_input->tile_c;
#if JAPARI
    if (conv_params->conv_input_is_intermediate_data) {
        input_tile_c *= 2;
    }
#endif
    output->tile_c = conv_params->N_FILTERS;
    determine_tile_c(output, conv_filter);
    uint16_t output_tile_c = output->tile_c;
    my_printf_debug("output_tile_c = %d" NEWLINE, output_tile_c);

#if MY_DEBUG >= 1
    last_output_data_offset = -1;
#endif

    conv_params->input_tile_c_offset = 0;
    conv_params->input_tile_c_index = 0;
    conv_params->input_w = conv_params->offset_w;
    conv_params->input_h = conv_params->offset_h;
    conv_params->conv_idx_base = 0;
    conv_params->conv_idx = 0;
#if INTERMITTENT

    uint32_t first_unfinished_value_offset = run_recovery(model, output);
    // Force recovery from an even OFM index as most DSPLib function does not like odd dimensions
    first_unfinished_value_offset = first_unfinished_value_offset / 2 * 2;

#if STATEFUL
    find_initial_state_bit(&conv_params->old_output_offset, &conv_params->turning_point_idx, &conv_params->next_turning_point,
                           &conv_params->cur_slot_info, first_unfinished_value_offset, model, output);

    my_printf_debug("old_output_offset = %d" NEWLINE, conv_params->old_output_offset);
#endif

    // Dimensions: channel-tiled NWHC
    uint16_t slice_size_input_channel_tiling = conv_params->OUTPUT_W * conv_params->OUTPUT_H * conv_params->OUTPUT_CHANNEL;
    conv_params->input_tile_c_index = first_unfinished_value_offset / slice_size_input_channel_tiling;
    conv_params->input_tile_c_offset = conv_params->input_tile_c_index * input_tile_c;
    first_unfinished_value_offset %= slice_size_input_channel_tiling;

    uint16_t slice_size_output_channel_tiling = conv_params->OUTPUT_W * conv_params->OUTPUT_H * output_tile_c;
    conv_params->conv_idx_base = first_unfinished_value_offset / slice_size_output_channel_tiling * output_tile_c;
    conv_params->conv_idx = conv_params->conv_idx_base;
    first_unfinished_value_offset %= slice_size_output_channel_tiling;

    uint16_t cur_output_tile_c = MIN_VAL(output_tile_c, conv_params->N_FILTERS - conv_params->conv_idx_base);
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

    int16_t input_channels = CHANNEL;
#if JAPARI
    if (conv_params->conv_input_is_intermediate_data) {
        input_channels *= 2;
    }
#endif
    for (; conv_params->input_tile_c_offset < input_channels; conv_params->input_tile_c_offset += input_tile_c, conv_params->input_tile_c_index++) {
        conv_params->cur_input_tile_c = MIN_VAL(input_tile_c, input_channels - conv_params->input_tile_c_offset);
        conv_params->cur_filter_tile_c = conv_params->cur_input_tile_c;
#if JAPARI
        if (conv_params->conv_input_is_intermediate_data) {
            conv_params->cur_filter_tile_c /= 2;
        }
#endif
        my_printf_debug("cur_input_tile_c = %d" NEWLINE, conv_params->cur_input_tile_c);
        conv_params->dest_offset = conv_params->kH * conv_params->cur_input_tile_c;
#if JAPARI
        if (conv_params->conv_input_is_intermediate_data) {
            conv_params->dest_offset /= 2;
        }
#endif
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

        while (conv_params->conv_idx_base < conv_params->N_FILTERS) {
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

#if STATEFUL
    flip_state_bit(model, output);
#endif

#if MY_DEBUG >= 2
    uint32_t tiling_results_len = conv_params->OUTPUT_CHANNEL * conv_params->OUTPUT_H * conv_params->OUTPUT_W;

    my_printf_debug("handle_conv output" NEWLINE);
    for (uint16_t input_tile_c_index = 0; input_tile_c_index * input_tile_c < input_channels; input_tile_c_index++) {
        dump_params_nhwc_debug(model, output, input_tile_c_index * tiling_results_len);
    }
#endif
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
        my_offset_q15(to_offset, -0x4000, to_offset, range_len);
    }
}

struct ConvMergeOutputChunkHandlerParams {
    uint32_t tiling_results_offset;
};

void ConvMergeOutputChunkHandler(uint32_t range_offset, uint16_t range_len, uint8_t state_bit, void* _params) {
    ConvMergeOutputChunkHandlerParams* params = reinterpret_cast<ConvMergeOutputChunkHandlerParams*>(_params);
    my_printf_debug("output range_offset=%d range_len=%d state_bit=%d" NEWLINE, range_offset, range_len, state_bit);
    int16_t *to_offset = lea_buffer + range_offset - params->tiling_results_offset;
    // output state bit has not been flipped yet
    if (!state_bit) {
        my_offset_q15(to_offset, 0x4000, to_offset, range_len);
    }
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

    uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / n_tiles_c / 2 * 2);

    float scale_f = 1.0 * find_max_multiplier(model, data) / n_tiles_c;
    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, scale_f);

    // XXX: use iterate_chunks() for the outer loop?
    for (uint32_t tiling_results_offset = 0; tiling_results_offset < tiling_results_len; tiling_results_offset += chunk_len) {
        uint32_t real_chunk_len = MIN_VAL(chunk_len, tiling_results_len - tiling_results_offset);
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
#if STATEFUL
        ConvMergeOutputChunkHandlerParams params({tiling_results_offset});
        iterate_chunks(model, output, tiling_results_offset, real_chunk_len, ConvMergeOutputChunkHandler, &params);
#endif
#if JAPARI
        for (uint16_t idx = 1; idx < real_chunk_len; idx += 2) {
            lea_buffer[idx] = get_layer_sign(model);
        }
#endif
        my_memcpy_to_param(output, tiling_results_offset, lea_buffer, real_chunk_len * sizeof(int16_t));
    }

    my_printf_debug("After scaling up back and merging tiling results" NEWLINE);

    output->scale /= scale_f;

#if STATEFUL
    flip_state_bit(model, output);
#endif

    dump_params_nhwc_debug(model, output, 0);
}
