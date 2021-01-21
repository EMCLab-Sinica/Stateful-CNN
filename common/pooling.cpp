#include <stdint.h>
#include "data.h"
#include "cnn_common.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "op_utils.h"
#include "my_dsplib.h"

struct MaxPoolParams {
    uint16_t output_h;
    uint16_t output_w;
    uint16_t start_channel;
    uint8_t n_channels;
    uint8_t need_nhwc2nchw;
    uint16_t new_W;
    const NodeFlags* flags;
    const ParameterInfo *data;
    const ParameterInfo *output;
    Model *model;
};
static MaxPoolParams maxpool_params_obj;

void alloc_maxpool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    uint16_t stride = flags->stride;

    const ParameterInfo *data = input[0];

    const uint16_t H = data->dims[2], W = data->dims[3];
    uint16_t CHANNEL = data->dims[1];
    uint16_t new_H = H / stride;

    MaxPoolParams* maxpool_params = &maxpool_params_obj;
    maxpool_params->new_W = W / stride;
    maxpool_params->need_nhwc2nchw = (flags->generic == NHWC2NCHW);

#if JAPARI
    if (maxpool_params->need_nhwc2nchw) {
        maxpool_params->new_W = extend_for_footprints(maxpool_params->new_W);
        CHANNEL = CHANNEL / (BATCH_SIZE + 1) * BATCH_SIZE;
    }
#endif

    output->params_len = new_H * maxpool_params->new_W * CHANNEL * sizeof(int16_t);
    output->slot = get_next_slot(model, data);
    output->dims[0] = 1;
    output->dims[1] = CHANNEL;
    output->dims[2] = new_H;
    output->dims[3] = maxpool_params->new_W;
}

static uint8_t maxpool_patch(MaxPoolParams *maxpool_params) {
    const uint16_t CHANNEL = maxpool_params->data->dims[1], W = maxpool_params->data->dims[3];
    uint16_t stride = maxpool_params->flags->stride;
    uint16_t kernel_size = maxpool_params->flags->kernel_size;

    int16_t offset_h, offset_w;
    offset_h = W * CHANNEL;
    offset_w = CHANNEL;

    my_printf_debug("output_h=% 3d ", maxpool_params->output_h);
    my_printf_debug("output_w=% 3d ", maxpool_params->output_w);
    my_printf_debug("c=[% 3d, % 3d) ", maxpool_params->start_channel, maxpool_params->start_channel + maxpool_params->n_channels);

    int16_t* const input_buffer = lea_buffer + maxpool_params->n_channels;
    int16_t* const output_buffer = lea_buffer;
    my_fill_q15(INT16_MIN, output_buffer, maxpool_params->n_channels);

    // explicitly initialize this as -Wmaybe-uninitialized may be triggered with -O3
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60165
    uint8_t output_channel_offset = 0;

    for (uint16_t sH = 0; sH < kernel_size; sH++) {
        for (uint16_t sW = 0; sW < kernel_size; sW++) {
            uint16_t val_offset = (maxpool_params->output_h*stride+sH) * offset_h + (maxpool_params->output_w*stride+sW) * offset_w + maxpool_params->start_channel;
            my_memcpy_from_param(maxpool_params->model, input_buffer, maxpool_params->data, val_offset, maxpool_params->n_channels * sizeof(int16_t));
            output_channel_offset = 0;
            for (uint8_t input_channel_offset = 0; input_channel_offset < maxpool_params->n_channels; input_channel_offset++) {
#if JAPARI
                if ((maxpool_params->start_channel + input_channel_offset) % (BATCH_SIZE + 1) == BATCH_SIZE) {
                    // not checking need_nhwc2nchw here - if that is true, input footprint channels should already be skipped
                    // before maxpool_patch is called
                    output_channel_offset++;
                    continue;
                }
#endif
                int16_t val = input_buffer[input_channel_offset];
#if STATEFUL
                if (get_value_state_bit(val)) {
                    // assuming input state bits are correct...
                    val -= 0x4000;
                }
#endif
                // dump_value_debug(model, maxpool_params->data, val_offset);
                my_printf_debug("% 6d ", val);
                // XXX: use LEA?
                if (val > output_buffer[output_channel_offset]) {
                    output_buffer[output_channel_offset] = val;
                }
                output_channel_offset++;
            }
            my_printf_debug("; ");
        }
    }
    return output_channel_offset;
}

#if STATEFUL
static inline void offset_vector(int16_t* const buffer, int16_t offset, uint8_t len, const uint16_t output_offset, const uint16_t next_output_turning_point) {
    int16_t cur_offset = offset;
    for (uint8_t idx = BATCH_SIZE - 1; idx < len; idx += BATCH_SIZE) {
        if (output_offset + idx == next_output_turning_point + BATCH_SIZE - 1) {
            cur_offset ^= 0x4000;
        }
        buffer[idx] += cur_offset;
    }
}
#endif
#if JAPARI
static inline void offset_vector(int16_t* const buffer, int16_t offset, uint8_t len, const uint16_t output_offset, const uint16_t next_output_turning_point) {
    int16_t cur_footprint = (offset == 0x4000 ? 1 : -1);
    uint8_t reverted = 0;
    for (uint8_t idx = BATCH_SIZE; idx < len; idx += BATCH_SIZE + 1) {
        if (output_offset + idx >= next_output_turning_point && !reverted) {
            cur_footprint = -cur_footprint;
            reverted = 1;
        }
        buffer[idx] = cur_footprint;
    }
}
#endif

void handle_maxpool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    my_printf_debug("MaxPool!" NEWLINE);

    uint16_t stride = flags->stride;

    /* XXX: add flags; assume no padding for now */
    const ParameterInfo *data = input[0];

    MaxPoolParams* maxpool_params = &maxpool_params_obj;
    maxpool_params->data = data;
    maxpool_params->output = output;
    maxpool_params->flags = flags;
    maxpool_params->model = model;

    const uint16_t CHANNEL = data->dims[1], H = data->dims[2], OUTPUT_CHANNEL = output->dims[1];
    uint16_t new_H = H / stride;

    uint16_t output_h = 0, output_w = 0, c = 0;
    uint16_t output_offset = 0;

#if INTERMITTENT
    uint32_t first_unfinished_value_offset = job_index_to_offset(output, run_recovery(model, output));
#if JAPARI
    first_unfinished_value_offset -= BATCH_SIZE;
#else
    first_unfinished_value_offset -= (BATCH_SIZE - 1);
#endif
    if (first_unfinished_value_offset * sizeof(int16_t) == output->params_len) {
        // give up early, or initial_real_tile_c may be zero and results in SIGFPE
        goto finished;
    }

    uint16_t initial_c, initial_h, initial_w;

#if INDIRECT_RECOVERY
    int16_t offset;
    uint16_t next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, first_unfinished_value_offset, model, output);
    offset ^= 0x4000;
#endif

    output_offset = first_unfinished_value_offset;
    if (!maxpool_params->need_nhwc2nchw) {
        initial_c = first_unfinished_value_offset % OUTPUT_CHANNEL;
        first_unfinished_value_offset /= OUTPUT_CHANNEL;
        initial_w = first_unfinished_value_offset % maxpool_params->new_W;
        first_unfinished_value_offset /= maxpool_params->new_W;
        initial_h = first_unfinished_value_offset % new_H;
    } else {
        initial_w = first_unfinished_value_offset % maxpool_params->new_W;
        first_unfinished_value_offset /= maxpool_params->new_W;
        initial_h = first_unfinished_value_offset % new_H;
        first_unfinished_value_offset /= new_H;
        initial_c = first_unfinished_value_offset % OUTPUT_CHANNEL;
    }
    output_h = initial_h;
    output_w = initial_w;
    c = initial_c;
    my_printf_debug("initial_h = %d" NEWLINE, initial_h);
    my_printf_debug("initial_w = %d" NEWLINE, initial_w);
    my_printf_debug("initial_c = %d" NEWLINE, initial_c);
#endif

    {
        if (!maxpool_params->need_nhwc2nchw) {
            // NHWC
            for (; output_h < new_H; output_h++) {
                maxpool_params->output_h = output_h;
                for (; output_w < maxpool_params->new_W; output_w++) {
                    uint8_t len = OUTPUT_CHANNEL - c;
                    maxpool_params->output_w = output_w;
                    maxpool_params->n_channels = len;
                    maxpool_params->start_channel = c;
                    len = maxpool_patch(maxpool_params);
                    my_printf_debug("output_offset=[% 5d, % 5d) ", output_offset, output_offset + len);
#if INDIRECT_RECOVERY
                    check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
                    offset_vector(lea_buffer, offset, len, output_offset, next_output_turning_point);
#endif
#if MY_DEBUG >= 1
                    // need a space as dump_value does not append spaces when DUMP_INTEGERS is not defined
                    my_printf_debug(" max=");
                    for (uint8_t idx = 0; idx < len; idx++) {
                        my_printf_debug("% 6d ", lea_buffer[idx]);
                    }
                    my_printf_debug(NEWLINE);
#endif
#if HAWAII
                    hawaii_preserve_vector(model, output, output_offset, lea_buffer, len);
#else
                    my_memcpy_to_param(output, output_offset, lea_buffer, len * sizeof(int16_t));
#endif
                    output_offset += len;
                    c = 0;
                }
                output_w = 0;
            }
            output_h = 0;
        } else {
            // NCHW
#if JAPARI
            // extend c as input footprint channels are skipped.
            // Not using extend_for_footprints() as the initial c may not be on a footprint channel
            c += c / BATCH_SIZE;
#endif
            uint8_t channel_stride = 1;
            for (; c < CHANNEL; c += channel_stride) {
#if JAPARI
                if (c % (BATCH_SIZE + 1) == BATCH_SIZE) {
                    continue;
                }
#endif
                for (; output_h < new_H; output_h++) {
                    maxpool_params->output_h = output_h;
#if !JAPARI
                    maxpool_params->output_w = output_w;
#else
                    maxpool_params->output_w = output_w / (BATCH_SIZE + 1) * BATCH_SIZE + output_w % (BATCH_SIZE +1);
#endif
                    for (; output_w < maxpool_params->new_W; output_w++) {
#if JAPARI
                        if (output_offset % (BATCH_SIZE + 1) == BATCH_SIZE) {
                            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
                            put_q15_param(output, output_offset, (offset == 0x4000 ? 1 : -1));
                            output_offset++;
                            continue;
                        }
#endif
                        maxpool_params->start_channel = c;
                        maxpool_params->n_channels = 1;
                        uint8_t len = maxpool_patch(maxpool_params);
                        if (!len) {
                            my_printf_debug(NEWLINE);
                            continue;
                        }
                        my_printf_debug("output_offset=% 5d ", output_offset);
#if STATEFUL
                        check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
                        lea_buffer[0] += offset;
#endif
                        my_printf_debug("max=% 6d " NEWLINE, lea_buffer[0]);
                        put_q15_param(output, output_offset, lea_buffer[0]);
#if HAWAII
                        if (output_offset % BATCH_SIZE == (BATCH_SIZE - 1)) {
                            write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
                        }
#endif
                        output_offset++;
                        maxpool_params->output_w++;
                    }
                    output_w = 0;
                }
                output_h = 0;
            }
            c = 0;
        }
    }

    MY_ASSERT(output_offset == output->params_len / sizeof(int16_t),
              "Expect output offset %d, got %d" NEWLINE, output->params_len / sizeof(int16_t), output_offset);

#if INTERMITTENT
finished:
    flip_state_bit(model, output);
#endif

    my_printf_debug("handle_maxpool output" NEWLINE);
    if (!maxpool_params->need_nhwc2nchw) {
        dump_params_nhwc_debug(model, output);
    } else {
        dump_params_debug(model, output);
    }
}

void alloc_globalaveragepool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *data = input[0];

    MY_ASSERT(data->dims[0] == 1);
    uint16_t output_len = data->dims[1];

    output->dims[0] = output->dims[2] = output->dims[3] = 1;
    output->dims[1] = output_len;
    output->params_len = output_len * sizeof(int16_t);
    output->bitwidth = 16;
    output->slot = get_next_slot(model, data);
}

void handle_globalaveragepool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("GlobalAveragePool!" NEWLINE);

    const ParameterInfo *data = input[0];

#if STATEFUL
    int16_t offset;
    uint16_t next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, 0 /*TODO: first_unfinished_value_offset*/, model, output);
    offset ^= 0x4000;
#endif

    uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t len = H * W;
    uint16_t output_channel = 0;
    for (uint16_t input_channel = 0; input_channel < CHANNEL; input_channel++) {
        int16_t output_val;
#if JAPARI
        if (input_channel % (BATCH_SIZE + 1) == BATCH_SIZE) {
            output_val = (param_state_bit(model, output, output_channel) ? -1 : 1);
        } else
#endif
        {
            uint32_t total = 0;
            for (uint16_t h = 0; h < H; h++) {
                for (uint16_t w = 0; w < W; w++) {
                    // Input is from Conv, which uses NHWC
                    int16_t val = get_q15_param(model, data, h * W * CHANNEL + w * CHANNEL + input_channel);
#if STATEFUL
                    if (get_value_state_bit(val)) {
                        val -= 0x4000;
                    }
#endif
                    total += val;
                }
            }
            output_val = total / len;
#if STATEFUL
            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_channel);
            output_val += offset;
#endif
        }
        put_q15_param(output, output_channel, output_val);
        output_channel++;
    }

    flip_state_bit(model, output);

    dump_params_debug(model, output);
}
