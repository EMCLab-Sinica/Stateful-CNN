#include <cstdint>
#include "cnn_common.h"
#include "op_utils.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"

#define RESHAPE_AUTO_DIM static_cast<uint16_t>(-1)

void alloc_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *data = input[0];
    output->slot = get_next_slot(model, data);
    output->flags &= ~TRANSPOSED;
}

void handle_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("ReLu!" NEWLINE);

    const ParameterInfo *X = input[0];

    uint16_t CHANNEL = X->dims[1];
    uint16_t OUTPUT_CHANNEL = output->dims[1];

    /* XXX: use LEA? */
    uint16_t bitwidth = X->bitwidth;
    MY_ASSERT(bitwidth == 16);
    int16_t data_len = X->params_len / (bitwidth / 8);

    uint16_t data_offset = 0;
    uint16_t output_offset = 0;
#if INTERMITTENT

    uint32_t first_unfinished_value_offset = job_index_to_offset(output, run_recovery(model, output));

#if JAPARI
    first_unfinished_value_offset -= BATCH_SIZE;
#else
    first_unfinished_value_offset -= (BATCH_SIZE - 1);
#endif
    data_offset += first_unfinished_value_offset;
    output_offset += first_unfinished_value_offset;

#if INDIRECT_RECOVERY
    uint16_t next_output_turning_point;
    int16_t offset;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           first_unfinished_value_offset, model, output);
    offset = -offset;
#endif

#endif

    if (X->flags & TRANSPOSED) {
        // input is in NWHC
        // TODO: state-aware recovery
        uint16_t H = X->dims[2], W = X->dims[3];
        uint16_t output_h = 0, output_w = 0, c = 0;
#if INTERMITTENT
        output_h = first_unfinished_value_offset / (W * CHANNEL);
        first_unfinished_value_offset %= (W * CHANNEL);
        output_w = first_unfinished_value_offset / CHANNEL;
        c = first_unfinished_value_offset % CHANNEL;
        my_printf_debug("initial output_h = %d, ", output_h);
        my_printf_debug("initial output_w = %d, ", output_w);
        my_printf_debug("initial c = %d" NEWLINE, c);
#endif

        for (; output_h < H; output_h++) {
            for (; output_w < W; output_w++) {
                    int16_t val_offset = output_w * H * CHANNEL + output_h * CHANNEL + c;
                    output_offset = output_h * W * OUTPUT_CHANNEL + output_w * OUTPUT_CHANNEL + c;
#if INDIRECT_RECOVERY
                    check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif
                    uint16_t len = CHANNEL - c;
                    my_memcpy_from_param(model, lea_buffer, X, val_offset, len * sizeof(int16_t));

                    my_printf_debug("output_h=% 3d, output_w=% 3d, c=[% 3d, % 3d), val_offset=[% 6d, % 6d), input val=",
                                    output_h, output_w, c, c + len, val_offset, val_offset + len);
                    for (uint16_t idx = 0; idx < len; idx++) {
                        my_printf_debug("% 6d ", lea_buffer[idx]);
                    }

                    uint16_t output_idx = 0;
                    for (uint16_t idx = 0; idx < len; idx++) {
                        int16_t input_val = 0, output_val;
#if JAPARI
                        if (offset_has_state(c + idx)) {
                            output_val = (offset > 0? 1 : -1);
                            if (next_output_turning_point != INVALID_TURNING_POINT && (output_offset + idx >= next_output_turning_point)) {
                                output_val = -output_val;
                            }
                        } else
#endif
                        {
                            input_val = lea_buffer[idx];
#if STATEFUL
                            if (offset_has_state(c + idx)) {
                                strip_state(&input_val);
                            }
#endif
                            output_val = MAX_VAL(input_val, 0);
                        }
                        lea_buffer[output_idx] = output_val;
                        output_idx++;
                    }
#if STATEFUL
                    uint8_t block_size;
                    if (next_output_turning_point == INVALID_TURNING_POINT) {
                        block_size = len;
                    } else {
                        block_size = MIN_VAL(len, next_output_turning_point - output_offset);
                    }
                    my_offset_q15_batched(lea_buffer, offset, lea_buffer, block_size);
                    if (next_output_turning_point < output_offset + len) {
                        int16_t* to_offset = lea_buffer + next_output_turning_point - output_offset;
                        my_offset_q15_batched(to_offset, -offset, to_offset, output_offset + len - next_output_turning_point);
                    }
#endif
                    my_memcpy_to_param(output, output_offset, lea_buffer, output_idx * sizeof(int16_t), 0);
#if HAWAII
                    hawaii_record_footprints(model, len);
#endif

                    my_printf_debug("output_offset=[% 6d, % 6d), output val=", output_offset, output_offset + output_idx);
#if MY_DEBUG >= 1
                    for (uint16_t idx = 0; idx < output_idx; idx++) {
                        my_printf_debug("% 6d ", lea_buffer[idx]);
                    }
#endif
                    my_printf_debug(NEWLINE);
                c = 0;
            }
            output_w = 0;
        }
    } else {
        uint16_t i = output_offset;
#if JAPARI
        uint8_t cur_batch_offset = i % (BATCH_SIZE + 1);
#else
        uint8_t cur_batch_offset = i % BATCH_SIZE;
#endif
        for (; i < data_len; i++) {
            int16_t output_val;
#if JAPARI
            if (cur_batch_offset == BATCH_SIZE) {
                cur_batch_offset -= BATCH_SIZE + 1;
                output_val = (offset > 0? 1 : -1);
            } else
#endif
            {
                int16_t input_val = get_q15_param(model, X, data_offset);
#if INDIRECT_RECOVERY
#if STATEFUL
                if (offset_has_state(data_offset)) {
                    strip_state(&input_val);
                }
#endif
                check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif
                output_val = MAX_VAL(input_val, 0);
            }
#if STATEFUL
            if (cur_batch_offset == BATCH_SIZE - 1) {
                cur_batch_offset -= BATCH_SIZE;
                output_val += offset;
            }
#endif
            my_printf_debug("output_offset=%d output_val=%d" NEWLINE, output_offset, output_val);
            put_q15_param(output, output_offset, output_val);
#if HAWAII
            if (cur_batch_offset == BATCH_SIZE - 1) {
                write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
                cur_batch_offset -= BATCH_SIZE;
            }
#endif
            data_offset++;
            output_offset++;
            cur_batch_offset++;
        }
    }

    flip_state_bit(model, output);

    my_printf_debug("handle_relu output" NEWLINE);
    if (X->flags & TRANSPOSED) {
        dump_params_nhwc_debug(model, output);
    } else {
        dump_params_debug(model, output);
    }
}

void handle_reshape(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Reshape!" NEWLINE);

    const ParameterInfo *data = input[0], *shape = input[1];
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    if (cur_slot_info) {
        cur_slot_info->user = model->layer_idx;
    }
    MY_ASSERT(shape->bitwidth == 64);
    /*
     * At most one dimension of the new shape can be -1. In this case, the
     * value is inferred from the size of the tensor and the remaining
     * dimensions.
     *
     * A dimension could also be 0, in which case the actual dimension value
     * is unchanged (i.e. taken from the input tensor).
     * */
    uint32_t new_len = 1;
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = get_int64_param(shape, i);
        if (!output->dims[i]) {
            output->dims[i] = data->dims[i];
        }
        if (output->dims[i] != RESHAPE_AUTO_DIM) {
            new_len *= output->dims[i];
        }
    }
    for (uint8_t i = shape->dims[0]; i < 4; i++) {
        output->dims[i] = 0;
    }
    uint16_t inferred_dim = output->params_len / sizeof(int16_t);
    int8_t auto_idx = -1;
#if JAPARI
    uint8_t last_dim_idx;
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i]) {
            last_dim_idx = i;
        }
    }
#endif
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i] != RESHAPE_AUTO_DIM && output->dims[i] != 0) {
#if JAPARI
            if (i == last_dim_idx) {
                inferred_dim /= extend_for_footprints(output->dims[i]);
            } else
#endif
            {
                inferred_dim /= output->dims[i];
            }
        } else if (output->dims[i] == RESHAPE_AUTO_DIM) {
            auto_idx = i;
        }
    }
    if (auto_idx != -1) {
        output->dims[auto_idx] = inferred_dim;
        new_len *= inferred_dim;
    }
#if JAPARI
    new_len = extend_for_footprints(new_len);
#endif
    MY_ASSERT(new_len * sizeof(int16_t) == output->params_len);
}

void handle_squeeze(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    my_printf_debug("Squeeze!" NEWLINE);

    const ParameterInfo *data = input[0];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth = data->bitwidth;
    output->slot = data->slot;
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    if (cur_slot_info) {
        cur_slot_info->user = model->layer_idx;
    }
    uint8_t axes = flags->extra.squeeze.axes;
    // If axes is not provided, all the single dimensions will be removed from the shape.
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#squeeze
    uint8_t j = 0;
    if (axes == 0) {
        for (uint8_t i = 0; i < 4; i++) {
            if (input[0]->dims[i] != 1) {
                output->dims[j] = input[0]->dims[i];
                j++;
            }
        }
    } else {
        for (uint8_t i = 0; i < 4; i++) {
            if (axes & (1 << i)) {
                MY_ASSERT(input[0]->dims[i] == 1);
            } else {
                output->dims[j] = input[0]->dims[i];
                j++;
            }
        }
    }
    for (; j < 4; j++) {
        output->dims[j] = 0;
    }
}

void alloc_concat(Model *, const ParameterInfo *[], ParameterInfo*, const NodeFlags*) {
}

void handle_concat(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Concat!" NEWLINE);

    const ParameterInfo *A = input[0], *B = input[1];
    // XXX: assume concatenating 2 tensors at the CHANNEL dimension and they
    // have the same number of channels.
    MY_ASSERT(A->dims[1] == B->dims[1]);
    output->dims[1] *= 2;
    output->flags |= SEPARATE_TILING;

    // The one with smaller `scale` (with larger values) is scaled down
    output->scale = MAX_VAL(A->scale, B->scale);

    // saving slots here as it might be changed during the downscaling loop above
    output->extra_info[0] = A->parameter_info_idx;
    output->extra_info[1] = B->parameter_info_idx;
    output->slot = A->slot;

    dump_params_nhwc_debug(model, A);
    dump_params_nhwc_debug(model, B);
}

void handle_softmax(Model*, const ParameterInfo*[], ParameterInfo*, const NodeFlags*) {
    // Do nothing - softmax does not change the relative order of values.
    // Just let run_model determine the max value
}

void handle_transpose(Model*, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Transpose!" NEWLINE);

    const ParameterInfo *X = input[0];
    // not actually transpose data as we happen to need NHWC
    // XXX: assume NHWC -> NCHW
    output->dims[1] = X->dims[3];
    output->dims[2] = X->dims[1];
    output->dims[3] = X->dims[2];
}
