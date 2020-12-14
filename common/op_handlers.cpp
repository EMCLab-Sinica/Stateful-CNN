#include "cnn_common.h"
#include "op_utils.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"

#define RESHAPE_AUTO_DIM static_cast<uint16_t>(-1)

void alloc_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *A = input[0];
    MY_ASSERT(A->bitwidth == 16 && input[1]->bitwidth == 16);

    output->slot = get_next_slot(model, A);
}

void handle_add(Model* model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    /* Add: Y = X + W */
    my_printf_debug("Add!" NEWLINE);

    const ParameterInfo *A = input[0], *B = input[1];

    my_printf_debug("handle_add input A" NEWLINE);
    dump_params_debug(model, A);
    my_printf_debug("handle_add input B" NEWLINE);
    dump_params_debug(model, B);

    uint16_t vector_size = A->dims[1];

    int16_t *buffer_a = lea_buffer,
            *buffer_b = lea_buffer + vector_size;
    my_memcpy_from_param(model, buffer_a, A, 0, output->params_len);
    my_memcpy_from_param(model, buffer_b, B, 0, output->params_len);

#if STATEFUL
    // XXX: use LEA?
    for (uint16_t idx = 0; idx < vector_size; idx++) {
        if (get_value_state_bit(buffer_a[idx])) {
            buffer_a[idx] -= 0x4000;
        }
        if (get_value_state_bit(buffer_b[idx])) {
            buffer_a[idx] -= 0x4000;
        }
    }
#endif

    int16_t scaleFract;
    uint8_t shift;
    if (A->scale > B->scale) {
        float_to_scale_params(&scaleFract, &shift, 1.0f * B->scale / A->scale);
        my_scale_q15(buffer_b, scaleFract, shift, buffer_b, vector_size);
    } else if (B->scale > A->scale) {
        float_to_scale_params(&scaleFract, &shift, 1.0f * A->scale / B->scale);
        my_scale_q15(buffer_a, scaleFract, shift, buffer_a, vector_size);
    }
    my_add_q15(buffer_a, buffer_b, buffer_a, vector_size);

#if INDIRECT_RECOVERY
    iterate_chunks(model, output, 0, vector_size, OutputChunkHandler, buffer_a);
#endif

    my_memcpy_to_param(output, 0, buffer_a, output->params_len);

    flip_state_bit(model, output);

    my_printf_debug("handle_add output" NEWLINE);
    dump_params_debug(model, output);
}

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
    int16_t offset, next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           first_unfinished_value_offset, model, output);
    offset ^= 0x4000;
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
                        if ((c + idx) % (BATCH_SIZE + 1) == BATCH_SIZE) {
                            output_val = (offset ? 1 : -1);
                            if (next_output_turning_point > 0 && (output_offset + idx >= next_output_turning_point)) {
                                output_val = -output_val;
                            }
                        } else
#endif
                        {
                            input_val = lea_buffer[idx];
#if STATEFUL
                            // assuming input state bits are correct...
                            if (get_value_state_bit(input_val)) {
                                input_val -= 0x4000;
                            }
#endif
                            output_val = MAX_VAL(input_val, 0);
                        }
                        lea_buffer[output_idx] = output_val;
                        output_idx++;
                    }
#if STATEFUL
                    if (offset) {
                        uint8_t block_size;
                        if (next_output_turning_point < 0) {
                            block_size = len;
                        } else {
                            block_size = MIN_VAL(len, next_output_turning_point - output_offset);
                        }
                        my_offset_q15(lea_buffer, offset, lea_buffer, block_size);
                    } else if (next_output_turning_point < output_offset + len) {
                        int16_t* to_offset = lea_buffer + next_output_turning_point - output_offset;
                        my_offset_q15(to_offset, 0x4000, to_offset, output_offset + len - next_output_turning_point);
                    }
#endif
#if HAWAII
                    hawaii_preserve_vector(model, output, output_offset, lea_buffer, len);
#else
                    my_memcpy_to_param(output, output_offset, lea_buffer, output_idx * sizeof(int16_t));
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
        uint16_t i = 0;
        for (; i < data_len; i++) {
            int16_t output_val;
#if JAPARI
            if (i % 2) {
                output_val = 1;
            } else
#endif
            {
                int16_t input_val = get_q15_param(model, X, data_offset);
#if STATEFUL
                if (get_value_state_bit(input_val)) {
                    input_val -= 0x4000;
                }
                check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif
                output_val = MAX_VAL(input_val, 0);
            }
#if STATEFUL
            output_val += offset;
#endif
            put_q15_param(output, output_offset, output_val);
#if HAWAII
            write_hawaii_layer_footprint(model->layer_idx, 1);
#endif
            data_offset++;
            output_offset++;
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
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i] != RESHAPE_AUTO_DIM && output->dims[i] != 0) {
            inferred_dim /= output->dims[i];
        } else if (output->dims[i] == RESHAPE_AUTO_DIM) {
            auto_idx = i;
        }
    }
    if (auto_idx != -1) {
        output->dims[auto_idx] = inferred_dim;
        new_len *= inferred_dim;
    }
#if JAPARI
    else {
        new_len = extend_for_footprints(new_len);
    }
#endif
    MY_ASSERT(new_len * sizeof(int16_t) == output->params_len);
}

void handle_squeeze(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Squeeze!" NEWLINE);

    const ParameterInfo *data = input[0];
    /* XXX: add flags; assume squeeze all one-size axes */
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth = data->bitwidth;
    output->slot = data->slot;
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    if (cur_slot_info) {
        cur_slot_info->user = model->layer_idx;
    }
    for (uint8_t i = 0, j = 0; i < 4; i++) {
        if (input[0]->dims[i] != 1) {
            output->dims[j] = input[0]->dims[i];
            j++;
        }
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

void handle_dropout(Model*, const ParameterInfo*[], ParameterInfo*, const NodeFlags*) {
    ERROR_OCCURRED();
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
