#include "cnn_common.h"
#include "op_utils.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"

#define RESHAPE_AUTO_DIM static_cast<uint16_t>(-1)

void alloc_maxpool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    uint16_t stride = flags->stride;

    const ParameterInfo *data = input[0];

    const uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t new_H = H / stride;
    uint16_t new_W = W / stride;

    output->params_len = new_H * new_W * CHANNEL * sizeof(int16_t);
    output->slot = get_next_slot(model, data);
    output->dims[0] = 1;
    output->dims[1] = CHANNEL;
    output->dims[2] = new_H;
    output->dims[3] = new_W;
}

static int16_t maxpool_patch(uint16_t output_h, uint16_t output_w, uint16_t c, const NodeFlags* flags, const ParameterInfo *data, Model *model) {
    const uint16_t CHANNEL = data->dims[1], W = data->dims[3];
    uint16_t stride = flags->stride;
    uint16_t kernel_size = flags->kernel_size;

    int16_t offset_h, offset_w;
    offset_h = W * CHANNEL;
    offset_w = CHANNEL;

    my_printf_debug("output_h=% 3d ", output_h);
    my_printf_debug("output_w=% 3d ", output_w);
    my_printf_debug("c=% 3d ", c);

    int16_t max_val = INT16_MIN;
    for (uint16_t sH = 0; sH < kernel_size; sH++) {
        for (uint16_t sW = 0; sW < kernel_size; sW++) {
            uint16_t val_offset = (output_h*stride+sH) * offset_h + (output_w*stride+sW) * offset_w + c;
            int16_t val = get_q15_param(model, data, val_offset);
#if STATEFUL_CNN
            if (get_value_state_bit(val)) {
                // assuming input state bits are correct...
                val -= 0x4000;
            }
#endif
            // dump_value_debug(model, data, val_offset);
            my_printf_debug("% 5d ", val);
            // XXX: use LEA?
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    // need a space as dump_value does not append spaces when DUMP_INTEGERS is not defined
    my_printf_debug(" max=% 5d ", max_val);
    return max_val;
}

void handle_maxpool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags* flags) {
    my_printf_debug("MaxPool!" NEWLINE);

    uint16_t stride = flags->stride;
    uint8_t need_nhwc2nchw = (flags->generic == NHWC2NCHW);

    /* XXX: add flags; assume no padding for now */
    const ParameterInfo *data = input[0];

    my_printf_debug("handle_maxpool input" NEWLINE);
    dump_params_debug(model, data);

    const uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t new_H = H / stride;
    uint16_t new_W = W / stride;

    determine_tile_c(output);
    uint16_t tile_c = output->tile_c;
    my_printf_debug("tile_c = %d" NEWLINE, tile_c);

    uint16_t tile_c_offset = 0;

    uint16_t output_h = 0, output_w = 0, c = 0;
    uint16_t output_offset = 0;

#if STATEFUL_CNN
    uint16_t initial_real_tile_c;

    uint32_t first_unfinished_value_offset = recovery_from_state_bits(model, output);
    if (first_unfinished_value_offset * sizeof(int16_t) == output->params_len) {
        // give up early, or initial_real_tile_c may be zero and results in SIGFPE
        goto finished;
    }

    uint16_t initial_n, initial_c, initial_h, initial_w;
    initial_n = first_unfinished_value_offset / (new_H * new_W * tile_c);

    tile_c_offset = initial_n * tile_c;

    int16_t offset, next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, first_unfinished_value_offset, model, output);
    offset ^= 0x4000;

    initial_real_tile_c = MIN_VAL(tile_c, CHANNEL - tile_c_offset);
    output_offset = first_unfinished_value_offset;
    if (!need_nhwc2nchw) {
        initial_c = first_unfinished_value_offset % initial_real_tile_c;
        first_unfinished_value_offset /= initial_real_tile_c;
        initial_w = first_unfinished_value_offset % new_W;
        first_unfinished_value_offset /= new_W;
        initial_h = first_unfinished_value_offset % new_H;
    } else {
        initial_w = first_unfinished_value_offset % new_W;
        first_unfinished_value_offset /= new_W;
        initial_h = first_unfinished_value_offset % new_H;
        first_unfinished_value_offset /= new_H;
        initial_c = first_unfinished_value_offset % initial_real_tile_c;
    }
    output_h = initial_h;
    output_w = initial_w;
    c = initial_c;
    my_printf_debug("initial_n = %d" NEWLINE, initial_n);
    my_printf_debug("initial_h = %d" NEWLINE, initial_h);
    my_printf_debug("initial_w = %d" NEWLINE, initial_w);
    my_printf_debug("initial_c = %d" NEWLINE, initial_c);
#endif

    for (; tile_c_offset < CHANNEL; tile_c_offset += tile_c) {
        uint16_t real_tile_c = MIN_VAL(tile_c, CHANNEL - tile_c_offset);
        if (!need_nhwc2nchw) {
            // NHWC
            for (; output_h < new_H; output_h++) {
                for (; output_w < new_W; output_w++) {
                    for (; c < real_tile_c; c++) {
                        int16_t max_val = maxpool_patch(output_h, output_w, c + tile_c_offset, flags, data, model);
                        my_printf_debug("output_offset=%d" NEWLINE, output_offset);
#if STATEFUL_CNN
                        check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
                        max_val += offset;
#endif
                        put_q15_param(output, output_offset, max_val);
                        output_offset++;
                    }
                    c = 0;
                }
                output_w = 0;
            }
            output_h = 0;
        } else {
            // NCHW
            for (; c < real_tile_c; c++) {
                for (; output_h < new_H; output_h++) {
                    for (; output_w < new_W; output_w++) {
                        int16_t max_val = maxpool_patch(output_h, output_w, c + tile_c_offset, flags, data, model);
                        my_printf_debug("output_offset=%d" NEWLINE, output_offset);
#if STATEFUL_CNN
                        check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
                        max_val += offset;
#endif
                        put_q15_param(output, output_offset, max_val);
                        output_offset++;
                    }
                    output_w = 0;
                }
                output_h = 0;
            }
            c = 0;
        }
    }

    MY_ASSERT(output_offset == output->params_len / sizeof(int16_t));

#if STATEFUL_CNN
finished:
    flip_state_bit(model, output);
#endif

    my_printf_debug("handle_maxpool output" NEWLINE);
    if (!need_nhwc2nchw) {
        dump_params_nhwc_debug(model, output, 0);
    } else if (tile_c == CHANNEL) {
        dump_params_debug(model, output);
    }
}

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

#if STATEFUL_CNN
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

#if STATEFUL_CNN
    iterate_chunks(model, output, 0, vector_size, OutputChunkHandler(buffer_a));
#endif

    my_memcpy_to_param(output, 0, buffer_a, output->params_len);

#if STATEFUL_CNN
    flip_state_bit(model, output);
#endif

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

    /* XXX: use LEA? */
    uint16_t bitwidth = X->bitwidth;
    MY_ASSERT(bitwidth == 16);
    int16_t data_len = X->params_len / (bitwidth / 8);

#if STATEFUL_CNN
#endif

    uint16_t data_offset = 0;
    uint16_t output_offset = 0;
#if STATEFUL_CNN
    uint32_t first_unfinished_value_offset = recovery_from_state_bits(model, output);
    data_offset += first_unfinished_value_offset;
    output_offset += first_unfinished_value_offset;
#endif

    my_printf_debug("handle_relu input" NEWLINE);
    if (X->flags & TRANSPOSED) {
        dump_params_nhwc_debug(model, X, 0);
        // input is in NWHC
        // TODO: state-aware recovery
        uint16_t H = X->dims[2], W = X->dims[3];
        uint16_t output_h = 0, output_w = 0, c = 0;
#if STATEFUL_CNN
        output_h = first_unfinished_value_offset / (W * CHANNEL);
        first_unfinished_value_offset %= (W * CHANNEL);
        output_w = first_unfinished_value_offset / CHANNEL;
        c = first_unfinished_value_offset % CHANNEL;
        my_printf_debug("initial output_h = %d, ", output_h);
        my_printf_debug("initial output_w = %d, ", output_w);
        my_printf_debug("initial c = %d" NEWLINE, c);

        int16_t offset, next_output_turning_point;
        uint8_t output_turning_point_idx;
        SlotInfo *output_slot_info;
        find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, first_unfinished_value_offset, model, output);
        offset ^= 0x4000;
#endif
        for (; output_h < H; output_h++) {
            for (; output_w < W; output_w++) {
                for (; c < CHANNEL; c++) {
                    int16_t input_tile_c_index = c / X->tile_c;
                    int16_t input_tile_c_offset = c % X->tile_c;
                    uint16_t cur_input_tile_c = MIN_VAL(X->tile_c, CHANNEL - input_tile_c_index * X->tile_c);
                    int16_t val_offset = input_tile_c_index * W * H * X->tile_c + output_w * H * cur_input_tile_c + output_h * cur_input_tile_c + input_tile_c_offset;
                    int16_t input_val = get_q15_param(model, X, val_offset);
                    output_offset = output_h * W * CHANNEL + output_w * CHANNEL + c;
#if STATEFUL_CNN
                    // assuming input state bits are correct...
                    if (get_value_state_bit(input_val)) {
                        input_val -= 0x4000;
                    }
                    check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset);
#endif
                    int16_t output_val = MAX_VAL(input_val, 0);
#if STATEFUL_CNN
                    output_val += offset;
#endif
                    put_q15_param(output, output_offset, output_val);
#if STATEFUL_CNN
                    my_printf_debug(
                        "output_h=% 3d, output_w=% 3d, c=% 3d, val_offset=% 6d, offset=% 6d, input val=% 6d, output_offset=% 6d, output val=% 6d" NEWLINE,
                        output_h, output_w, c, val_offset, offset, input_val, output_offset, output_val);
#else
                    my_printf_debug(
                        "output_h=% 3d, output_w=% 3d, c=% 3d, val_offset=% 6d, input val=% 6d, output_offset=% 6d, output val=% 6d" NEWLINE,
                        output_h, output_w, c, val_offset, input_val, output_offset, output_val);
#endif
                }
                c = 0;
            }
            output_w = 0;
        }
    } else {
        dump_params_debug(model, X);
        uint16_t i = 0;
#if STATEFUL_CNN
        MY_ASSERT(false); // TODO: adapt to range-based state assignments
#endif
        for (; i < data_len; i++) {
            put_q15_param(output, output_offset, MAX_VAL(get_q15_param(model, X, data_offset), 0));
            data_offset++;
            output_offset++;
        }
    }

    output->tile_c = CHANNEL;

#if STATEFUL_CNN
    flip_state_bit(model, output);
#endif

    my_printf_debug("handle_relu output" NEWLINE);
    if (X->flags & TRANSPOSED) {
        dump_params_nhwc_debug(model, output, 0);
    } else {
        dump_params_debug(model, output);
    }
}

void handle_reshape(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Reshape!" NEWLINE);

    const ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth = data->bitwidth;
    output->slot = data->slot;
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

class ConcatOutputChunkHandler : public ChunkHandler {
public:
    ConcatOutputChunkHandler(uint32_t _offset) : offset(_offset) {}

    void operator () (uint32_t output_offset, uint16_t output_chunk_len, uint8_t old_output_state_bit) const override {
        my_printf_debug("output output_offset=%d output_chunk_len=%d old_output_state_bit=%d" NEWLINE, output_offset, output_chunk_len, old_output_state_bit);
        // every output chunk has the same starting offset as corresponding scaled input chunk
        int16_t *output_to_offset = lea_buffer + output_offset - offset;
        if (!old_output_state_bit) {
            my_offset_q15(output_to_offset, 0x4000, output_to_offset, output_chunk_len);
        }
    }

private:
    uint32_t offset;
};

void handle_concat(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("Concat!" NEWLINE);

    const ParameterInfo *A = input[0], *B = input[1];
    // XXX: assume concatenating 2 tensors at the CHANNEL dimension and they
    // have the same number of channels.
    MY_ASSERT(A->dims[1] == B->dims[1]);
    output->tile_c = A->dims[1];
    output->dims[1] *= 2;
    output->flags |= SEPARATE_TILING;

    // The one with smaller `scale` (with larger values) is scaled down
    output->scale = MAX_VAL(A->scale, B->scale);

    // saving slots here as it might be changed during the downscaling loop above
    output->extra_info[0] = A->parameter_info_idx;
    output->extra_info[1] = B->parameter_info_idx;
    output->slot = A->slot;

    dump_params_nhwc_debug(model, A, 0);
    dump_params_nhwc_debug(model, B, 0);
}

void handle_dropout(Model*, const ParameterInfo*[], ParameterInfo*, const NodeFlags*) {
    ERROR_OCCURRED();
}

void alloc_globalaveragepool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    const ParameterInfo *data = input[0];

    MY_ASSERT(data->dims[0] == 1);
    uint16_t output_len = data->dims[1];

    output->dims[0] = output->dims[2] = output->dims[3] = 1;
    output->dims[1] = data->dims[1];
    output->params_len = output_len * sizeof(int16_t);
    output->bitwidth = 16;
    output->slot = get_next_slot(model, data);
}

void handle_globalaveragepool(Model *model, const ParameterInfo *input[], ParameterInfo *output, const NodeFlags*) {
    my_printf_debug("GlobalAveragePool!" NEWLINE);

    const ParameterInfo *data = input[0];

#if STATEFUL_CNN
    int16_t offset, next_output_turning_point;
    uint8_t output_turning_point_idx;
    SlotInfo *output_slot_info;
    find_initial_state_bit(&offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info, 0 /*TODO: first_unfinished_value_offset*/, model, output);
    offset ^= 0x4000;
#endif

    uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t len = H * W;
    for (uint16_t c = 0; c < CHANNEL; c++) {
        uint32_t total = 0;
        for (uint16_t h = 0; h < H; h++) {
            for (uint16_t w = 0; w < W; w++) {
                // Input is from Conv, which uses NHWC
                int16_t val = get_q15_param(model, data, h * W * CHANNEL + w * CHANNEL + c);
#if STATEFUL_CNN
                if (get_value_state_bit(val)) {
                    val -= 0x4000;
                }
#endif
                total += val;
            }
        }
        int16_t avg = total / len;
#if STATEFUL_CNN
        check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, c);
        avg += offset;
#endif
        put_q15_param(output, c, avg);
    }

#if STATEFUL_CNN
    flip_state_bit(model, output);
#endif

    dump_params_debug(model, output);
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
