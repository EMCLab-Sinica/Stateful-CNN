#include <cstdint>
#include "cnn_common.h"
#include "data.h"
#include "op_utils.h"
#include "my_debug.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "platform.h"

#define RESHAPE_AUTO_DIM static_cast<uint16_t>(-1)

const uint8_t RELU_TILE_SIZE = 16;
static_assert(RELU_TILE_SIZE % BATCH_SIZE == 0, "Incorrect tile size for ReLU");

void alloc_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    const ParameterInfo *data = input[0];
    output->slot = get_next_slot(model, data);
}

void handle_relu(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("ReLu!" NEWLINE);

    const ParameterInfo *X = input[0];

    uint16_t bitwidth = X->bitwidth;
    MY_ASSERT(bitwidth == 16);
    int16_t data_len = X->params_len / (bitwidth / 8);

    uint16_t output_offset = 0;
#if INTERMITTENT

    uint32_t first_unfinished_value_offset = batch_start(job_index_to_offset(output, run_recovery(model, output)));
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

    int16_t vals[32];
    uint16_t i = output_offset;
#if JAPARI
    const uint8_t real_relu_tile_size = extend_for_footprints(RELU_TILE_SIZE);
#else
    const uint8_t real_relu_tile_size = RELU_TILE_SIZE;
#endif
    for (; i < data_len; i += real_relu_tile_size) {
        uint8_t cur_tile_size = MIN_VAL(real_relu_tile_size, data_len - i);
        my_memcpy_from_param(model, vals, X, output_offset, cur_tile_size*sizeof(int16_t));

#if STATEFUL
        start_cpu_counter(&Counters::stripping);
        for (uint8_t j = 0; j < cur_tile_size; j++) {
            if (offset_has_state(output_offset+j)) {
                strip_state(&vals[j]);
            }
            vals[j] *= 2;
        }
        stop_cpu_counter();
#endif

        for (uint8_t j = 0; j < cur_tile_size; j++) {
            vals[j] = MAX_VAL(vals[j], 0);
        }

#if INDIRECT_RECOVERY
        start_cpu_counter(&Counters::embedding);
#if STATEFUL
        const uint8_t embedding_shift = BATCH_SIZE;
#else
        const uint8_t embedding_shift = BATCH_SIZE + 1;
#endif
        for (uint8_t j = 0; j < cur_tile_size; j += embedding_shift) {
            uint8_t tile_last = j + embedding_shift - 1;
            check_next_turning_point(offset, output_turning_point_idx, next_output_turning_point, output_slot_info, output_offset + tile_last);
#if STATEFUL
            for (uint8_t k = j; k < tile_last; k++) {
                vals[k] /= 2;
            }
            vals[tile_last] = vals[tile_last] / 2 + offset;
#else
            vals[tile_last] = (offset > 0 ? 1 : -1);
#endif
        }
        stop_cpu_counter();
#endif

#if MY_DEBUG >= MY_DEBUG_VERBOSE
        my_printf_debug("output_offset=[% 6d, % 6d), output_val=", output_offset, output_offset+cur_tile_size);
        for (uint8_t j = 0; j < cur_tile_size; j++) {
            my_printf_debug("% 6d", vals[j]);
            if (j != cur_tile_size - 1) {
                my_printf_debug(", ");
            }
        }
        my_printf_debug(NEWLINE);
#endif

        my_memcpy_to_param(output, output_offset, vals, cur_tile_size*sizeof(int16_t), 0);
        output_offset += cur_tile_size;
#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, cur_tile_size/BATCH_SIZE*BATCH_SIZE);
#endif
    }

    flip_state_bit(model, output);

    my_printf_debug("handle_relu output" NEWLINE);
    dump_params_nhwc_debug(model, output, node->output_name);
}

void handle_reshape(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    my_printf_debug("Reshape!" NEWLINE);

    const ParameterInfo *data = input[0], *shape = input[1];
    MY_ASSERT(shape->bitwidth == 64);
    /*
     * At most one dimension of the new shape can be -1. In this case, the
     * value is inferred from the size of the tensor and the remaining
     * dimensions.
     *
     * A dimension could also be 0, in which case the actual dimension value
     * is unchanged (i.e. taken from the input tensor).
     * */
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = get_int64_param(shape, i);
        if (!output->dims[i]) {
            output->dims[i] = data->dims[i];
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
            if (i == last_dim_idx && data->slot != SLOT_TEST_SET) {
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
    }
}

void handle_squeeze(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node) {
    my_printf_debug("Squeeze!" NEWLINE);

    uint8_t axes = node->flags.extra.squeeze.axes;
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
#if !JAPARI
                MY_ASSERT(input[0]->dims[i] == 1);
#endif
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

void handle_unsqueeze(Model* model, const ParameterInfo* input[], ParameterInfo* output, const Node* node) {
    my_printf_debug("Unsqueeze!" NEWLINE);
    uint8_t axes = node->flags.extra.squeeze.axes;
    uint8_t input_dim_offset = 0, output_dim_offset = 0;
    for (uint8_t i = 0; i < 4; i++) {
        if (axes & (1 << i)) {
            output->dims[output_dim_offset] = 1;
            output_dim_offset++;
        } else {
            output->dims[output_dim_offset] = input[0]->dims[input_dim_offset];
            input_dim_offset++;
            output_dim_offset++;
        }
    }
}

void alloc_concat(Model *, const ParameterInfo *[], ParameterInfo*, const Node*) {
}

void handle_concat(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    my_printf_debug("Concat!" NEWLINE);

    const ParameterInfo *A = input[0], *B = input[1];
    // XXX: assume concatenating 2 tensors at the CHANNEL dimension and they
    // have the same number of channels.
    MY_ASSERT(A->dims[1] == B->dims[1]);
    output->dims[1] *= 2;
    output->param_flags |= SEPARATE_TILING;

    // The one with smaller `scale` (with larger values) is scaled down
    output->scale = MAX_VAL(A->scale, B->scale);

    // saving slots here as it might be changed during the downscaling loop above
    output->extra_info[0] = A->parameter_info_idx;
    output->extra_info[1] = B->parameter_info_idx;
    output->slot = A->slot;

    dump_params_nhwc_debug(model, A);
    dump_params_nhwc_debug(model, B);
}

void handle_softmax(Model*, const ParameterInfo*[], ParameterInfo*, const Node*) {
    // Do nothing - softmax does not change the relative order of values.
    // Just let run_model determine the max value
}

void handle_transpose(Model*, const ParameterInfo *input[], ParameterInfo *output, const Node*) {
    my_printf_debug("Transpose!" NEWLINE);

    const ParameterInfo *X = input[0];
    // not actually transpose data as we happen to need NHWC
    // XXX: assume NHWC -> NCHW
    output->dims[1] = X->dims[3];
    output->dims[2] = X->dims[1];
    output->dims[3] = X->dims[2];
}

void alloc_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node) {
    output->slot = get_next_slot(model, input[0]);
}

void handle_add(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node *node) {
    my_printf_debug("Add!" NEWLINE);

    const ParameterInfo *X = input[0], *Y = input[1];

    uint32_t data_offset = 0;
#if INTERMITTENT
    uint32_t first_unfinished_job_idx = run_recovery(model, output);
    data_offset = batch_start(job_index_to_offset(output, first_unfinished_job_idx));

#if INDIRECT_RECOVERY
    uint16_t next_input_turning_point, next_output_turning_point;
    int16_t input_offset, output_offset;
    uint8_t input_turning_point_idx, output_turning_point_idx;
    SlotInfo *input_slot_info, *output_slot_info;
    find_initial_state_bit(&input_offset, &input_turning_point_idx, &next_input_turning_point, &input_slot_info,
                           data_offset, model, X);
    find_initial_state_bit(&output_offset, &output_turning_point_idx, &next_output_turning_point, &output_slot_info,
                           data_offset, model, output);
#endif

#endif

    uint16_t buffer_size = X->dims[1];
    int16_t *buffer_a = lea_buffer,
            *buffer_b = buffer_a + buffer_size;
    my_memcpy_from_param(model, buffer_b, Y, 0, buffer_size * sizeof(int16_t));
#if JAPARI
    move_weights(buffer_b, false, extend_for_footprints(buffer_size), buffer_size);
#endif
    my_printf_debug("weights" NEWLINE);
    dump_matrix_debug(buffer_b, buffer_size, ValueInfo(Y), false);

    int16_t scaleFract;
    uint8_t shift;
    float_to_scale_params(&scaleFract, &shift, 1.0f*Y->scale/X->scale);
    my_scale_q15(buffer_b, scaleFract, shift, buffer_b, buffer_size);

    for (uint16_t idx = data_offset / buffer_size; idx < X->dims[2]; idx++) {
        my_printf_debug("data_offset=%d" NEWLINE, data_offset);
        my_memcpy_from_param(model, buffer_a, X, data_offset, buffer_size * sizeof(int16_t));
#if STATEFUL
        check_next_turning_point(input_offset, input_turning_point_idx, next_input_turning_point, input_slot_info, data_offset);
        update_states(buffer_a, buffer_size, data_offset, input_offset, next_input_turning_point, false);
        my_printf_debug("After strip states" NEWLINE);
        dump_matrix_debug(buffer_a, buffer_size, ValueInfo(output), false);
#endif

        my_add_q15(buffer_a, buffer_b, buffer_a, buffer_size);
        my_printf_debug("After add" NEWLINE);
        dump_matrix_debug(buffer_a, buffer_size, ValueInfo(output), false);

#if INDIRECT_RECOVERY
        check_next_turning_point(output_offset, output_turning_point_idx, next_output_turning_point, output_slot_info, data_offset);
        update_states(buffer_a, buffer_size, data_offset, output_offset, next_output_turning_point, true);
        my_printf_debug("After embedding states" NEWLINE);
        dump_matrix_debug(buffer_a, buffer_size, ValueInfo(output), true);
#endif

        my_memcpy_to_param(output, data_offset, buffer_a, buffer_size * sizeof(int16_t), 0);
        data_offset += buffer_size;
#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, buffer_size/BATCH_SIZE*BATCH_SIZE);
#endif
    }

    flip_state_bit(model, output);

    dump_params_nhwc_debug(model, output, node->output_name);
}
