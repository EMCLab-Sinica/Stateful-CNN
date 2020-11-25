#include <stdint.h>
#include <string.h>
#include <inttypes.h> // for PRId32

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"

uint16_t sample_idx;

const uint8_t MAX_CLASSES = 20;

static void handle_node(Model *model, uint16_t node_idx) {
    my_printf_debug("Current node: %d, ", node_idx);

    /* schedule it */
    const Node *cur_node = get_node(node_idx);
    my_printf_debug("name = %.*s, ", NODE_NAME_LEN, cur_node->name);
    my_printf_debug("op_type = %d" NEWLINE, cur_node->op_type);

    int16_t input_id[3];
    const ParameterInfo *input[3];
    MY_ASSERT(cur_node->inputs_len == expected_inputs_len[cur_node->op_type]);
    for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
        input_id[j] = cur_node->inputs[j];
        my_printf_debug("input_id[%d] = %d" NEWLINE, j, input_id[j]);
        input[j] = get_parameter_info(input_id[j]);
        // dump_params(input[j]);
    }
    my_printf_debug(NEWLINE);

    /* Allocate an ParameterInfo for output. Details are filled by
     * individual operation handlers */
    ParameterInfo *output = get_intermediate_parameter_info(node_idx);
    my_memcpy(output, input[0], sizeof(ParameterInfo) - sizeof(uint16_t)); // don't overwrite parameter_info_idx
    output->params_offset = 0;
    allocators[cur_node->op_type](model, input, output, &cur_node->flags);
    my_printf_debug("Needed mem = %d" NEWLINE, output->params_len);
    MY_ASSERT(output->params_len < INTERMEDIATE_VALUES_SIZE);
    if (output->slot == SLOT_INTERMEDIATE_VALUES) {
        my_printf_debug("New params_offset = %d" NEWLINE, output->params_offset);
    }

#if STATEFUL
    my_printf_debug("Old output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif
    handlers[cur_node->op_type](model, input, output, &cur_node->flags);
    // For some operations (e.g., ConvMerge), scale is determined in the handlers
    my_printf_debug("Output scale = %d" NEWLINE, output->scale);
    MY_ASSERT(output->scale > 0);  // fail when overflow
#if STATEFUL
    my_printf_debug("New output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif

    counters()->counter_idx++;
    MY_ASSERT(counters()->counter_idx < COUNTERS_LEN);

#if MY_DEBUG >= 1
    my_printf_debug("output dims: ");
    uint8_t has_dims = 0;
    for (uint8_t j = 0; j < 4; j++) {
        if (output->dims[j]) {
            has_dims = 1;
            my_printf_debug("%d, ", output->dims[j]);
        }
    }
    my_printf_debug(NEWLINE);
    MY_ASSERT(has_dims);
    MY_ASSERT(output->bitwidth);
#endif

    commit_intermediate_parameter_info(node_idx);

    if (node_idx == MODEL_NODES_LEN - 1) {
        model->running = 0;
        model->run_counter++;
    }
}

static void run_model(int8_t *ansptr, const ParameterInfo **output_node_ptr) {
    my_printf_debug("N_INPUT = %d" NEWLINE, N_INPUT);

    Model *model = get_model();
    if (!model->running) {
        counters()->counter_idx = 0;
        // reset model
        model->layer_idx = 0;
        for (uint8_t idx = 0; idx < NUM_SLOTS; idx++) {
            SlotInfo *cur_slot_info = get_slot_info(model, idx);
            cur_slot_info->user = -1;
        }
#if HAWAII
        for (uint16_t node_idx = 0; node_idx < MODEL_NODES_LEN; node_idx++) {
            reset_hawaii_layer_footprint(node_idx);
        }
#endif
        model->running = 1;
        commit_model();
    }

    counters()->power_counters[counters()->counter_idx]++;

    dump_model_debug(model);

    for (uint16_t node_idx = model->layer_idx; node_idx < MODEL_NODES_LEN; node_idx++) {
        handle_node(model, node_idx);
        model->layer_idx++;

        commit_model();

        dump_model_debug(model);
    }

    /* XXX: is the last node always the output node? */
    const ParameterInfo *output_node = get_parameter_info(MODEL_NODES_LEN + N_INPUT - 1);
    if (output_node_ptr) {
        *output_node_ptr = output_node;
    }
    int16_t max = INT16_MIN;
    uint16_t u_ans;
    uint8_t buffer_len = MIN_VAL(output_node->dims[1], MAX_CLASSES);
    my_memcpy_from_param(model, lea_buffer, output_node, 0, buffer_len * sizeof(int16_t));
    my_max_q15(lea_buffer, buffer_len, &max, &u_ans);
    *ansptr = u_ans;
}

#if MY_DEBUG >= 1
static void print_results(const ParameterInfo *output_node) {
    Model *model = get_model();

    dump_params(model, output_node);

    my_printf("op types:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 5d ", get_node(i)->op_type);
        if (i % 16 == 15) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE "ticks:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 5d ", counters()->time_counters[i]);
        if (i % 16 == 15) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE "power counters: ");
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("%d ", counters()->power_counters[i]);
    }
    my_printf(NEWLINE "DMA invocations:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 8d", counters()->dma_invocations[i]);
    }
    my_printf(NEWLINE "DMA bytes:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 8d", counters()->dma_bytes[i]);
    }
    my_printf(NEWLINE "run_counter: %d", model->run_counter);
    my_printf(NEWLINE);
}
#endif

uint8_t run_cnn_tests(uint16_t n_samples) {
    int8_t label = -1, predicted = -1;
    uint32_t correct = 0, total = 0;
    if (!n_samples) {
        n_samples = PLAT_LABELS_DATA_LEN;
    }
    const ParameterInfo *output_node;
    const uint8_t *labels = labels_data;
    for (uint16_t i = 0; i < n_samples; i++) {
        sample_idx = i;
        label = labels[i];
        run_model(&predicted, &output_node);
        total++;
        if (label == predicted) {
            correct++;
        }
#if MY_DEBUG >= 1
        if (i % 100 == 0) {
            my_printf("Sample %d finished" NEWLINE, sample_idx);
            // stdout is not flushed at \n if it is not a terminal
            my_flush();
        }
        my_printf_debug("idx=%d label=%d predicted=%d correct=%d" NEWLINE, i, label, predicted, label == predicted);
#endif
    }
#if MY_DEBUG >= 1
    if (n_samples == 1) {
        print_results(output_node);
    }
    my_printf("correct=%" PRId32 " ", correct);
    my_printf("total=%" PRId32 " ", total);
    my_printf("rate=%f" NEWLINE, 1.0*correct/total);
#endif

    // Allow only 1% of accuracy drop
    if (N_SAMPLES == N_ALL_SAMPLES && correct < (FP32_ACCURACY - 0.01) * total) {
        return 1;
    }
    return 0;
}

#if STATEFUL

uint32_t remap_offset(const ParameterInfo* output, uint32_t offset) {
    const Node* node = get_node(output->parameter_info_idx - N_INPUT);
    if (node->op_type != Conv) {
        return offset;
    }
    if (offset >= output->params_len / sizeof(int16_t)) {
        return offset;
    }
    uint32_t orig_offset = offset;

    uint16_t OUTPUT_CHANNEL = output->dims[1], OUTPUT_H = output->dims[2], OUTPUT_W = output->dims[3];
    uint16_t input_tile_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;
    uint8_t input_tile_c_index = offset / input_tile_len;
    offset = offset % input_tile_len;
    uint16_t output_tile_c = node->flags.conv_output_tile_c;
    uint16_t tile_len = OUTPUT_H * OUTPUT_W * output_tile_c;
    uint16_t channel_offset = offset / tile_len * output_tile_c;
    uint16_t cur_tile_offset = offset % tile_len;
    uint32_t remapped_offset = input_tile_c_index * input_tile_len +
                               OUTPUT_CHANNEL * (cur_tile_offset / output_tile_c) +
                               channel_offset + (cur_tile_offset % output_tile_c);
    if (output_tile_c == OUTPUT_CHANNEL) {
        MY_ASSERT(orig_offset == remapped_offset);
    }
    // my_printf_debug("Offset mapping %d => %d" NEWLINE, orig_offset, remapped_offset);
    return remapped_offset;
}

static void check_feature_map_states(Model *model, const ParameterInfo* output, uint32_t first_unfinished_job_index, uint32_t len, const char* func) {
#if MY_DEBUG >= 1
    my_printf_debug("Running check_feature_map_states..." NEWLINE);
#if 0
    for (uint32_t idx = 0; idx < len; idx++) {
        my_printf_debug("% 6d ", get_q15_param(model, output, idx));
        if (idx % 16 == 15) {
            my_printf_debug(NEWLINE);
        }
    }
#endif
    for (uint32_t idx = 0; idx < len; idx++) {
        uint32_t remapped_index = remap_offset(output, idx);
        int16_t val = get_q15_param(model, output, remapped_index);
        uint8_t cur_state_bit = param_state_bit(model, output, remapped_index);
        if (idx < first_unfinished_job_index) {
            cur_state_bit ^= 1;
        }
        MY_ASSERT(get_value_state_bit(val) == cur_state_bit,
            "Value %d at index %d (remapped to %d) does not have expected state bit %d" NEWLINE, val, idx, remapped_index, cur_state_bit);
    }
#endif
}

void flip_state_bit(Model *model, const ParameterInfo *output) {
    int16_t new_turning_point = output->params_len / 2;
    my_printf_debug("New turning point=%d" NEWLINE, new_turning_point);
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    // XXX: better way than copying the array?
    uint8_t new_turning_point_inserted = 0;
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        if (new_turning_point < cur_slot_info->turning_points[idx]) {
            uint8_t new_turning_point_idx = idx;
            cur_slot_info->n_turning_points++;
            MY_ASSERT(cur_slot_info->n_turning_points <= TURNING_POINTS_LEN);
            for (uint8_t idx2 = cur_slot_info->n_turning_points - 1; idx2 > new_turning_point_idx; idx2--) {
                cur_slot_info->turning_points[idx2] = cur_slot_info->turning_points[idx2 - 1];
            }
            cur_slot_info->turning_points[new_turning_point_idx] = new_turning_point;
            new_turning_point_inserted = 1;
            break;
        } else if (new_turning_point == cur_slot_info->turning_points[idx]) {
            cur_slot_info->n_turning_points--;
            for (uint8_t idx2 = idx; idx2 < cur_slot_info->n_turning_points; idx2++) {
                cur_slot_info->turning_points[idx2] = cur_slot_info->turning_points[idx2 + 1];
            }
            new_turning_point_inserted = 1;
            break;
        }
    }
    if (!new_turning_point_inserted) {
        cur_slot_info->n_turning_points++;
        cur_slot_info->turning_points[cur_slot_info->n_turning_points - 1] = new_turning_point;
    }

    dump_turning_points_debug(model, output);

    cur_slot_info->state_bit ^= 1;

    // Use first_unfinished_job_index = 0 here as all values finished and the initial state bit is flipped above
    check_feature_map_states(model, output, 0, INTERMEDIATE_VALUES_SIZE / sizeof(int16_t), __func__);
}

uint8_t get_state_bit(Model *model, uint8_t slot_id) {
    switch (slot_id) {
        case SLOT_PARAMETERS:
        case SLOT_TEST_SET:
            return 0;
        default:
            return get_slot_info(model, slot_id)->state_bit;
    }
}

uint8_t param_state_bit(Model *model, const ParameterInfo *param, uint16_t offset) {
    uint8_t ret = get_state_bit(model, param->slot);
    SlotInfo *cur_slot_info = get_slot_info(model, param->slot);
    if (!cur_slot_info) {
        return 0;
    }
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        if (offset >= cur_slot_info->turning_points[idx]) {
            ret = ret ^ 1;
        } else {
            break;
        }
    }
    return ret;
}

static uint8_t after_recovery = 1;

static uint8_t value_finished(Model* model, const ParameterInfo* output, uint32_t offset) {
    uint32_t remapped_offset = remap_offset(output, offset);
    return get_value_state_bit(get_q15_param(model, output, remapped_offset)) != param_state_bit(model, output, remapped_offset);
}

uint32_t run_recovery(Model *model, ParameterInfo *output) {
#if MY_DEBUG < 1
    if (!after_recovery) {
        return 0;
    }
#endif

    // recovery from state bits
    uint32_t end_offset = output->params_len / 2;
    uint32_t cur_begin_offset = 0;
    uint32_t cur_end_offset = end_offset;
    uint32_t first_unfinished_job_index;
    my_printf_debug("new_output_state_bit for first value = %d" NEWLINE, param_state_bit(model, output, 0) ^ 1);

    dump_turning_points_debug(model, output);

    while (1) {
#if 0
        dump_matrix_debug(model, output, cur_begin_offset, cur_end_offset - cur_begin_offset, ValueInfo(output));
#endif
        if (cur_end_offset - cur_begin_offset <= 1) {
            if (!value_finished(model, output, cur_begin_offset)) {
                first_unfinished_job_index = 0;
            } else if (!value_finished(model, output, cur_end_offset)) {
                first_unfinished_job_index = cur_end_offset;
            } else if (cur_end_offset == end_offset) {
                // all values finished - power failure just before the state
                // bit for the output is flipped
                first_unfinished_job_index = end_offset;
            } else {
                ERROR_OCCURRED();
            }
            break;
        }
        uint32_t middle_offset = cur_begin_offset + (cur_end_offset - cur_begin_offset) / 2;
        if (value_finished(model, output, middle_offset)) {
            cur_begin_offset = middle_offset;
        } else {
            cur_end_offset = middle_offset;
        }
        my_printf_debug(
            "offset of begin = %" PRId32 ", offset of end = %" PRId32 NEWLINE,
            cur_begin_offset, cur_end_offset
        );
    }

    my_printf_debug("first_unfinished_job_index = %d" NEWLINE, first_unfinished_job_index);

    if (!after_recovery) {
        MY_ASSERT(first_unfinished_job_index == 0);
    } else {
        after_recovery = 0;
    }

    check_feature_map_states(model, output, first_unfinished_job_index, output->params_len / 2, __func__);

    return first_unfinished_job_index;
}

#endif

#if HAWAII
uint32_t run_recovery(Model* model, ParameterInfo*) {
    return read_hawaii_layer_footprint(model->layer_idx);
}
#endif

#if JAPARI
int16_t get_layer_sign(Model *model) {
    return get_node(model->layer_idx)->layer_sign;
}
uint32_t run_recovery(Model *model, ParameterInfo *output) {
    // TODO
    return 0;
}
#endif
