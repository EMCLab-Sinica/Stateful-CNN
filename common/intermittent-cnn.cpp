#include <stdint.h>
#include <string.h>
#include <inttypes.h> // for PRId32

#include "intermittent-cnn.h"
#include "op_handlers.h"
#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"

static void handle_node(Model *model, Node *nodes, uint16_t node_idx) {
    my_printf_debug("Current node: %d, ", node_idx);

    /* schedule it */
    Node *cur_node = &(nodes[node_idx]);
    my_printf_debug("name = %s, ", cur_node->name);
    my_printf_debug("op_type = %d" NEWLINE, cur_node->op_type);

    int16_t input_id[3];
    ParameterInfo *input[3];
    MY_ASSERT(cur_node->inputs_len == expected_inputs_len[cur_node->op_type]);
    for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
        input_id[j] = node_input(cur_node, j);
        my_printf_debug("input_id[%d] = %d ", j, input_id[j]);
        input[j] = get_parameter_info(input_id[j]);
        if (input[j]->slot == SLOT_TEST_SET) {
            input[j]->params_offset = (model->sample_idx % PLAT_LABELS_DATA_LEN) * input[j]->params_len;
        }
        // dump_params(input[j]);
    }
    my_printf_debug(NEWLINE);

    /* Allocate an ParameterInfo for output. Details are filled by
     * individual operation handlers */
    ParameterInfo *output = get_parameter_info(node_idx + model->n_input);
    uint16_t parameter_info_idx_saved = output->parameter_info_idx;
    my_memcpy(output, input[0], sizeof(ParameterInfo));
    output->parameter_info_idx = parameter_info_idx_saved;
    output->params_offset = 0;
    allocators[cur_node->op_type](model, input, output, &cur_node->flags);
    my_printf_debug("Needed mem = %d" NEWLINE, output->params_len);
    if (output->slot == SLOT_INTERMEDIATE_VALUES) {
        my_printf_debug("New params_offset = %d" NEWLINE, output->params_offset);
    }

#ifdef WITH_PROGRESS_EMBEDDING
    my_printf_debug("Old output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif
    handlers[cur_node->op_type](model, input, output, &cur_node->flags);
    // For some operations (e.g., ConvMerge), scale is determined in the handlers
    my_printf_debug("Scale = %d" NEWLINE, output->scale);
#ifdef WITH_PROGRESS_EMBEDDING
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

    if (node_idx == model->nodes_len - 1) {
        model->running = 0;
        model->run_counter++;
    }
}

int run_model(Model *model, int8_t *ansptr, ParameterInfo **output_node_ptr) {
    Node *nodes = (Node*)(model + 1);
    inputs_data = reinterpret_cast<uint8_t*>(nodes + model->nodes_len);

    my_printf_debug("model->n_input = %d" NEWLINE, model->n_input);

    if (!model->running) {
        counters()->counter_idx = 0;
        // reset model
        model->layer_idx = 0;
        for (uint8_t idx = 0; idx < NUM_SLOTS; idx++) {
            SlotInfo *cur_slot_info = get_slot_info(idx);
            cur_slot_info->user = -1;
        }
        for (uint16_t node_idx = 0; node_idx < model->nodes_len; node_idx++) {
            nodes[node_idx].max_output_id &= ~MAX_OUTPUT_ID_INVALID;
        }
        model->running = 1;
    }

    counters()->power_counters[counters()->counter_idx]++;

    dump_model_debug(model, nodes);

    for (uint16_t node_idx = model->layer_idx; node_idx < model->nodes_len; node_idx++) {
        handle_node(model, nodes, node_idx);
        model->layer_idx++;

        dump_model_debug(model, nodes);
    }

    /* XXX: is the last node always the output node? */
    ParameterInfo *output_node = get_parameter_info(model->nodes_len + model->n_input - 1);
    if (output_node_ptr) {
        *output_node_ptr = output_node;
    }
    int16_t max = INT16_MIN;
    for (uint16_t i = 0; i < output_node->dims[1]; i++) {
        int16_t val = get_q15_param(output_node, i);
        if (val > max) {
            *ansptr = (uint8_t)i;
            max = val;
        }
    }

    return 0;
}

#if MY_DEBUG >= 1
static void print_results(Model *model, ParameterInfo *output_node) {
    dump_params(model, output_node);

    my_printf("op types:" NEWLINE);
    Node *nodes = (Node*)(model + 1);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 5d ", nodes[i].op_type);
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
    plat_print_results();
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
    ParameterInfo *output_node;
    Model *model = (Model*)model_data;
    const uint8_t *labels = labels_data;
    for (uint16_t i = 0; i < n_samples; i++) {
        model->sample_idx = i;
        label = labels[i];
        run_model(model, &predicted, &output_node);
        total++;
        if (label == predicted) {
            correct++;
        }
#if MY_DEBUG >= 1
        if (i % 100 == 0) {
            my_printf("Sample %d finished" NEWLINE, model->sample_idx);
            // stdout is not flushed at \n if it is not a terminal
            my_flush();
        }
        my_printf_debug("idx=%d label=%d predicted=%d correct=%d" NEWLINE, i, label, predicted, label == predicted);
#endif
    }
#if MY_DEBUG >= 1
    if (n_samples == 1) {
        print_results(model, output_node);
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

void set_sample_index(Model *model, uint8_t index) {
    model->sample_idx = index;
}

void reset_everything(Model *model) {
    model->first_time = 1;
    model->running = 0;
#ifdef WITH_PROGRESS_EMBEDDING
    for (uint8_t idx = 0; idx < NUM_SLOTS; idx++) {
        SlotInfo *cur_slot_info = get_slot_info(idx);
        cur_slot_info->state_bit = 0;
        cur_slot_info->n_turning_points = 0;
    }
#endif
}

#ifdef WITH_PROGRESS_EMBEDDING
void flip_state_bit(Model *model, ParameterInfo *output) {
    int16_t new_turning_point = output->params_len / 2;
    my_printf_debug("New turning point=%d" NEWLINE, new_turning_point);
    SlotInfo *cur_slot_info = get_slot_info(output->slot);
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

    dump_turning_points_debug(output);

    cur_slot_info->state_bit ^= 1;

#if MY_DEBUG >= 1
    // dump_matrix_debug(output, 0, output->params_len / 2, ValueInfo(output));
    uint8_t cur_state_bit = get_state_bit(model, output->slot);
    uint8_t turning_point_idx = 0;
    uint16_t next_turning_point = -1;
    if (turning_point_idx < cur_slot_info->n_turning_points) {
        next_turning_point = cur_slot_info->turning_points[turning_point_idx];
    }
    for (uint32_t idx = 0; idx < INTERMEDIATE_VALUES_SIZE / sizeof(int16_t); idx++) {
        if (idx == next_turning_point) {
            cur_state_bit ^= 1;
            turning_point_idx++;
            if (turning_point_idx < cur_slot_info->n_turning_points) {
                next_turning_point = cur_slot_info->turning_points[turning_point_idx];
            }
        }
        int16_t val = get_q15_param(output, idx);
        MY_ASSERT(get_value_state_bit(val) == cur_state_bit);
    }
    // dump_matrix_debug(output, 0, new_turning_point, ValueInfo(output));
#endif
}

uint8_t get_state_bit(Model *model, uint8_t slot_id) {
    switch (slot_id) {
        case SLOT_PARAMETERS:
        case SLOT_PARAMETERS2:
        case SLOT_TEST_SET:
            return 0;
        default:
            return get_slot_info(slot_id)->state_bit;
    }
}

uint8_t param_state_bit(Model *model, ParameterInfo *param, uint16_t offset) {
    uint8_t ret = get_state_bit(model, param->slot);
    SlotInfo *cur_slot_info = get_slot_info(param->slot);
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

uint32_t recovery_from_state_bits(Model *model, ParameterInfo *output) {
#if MY_DEBUG >= 1
    if (!after_recovery) {
        return 0;
    }
#endif

    // recovery from state bits
    uint32_t end_offset = output->params_len / 2;
    uint32_t cur_begin_offset = 0;
    uint32_t cur_end_offset = end_offset;
    uint32_t first_unfinished_value_offset;
    my_printf_debug("new_output_state_bit for first value = %d" NEWLINE, param_state_bit(model, output, 0) ^ 1);

    dump_turning_points_debug(output);

    while (1) {
#if 1
        dump_matrix_debug(output, cur_begin_offset, cur_end_offset - cur_begin_offset, ValueInfo(output));
#endif
        if (cur_end_offset - cur_begin_offset <= 1) {
            if (get_value_state_bit(get_q15_param(output, cur_begin_offset)) == param_state_bit(model, output, cur_begin_offset)) {
                first_unfinished_value_offset = 0;
            } else if (get_value_state_bit(get_q15_param(output, cur_end_offset)) == param_state_bit(model, output, cur_end_offset)) {
                first_unfinished_value_offset = cur_end_offset;
            } else if (cur_end_offset == end_offset) {
                // all values finished - power failure just before the state
                // bit for the output is flipped
                first_unfinished_value_offset = end_offset;
            } else {
                ERROR_OCCURRED();
            }
            break;
        }
        uint32_t middle_offset = cur_begin_offset + (cur_end_offset - cur_begin_offset) / 2;
        if (get_value_state_bit(get_q15_param(output, middle_offset)) != param_state_bit(model, output, middle_offset)) {
            cur_begin_offset = middle_offset;
        } else {
            cur_end_offset = middle_offset;
        }
        my_printf_debug(
            "offset of begin = %" PRId32 ", offset of end = %" PRId32 NEWLINE,
            cur_begin_offset, cur_end_offset
        );
    }

    my_printf_debug("first_unfinished_value_offset = %d" NEWLINE, first_unfinished_value_offset);

    if (!after_recovery) {
        MY_ASSERT(first_unfinished_value_offset == 0);
    } else {
        after_recovery = 0;
    }

#if MY_DEBUG >= 1
    for (uint32_t idx = 0; idx < first_unfinished_value_offset; idx++) {
        MY_ASSERT(get_value_state_bit(get_q15_param(output, idx)) != param_state_bit(model, output, idx));
    }
    for (uint32_t idx = first_unfinished_value_offset; idx < output->params_len / 2; idx++) {
        MY_ASSERT(get_value_state_bit(get_q15_param(output, idx)) == param_state_bit(model, output, idx));
    }
#endif

    return first_unfinished_value_offset;
}

#endif
