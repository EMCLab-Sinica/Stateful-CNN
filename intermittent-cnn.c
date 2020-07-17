#include <stdint.h>
#include <string.h>

#include "intermittent-cnn.h"
#include "op_handlers.h"
#include "cnn_common.h"
#include "data.h"
#include "debug.h"

static void handle_node(Model *model, Node *nodes, ParameterInfo* parameter_info, uint16_t node_idx) {
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
        input[j] = &(parameter_info[input_id[j]]);
        if (input[j]->slot == SLOT_TEST_SET) {
            input[j]->params_offset = (model->sample_idx % LABELS_DATA_LEN) * input[j]->params_len;
        }
        // dump_params(input[j]);
    }
    my_printf_debug(NEWLINE);

    /* Allocate an ParameterInfo for output. Details are filled by
     * individual operation handlers */
    ParameterInfo *output = &(parameter_info[node_idx + model->n_input]);
    output->params_offset = 0;
    allocators[cur_node->op_type](input, output, cur_node->flags);
    my_printf_debug("Needed mem = %d" NEWLINE, output->params_len);
    if (output->slot == SLOT_INTERMEDIATE_VALUES) {
        my_printf_debug("New params_offset = %d" NEWLINE, output->params_offset);
    }

#ifdef WITH_PROGRESS_EMBEDDING
    my_printf_debug("State bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif
    handlers[cur_node->op_type](model, input, output, cur_node->flags);
#ifdef WITH_PROGRESS_EMBEDDING
    my_printf_debug("State bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif

    counters()->counter_idx++;
    MY_ASSERT(counters()->counter_idx < COUNTERS_LEN);

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

    if (node_idx == model->nodes_len - 1) {
        model->running = 0;
        model->run_counter++;
    }
}

int run_model(Model *model, int8_t *ansptr, ParameterInfo **output_node_ptr) {
    Node *nodes = (Node*)(model + 1);
    ParameterInfo *parameter_info = (ParameterInfo*)(nodes + model->nodes_len);
    inputs_data = (uint8_t*)(parameter_info + model->nodes_len + model->n_input);

    my_printf_debug("model->n_input = %d" NEWLINE, model->n_input);

    if (!model->running) {
        counters()->counter_idx = 0;
        // reset model
        model->layer_idx = 0;
        for (uint8_t idx = 0; idx < NUM_SLOTS; idx++) {
            model->state_bit[idx] = 0;
        }
#ifdef WITH_PROGRESS_EMBEDDING
        fill_int16((int16_t*)intermediate_values(0), NUM_SLOTS * INTERMEDIATE_VALUES_SIZE / sizeof(int16_t), 0);
#endif
        model->running = 1;
    } else {
        model->recovery = 1;
    }

    counters()->power_counters[counters()->counter_idx]++;

    dump_model(model, nodes);

    for (uint16_t node_idx = model->layer_idx; node_idx < model->nodes_len; node_idx++) {
        Node *cur_node = &(nodes[node_idx]);

        handle_node(model, nodes, parameter_info, node_idx);
        model->layer_idx++;

        dump_model(model, nodes);
    }

    /* XXX: is the last node always the output node? */
    ParameterInfo *output_node = &(parameter_info[model->nodes_len + model->n_input - 1]);
    if (output_node_ptr) {
        *output_node_ptr = output_node;
    }
    int16_t max = INT16_MIN;
    for (uint16_t i = 0; i < output_node->dims[1]; i++) {
        int16_t val = *get_q15_param(output_node, i);
        if (val > max) {
            *ansptr = (uint8_t)i;
            max = val;
        }
    }

    return 0;
}

void print_results(Model *model, ParameterInfo *output_node) {
    for (uint16_t i = 0; i < output_node->dims[1]; i++) {
        print_q15(*get_q15_param(output_node, i));
    }
    my_printf(NEWLINE);

    my_printf("ticks: ");
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("%d ", counters()->time_counters[i]);
    }
    my_printf(NEWLINE "power counters: ");
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("%d ", counters()->power_counters[i]);
    }
    plat_print_results();
    my_printf(NEWLINE "run_counter: %d", model->run_counter);
    my_printf(NEWLINE);
}

void run_cnn_tests(uint16_t n_samples) {
    int8_t label = -1, predicted = -1;
    uint32_t correct = 0, total = 0;
    if (!n_samples) {
        n_samples = LABELS_DATA_LEN;
    }
    ParameterInfo *output_node;
    Model *model = (Model*)model_data;
    uint8_t *labels = labels_data;
    for (uint16_t i = 0; i < n_samples; i++) {
        model->sample_idx = i;
        label = labels[i];
        run_model(model, &predicted, &output_node);
        total++;
        if (label == predicted) {
            correct++;
        }
        my_printf_debug("label=%d predicted=%d correct=%d" NEWLINE, label, predicted, label == predicted);
    }
    if (n_samples == 1) {
        print_results(model, output_node);
    }
    my_printf("correct=%" PRId32 " ", correct);
    my_printf("total=%" PRId32 " ", total);
    my_printf("rate=%f" NEWLINE, 1.0*correct/total);
}

void set_sample_index(Model *model, uint8_t index) {
    model->sample_idx = index;
}

void flip_state_bit(Model *model, ParameterInfo *output) {
#ifdef WITH_PROGRESS_EMBEDDING
    // XXX: reduce # of values to fill
    int16_t fill_value;
    if (!get_state_bit(model, output->slot)) {
        fill_value = 0x4000;
    } else {
        fill_value = 0;
    }
    int16_t *output_ptr = get_q15_param(output, 0);
    uint16_t fill_offset = output->params_len / sizeof(int16_t),
             end = INTERMEDIATE_VALUES_SIZE / sizeof(int16_t);
    my_printf_debug("Fill %d", fill_value);
    my_printf_debug(" from %d", fill_offset);
    my_printf_debug(" to %d" NEWLINE, end);
    fill_int16(output_ptr + fill_offset, end - fill_offset, fill_value);

    uint8_t slot_id = output->slot;
    if (model->state_bit[slot_id]) {
        model->state_bit[slot_id] = 0;
    } else {
        model->state_bit[slot_id] = 1;
    }
#else
    UNUSED(model);
    UNUSED(output);
#endif
}

uint8_t get_state_bit(Model *model, uint8_t slot_id) {
#ifdef WITH_PROGRESS_EMBEDDING
    switch (slot_id) {
        case SLOT_PARAMETERS:
        case SLOT_PARAMETERS2:
        case SLOT_TEST_SET:
            return 0;
        default:
            return model->state_bit[slot_id];
    }
#else
    UNUSED(model);
    UNUSED(slot_id);
    ERROR_OCCURRED();
#endif
}

uint8_t get_value_state_bit(int16_t val) {
    if (val < 0x2000 && val >= -0x2000) {
        return 0;
    } else {
        return 1;
    }
}

#ifdef WITH_PROGRESS_EMBEDDING
static uint8_t after_recovery = 1;
#endif

uint32_t recovery_from_state_bits(Model *model, ParameterInfo *output) {
#ifdef WITH_PROGRESS_EMBEDDING
    // recovery from state bits
    int16_t *baseptr = get_q15_param(output, 0);
    int16_t *baseptr_end = baseptr + output->params_len / 2;
    int16_t *start = baseptr;
    int16_t *end = baseptr_end;
    uint8_t new_output_state_bit = get_state_bit(model, output->slot) ? 0 : 1;
    uint32_t first_unfinished_value_offset;
    my_printf_debug("new_output_state_bit = %d" NEWLINE, new_output_state_bit);

    while (1) {
        // dump_matrix(start, end - start);
        int16_t *middle = start + (end - start) / 2;
        if (middle == start || middle == end) {
            if (get_value_state_bit(*start) != new_output_state_bit) {
                first_unfinished_value_offset = start - baseptr;
            } else if (get_value_state_bit(*end) != new_output_state_bit) {
                first_unfinished_value_offset = end - baseptr;
            } else if (end == baseptr_end) {
                // all values finished - power failure just before the state
                // bit for the output is flipped
                first_unfinished_value_offset = baseptr_end - baseptr;
            } else {
                ERROR_OCCURRED();
            }
            break;
        }
        if (get_value_state_bit(*middle) == new_output_state_bit) {
            start = middle;
        } else {
            end = middle;
        }
        my_printf_debug("offset of start = %" PRId64, start - baseptr);
        my_printf_debug(", offset of end = %" PRId64 NEWLINE, end - baseptr);
    }

#ifdef WITH_PROGRESS_EMBEDDING
    if (!after_recovery) {
        MY_ASSERT(first_unfinished_value_offset == 0);
    } else {
        after_recovery = 0;
    }
#endif

    my_printf_debug("first_unfinished_value_offset = %d" NEWLINE, first_unfinished_value_offset);

#ifndef MY_NDEBUG
    for (uint32_t idx = 0; idx < first_unfinished_value_offset; idx++) {
        MY_ASSERT(get_value_state_bit(baseptr[idx]) == new_output_state_bit);
    }
    for (uint32_t idx = first_unfinished_value_offset; idx < output->params_len / 2; idx++) {
        MY_ASSERT(get_value_state_bit(baseptr[idx]) != new_output_state_bit);
    }
#endif

    return first_unfinished_value_offset;
#else
    UNUSED(model);
    UNUSED(output);
    return 0;
#endif
}
