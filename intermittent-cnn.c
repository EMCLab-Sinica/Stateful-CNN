#include <stdint.h>
#include <string.h>

#include "intermittent-cnn.h"
#include "op_handlers.h"
#include "cnn_common.h"
#include "data.h"
#include "ops.h"
#include "debug.h"

static void handle_node(Model *model, Node *nodes, ParameterInfo* parameter_info, uint16_t node_idx) {
    my_printf_debug("Current node: %d ", node_idx);

    /* schedule it */
    Node *cur_node = &(nodes[node_idx]);
    my_printf_debug("op_type = %d" NEWLINE, cur_node->op_type);

    int16_t input_id[3];
    ParameterInfo *input[3];
    if (cur_node->inputs_len != expected_inputs_len[cur_node->op_type]) {
        // unexpected input length
        ERROR_OCCURRED();
    }
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
    uint32_t needed_mem = allocators[cur_node->op_type](input, output, cur_node->flags);
    if (!needed_mem && !inplace_update[cur_node->op_type]) {
        needed_mem = output->params_len;
    }
    my_printf_debug("Needed mem = %d" NEWLINE, needed_mem);
    if (needed_mem) {
        for (int16_t prev_node_idx = node_idx - 1; prev_node_idx >= 0; prev_node_idx--) {
            if (!inplace_update[nodes[prev_node_idx].op_type]) {
                ParameterInfo *prev_node = &(parameter_info[prev_node_idx + model->n_input]);
                if (prev_node->slot != SLOT_INTERMEDIATE_VALUES) {
                    continue;
                }
                output->params_offset = prev_node->params_offset + prev_node->params_len;
                if (output->params_offset + needed_mem >= INTERMEDIATE_VALUES_SIZE) {
                    // reuse the ring buffer
                    // TODO: check if this is OK
                    output->params_offset = 0;
                }
                break;
            }
        }
    }
    if (output->slot == SLOT_INTERMEDIATE_VALUES) {
        my_printf_debug("New params_offset = %d" NEWLINE, output->params_offset);
    }

    my_printf_debug("State bit=%d" NEWLINE, model->state_bit);
    handlers[cur_node->op_type](model, input, output, cur_node->flags);
    my_printf_debug("State bit=%d" NEWLINE, model->state_bit);

    counters()->counter_idx++;
    if (counters()->counter_idx >= COUNTERS_LEN) {
        ERROR_OCCURRED();
    }

    my_printf_debug("output dims: ");
    uint8_t has_dims = 0;
    for (uint8_t j = 0; j < 4; j++) {
        if (output->dims[j]) {
            has_dims = 1;
            my_printf_debug("%d, ", output->dims[j]);
        }
    }
    my_printf_debug(NEWLINE);
    if (!has_dims) {
        // missing dims
        ERROR_OCCURRED();
    }
    if (output->bitwidth == 0) {
        // invalid bitwidth
        ERROR_OCCURRED();
    }

    if (node_idx == model->nodes_len - 1) {
        model->running = 0;
        model->run_counter++;
    }
    cur_node->scheduled = 1;
}

int run_model(Model *model, int8_t *ansptr, ParameterInfo **output_node_ptr) {
    Node *nodes = (Node*)(model + 1);
    ParameterInfo *parameter_info = (ParameterInfo*)(nodes + model->nodes_len);
    inputs_data = (uint8_t*)(parameter_info + model->nodes_len + model->n_input);

    my_printf_debug("model->n_input = %d" NEWLINE, model->n_input);

    if (!model->running) {
        // reset model
        for (uint16_t i = 0; i < model->nodes_len; i++) {
            Node *cur_node = &(nodes[i]);
            cur_node->scheduled = 0;
        }
        counters()->counter_idx = 0;
        model->running = 1;
        model->state_bit = 0;
    } else {
        model->recovery = 1;
    }

    counters()->power_counters[counters()->counter_idx]++;

    dump_model(model, nodes);

    for (uint16_t node_idx = 0; node_idx < model->nodes_len; node_idx++) {
        Node *cur_node = &(nodes[node_idx]);

        if (cur_node->scheduled) {
            continue;
        }

        handle_node(model, nodes, parameter_info, node_idx);

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
#ifndef MY_NDEBUG
    my_printf(NEWLINE "DMA invocations:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 8d", counters()->dma_invocations[i]);
    }
    my_printf(NEWLINE "DMA bytes:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 8d", counters()->dma_bytes[i]);
    }
#endif
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
