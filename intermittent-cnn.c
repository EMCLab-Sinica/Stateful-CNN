#include <stdint.h>
#include <string.h>
#include "platform.h" // for WITH_FREERTOS
#ifdef WITH_FREERTOS
#include <FreeRTOS.h>
#include <task.h>
#endif

#ifdef WITH_FAILURE_RESILIENT_OS
#include "SharedDB.h"
#endif

#include "intermittent-cnn.h"
#include "op_handlers.h"
#include "cnn_common.h"
#include "data.h"
#include "ops.h"
#include "debug.h"

typedef struct {
    Model *model;
    Node *nodes;
    ParameterInfo* parameter_info;
    uint16_t *cur_group;
    uint8_t grp_index;
} handle_cur_group_params;

static void handle_cur_group(void *pvParameters) {
    handle_cur_group_params *params = (handle_cur_group_params*)pvParameters;
    Model *model = params->model;
    ParameterInfo *parameter_info = params->parameter_info;

    uint16_t intermediate_values_offset = 0;

    my_printf_debug("Current group: ");

    for (uint8_t i = 0; i < params->grp_index; i++) {
        uint16_t cur_node_id = params->cur_group[i];
        my_printf_debug("%d ", cur_node_id);

        /* schedule it */
        Node *cur_node = &(params->nodes[cur_node_id]);
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
            if (input[j]->slot == FLAG_TEST_SET) {
                input[j]->params_offset = (model->sample_idx % LABELS_DATA_LEN) * input[j]->params_len;
            }
            // dump_params(input[j]);
        }

        /* Allocate an ParameterInfo for output. Details are filled by
         * individual operation handlers */
        ParameterInfo *output = &(parameter_info[cur_node_id + model->n_input]);
        output->params_offset = intermediate_values_offset;

        uint32_t new_intermediate_values_offset = (uint32_t)(
            /* use uint32_t here to avoid overflow */
            intermediate_values_offset + output->params_len
        );
        if (new_intermediate_values_offset >= INTERMEDIATE_VALUES_SIZE) {
            /* TODO: reuse the ring buffer */
            // too many immediate values
            ERROR_OCCURRED();
        }

        handlers[cur_node->op_type](input, output, cur_node->flags);

        counters()->counter_idx++;
        if (counters()->counter_idx >= COUNTERS_LEN) {
            ERROR_OCCURRED();
        }

        intermediate_values_offset = (uint16_t)new_intermediate_values_offset;

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

        if (cur_node_id == model->nodes_len - 1) {
            model->running = 0;
            model->run_counter++;
        }
        cur_node->scheduled = 1;
    }
    my_printf_debug(" - %d element(s)." NEWLINE, params->grp_index);
}

int run_model(Model *model, int8_t *ansptr, ParameterInfo **output_node_ptr) {
    uint16_t cur_group[16] = { 0 };
    uint8_t grp_index = 0;

#ifdef WITH_FAILURE_RESILIENT_OS
    if (!model->n_input) {
        memcpy(model, model_data, MODEL_DATA_LEN);
    }
#endif

    Node *nodes = (Node*)(model + 1);
    ParameterInfo *parameter_info = (ParameterInfo*)(nodes + model->nodes_len);
    inputs_data = (uint8_t*)(parameter_info + model->nodes_len + model->n_input);

    my_printf_debug("model->n_input = %d" NEWLINE, model->n_input);

    if (!model->running) {
        // reset model
        for (uint16_t i = 0; i < model->nodes_len; i++) {
            Node *cur_node = &(nodes[i]);
            node_input_unmark_all(cur_node);
            cur_node->scheduled = 0;
        }
        counters()->counter_idx = 0;
        model->running = 1;
    }

    counters()->power_counters[counters()->counter_idx]++;

    dump_model(model, nodes);

    uint16_t next_node_idx = 0;
    while (next_node_idx < model->nodes_len) {
        for (uint16_t i = next_node_idx; i < model->nodes_len; i++) {
            Node *cur_node = &(nodes[i]);
            uint8_t no_inputs = 1;

            if (cur_node->scheduled) {
                continue;
            }

            for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
                if (node_input(cur_node, j) >= model->n_input && !node_input_marked(cur_node, j)) {
                    no_inputs = 0;
                }
            }
            if (no_inputs) {
                my_printf_debug("Node %d has no inputs." NEWLINE, i);
                cur_group[grp_index] = i;
                grp_index++;
                /* https://stackoverflow.com/a/47417220 */
                next_node_idx = (uint16_t)(i + 1);
                if (grp_index == 16) {
                    break;
                }
            }
        }

        if (!grp_index) {
            // unable to establish a group
            ERROR_OCCURRED();
        }

        if (grp_index < 16) {
            next_node_idx = 0;
        }

        handle_cur_group_params params;
        params.model = model;
        params.nodes = nodes;
        params.parameter_info = parameter_info;
        params.cur_group = cur_group;
        params.grp_index = grp_index;
        handle_cur_group(&params);

        if (cur_group[grp_index - 1] == model->nodes_len - 1) {
            break;
        }

        /**
         * topological sort: remove handled (scheduled) dependent nodes
         */
        for (uint16_t i = cur_group[0]; i < model->nodes_len; i++) {
            Node *cur_node = &(nodes[i]);
            for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
                for (uint8_t k = 0; k < grp_index; k++) {
                    if (node_input(cur_node, j) == cur_group[k] + model->n_input) {
                        node_input_mark(cur_node, j);
                    }
                }
            }
        }

        grp_index = 0;
        memset(cur_group, 0, sizeof(cur_group));

        dump_model(model, nodes);

#ifdef WITH_FAILURE_RESILIENT_OS
        int objId = OBJ_CNN_MODEL;
        commit(DB, IDCNN, &objId, 1, MODEL_DATA_LEN, 0, 0, MODEL_DATA_LEN);
#endif
    }

    /* XXX: is the last node always the output node? */
    ParameterInfo *output_node = &(parameter_info[model->nodes_len + model->n_input - 1]);
    if (output_node_ptr) {
        *output_node_ptr = output_node;
    }
    int16_t max = INT16_MIN;
    for (uint16_t i = 0; i < output_node->dims[1]; i++) {
        int16_t val = *get_q15_param(output_node, i, WILL_NOT_WRITE);
        if (val > max) {
            *ansptr = (uint8_t)i;
            max = val;
        }
    }

    return 0;
}

void print_results(Model *model, ParameterInfo *output_node) {
    for (uint16_t i = 0; i < output_node->dims[1]; i++) {
        print_q15(*get_q15_param(output_node, i, WILL_NOT_WRITE));
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
