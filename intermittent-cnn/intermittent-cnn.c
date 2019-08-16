#include <stdint.h>
#include <string.h>

#include "intermittent-cnn.h"
#include "common.h"
#include "data.h"
#include "ops.h"

static uint16_t cur_group[16] = { 0 };
static uint8_t grp_index = 0;
static uint16_t group_last_item;

static void dump_model(void) {
#ifndef NDEBUG
    uint16_t i, j;
    for (i = 0; i < model->nodes_len; i++) {
        Node *cur_node = &(nodes[i]);
        my_printf("(");
        for (j = 0; j < cur_node->inputs_len; j++) {
            my_printf("%d", node_input(cur_node, j));
            if (j != cur_node->inputs_len - 1) {
                my_printf(", ");
            }
        }
        my_printf(") ");
    }
    my_printf(NEWLINE);
#endif
}

static void dump_params(ParameterInfo *cur_param) {
#ifndef NDEBUG
    my_printf("offset=%d len=%d" NEWLINE, cur_param->params_offset, cur_param->params_len);
    uint16_t bitwidth = cur_param->bitwidth_and_flags >> 1;
    for (uint32_t k = 0; k < cur_param->params_len / (bitwidth / 8); k++) {
        if (k && (k % 16 == 0)) {
            my_printf(NEWLINE);
        }
        if (bitwidth == 16) {
            my_printf("%d ", *get_q15_param(cur_param, k));
        } else if (bitwidth == 32) {
            my_printf("%d ", *get_iq31_param(cur_param, k));
        } else if (bitwidth == 64) {
            my_printf("%ld ", get_int64_param(cur_param, k));
        }
    }
    my_printf(NEWLINE);
#else
    (void)cur_param;
#endif
}

static uint8_t handle_cur_group(void) {
    uint16_t intermediate_values_offset = 0;

    my_printf("Current group: ");
    for (uint8_t i = 0; i < grp_index; i++) {
        uint16_t cur_node_id = cur_group[i];
        my_printf("%d ", cur_node_id);
        /* schedule it */
        Node *cur_node = &(nodes[cur_node_id]);
#ifndef NDEBUG
        my_printf("op_type = %d" NEWLINE, cur_node->op_type);
#endif
        int16_t input_id[3];
        ParameterInfo *input[3];
        if (cur_node->inputs_len != expected_inputs_len[cur_node->op_type]) {
            my_printf("Error: unexpected input length." NEWLINE);
            return 1;
        }
        for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
            input_id[j] = node_input(cur_node, j);
#ifndef NDEBUG
            my_printf("input_id[%d] = %d ", j, input_id[j]);
#endif
            input[j] = &(parameter_info[input_id[j]]);
            dump_params(input[j]);
        }

        /* Allocate an ParameterInfo for output. Details are filled by
         * individual operation handlers */
        ParameterInfo *output = &(parameter_info[cur_node_id + model->n_input]);
        output->params_offset = intermediate_values_offset;

        uint32_t new_intermediate_values_offset = (uint32_t)(
            /* use uint32_t here to avoid overflow */
            intermediate_values_offset + output->params_len
        );
        if (new_intermediate_values_offset >= 65536) {
            /* TODO: reuse the ring buffer */
            my_printf("Error: too many immediate values" NEWLINE);
        }
        if (handlers[cur_node->op_type](input, output) != 0) {
            return 1;
        }
        intermediate_values_offset = (uint16_t)new_intermediate_values_offset;

#ifndef NDEBUG
        my_printf("output dims: ");
#endif
        uint8_t has_dims = 0;
        for (uint8_t j = 0; j < 4; j++) {
            if (output->dims[j]) {
                has_dims = 1;
#ifndef NDEBUG
                my_printf("%d, ", output->dims[j]);
#endif
            }
        }
#ifndef NDEBUG
        my_printf(NEWLINE);
#endif
        if (!has_dims) {
            my_printf("Error: missing dims." NEWLINE);
            return 1;
        }
        if (output->bitwidth_and_flags >> 1 == 0) {
            my_printf("Error: invalid bitwidth." NEWLINE);
            return 1;
        }

        cur_node->scheduled = 1;
    }
    my_printf(" - %d element(s)." NEWLINE, grp_index);
    return 0;
}

int run_model(void) {
    model = (Model*)model_data;
    inputs = (uint16_t*)inputs_data;
    parameters = (uint16_t*)parameters_data;

    memset(intermediate_values, 0, sizeof(intermediate_values));

    nodes = (Node*)(model + 1);
    parameter_info = (ParameterInfo*)(nodes + model->nodes_len);

    my_printf("model->n_input = %d" NEWLINE, model->n_input);

    /* initialize - the first node must have no inputs as
     * ONNX already sort nodes topologically */
    cur_group[0] = 0;
    grp_index = 1;

    dump_model();

    uint16_t next_node_idx = 1;
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
#ifndef NDEBUG
                my_printf("Node %d has no inputs." NEWLINE, i);
#endif
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
            my_printf("Error: unable to establish a group." NEWLINE);
            break;
        }

        if (grp_index < 16) {
            next_node_idx = 0;
        }

        if (handle_cur_group() != 0) {
            return 1;
        }

        group_last_item = cur_group[grp_index - 1];

        if (group_last_item == model->nodes_len - 1) {
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

        dump_model();
    }

    return 0;
}
