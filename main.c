#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "common.h"
#include "ops.h"

static Model *model;
uint16_t *inputs;
uint16_t *parameters;

static uint16_t cur_group[16] = { 0 };
static uint8_t grp_index = 0;
static uint16_t group_last_item;

/* on FRAM */
uint8_t intermediate_values[65536] = { 0 };

static void dump_model(void) {
#ifndef NDEBUG
    uint16_t i, j;
    for (i = 0; i < model->nodes_len; i++) {
        Node *cur_node = &(model->nodes[i]);
        printf("(");
        for (j = 0; j < cur_node->inputs_len; j++) {
            printf("%d", node_input(cur_node, j));
            if (j != cur_node->inputs_len - 1) {
                printf(", ");
            }
        }
        printf(") ");
    }
    printf(NEWLINE);
#endif
}

static int map_files(void) {
    int fd_model = -1, fd_inputs = -1, fd_parameters = -1;
    int ret = 0;

    fd_model = open("model.bin", O_RDONLY);
    fd_inputs = open("inputs.bin", O_RDONLY);
    fd_parameters = open("parameters.bin", O_RDONLY);
    if (fd_model == -1 || fd_inputs == -1 || fd_parameters == -1) {
        perror("open files failed");
        ret = -1;
        goto error;
    }

    long val = sysconf(_SC_PAGESIZE);
    if (val < 0) {
        perror("Failed to get page size");
        goto error;
    }
    size_t map_size = 4 * (size_t)val;
    model = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd_model, 0);
    inputs = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd_inputs, 0);
    parameters = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd_parameters, 0);

    if (model == MAP_FAILED || inputs == MAP_FAILED) {
        perror("mmap failed");
        ret = -1;
    }

error:
    if (fd_model != -1) {
        close(fd_model);
    }
    if (fd_inputs != -1) {
        close(fd_inputs);
    }
    if (fd_parameters != -1) {
        close(fd_parameters);
    }

    return ret;
}

static void dump_params(ParameterInfo *cur_param) {
#ifndef NDEBUG
    printf("offset=%d len=%d" NEWLINE, cur_param->params_offset, cur_param->params_len);
    uint16_t bitwidth = cur_param->bitwidth_and_flags >> 1;
    for (uint32_t k = 0; k < cur_param->params_len / (bitwidth / 8); k++) {
        if (bitwidth == 16) {
            printf("%f ", *get_q15_param(cur_param, k) / 32768.0f);
        } else if (bitwidth == 32) {
            printf("%f ", (float)(*get_iq31_param(cur_param, k)) / 2147483648.0f);
        } else if (bitwidth == 64) {
            printf("%ld ", get_int64_param(cur_param, k));
        }
    }
    printf(NEWLINE);
#else
    (void)cur_param;
#endif
}

static uint8_t handle_cur_group(void) {
    uint16_t intermediate_values_offset = 0;

    printf("Current group: ");
    for (uint8_t i = 0; i < grp_index; i++) {
        uint16_t cur_node_id = cur_group[i];
        printf("%d ", cur_node_id);
        /* schedule it */
        Node *cur_node = &(model->nodes[cur_node_id]);
#ifndef NDEBUG
        printf("op_type = %d" NEWLINE, cur_node->op_type);
#endif
        int16_t input_id[3];
        ParameterInfo *input[3];
        if (cur_node->inputs_len != expected_inputs_len[cur_node->op_type]) {
            printf("Error: unexpected input length." NEWLINE);
            return 1;
        }
        for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
            input_id[j] = node_input(cur_node, j);
#ifndef NDEBUG
            printf("input_id[%d] = %d ", j, input_id[j]);
#endif
            input[j] = &(model->parameter_info[input_id[j]]);
            dump_params(input[j]);
        }

        /* Allocate an ParameterInfo for output. Details are filled by
         * individual operation handlers */
        ParameterInfo *output = &(model->parameter_info[cur_node_id + model->n_input]);
        output->params_offset = intermediate_values_offset;

        uint32_t new_intermediate_values_offset = (uint32_t)(
            /* use uint32_t here to avoid overflow */
            intermediate_values_offset + output->params_len
        );
        if (new_intermediate_values_offset >= 65536) {
            /* TODO: reuse the ring buffer */
            printf("Error: too many immediate values" NEWLINE);
        }
        if (handlers[cur_node->op_type](input, output) != 0) {
            return 1;
        }
        intermediate_values_offset = (uint16_t)new_intermediate_values_offset;

#ifndef NDEBUG
        printf("output dims: ");
#endif
        uint8_t has_dims = 0;
        for (uint8_t j = 0; j < 4; j++) {
            if (output->dims[j]) {
                has_dims = 1;
#ifndef NDEBUG
                printf("%d, ", output->dims[j]);
#endif
            }
        }
#ifndef NDEBUG
        printf(NEWLINE);
#endif
        if (!has_dims) {
            printf("Error: missing dims." NEWLINE);
            return 1;
        }
        if (output->bitwidth_and_flags >> 1 == 0) {
            printf("Error: invalid bitwidth." NEWLINE);
            return 1;
        }

        cur_node->scheduled = 1;
    }
    printf(" - %d element(s)." NEWLINE, grp_index);
    return 0;
}

int main (void) {
    if (map_files() != 0) {
        return 1;
    }

    model->nodes = (Node*)(model + 1);
    model->parameter_info = (ParameterInfo*)(model->nodes + model->nodes_len);

    printf("model->n_input = %d" NEWLINE, model->n_input);

    /* initialize - the first node must have no inputs as
     * ONNX already sort nodes topologically */
    cur_group[0] = 0;
    grp_index = 1;

    dump_model();

    uint16_t next_node_idx = 1;
    while (next_node_idx < model->nodes_len) {
        for (uint16_t i = next_node_idx; i < model->nodes_len; i++) {
            Node *cur_node = &(model->nodes[i]);
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
                printf("Node %d has no inputs." NEWLINE, i);
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
            printf("Error: unable to establish a group." NEWLINE);
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
            Node *cur_node = &(model->nodes[i]);
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
