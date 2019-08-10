#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <DSPLib.h>

#include "ops.h"

#define NEWLINE "\n"

const uint16_t FLAG_INTERMEDIATE_VALUES = 1;

typedef struct {
    uint16_t inputs_len;
    uint16_t inputs_offset;
    uint16_t op_type;
    uint16_t scheduled;  /* 16 bits for aligned memory */
} Node;

/* ParameterInfo may indicate data from the model (parameters) or intermediate values */
typedef struct __attribute__((__packed__)) {
    uint32_t params_offset;
    uint32_t params_len;  /* in bytes */
    /* Known bitwidth values:
     * 16: q15
     * 32: iq31
     * 64: INT64 (from ONNX)
     *
     * The least sigfinicant bit is a flag to indicate where the data are - parameters or intermediate_values
     */
    uint16_t bitwidth_and_flags;
    uint16_t dims[4];
} ParameterInfo;

typedef struct __attribute__((__packed__)) {
    uint16_t nodes_len;
    uint16_t n_input;
    Node *nodes;
    ParameterInfo *parameter_info;
} Model;

static Model *model;
static uint16_t *inputs;
static uint16_t *parameters;

static uint16_t cur_group[16] = { 0 };
static uint8_t grp_index = 0;
static uint16_t group_last_item;

/* on FRAM */
static uint8_t intermediate_values[65536] = { 0 };

static inline int16_t* node_input_ptr(Node *node, size_t i) {
    return (int16_t*)((uint8_t*)inputs + node->inputs_offset) + i;
}

static inline int16_t node_input(Node *node, size_t i) {
    return *node_input_ptr(node, i) / 2;
}

static inline void node_input_mark(Node *node, size_t i) {
    int16_t *ptr = node_input_ptr(node, i);
    *ptr |= 1;
}

static inline uint8_t node_input_marked(Node *node, size_t i) {
    int16_t *ptr = node_input_ptr(node, i);
    return *ptr & 0x1;
}

static inline uint8_t* get_param_base_pointer(ParameterInfo *param) {
    if (param->bitwidth_and_flags & FLAG_INTERMEDIATE_VALUES) {
        return &(intermediate_values[0]);
    } else {
        return (uint8_t*)parameters;
    }
}

static inline int16_t* get_q15_param(ParameterInfo *param, size_t i) {
    if ((param->bitwidth_and_flags >> 1) != 16) {
        printf("Error: incorrect param passed to %s" NEWLINE, __func__);
        return NULL;
    }
    return (int16_t*)(get_param_base_pointer(param) + param->params_offset) + i;
}

static inline int32_t* get_iq31_param(ParameterInfo *param, size_t i) {
    if ((param->bitwidth_and_flags >> 1) != 32) {
        printf("Error: incorrect param passed to %s" NEWLINE, __func__);
        return NULL;
    }
    return (int32_t*)(get_param_base_pointer(param) + param->params_offset) + i;
}

static inline int64_t get_int64_param(ParameterInfo *param, size_t i) {
    return *((int64_t*)((uint8_t*)parameters + param->params_offset) + i);
}

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

static void handle_conv(ParameterInfo *conv_input, ParameterInfo *conv_filter, ParameterInfo *output) {
    uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
             kH = conv_filter->dims[2], kW = conv_filter->dims[3],
             C = conv_filter->dims[1];
    /* MSP430 LEA requires length to be even */
    msp_mac_q15_params params = { .length = (uint16_t)(kH * kW / 2 * 2) };
    uint8_t truncated = (params.length != kH * kW);
    uint16_t buffer_size = (uint16_t)(sizeof(uint16_t) * params.length);
    int16_t *lea_buffer_input = malloc(buffer_size),
            *lea_buffer_filter = malloc(buffer_size);
    int32_t *output_data = get_iq31_param(output, 0);
    for (uint16_t conv_idx = 0; conv_idx < conv_filter->dims[0]; conv_idx++) {
        for (uint16_t channel = 0; channel < C; channel++) {
            /* copy filter data */
            memcpy(lea_buffer_filter,
                   get_q15_param(conv_filter, (size_t)((conv_idx * C + channel) * params.length)),
                   buffer_size);
            for (uint16_t output_h = 0; output_h < H; output_h++) {
                for (uint16_t output_w = 0; output_w < W; output_w++) {
                    /* copy input data, row by row */
                    for (uint16_t h = 0; h < kH; h++) {
                        size_t size = kW;
                        if (truncated && h == kH - 1) {
                            size--;
                        }
                        memcpy(lea_buffer_input + h * kW,  /* dest */
                               get_q15_param(conv_input, (size_t)(output_h * W + output_w)),  /* src */
                               size * sizeof(uint16_t));  /* size */
                    }
                    int32_t mac_result;
                    msp_status status = msp_mac_q15(&params, lea_buffer_input, lea_buffer_filter, &mac_result);
                    msp_checkStatus(status);
                    if (truncated) {
#ifndef NDEBUG
                        // printf("Adding truncated product back" NEWLINE);
#endif
                        uint16_t last_idx = (uint16_t)(kH * kW - 1);
                        mac_result += (*get_q15_param(conv_input, last_idx)) * (*get_q15_param(conv_filter, last_idx)) * 2;
                    }
#ifndef NDEBUG
                    printf("%f ", (float)mac_result / 2147483648.0f);
#endif
                    output_data[conv_idx * H * W + output_h * W + output_w] = mac_result;
                }
            }
#ifndef NDEBUG
            printf(NEWLINE);
#endif
        }
    }

    free(lea_buffer_input);
    free(lea_buffer_filter);
}

static void handle_maxpool(Node *cur_node) {
    printf("MaxPool!" NEWLINE);
    /* TODO */
    (void)cur_node;
}

static void handle_cur_group(void) {
    uint16_t intermediate_values_offset = 0;

    printf("Current group: ");
    for (uint8_t i = 0; i < grp_index; i++) {
        printf("%d ", cur_group[i]);
        /* schedule it */
        Node *cur_node = &(model->nodes[cur_group[i]]);
#ifndef NDEBUG
        printf("op_type = %d" NEWLINE, cur_node->op_type);
#endif
        if (cur_node->op_type == Reshape) {
            cur_node->scheduled = 1;
            continue;
        }
        if (cur_node->op_type == Conv) {
            printf("Conv!" NEWLINE);
            if (cur_node->inputs_len != 2) {
                printf("Error!" NEWLINE);
                return;
            }
            int16_t conv_input_id = node_input(cur_node, 0),
                    conv_filter_id = node_input(cur_node, 1);
#ifndef NDEBUG
            printf("conv_input_id = %d, conv_filter_id = %d" NEWLINE, conv_input_id, conv_filter_id);
#endif
            if (conv_filter_id >= model->n_input) {
                printf("Not implemented: intermediate data" NEWLINE);
                cur_node->scheduled = 1;
                return;
            }
            /* input: N x C x H x W, filter: M x C x kH x kW */
            ParameterInfo *conv_input = &(model->parameter_info[conv_input_id]),
                          *conv_filter = &(model->parameter_info[conv_filter_id]);
            dump_params(conv_input);
            dump_params(conv_filter);
            uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                     output_C = conv_filter->dims[0]; // output_C = input_N
            /* TODO: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
            ParameterInfo *output = (ParameterInfo*)(intermediate_values + intermediate_values_offset);
            output->params_len = (uint16_t)(output_C * H * W * 4); /* 4 bytes as IQ31 values are stored */
            output->params_offset = intermediate_values_offset;
            output->bitwidth_and_flags = 32 << 1 | FLAG_INTERMEDIATE_VALUES;
            uint32_t new_intermediate_values_offset = (uint32_t)(
                /* use uint32_t here to avoid overflow */
                intermediate_values_offset + sizeof(ParameterInfo) + output->params_len
            );
            if (new_intermediate_values_offset >= 65536) {
                /* TODO: reuse the ring buffer */
                printf("Error: too many immediate values" NEWLINE);
            }
            handle_conv(conv_input, conv_filter, output);
            intermediate_values_offset = (uint16_t)new_intermediate_values_offset;
        }
        if (cur_node->op_type == MaxPool) {
            handle_maxpool(cur_node);
        }
        cur_node->scheduled = 1;
    }
    printf(" - %d element(s)." NEWLINE, grp_index);
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
            printf("Error!" NEWLINE);
            break;
        }

        if (grp_index < 16) {
            next_node_idx = 0;
        }

        handle_cur_group();

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
                if (node_input(cur_node, j) > group_last_item + model->n_input) {
                    break;
                }
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
