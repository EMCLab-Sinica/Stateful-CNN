#include "common.h"

Model *model;
Node *nodes;
ParameterInfo *parameter_info;
uint16_t *inputs;
uint16_t *parameters;
uint8_t *labels;

static int16_t* node_input_ptr(Node *node, size_t i) {
    return (int16_t*)((uint8_t*)inputs + node->inputs_offset) + i;
}

int16_t node_input(Node *node, size_t i) {
    return *node_input_ptr(node, i) / 2;
}

void node_input_mark(Node *node, size_t i) {
    int16_t *ptr = node_input_ptr(node, i);
    *ptr |= 1;
}

void node_input_unmark_all(Node *node) {
    for (uint16_t i = 0; i < node->inputs_len; i++) {
        int16_t *ptr = node_input_ptr(node, i);
        *ptr &= ~1;
    }
}

uint8_t node_input_marked(Node *node, size_t i) {
    int16_t *ptr = node_input_ptr(node, i);
    return *ptr & 0x1;
}

int32_t* get_iq31_param(ParameterInfo *param, size_t i) {
    if (param->bitwidth != 32) {
        // incorrect param passed
        ERROR_OCCURRED();
    }
    return (int32_t*)(get_param_base_pointer(param) + param->params_offset) + i;
}

int64_t get_int64_param(ParameterInfo *param, size_t i) {
    return *((int64_t*)((uint8_t*)parameters + param->params_offset) + i);
}
