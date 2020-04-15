#include "cnn_common.h"

static int16_t* node_input_ptr(Node *node, size_t i) {
    return (int16_t*)(inputs_data + node->inputs_offset) + i;
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
    return *((int64_t*)(parameters_data + param->params_offset) + i);
}
