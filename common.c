#include "common.h"

Model *model;
Node *nodes;
ParameterInfo *parameter_info;
uint16_t *inputs;
uint16_t *parameters;

/* on FRAM */
#ifdef __MSP430__
#pragma NOINIT(intermediate_values)
#endif
uint8_t intermediate_values[INTERMEDIATE_VALUES_SIZE];

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

uint8_t node_input_marked(Node *node, size_t i) {
    int16_t *ptr = node_input_ptr(node, i);
    return *ptr & 0x1;
}

int16_t iq31_to_q15(int32_t *iq31_val_ptr) {
    return *(int16_t*)iq31_val_ptr;
}

int32_t* get_iq31_param(ParameterInfo *param, size_t i) {
    if ((param->bitwidth_and_flags >> 1) != 32) {
        my_printf("Error: incorrect param passed to %s" NEWLINE, __func__);
        return NULL;
    }
    return (int32_t*)(get_param_base_pointer(param) + param->params_offset) + i;
}

int64_t get_int64_param(ParameterInfo *param, size_t i) {
    return *((int64_t*)((uint8_t*)parameters + param->params_offset) + i);
}

#if !defined(MY_NDEBUG) && defined(DUMP_PARAMS)
void dump_params(ParameterInfo *cur_param) {
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
}
#endif
