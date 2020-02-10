#include "common.h"

Model *model;
Node *nodes;
ParameterInfo *parameter_info;
uint16_t *inputs;
uint16_t *parameters;

/* on FRAM */
#ifdef __MSP430__
#pragma NOINIT(_intermediate_values)
static uint8_t _intermediate_values[NUM_SLOTS * INTERMEDIATE_VALUES_SIZE];
uint8_t *intermediate_values = _intermediate_values;
#pragma NOINIT(_task_flags)
static uint8_t _task_flags[TASK_FLAGS_SIZE];
uint8_t *task_flags = _task_flags;
#endif

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
    if (get_param_bitwidth(param) != 32) {
        my_printf("Error: incorrect param passed to %s" NEWLINE, __func__);
        return NULL;
    }
    return (int32_t*)(get_param_base_pointer(param) + param->params_offset) + i;
}

int64_t get_int64_param(ParameterInfo *param, size_t i) {
    return *((int64_t*)((uint8_t*)parameters + param->params_offset) + i);
}

#if !defined(MY_NDEBUG) && defined(DUMP_PARAMS)
// dump in NCHW format
void dump_params(ParameterInfo *cur_param) {
    uint16_t NUM, H, W, CHANNEL;
    if (cur_param->dims[2] && cur_param->dims[3]) {
        // tensor
        NUM = cur_param->dims[0];
        H = cur_param->dims[1];
        W = cur_param->dims[2];
        CHANNEL = cur_param->dims[3];
    } else {
        // matrix
        NUM = CHANNEL = 1;
        H = cur_param->dims[0];
        W = cur_param->dims[1];
    }
    uint16_t bitwidth = get_param_bitwidth(cur_param);
    for (uint16_t i = 0; i < NUM; i++) {
        for (uint16_t j = 0; j < CHANNEL; j++) {
            for (uint16_t k = 0; k < H; k++) {
                for (uint16_t l = 0; l < W; l++) {
                    // internal format is NHWC
                    size_t offset = (size_t)(i * H * W * CHANNEL + k * W * CHANNEL + l * CHANNEL + j);
                    if (bitwidth == 16) {
                        print_q15(*get_q15_param(cur_param, offset));
                    } else if (bitwidth == 32) {
                        print_iq31(*get_iq31_param(cur_param, offset));
                    } else if (bitwidth == 64) {
                        my_printf("%ld ", get_int64_param(cur_param, offset));
                    }
                }
                my_printf(NEWLINE);
            }
            my_printf(NEWLINE);
        }
        my_printf(NEWLINE);
    }
}
#endif
