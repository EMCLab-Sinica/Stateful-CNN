#include "cnn_common.h"

uint8_t *inputs_data;

static int16_t* node_input_ptr(Node *node, size_t i) {
    return (int16_t*)(inputs_data + node->inputs_offset) + i;
}

int16_t node_input(Node *node, size_t i) {
    return *node_input_ptr(node, i) / 2;
}

static void get_param_base_pointer(ParameterInfo *param, uint8_t **baseptr_p, uint32_t *limit_p) {
    uint16_t slot_id = param->slot;
    switch (slot_id) {
        case SLOT_PARAMETERS:
            *baseptr_p = parameters_data;
            *limit_p = PARAMETERS_DATA_LEN;
            break;
        case SLOT_TEST_SET:
            *baseptr_p = samples_data;
            *limit_p = SAMPLES_DATA_LEN;
            break;
        case SLOT_INTERMEDIATE_VALUES:
            *baseptr_p = intermediate_values();
            *limit_p = INTERMEDIATE_VALUES_SIZE;
            break;
        default:
            ERROR_OCCURRED();
    }
}

int16_t* get_q15_param(ParameterInfo *param, size_t i) {
    if (param->bitwidth != 16) {
        // incorrect param passed
        ERROR_OCCURRED();
    }
    uint8_t *baseptr;
    uint32_t limit;
    get_param_base_pointer(param, &baseptr, &limit);
    int16_t *ret = (int16_t*)(baseptr + param->params_offset) + i;
    if ((uint8_t*)ret >= baseptr + limit) {
        ERROR_OCCURRED();
    }
    return ret;
}

int32_t* get_iq31_param(ParameterInfo *param, size_t i) {
    if (param->bitwidth != 32) {
        // incorrect param passed
        ERROR_OCCURRED();
    }
    uint8_t *baseptr;
    uint32_t limit;
    get_param_base_pointer(param, &baseptr, &limit);
    int32_t *ret = (int32_t*)(baseptr + param->params_offset) + i;
    if ((uint8_t*)ret >= baseptr + limit) {
        ERROR_OCCURRED();
    }
    return ret;
}

int64_t get_int64_param(ParameterInfo *param, size_t i) {
    return *((int64_t*)(parameters_data + param->params_offset) + i);
}
