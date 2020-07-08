#include "cnn_common.h"
#include "debug.h"

uint8_t *inputs_data;

static int16_t* node_input_ptr(Node *node, size_t i) {
    return (int16_t*)(inputs_data + node->inputs_offset) + i;
}

int16_t node_input(Node *node, size_t i) {
    return *node_input_ptr(node, i) / 2;
}

static uint8_t* get_param_base_pointer(ParameterInfo *param, uint32_t *limit_p) {
    uint16_t slot_id = param->slot;
    switch (slot_id) {
        case SLOT_PARAMETERS:
            *limit_p = PARAMETERS_DATA_LEN;
            return parameters_data;
        case SLOT_PARAMETERS2:
            *limit_p = PARAMETERS2_DATA_LEN;
            return parameters2_data;
        case SLOT_TEST_SET:
            *limit_p = SAMPLES_DATA_LEN;
            return samples_data;
        default:
            *limit_p = INTERMEDIATE_VALUES_SIZE;
            return intermediate_values(slot_id);
    }
}

int16_t* get_q15_param(ParameterInfo *param, size_t i) {
    MY_ASSERT(param->bitwidth == 16);
    uint32_t limit;
    uint8_t *baseptr = get_param_base_pointer(param, &limit);
    int16_t *ret = (int16_t*)(baseptr + param->params_offset) + i;
    MY_ASSERT((uint8_t*)ret < baseptr + limit);
    return ret;
}

int32_t* get_iq31_param(ParameterInfo *param, size_t i) {
    MY_ASSERT(param->bitwidth == 32);
    uint32_t limit;
    uint8_t *baseptr = get_param_base_pointer(param, &limit);
    int32_t *ret = (int32_t*)(baseptr + param->params_offset) + i;
    MY_ASSERT((uint8_t*)ret < baseptr + limit);
    return ret;
}

int64_t get_int64_param(ParameterInfo *param, size_t i) {
    MY_ASSERT(param->bitwidth == 64);
    uint32_t limit;
    uint8_t *baseptr = get_param_base_pointer(param, &limit);
    int64_t *ret = (int64_t*)(baseptr + param->params_offset) + i;
    MY_ASSERT((uint8_t*)ret < baseptr + limit);
    return *ret;
}

uint16_t get_next_slot(ParameterInfo *param) {
    uint16_t slot_id = param->slot;
    /* pick the next slot */
    uint16_t next_slot_id = slot_id + 1;
    if (next_slot_id >= NUM_SLOTS) {
        next_slot_id = 0;
    }
    my_printf_debug("next_slot_id = %d" NEWLINE, next_slot_id);
    return next_slot_id;
}
