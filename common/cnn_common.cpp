#include "cnn_common.h"
#include "debug.h"

uint8_t *inputs_data;

static int16_t* node_input_ptr(Node *node, size_t i) {
    return (int16_t*)(inputs_data + node->inputs_offset) + i;
}

ParameterInfo* get_parameter_info(size_t i) {
    return reinterpret_cast<ParameterInfo*>(parameters_info_data) + i;
}

SlotInfo* get_slot_info(uint8_t i) {
    if (i < NUM_SLOTS) {
        return reinterpret_cast<SlotInfo*>(slots_info_data) + i;
    } else if (i >= SLOT_CONSTANTS_MIN) {
        return nullptr;
    } else {
        ERROR_OCCURRED();
    }
}

int16_t node_input(Node *node, size_t i) {
    return *node_input_ptr(node, i) / 2;
}

const uint8_t* get_param_base_pointer(ParameterInfo *param, uint32_t *limit_p) {
    uint16_t slot_id = param->slot;
    switch (slot_id) {
        case SLOT_PARAMETERS:
            *limit_p = PARAMETERS_DATA_LEN;
            return parameters_data;
        case SLOT_PARAMETERS2:
            *limit_p = PARAMETERS2_DATA_LEN;
            return parameters2_data;
        case SLOT_TEST_SET:
            *limit_p = PLAT_SAMPLES_DATA_LEN;
            return samples_data;
        default:
            ERROR_OCCURRED();
    }
}

int16_t get_q15_param(ParameterInfo *param, uint16_t i) {
    MY_ASSERT(param->bitwidth == 16);
    if (param->slot >= SLOT_CONSTANTS_MIN) {
        uint32_t limit;
        const uint8_t *baseptr = get_param_base_pointer(param, &limit);
        const int16_t *ret = (int16_t*)(baseptr + param->params_offset) + i;
        MY_ASSERT(param->params_offset + i * sizeof(int16_t) < limit);
        return *ret;
    } else {
        int16_t ret;
        my_memcpy_from_param(&ret, param, i, sizeof(int16_t));
        return ret;
    }
}

void put_q15_param(ParameterInfo *param, uint16_t i, int16_t val) {
    my_memcpy_to_param(param, i, &val, sizeof(int16_t));
}

int64_t get_int64_param(ParameterInfo *param, size_t i) {
    MY_ASSERT(param->bitwidth == 64);
    uint32_t limit;
    const uint8_t *baseptr = get_param_base_pointer(param, &limit);
    int64_t *ret = (int64_t*)(baseptr + param->params_offset) + i;
    MY_ASSERT((uint8_t*)ret < baseptr + limit);
    return *ret;
}

uint16_t get_next_slot(Model *model, ParameterInfo *param) {
    Node *nodes = (Node*)(model + 1);
    uint16_t slot_id = param->slot;
    /* pick the next unused slot */
    uint16_t next_slot_id = slot_id;
    uint8_t cycle_count = 0;
    while (1) {
        next_slot_id++;
        // Fail if the loop has run a cycle
        if (next_slot_id >= NUM_SLOTS) {
            next_slot_id = 0;
            cycle_count++;
            MY_ASSERT(cycle_count <= 1);
        }
        int16_t slot_user_id = get_slot_info(next_slot_id)->user;
        if (slot_user_id < 0) {
            break;
        }
        // previously allocated, most likely in a previous power cycle
        if (slot_user_id == model->layer_idx) {
            break;
        }
        Node *slot_user = &(nodes[slot_user_id]);
        if ((slot_user->max_output_id & MAX_OUTPUT_ID_INVALID) || (slot_user->max_output_id < model->layer_idx)) {
            break;
        }
    }
    my_printf_debug("next_slot_id = %d" NEWLINE, next_slot_id);
    get_slot_info(next_slot_id)->user = model->layer_idx;
    return next_slot_id;
}
