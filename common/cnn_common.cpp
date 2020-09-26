#include "cnn_common.h"
#include "my_debug.h"
#include "platform.h"

ParameterInfo* get_parameter_info(size_t i) {
    return reinterpret_cast<ParameterInfo*>(parameters_info_data) + i;
}

const Node* get_node(size_t i) {
    return reinterpret_cast<const Node*>(nodes_data) + i;
}

SlotInfo* get_slot_info(Model* model, uint8_t i) {
    if (i < NUM_SLOTS) {
        return model->slots_info + i;
    } else if (i >= SLOT_CONSTANTS_MIN) {
        return nullptr;
    } else {
        ERROR_OCCURRED();
    }
}

const uint8_t* get_param_base_pointer(const ParameterInfo *param, uint32_t *limit_p) {
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

int16_t get_q15_param(Model* model, const ParameterInfo *param, uint16_t i) {
    MY_ASSERT(param->bitwidth == 16);
    if (param->slot >= SLOT_CONSTANTS_MIN) {
        uint32_t limit;
        const uint8_t *baseptr = get_param_base_pointer(param, &limit);
        const int16_t *ret = (int16_t*)(baseptr + param->params_offset) + i;
        MY_ASSERT(param->params_offset + i * sizeof(int16_t) < limit);
        return *ret;
    } else {
        int16_t ret;
        my_memcpy_from_param(model, &ret, param, i, sizeof(int16_t));
        return ret;
    }
}

void put_q15_param(ParameterInfo *param, uint16_t i, int16_t val) {
    my_memcpy_to_param(param, i, &val, sizeof(int16_t));
}

int64_t get_int64_param(const ParameterInfo *param, size_t i) {
    MY_ASSERT(param->bitwidth == 64);
    uint32_t limit;
    const uint8_t *baseptr = get_param_base_pointer(param, &limit);
    int64_t *ret = (int64_t*)(baseptr + param->params_offset) + i;
    MY_ASSERT((uint8_t*)ret < baseptr + limit);
    return *ret;
}

uint16_t get_next_slot(Model *model, const ParameterInfo *param) {
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
        int16_t slot_user_id = get_slot_info(model, next_slot_id)->user;
        if (slot_user_id < 0) {
            break;
        }
        // previously allocated, most likely in a previous power cycle
        if (slot_user_id == model->layer_idx) {
            break;
        }
        const Node *slot_user = get_node(slot_user_id);
        if (slot_user->max_output_id < model->layer_idx) {
            break;
        }
        // The recorded slot user is not the actual user. This happens when Concat
        // uses a new slot for scaled IFM. The old slot is actually used by nobody
        // and available for allocation.
        if (get_parameter_info(model->n_input + slot_user_id)->slot != next_slot_id) {
            break;
        }
    }
    my_printf_debug("next_slot_id = %d" NEWLINE, next_slot_id);
    get_slot_info(model, next_slot_id)->user = model->layer_idx;
    return next_slot_id;
}

void my_memcpy_from_param(Model* model, void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n) {
    if (param->slot >= SLOT_CONSTANTS_MIN) {
        uint32_t limit;
        const uint8_t *baseptr = get_param_base_pointer(param, &limit);
        uint32_t total_offset;
        if (param->slot == SLOT_TEST_SET) {
            total_offset = (model->sample_idx % PLAT_LABELS_DATA_LEN) * param->params_len;
        } else {
            total_offset = param->params_offset ;
        }
        total_offset += offset_in_word * sizeof(int16_t);
        MY_ASSERT(total_offset + n <= limit);
        my_memcpy(dest, baseptr + total_offset, n);
    } else {
        my_memcpy_from_intermediate_values(dest, param, offset_in_word, n);
    }
}
