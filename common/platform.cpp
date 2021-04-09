#include <string.h>
#include "platform.h"
#include "platform-private.h"
#include "cnn_common.h"
#include "my_debug.h"
#include "intermittent-cnn.h" // for sample_idx

// put offset checks here as extra headers are used
static_assert(NODES_OFFSET > SAMPLES_OFFSET + SAMPLES_DATA_LEN, "Incorrect NVM layout");

Model model_vm;
uint8_t dma_counter_enabled = 1;

template<typename T>
static uint32_t nvm_addr(uint8_t, uint16_t);

template<typename T>
T* vm_addr(uint16_t data_idx);

// typeinfo does not always give names I want
template<typename T>
const char* datatype_name(void);

static uint32_t intermediate_values_offset(uint8_t slot_id) {
    return INTERMEDIATE_VALUES_OFFSET + slot_id * INTERMEDIATE_VALUES_SIZE;
}

static uint32_t intermediate_parameters_info_addr(uint8_t i) {
    return INTERMEDIATE_PARAMETERS_INFO_OFFSET + i * sizeof(ParameterInfo);
}

template<>
uint32_t nvm_addr<Model>(uint8_t i, uint16_t) {
    return MODEL_OFFSET + i * sizeof(Model);
}

template<>
Model* vm_addr<Model>(uint16_t data_idx) {
    return &model_vm;
}

template<>
const char* datatype_name<Model>(void) {
    return "model";
}

void my_memcpy_to_param(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n, uint16_t timer_delay) {
    MY_ASSERT(param->bitwidth == 16);
    MY_ASSERT(param->slot < SLOT_CONSTANTS_MIN);
    uint32_t total_offset = param->params_offset + offset_in_word * sizeof(int16_t);
    MY_ASSERT(total_offset + n <= param->params_len);
    write_to_nvm(src, intermediate_values_offset(param->slot) + total_offset, n, timer_delay);
}

void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n) {
    read_from_nvm(dest, intermediate_values_offset(param->slot) + offset_in_word * sizeof(int16_t), n);
}

void read_from_samples(void *dest, uint16_t offset_in_word, size_t n) {
    read_from_nvm(dest, SAMPLES_OFFSET + (sample_idx % PLAT_LABELS_DATA_LEN) * SAMPLE_SIZE + offset_in_word * sizeof(int16_t), n);
}

ParameterInfo* get_intermediate_parameter_info(uint8_t i) {
    ParameterInfo* dst = intermediate_parameters_info_vm + i;
    read_from_nvm(dst, intermediate_parameters_info_addr(i), sizeof(ParameterInfo));
    my_printf_debug("Load intermediate parameter info %d from NVM" NEWLINE, i);
    MY_ASSERT(dst->parameter_info_idx == i + N_INPUT,
              "Expect parameter index %d but got %d" NEWLINE, i + N_INPUT, dst->parameter_info_idx);
    return dst;
}

void commit_intermediate_parameter_info(uint8_t i) {
    const ParameterInfo* src = intermediate_parameters_info_vm + i;
    MY_ASSERT(src->parameter_info_idx == i + N_INPUT);
    write_to_nvm(src, intermediate_parameters_info_addr(i), sizeof(ParameterInfo));
    my_printf_debug("Committing intermediate parameter info %d to NVM" NEWLINE, i);
}

template<typename T>
static uint8_t get_newer_copy_id(uint16_t data_idx) {
    uint8_t version1, version2;
    read_from_nvm(&version1, nvm_addr<T>(0, data_idx) + offsetof(T, version), sizeof(uint8_t));
    read_from_nvm(&version2, nvm_addr<T>(1, data_idx) + offsetof(T, version), sizeof(uint8_t));
    my_printf_debug("Versions of shadow %s copies for data item %d: %d, %d" NEWLINE, datatype_name<T>(), data_idx, version1, version2);

    if (abs(static_cast<int>(version1 - version2)) == 1) {
        if (version1 > version2) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (version1 > version2) {
            // ex: versions = 65535, 1
            return 1;
        } else {
            return 0;
        }
    }
}

template<typename T>
void bump_version(T *data) {
    data->version++;
    if (!data->version) {
        // don't use version 0 as it indicates the first run
        data->version++;
    }
}

template<typename T>
T* get_versioned_data(uint16_t data_idx) {
    T *dst = vm_addr<T>(data_idx);

    uint8_t newer_copy_id = get_newer_copy_id<T>(data_idx);
    read_from_nvm(dst, nvm_addr<T>(newer_copy_id, data_idx), sizeof(T));
    my_printf_debug("Using %s copy %d, version %d" NEWLINE, datatype_name<T>(), newer_copy_id, dst->version);
    return dst;
}

template<typename T>
void commit_versioned_data(uint16_t data_idx) {
    uint8_t newer_copy_id = get_newer_copy_id<T>(data_idx);
    uint8_t older_copy_id = newer_copy_id ^ 1;

    T* vm_ptr = vm_addr<T>(data_idx);
    bump_version<T>(vm_ptr);

    write_to_nvm(vm_ptr, nvm_addr<T>(older_copy_id, data_idx), sizeof(T));
    my_printf_debug("Committing version %d to %s copy %d" NEWLINE, vm_ptr->version, datatype_name<T>(), older_copy_id);
}

Model* get_model(void) {
    return get_versioned_data<Model>(0);
}

void commit_model(void) {
    if (!model_vm.running) {
        notify_model_finished();
    }
    return commit_versioned_data<Model>(0);
}

void first_run(void) {
    dma_counter_enabled = 0;
    my_printf_debug("First run, resetting everything..." NEWLINE);
    my_erase();
    copy_samples_data();
#if NON_VOLATILE_COUNTERS
    memset(counters(), 0, sizeof(Counters));
#endif

    write_to_nvm_segmented(intermediate_parameters_info_data, intermediate_parameters_info_addr(0),
                           INTERMEDIATE_PARAMETERS_INFO_DATA_LEN, sizeof(ParameterInfo));
    write_to_nvm(model_data, nvm_addr<Model>(0, 0), MODEL_DATA_LEN);
    write_to_nvm(model_data, nvm_addr<Model>(1, 0), MODEL_DATA_LEN);
    dma_counter_enabled = 1;

    get_model(); // refresh model_vm
    commit_model();

    my_printf("Init for " CONFIG "/" METHOD " with batch size=%d" NEWLINE, BATCH_SIZE);
}

static uint32_t max_multipler_offset(uint16_t layer_idx) {
    return NODES_OFFSET + layer_idx * sizeof(Node) + offsetof(Node, max_multiplier);
}

uint16_t read_max_multiplier(const ParameterInfo* param) {
    uint16_t max_multiplier = 0;
    if (param->slot < NUM_SLOTS) {
        uint16_t layer_idx = param->parameter_info_idx - N_INPUT;
        read_from_nvm(&max_multiplier, max_multipler_offset(layer_idx), sizeof(uint16_t));
    }
    return max_multiplier;
}

void write_max_multiplier(const ParameterInfo* param, uint16_t max_multiplier) {
    if (param->slot < NUM_SLOTS) {
        uint16_t layer_idx = param->parameter_info_idx - N_INPUT;
        write_to_nvm(&max_multiplier, max_multipler_offset(layer_idx), sizeof(uint16_t));
    }
}

void check_nvm_write_address(uint32_t nvm_offset, size_t n) {
    if (nvm_offset >= INTERMEDIATE_PARAMETERS_INFO_OFFSET && nvm_offset < MODEL_OFFSET) {
        MY_ASSERT((nvm_offset - INTERMEDIATE_PARAMETERS_INFO_OFFSET) % sizeof(ParameterInfo) == 0);
    } else if (nvm_offset < INTERMEDIATE_PARAMETERS_INFO_OFFSET) {
        MY_ASSERT(n <= INTERMEDIATE_PARAMETERS_INFO_OFFSET - nvm_offset, "Size %d too large!!! nvm_offset=%d" NEWLINE, n, nvm_offset);
    }
}

void write_to_nvm_segmented(const uint8_t* vm_buffer, uint32_t nvm_offset, uint16_t total_len, uint16_t segment_size) {
    for (uint16_t idx = 0; idx < total_len; idx += segment_size) {
        write_to_nvm(vm_buffer + idx, nvm_offset + idx, MIN_VAL(total_len - idx, segment_size));
    }
}

#if HAWAII
Node::Footprint footprints_vm[MODEL_NODES_LEN];

template<>
uint32_t nvm_addr<Node::Footprint>(uint8_t i, uint16_t layer_idx) {
    return NODES_OFFSET + layer_idx * sizeof(Node) + offsetof(Node, footprint) + i * sizeof(Node::Footprint);
}

template<>
Node::Footprint* vm_addr<Node::Footprint>(uint16_t layer_idx) {
    return &footprints_vm[layer_idx];
}

template<>
const char* datatype_name<Node::Footprint>(void) {
    return "footprint";
}

void write_hawaii_layer_footprint(uint16_t layer_idx, int16_t n_jobs) {
    Node::Footprint* footprint_vm = footprints_vm + layer_idx;
    footprint_vm->value += n_jobs;
    MY_ASSERT(footprint_vm->value < INTERMEDIATE_VALUES_SIZE);
    commit_versioned_data<Node::Footprint>(layer_idx);
    my_printf_debug("Write HAWAII layer footprint %d for layer %d" NEWLINE, footprint_vm->value, layer_idx);
}

uint16_t read_hawaii_layer_footprint(uint16_t layer_idx) {
    uint16_t footprint = get_versioned_data<Node::Footprint>(layer_idx)->value;
    my_printf_debug("HAWAII layer footprint=%d for layer %d" NEWLINE, footprint, layer_idx);
    MY_ASSERT(footprint % BATCH_SIZE == 0);
    return footprint;
}

void reset_hawaii_layer_footprint(uint16_t layer_idx) {
    Node::Footprint footprint;
    footprint.value = footprint.version = 0;
    write_to_nvm(&footprint, nvm_addr<Node::Footprint>(0, layer_idx), sizeof(Node::Footprint));
    write_to_nvm(&footprint, nvm_addr<Node::Footprint>(1, layer_idx), sizeof(Node::Footprint));
    my_printf_debug("Reset HAWAII layer footprint for layer %d" NEWLINE, layer_idx);
}
#endif
