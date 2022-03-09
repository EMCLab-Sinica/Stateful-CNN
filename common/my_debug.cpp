#include <cstddef>
#include <cstring>
#include <cinttypes> // for PRId64
#include <cstdint>
#include <memory>
#include "my_debug.h"
#include "cnn_common.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"
#ifdef USE_PROTOBUF
#include "model_output.pb.h"
#endif

uint8_t dump_integer = 1;

template<>
void my_printf_wrapper() {}

#ifdef USE_PROTOBUF
std::unique_ptr<ModelOutput> model_output_data;
#endif

#define PRINT_NEWLINE_IF_DATA_NOT_SAVED if (!layer_out) { my_printf(NEWLINE); }

ValueInfo::ValueInfo(const ParameterInfo *cur_param, Model *model) {
    this->scale = cur_param->scale;
}

static void print_q15(LayerOutput* layer_out, int16_t val, const ValueInfo& val_info, bool has_state) {
    uint8_t use_prefix = 0;
    float real_value = q15_to_float(val, val_info, &use_prefix, has_state);
#ifdef USE_PROTOBUF
    if (layer_out) {
        layer_out->add_value(real_value);
    } else
#endif
    if (dump_integer) {
        my_printf("% 6d ", val);
    } else {
        my_printf(use_prefix ? "   *% 9.6f" : "% 13.6f", real_value);
    }
}

void dump_value(Model *model, const ParameterInfo *cur_param, LayerOutput* layer_out, size_t offset, bool has_state) {
    if (cur_param->bitwidth == 16) {
        print_q15(layer_out, get_q15_param(model, cur_param, offset), ValueInfo(cur_param, model), has_state);
    } else if (cur_param->bitwidth == 64) {
        my_printf("%" PRId64 " ", get_int64_param(cur_param, offset));
    } else {
        MY_ASSERT(false);
    }
}

void dump_matrix(const int16_t *mat, size_t len, const ValueInfo& val_info, bool has_state) {
    my_printf("Scale: %d" NEWLINE, val_info.scale);
    for (size_t j = 0; j < len; j++) {
        print_q15(nullptr, mat[j], val_info, has_state && offset_has_state(j));
        if (j && (j % 16 == 15)) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE);
}

static void dump_params_common(Model* model, const ParameterInfo* cur_param, const char* layer_name, LayerOutput** p_layer_out) {
    my_printf("Slot: %d" NEWLINE, cur_param->slot);
    my_printf("Scale: %d" NEWLINE, cur_param->scale);
    my_printf("Params len: %" PRId32 NEWLINE, cur_param->params_len);
#if INDIRECT_RECOVERY
    if (cur_param->slot < NUM_SLOTS) {
        my_printf("State: %d" NEWLINE, get_slot_info(model, cur_param->slot)->state_bit);
    }
#endif
    my_printf("Dims: ");
    for (uint8_t j = 0; j < 4; j++) {
        if (cur_param->dims[j]) {
            my_printf("%d, ", cur_param->dims[j]);
        }
    }
    my_printf(NEWLINE);

#ifdef USE_PROTOBUF
    if (layer_name && model_output_data.get()) {
        LayerOutput* layer_out = *p_layer_out = model_output_data->add_layer_out();
        layer_out->set_name(layer_name);
        for (uint8_t idx = 0; idx < 4; idx++) {
            layer_out->add_dims(cur_param->dims[idx]);
        }
    }
#endif
}

static void extract_dimensions(const ParameterInfo* cur_param, uint16_t* NUM, uint16_t* H, uint16_t* W, uint16_t* CHANNEL) {
    if (cur_param->dims[3]) {
        // 4-D tensor, NCHW
        *NUM = cur_param->dims[0];
        *CHANNEL = cur_param->dims[1];
        *H = cur_param->dims[2];
        *W = cur_param->dims[3];
    } else if (cur_param->dims[2]) {
        // 3-D tensor, NCW
        *NUM = cur_param->dims[0];
        *CHANNEL = cur_param->dims[1];
        *H = 1;
        *W = cur_param->dims[2];
    } else if (cur_param->dims[1]) {
        // matrix, HW
        *NUM = *CHANNEL = 1;
        *H = cur_param->dims[0];
        *W = cur_param->dims[1];
    } else {
        // vector, W
        *NUM = *CHANNEL = *H = 1;
        *W = cur_param->dims[0];
    }

    // find real num
    uint32_t expected_params_len = sizeof(int16_t);
    for (uint8_t idx = 0; idx < 4; idx++) {
        if (cur_param->dims[idx]) {
            expected_params_len *= cur_param->dims[idx];
        }
    }
    if (expected_params_len != cur_param->params_len) {
        MY_ASSERT(cur_param->dims[0] == 1);
        *NUM = cur_param->params_len / expected_params_len;
    }
}

void dump_params_nhwc(Model *model, const ParameterInfo *cur_param, const char* layer_name) {
    dma_counter_enabled = 0;
    uint16_t NUM, H, W, CHANNEL;
    extract_dimensions(cur_param, &NUM, &H, &W, &CHANNEL);
    LayerOutput* layer_out = nullptr;
    dump_params_common(model, cur_param, layer_name, &layer_out);
    int16_t output_tile_c = cur_param->dims[1];
    for (uint16_t n = 0; n < NUM; n++) {
        my_printf("Matrix %d" NEWLINE, n);
        for (uint16_t tile_c_base = 0; tile_c_base < CHANNEL; tile_c_base += output_tile_c) {
            uint16_t cur_tile_c = MIN_VAL(output_tile_c, CHANNEL - tile_c_base);
            for (uint16_t c = 0; c < cur_tile_c; c++) {
                if (!layer_out) {
                    my_printf("Channel %d" NEWLINE, tile_c_base + c);
                }
                for (uint16_t h = 0; h < H; h++) {
                    for (uint16_t w = 0; w < W; w++) {
                        // internal format is NHWC
                        size_t offset2 = n * H * W * CHANNEL + H * W * tile_c_base + h * W * cur_tile_c + w * cur_tile_c + c;
                        dump_value(model, cur_param, layer_out, offset2, offset_has_state(offset2));
                    }
                    PRINT_NEWLINE_IF_DATA_NOT_SAVED
                }
                PRINT_NEWLINE_IF_DATA_NOT_SAVED
            }
        }
        PRINT_NEWLINE_IF_DATA_NOT_SAVED
    }
    dma_counter_enabled = 1;
}

void dump_model(Model *model) {
    uint16_t i, j;
    for (i = 0; i < MODEL_NODES_LEN; i++) {
        const Node *cur_node = get_node(i);
        if (model->layer_idx > i) {
            my_printf("scheduled     ");
        } else {
            my_printf("not scheduled ");
        }
        my_printf("(");
        for (j = 0; j < cur_node->inputs_len; j++) {
            my_printf("%d", cur_node->inputs[j]);
            if (j != cur_node->inputs_len - 1) {
                my_printf(", ");
            }
        }
        my_printf(")" NEWLINE);
    }
}

// dump in NCHW format
void dump_params(Model *model, const ParameterInfo *cur_param, const char* layer_name) {
    dma_counter_enabled = 0;
    uint16_t NUM, H, W, CHANNEL;
    extract_dimensions(cur_param, &NUM, &H, &W, &CHANNEL);
    LayerOutput* layer_out = nullptr;
    dump_params_common(model, cur_param, layer_name, &layer_out);
    for (uint16_t i = 0; i < NUM; i++) {
        my_printf("Matrix %d" NEWLINE, i);
        for (uint16_t j = 0; j < CHANNEL; j++) {
            if (!layer_out) {
                my_printf("Channel %d" NEWLINE, j);
            }
            for (uint16_t k = 0; k < H; k++) {
                for (uint16_t l = 0; l < W; l++) {
                    // internal format is NCHW
                    size_t offset = i * H * W * CHANNEL + j * H * W + k * W + l;
                    dump_value(model, cur_param, layer_out, offset, offset_has_state(offset));
                }
                PRINT_NEWLINE_IF_DATA_NOT_SAVED
            }
            PRINT_NEWLINE_IF_DATA_NOT_SAVED
        }
        PRINT_NEWLINE_IF_DATA_NOT_SAVED
    }
    dma_counter_enabled = 1;
}

void dump_turning_points(Model *model, const ParameterInfo *output) {
#if INDIRECT_RECOVERY
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    if (!cur_slot_info) {
        my_printf("%d is not a normal slot" NEWLINE, output->slot);
        return;
    }
    my_printf("Initial state bit for slot %d: %d" NEWLINE, output->slot, cur_slot_info->state_bit);
    my_printf("%d turning point(s) for slot %d: ", cur_slot_info->n_turning_points, output->slot);
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        uint16_t cur_turning_point = cur_slot_info->turning_points[idx];
        my_printf("%d ", cur_turning_point);
    }
    my_printf(NEWLINE);
#endif
}

void dump_matrix(const int16_t *mat, size_t rows, size_t cols, const ValueInfo& val_info, bool has_state) {
    my_printf("Scale: %d", val_info.scale);
    if (rows > cols) {
        my_printf(" (transposed)" NEWLINE);
        for (size_t j = 0; j < cols; j++) {
            for (size_t i = 0; i < rows; i++) {
                size_t offset = i * cols + j;
                print_q15(nullptr, mat[offset], val_info, has_state && offset_has_state(offset));
            }
            my_printf(NEWLINE);
        }
    } else {
        my_printf(NEWLINE);
        for (size_t j = 0; j < rows * cols; j++) {
            print_q15(nullptr, mat[j], val_info, has_state && offset_has_state(j));
            if ((j+1) % cols == 0) {
                my_printf(NEWLINE);
            }
        }
    }
    my_printf(NEWLINE);
}

static const uint16_t BUFFER_TEMP_SIZE = 256;
static int16_t buffer_temp[BUFFER_TEMP_SIZE];

void compare_vm_nvm_impl(int16_t* vm_data, Model* model, const ParameterInfo* output, uint16_t output_offset, uint16_t blockSize) {
    check_buffer_address(vm_data, blockSize);
    MY_ASSERT(blockSize <= BUFFER_TEMP_SIZE);

    memset(buffer_temp, 0, blockSize * sizeof(int16_t));
    my_memcpy_from_param(model, buffer_temp, output, output_offset, blockSize * sizeof(int16_t));
    for (uint16_t idx = 0; idx < blockSize; idx++) {
        MY_ASSERT_ALWAYS(vm_data[idx] == buffer_temp[idx]);
    }
}

void check_nvm_write_address_impl(uint32_t nvm_offset, size_t n) {
    if (nvm_offset >= INTERMEDIATE_PARAMETERS_INFO_OFFSET && nvm_offset < MODEL_OFFSET) {
        MY_ASSERT((nvm_offset - INTERMEDIATE_PARAMETERS_INFO_OFFSET) % sizeof(ParameterInfo) == 0);
    } else if (nvm_offset < INTERMEDIATE_PARAMETERS_INFO_OFFSET) {
        MY_ASSERT(n <= INTERMEDIATE_PARAMETERS_INFO_OFFSET - nvm_offset, "Size %d too large!!! nvm_offset=%d" NEWLINE, n, nvm_offset);
    }
}
