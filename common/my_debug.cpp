#include <inttypes.h> // for PRId64
#include "my_debug.h"
#include "cnn_common.h"
#include "intermittent-cnn.h"

uint8_t dump_integer = 1;

#if MY_DEBUG >= 1
template<>
void my_printf_wrapper() {}
#endif

ValueInfo::ValueInfo(ParameterInfo *cur_param, Model *model) {
    this->scale = cur_param->scale;
}

static void print_q15(int16_t val, const ValueInfo& val_info) {
    if (dump_integer) {
        my_printf("% 6d ", val);
    } else {
        uint8_t use_prefix = 0;
#if STATEFUL_CNN
        if (val != -0x8000) {
            if (val < -0x2000) {
                // happens in the last value of each filter column (state embedding)
                val += 0x4000;
                use_prefix = 1;
            }
            if (get_value_state_bit(val)) {
                // 2^15
                val -= 0x4000;
            }
        }
#endif
        my_printf(use_prefix ? "   *% 9.6f" : "% 13.6f", val_info.scale * val / 32768.0);
    }
}

void dump_value(Model *model, ParameterInfo *cur_param, size_t offset) {
    if (cur_param->bitwidth == 16) {
        print_q15(get_q15_param(cur_param, offset), ValueInfo(cur_param, model));
    } else if (cur_param->bitwidth == 64) {
        my_printf("%" PRId64 " ", get_int64_param(cur_param, offset));
    } else {
        ERROR_OCCURRED();
    }
}

void dump_matrix(const int16_t *mat, size_t len, const ValueInfo& val_info) {
    my_printf("Scale: %d" NEWLINE, val_info.scale);
    for (size_t j = 0; j < len; j++) {
        print_q15(mat[j], val_info);
        if (j && (j % 16 == 15)) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE);
}

void dump_matrix(ParameterInfo *param, uint16_t offset, uint16_t len, const ValueInfo& val_info) {
    my_printf("Scale: %d" NEWLINE, val_info.scale);
    for (size_t j = 0; j < len; j++) {
        print_q15(get_q15_param(param, offset + j), val_info);
        if (j && (j % 16 == 15)) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE);
}

void dump_params_nhwc(Model *model, ParameterInfo *cur_param, size_t offset) {
    uint16_t NUM, H, W, CHANNEL;
    // tensor
    NUM = cur_param->dims[0];
    CHANNEL = cur_param->dims[1];
    H = cur_param->dims[2];
    W = cur_param->dims[3];
    // XXX: re-enable this check
    // check_params_len(cur_param);
    my_printf("Slot: %d" NEWLINE, cur_param->slot);
    my_printf("Scale: %d" NEWLINE, cur_param->scale);
    for (uint16_t n = 0; n < NUM; n++) {
        my_printf("Matrix %d" NEWLINE, n);
        for (uint16_t tile_c_base = 0; tile_c_base < CHANNEL; tile_c_base += cur_param->tile_c) {
            uint16_t cur_tile_c = MIN_VAL(cur_param->tile_c, CHANNEL - tile_c_base);
            for (uint16_t c = 0; c < cur_tile_c; c++) {
                my_printf("Channel %d" NEWLINE, tile_c_base + c);
                for (uint16_t h = 0; h < H; h++) {
                    for (uint16_t w = 0; w < W; w++) {
                        // internal format is NWHC (transposed) or NHWC
                        size_t offset2 = n * W * H * CHANNEL + W * H * tile_c_base;
                        if (cur_param->flags & TRANSPOSED) {
                            offset2 += w * H * cur_tile_c + h * cur_tile_c + c;
                        } else {
                            offset2 += h * W * cur_tile_c + w * cur_tile_c + c;
                        }
                        dump_value(model, cur_param, offset + offset2);
                    }
                    my_printf(NEWLINE);
                }
                my_printf(NEWLINE);
            }
        }
        my_printf(NEWLINE);
    }
}

void dump_model(Model *model, Node *nodes) {
    uint16_t i, j;
    for (i = 0; i < model->nodes_len; i++) {
        Node *cur_node = &(nodes[i]);
        if (model->layer_idx > i) {
            my_printf("scheduled     ");
        } else {
            my_printf("not scheduled ");
        }
        my_printf("(");
        for (j = 0; j < cur_node->inputs_len; j++) {
            my_printf("%d", node_input(cur_node, j));
            if (j != cur_node->inputs_len - 1) {
                my_printf(", ");
            }
        }
        my_printf(")" NEWLINE);
    }
}

static void check_params_len(ParameterInfo *cur_param) {
    uint32_t expected_params_len = sizeof(int16_t);
    for (uint8_t i = 0; i < 4; i++) {
        if (cur_param->dims[i]) {
            expected_params_len *= cur_param->dims[i];
        }
    }
    MY_ASSERT(cur_param->params_len == expected_params_len);
}

// dump in NCHW format
void dump_params(Model *model, ParameterInfo *cur_param) {
    uint16_t NUM, H, W, CHANNEL;
    if (cur_param->dims[2] && cur_param->dims[3]) {
        // tensor
        NUM = cur_param->dims[0];
        CHANNEL = cur_param->dims[1];
        H = cur_param->dims[2];
        W = cur_param->dims[3];
    } else {
        // matrix
        NUM = CHANNEL = 1;
        H = cur_param->dims[0];
        W = cur_param->dims[1];
    }
    check_params_len(cur_param);
    my_printf("Slot: %d" NEWLINE, cur_param->slot);
    my_printf("Scale: %d" NEWLINE, cur_param->scale);
    for (uint16_t i = 0; i < NUM; i++) {
        my_printf("Matrix %d" NEWLINE, i);
        for (uint16_t j = 0; j < CHANNEL; j++) {
            my_printf("Channel %d" NEWLINE, j);
            for (uint16_t k = 0; k < H; k++) {
                for (uint16_t l = 0; l < W; l++) {
                    // internal format is NCHW
                    size_t offset = i * H * W * CHANNEL + j * H * W + k * W + l;
                    dump_value(model, cur_param, offset);
                }
                my_printf(NEWLINE);
            }
            my_printf(NEWLINE);
        }
        my_printf(NEWLINE);
    }
}

#if STATEFUL_CNN
void dump_turning_points(Model *model, ParameterInfo *output) {
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    if (!cur_slot_info) {
        my_printf("%d is not a normal slot" NEWLINE, output->slot);
        return;
    }
    my_printf("%d turning point(s) for slot %d: ", cur_slot_info->n_turning_points, output->slot);
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        my_printf("%d ", cur_slot_info->turning_points[idx]);
    }
    my_printf(NEWLINE);
}
#endif

void dump_matrix2(int16_t *mat, size_t rows, size_t cols, const ValueInfo& val_info) {
    my_printf("Scale: %d" NEWLINE, val_info.scale);
    for (size_t j = 0; j < rows * cols; j++) {
        print_q15(mat[j], val_info);
        if ((j+1) % cols == 0) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE);
}
