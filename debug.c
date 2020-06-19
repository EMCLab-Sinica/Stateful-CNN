#include "debug.h"
#include "cnn_common.h"

void print_q15(int16_t val) {
#if defined(__MSP430__)
    my_printf("%d ", val);
#elif defined(DUMP_INTEGERS)
    my_printf("% 6d ", val);
#else
    // 2^15
    my_printf("% 13.6f", SCALE * val / 32768.0);
#endif
}

void print_iq31(int32_t val) {
#if defined(__MSP430__) || defined(DUMP_INTEGERS)
    my_printf("%" PRId32 " ", val);
#else
    // 2^31
    my_printf("% f ", SCALE * val / 2147483648.0);
#endif
}

#ifndef MY_NDEBUG

static void dump_value(ParameterInfo *cur_param, size_t offset) {
    if (cur_param->bitwidth == 16) {
        print_q15_debug(*get_q15_param(cur_param, offset));
    } else if (cur_param->bitwidth == 32) {
        print_iq31_debug(*get_iq31_param(cur_param, offset));
    } else if (cur_param->bitwidth == 64) {
        my_printf_debug("%" PRId64 " ", get_int64_param(cur_param, offset));
    }
}

static void check_params_len(ParameterInfo *cur_param) {
    uint32_t expected_params_len = sizeof(int16_t);
    for (uint8_t i = 0; i < 4; i++) {
        if (cur_param->dims[i]) {
            expected_params_len *= cur_param->dims[i];
        }
    }
    if (cur_param->params_len != expected_params_len) {
        ERROR_OCCURRED();
    }
}

// dump in NCHW format
void dump_params(struct ParameterInfo *cur_param) {
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
    for (uint16_t i = 0; i < NUM; i++) {
        my_printf_debug("Matrix %d" NEWLINE, i);
        for (uint16_t j = 0; j < CHANNEL; j++) {
            my_printf_debug("Channel %d" NEWLINE, j);
            for (uint16_t k = 0; k < H; k++) {
                for (uint16_t l = 0; l < W; l++) {
                    // internal format is NCHW
                    size_t offset = i * H * W * CHANNEL + j * H * W + k * W + l;
                    dump_value(cur_param, offset);
                }
                my_printf_debug(NEWLINE);
            }
            my_printf_debug(NEWLINE);
        }
        my_printf_debug(NEWLINE);
    }
}

void dump_params_nhwc(struct ParameterInfo *cur_param, size_t offset) {
    uint16_t NUM, H, W, CHANNEL;
    // tensor
    NUM = cur_param->dims[0];
    CHANNEL = cur_param->dims[1];
    H = cur_param->dims[2];
    W = cur_param->dims[3];
    // check_params_len(cur_param);
    for (uint16_t i = 0; i < NUM; i++) {
        my_printf_debug("Matrix %d" NEWLINE, i);
        for (uint16_t j = 0; j < CHANNEL; j++) {
            my_printf_debug("Channel %d" NEWLINE, j);
            for (uint16_t k = 0; k < H; k++) {
                for (uint16_t l = 0; l < W; l++) {
                    // internal format is NHWC
                    dump_value(cur_param, offset + i * H * W * CHANNEL + k * W * CHANNEL + l * CHANNEL + j);
                }
                my_printf_debug(NEWLINE);
            }
            my_printf_debug(NEWLINE);
        }
        my_printf_debug(NEWLINE);
    }
}

void dump_matrix(int16_t *mat, size_t len) {
    for (size_t j = 0; j < len; j++) {
        print_q15_debug(mat[j]);
        if (j && (j % 16 == 15)) {
            my_printf_debug(NEWLINE);
        }
    }
    my_printf_debug(NEWLINE);
}

void dump_matrix2(int16_t *mat, size_t rows, size_t cols) {
    for (size_t j = 0; j < rows * cols; j++) {
        print_q15_debug(mat[j]);
        if ((j+1) % cols == 0) {
            my_printf_debug(NEWLINE);
        }
    }
    my_printf_debug(NEWLINE);
}

void dump_model(Model *model, Node *nodes) {
    uint16_t i, j;
    for (i = 0; i < model->nodes_len; i++) {
        Node *cur_node = &(nodes[i]);
        my_printf(cur_node->scheduled ? "scheduled     " : "not scheduled ");
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

#else

void dump_model(Model *model, Node *nodes) {
    UNUSED(model);
    UNUSED(nodes);
}

#endif
