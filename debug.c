#include "debug.h"

#ifndef MY_NDEBUG
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
                        print_q15_debug(*get_q15_param(cur_param, offset));
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

void dump_matrix(int16_t *mat, size_t len) {
    for (size_t j = 0; j < len; j++) {
        print_q15_debug(mat[j]);
        if (j && (j % 16 == 15)) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE);
}

void dump_model(void) {
    uint16_t i, j;
    for (i = 0; i < model->nodes_len; i++) {
        Node *cur_node = &(nodes[i]);
        my_printf_debug("(");
        for (j = 0; j < cur_node->inputs_len; j++) {
            my_printf_debug("%d", node_input(cur_node, j));
            if (node_input_marked(cur_node, j)) {
                my_printf_debug("M");
            } else {
                my_printf_debug("U");
            }
            if (j != cur_node->inputs_len - 1) {
                my_printf_debug(", ");
            }
        }
        my_printf_debug(") ");
    }
    my_printf_debug(NEWLINE);
}

#endif