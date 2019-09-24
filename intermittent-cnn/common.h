#pragma once

#include <stddef.h> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <stdint.h>

#ifdef __linux__
#include <stdio.h>
#elif defined(__MSP430__)
#include <Tools/myuart.h>
#endif

#define FLAG_INTERMEDIATE_VALUES 1

/**********************************
 *        Data structures         *
 **********************************/
typedef struct {
    uint16_t inputs_len;
    uint16_t inputs_offset;
    uint16_t op_type;
    uint16_t scheduled;  /* 16 bits for aligned memory */
} Node;

/* ParameterInfo may indicate data from the model (parameters) or intermediate values */
typedef struct __attribute__((__packed__)) {
    uint32_t params_offset;
    uint32_t params_len;  /* in bytes */
    /* Known bitwidth values:
     * 16: q15
     * 32: iq31
     * 64: INT64 (from ONNX)
     *
     * The least sigfinicant bit is a flag to indicate where the data are - parameters or intermediate_values
     */
    uint16_t bitwidth_and_flags;
    uint16_t dims[4];
} ParameterInfo;

typedef struct __attribute__((__packed__)) {
    uint16_t nodes_len;
    uint16_t n_input;
} Model;

/**********************************
 *          Global data           *
 **********************************/
#define INTERMEDIATE_VALUES_SIZE 65536
extern Model *model;
extern Node *nodes;
extern ParameterInfo *parameter_info;
extern uint16_t *inputs;
extern uint16_t *parameters;
extern uint8_t intermediate_values[INTERMEDIATE_VALUES_SIZE];

/**********************************
 *          Miscellaneous         *
 **********************************/
#if defined(__linux__)
#  define my_printf printf
#  define NEWLINE "\n"
#elif defined(__MSP430__)
#  define my_printf print2uart
#  define NEWLINE "\r\n"
#endif

/* MSP430 SDK already defines MIN, which means minutes */
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))

/**********************************
 *       Helpers for nodes        *
 **********************************/
static inline uint8_t* get_param_base_pointer(ParameterInfo *param) {
    if (param->bitwidth_and_flags & FLAG_INTERMEDIATE_VALUES) {
        return &(intermediate_values[0]);
    } else {
        return (uint8_t*)parameters;
    }
}

static int16_t* get_q15_param(ParameterInfo *param, size_t i) {
    if ((param->bitwidth_and_flags >> 1) != 16) {
        my_printf("Error: incorrect param passed to %s" NEWLINE, __func__);
        return NULL;
    }
    return (int16_t*)(get_param_base_pointer(param) + param->params_offset) + i;
}


int32_t* get_iq31_param(ParameterInfo *param, size_t i);
int64_t get_int64_param(ParameterInfo *param, size_t i);
int16_t node_input(Node *node, size_t i);
void node_input_mark(Node *node, size_t i);
uint8_t node_input_marked(Node *node, size_t i);
int16_t iq31_to_q15(int32_t *iq31_val_ptr);

/**********************************
 *       Operation handlers       *
 **********************************/
typedef uint8_t (*handler)(ParameterInfo *input[], ParameterInfo *output);
extern uint8_t expected_inputs_len[];
extern handler handlers[];
