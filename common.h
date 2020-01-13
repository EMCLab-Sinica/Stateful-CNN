#pragma once

#define MY_NDEBUG
#ifndef MY_NDEBUG
#define DUMP_PARAMS
#endif

#include <stddef.h> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <stdint.h>
#include <msp430.h> /* __no_operation() */

#ifdef __linux__
#include <stdio.h>
#elif defined(__MSP430__)
#include <Tools/myuart.h>
#endif

#define NUM_SLOTS 2
#define FLAG_SLOTS 0b11
#define FLAG_SLOTS_WIDTH 2

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
     * The least two sigfinicant bits contains a flag to indicate where the
     * data are. All 1's indicate data are in parameters, otherwise it's the
     * slot number for one of intermediate_values.
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
// similar to double buffering
extern uint8_t intermediate_values[NUM_SLOTS][INTERMEDIATE_VALUES_SIZE];

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

#define ERROR_OCCURRED() for (;;) { __no_operation(); }

/**********************************
 *       Helpers for nodes        *
 **********************************/
static inline uint16_t get_param_bitwidth(ParameterInfo *param) {
    return param->bitwidth_and_flags >> FLAG_SLOTS_WIDTH;
}

static inline uint16_t get_param_slot_id(ParameterInfo *param) {
    return param->bitwidth_and_flags & FLAG_SLOTS;
}

static inline uint16_t get_next_slot(ParameterInfo *param) {
    uint16_t slot_id = get_param_slot_id(param);
    /* Some cases:
     * 1. slot_id == FLAG_SLOTS -> pick the first slot as the current param is input
     * 2. Otherwise, pick the next slot
     */
    uint16_t next_slot_id = (slot_id + 1) & FLAG_SLOTS;
    if (next_slot_id >= NUM_SLOTS) {
        next_slot_id = 0;
    }
    return next_slot_id;
}

static inline uint8_t* get_param_base_pointer(ParameterInfo *param) {
    uint16_t slot_id = get_param_slot_id(param);
    if (slot_id != FLAG_SLOTS) {
        return &(intermediate_values[slot_id][0]);
    } else {
        return (uint8_t*)parameters;
    }
}

static inline int16_t* get_q15_param(ParameterInfo *param, size_t i) {
    if (get_param_bitwidth(param) != 16) {
        my_printf("Error: incorrect param passed to %s" NEWLINE, __func__);
        ERROR_OCCURRED();
    }
    return (int16_t*)(get_param_base_pointer(param) + param->params_offset) + i;
}


int32_t* get_iq31_param(ParameterInfo *param, size_t i);
int64_t get_int64_param(ParameterInfo *param, size_t i);
int16_t node_input(Node *node, size_t i);
void node_input_mark(Node *node, size_t i);
uint8_t node_input_marked(Node *node, size_t i);
int16_t iq31_to_q15(int32_t *iq31_val_ptr);
#if !defined(MY_NDEBUG) && defined(DUMP_PARAMS)
void dump_params(ParameterInfo *cur_param);
#else
#define dump_params(cur_param)
#endif

#ifdef DUMP_PARAMS
static inline void dump_matrix(int16_t *mat, size_t len) {
    for (size_t j = 0; j < len; j++) {
        my_printf("%d ", mat[j]);
        if (j && (j % 16 == 0)) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE);
}
#else
#define dump_matrix(mat, len)
#endif

/**********************************
 *       Operation handlers       *
 **********************************/
typedef uint8_t (*handler)(ParameterInfo *input[], ParameterInfo *output);
extern uint8_t expected_inputs_len[];
extern handler handlers[];
