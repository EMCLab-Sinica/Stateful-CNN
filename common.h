#pragma once

#include <stddef.h> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <stdint.h>
#include <msp430.h> /* __no_operation() */

#ifdef __linux__
#include <stdio.h>
#include <inttypes.h>
#include <signal.h>
#elif defined(__MSP430__)
#include <Tools/myuart.h>
#endif

#define NUM_SLOTS 2
#define FLAG_SLOTS 0b11
#define FLAG_SLOTS_WIDTH 2
#define SCALE 16
#define INTERMEDIATE_VALUES_SIZE 65536
#define TASK_FLAGS_SIZE 1024

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
extern Model *model;
extern Node *nodes;
extern ParameterInfo *parameter_info;
extern uint16_t *inputs;
extern uint16_t *parameters;
// similar to double buffering
extern uint8_t *intermediate_values;

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

/* Better to not use macros
 * https://stackoverflow.com/a/3437484/3786245
 */
static inline int16_t int16_min(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t int16_max(int16_t a, int16_t b) {
    return a > b ? a : b;
}

#ifdef __MSP430__
#define ERROR_OCCURRED() for (;;) { __no_operation(); }
#else
#define ERROR_OCCURRED() do { raise(SIGINT); } while (0);
#endif

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
        return intermediate_values + slot_id * INTERMEDIATE_VALUES_SIZE;
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
void node_input_unmark_all(Node *node);
uint8_t node_input_marked(Node *node, size_t i);
static inline int16_t iq31_to_q15(int32_t val) {
    return (int16_t)(val >> 16);
}

static inline void print_q15(int16_t val) {
#if defined(__MSP430__) || defined(DUMP_INTEGERS)
    my_printf("%d ", val);
#else
    // 2^15
    my_printf("% f ", SCALE * val / 32768.0);
#endif
}

static inline void print_iq31(int32_t val) {
#if defined(__MSP430__)
    my_printf("%L ", val); // see print2uart() in Tools/myuart.c
#elif defined(DUMP_INTEGERS)
    my_printf("%" PRId32 " ", val);
#else
    // 2^31
    my_printf("% f ", SCALE * val / 2147483648.0);
#endif
}

/**********************************
 *       Operation handlers       *
 **********************************/
typedef uint8_t (*handler)(ParameterInfo *input[], ParameterInfo *output);
extern uint8_t expected_inputs_len[];
extern handler handlers[];
