#pragma once

#include <stddef.h> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <stdint.h>

#define NEWLINE "\n"

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
    Node *nodes;
    ParameterInfo *parameter_info;
} Model;

/**********************************
 *          Global data           *
 **********************************/
extern uint16_t *inputs;
extern uint16_t *parameters;
extern uint8_t intermediate_values[];

/**********************************
 *       Helpers for nodes        *
 **********************************/
int16_t* get_q15_param(ParameterInfo *param, size_t i);
int32_t* get_iq31_param(ParameterInfo *param, size_t i);
int64_t get_int64_param(ParameterInfo *param, size_t i);
int16_t node_input(Node *node, size_t i);
void node_input_mark(Node *node, size_t i);
uint8_t node_input_marked(Node *node, size_t i);

/**********************************
 *       Operation handlers       *
 **********************************/
typedef uint8_t (*handler)(ParameterInfo *input[], ParameterInfo *output);
extern uint8_t expected_inputs_len[];
extern handler handlers[];
