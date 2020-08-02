#pragma once

#include <stddef.h> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <stdint.h>
#include "data.h"

/**********************************
 *        Data structures         *
 **********************************/
typedef struct Node {
    char name[NODE_NAME_LEN];
    uint16_t inputs_len;
    uint16_t inputs_offset;
    uint16_t max_output_id;
    uint16_t op_type;
    /* Layout of 16 bits in flags
     * 15-08 generic flags
     * 07-04 kernel_size (used in MaxPool)
     * 03-00 stride (used in Conv and MaxPool)
     **/
    uint16_t flags;
} Node;

// _Static_assert in C11 or static_assert in C++11 requires the message
static_assert(sizeof(Node) == 64, "Unexpected size for Node");

/* ParameterInfo may indicate data from the model (parameters) or intermediate values */
typedef struct ParameterInfo {
    uint32_t params_offset;
    uint32_t params_len;  /* in bytes */
    /* Known bitwidth values:
     * 16: q15
     * 32: iq31
     * 64: INT64 (from ONNX)
     */
    uint8_t bitwidth;
    /* A flag to indicate where the data are. Possible values are SLOT_TEST_SET,
     * SLOT_PARAMETERS and SLOT_INTERMEDIATE_VALUES.
     */
    uint8_t slot;
    /* Values are grouped each tile_c channels */
    uint16_t tile_c;
    // uint8_t is not enough. For example, fully connected layer in MNIST has dims 256x1
    uint16_t dims[4];
    uint8_t flags;
    uint8_t extra_info[3];
    // use signed type for scale as TI's compiler does not handle
    // multiplication/division with mixed signed and unsigned numbers correctly
    int16_t scale;
    uint8_t dummy[2]; // for memory alignment
} ParameterInfo;

static_assert(sizeof(ParameterInfo) == 28, "Unexpected size for ParameterInfo");

typedef struct Model {
    uint16_t nodes_len;
    uint16_t n_input;
    uint16_t running;
    uint16_t recovery;
    uint16_t run_counter;
#ifdef WITH_PROGRESS_EMBEDDING
    uint16_t state_bit[NUM_SLOTS];
#else
    uint16_t dummy[NUM_SLOTS];
#endif
    int16_t slot_users[NUM_SLOTS];
    uint16_t layer_idx;
    uint16_t sample_idx;
} Model;

static_assert(sizeof(Model) == 14 + NUM_SLOTS * 4, "Unexpected size for Model");

typedef struct {
    uint16_t time_counters[COUNTERS_LEN];
    uint16_t power_counters[COUNTERS_LEN];
    uint16_t counter_idx;
} Counters;

// Keep the following coefficients synced with transform.py
static_assert(sizeof(Counters) == 4 * COUNTERS_LEN + 2, "Unexpected size of Counters");

/**********************************
 *          Global data           *
 **********************************/
extern uint8_t *inputs_data;
Counters *counters(void);


/**********************************
 *          Miscellaneous         *
 **********************************/

/* MSP430 SDK already defines MIN, which means minutes */
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))
#define MAX_VAL(x, y) ((x) > (y) ? (x) : (y))

/* Better to not use macros
 * https://stackoverflow.com/a/3437484/3786245
 */
static inline int16_t int16_min(int16_t a, int16_t b) {
    return a < b ? a : b;
}

static inline int16_t int16_max(int16_t a, int16_t b) {
    return a > b ? a : b;
}

/**********************************
 *       Helpers for nodes        *
 **********************************/
const int16_t* get_q15_param(ParameterInfo *param, size_t i);
int16_t* get_q15_param_writable(ParameterInfo *param, size_t i);
int64_t get_int64_param(ParameterInfo *param, size_t i);
int16_t node_input(Node *node, size_t i);
uint16_t get_next_slot(Model *model, ParameterInfo *param);

/**********************************
 *       Operation handlers       *
 **********************************/
typedef void (*handler)(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
typedef void (*allocator)(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
// below are defined in ops.c
extern uint8_t expected_inputs_len[];
extern handler handlers[];
extern allocator allocators[];
