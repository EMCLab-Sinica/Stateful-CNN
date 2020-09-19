#pragma once

#include <stddef.h> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <stdint.h>
#include "data.h"

/**********************************
 *        Data structures         *
 **********************************/

struct NodeFlags {
    uint8_t generic : 8;
    uint8_t kernel_size : 4;    // used in MaxPool
    uint8_t stride : 4;         // used in Conv and MaxPool
};

typedef struct Node {
    char name[NODE_NAME_LEN];
    uint16_t inputs_len;
    uint16_t inputs_offset;
    uint16_t max_output_id;
    uint16_t op_type;
    NodeFlags flags;
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
    uint8_t extra_info[EXTRA_INFO_LEN];
    // use signed type for scale as TI's compiler does not handle
    // multiplication/division with mixed signed and unsigned numbers correctly
    int16_t scale;
    uint16_t parameter_info_idx;
} ParameterInfo;

static_assert(sizeof(ParameterInfo) == 28, "Unexpected size for ParameterInfo");

typedef struct SlotInfo {
    SlotInfo() {}
#if STATEFUL_CNN
    uint16_t state_bit;
    uint16_t n_turning_points;
    uint16_t turning_points[TURNING_POINTS_LEN];
#endif
    int16_t user;
} SlotInfo;

typedef struct Model {
    uint16_t nodes_len;
    uint16_t n_input;
    uint16_t running;
    uint16_t first_time;
    uint16_t run_counter;
    uint16_t layer_idx;
    uint16_t sample_idx;
    SlotInfo slots_info[NUM_SLOTS];
} Model;

static_assert(sizeof(Model) == 14 + NUM_SLOTS * (2 + STATEFUL_CNN * (4 + TURNING_POINTS_LEN * 2)), "Unexpected size for Model");

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
extern uint8_t *parameters_info_data;
Counters *counters(void);


/**********************************
 *          Miscellaneous         *
 **********************************/

/* MSP430 SDK already defines MIN, which means minutes */
#define MIN_VAL(x, y) ((x) < (y) ? (x) : (y))
#define MAX_VAL(x, y) ((x) > (y) ? (x) : (y))
// XXX: MSP432 driverlib requires DMA transfer size to be <= 1024. However,
// transfer size < 1024 may be broken as well - copying 1024 items works,
// copying 512 items works, copy a small number of items (e.g., 6, 10, ...)
// works, and copying 626 items (in ConvMerge of conv2 in MNIST) DOES NOT
// WORK (!?).
#define LIMIT_DMA_SIZE(x) MIN_VAL(512, x)

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
const uint8_t* get_param_base_pointer(ParameterInfo *param, uint32_t *limit_p);
int16_t get_q15_param(ParameterInfo *param, uint16_t offset_in_word);
void put_q15_param(ParameterInfo *param, uint16_t offset_in_word, int16_t val);
int64_t get_int64_param(ParameterInfo *param, size_t i);
int16_t node_input(Node *node, size_t i);
uint16_t get_next_slot(Model *model, ParameterInfo *param);
ParameterInfo* get_parameter_info(size_t i);
SlotInfo * get_slot_info(Model* model, uint8_t i);

/**********************************
 *       Operation handlers       *
 **********************************/
typedef void (*handler)(Model *model, ParameterInfo *input[], ParameterInfo *output, NodeFlags* flags);
typedef void (*allocator)(Model *model, ParameterInfo *input[], ParameterInfo *output, NodeFlags* flags);
// below are defined in ops.c
extern uint8_t expected_inputs_len[];
extern handler handlers[];
extern allocator allocators[];
