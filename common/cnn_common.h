#pragma once

#include <cstddef> /* size_t, see https://stackoverflow.com/a/26413264 */
#include <cstdint>
#include "data.h"

#define ENABLE_COUNTERS 0

/**********************************
 *        Data structures         *
 **********************************/

struct ConvNodeFlags {
    uint8_t input_tile_c;
    uint8_t output_tile_c;
    uint8_t pads[4];
};

struct GemmNodeFlags {
    uint16_t tile_channel;
};

struct GemmMergeNodeFlags {
    uint16_t tile_length;
};

struct SqueezeNodeFlags {
    uint8_t axes;
};

union ExtraNodeFlags {
    ConvNodeFlags conv;
    GemmNodeFlags gemm;
    GemmMergeNodeFlags gemmmerge;
    SqueezeNodeFlags squeeze;
};

struct NodeFlags {
    uint8_t generic : 8;
    uint8_t kernel_size : 4;    // used in MaxPool
    uint8_t stride : 4;         // used in Conv and MaxPool
    ExtraNodeFlags extra;
};

static_assert(sizeof(NodeFlags) == 8, "Unexpected size for NodeFlags");

typedef struct Node {
    char name[NODE_NAME_LEN];
    char output_name[NODE_NAME_LEN];
    uint16_t inputs_len;
    int16_t inputs[NUM_INPUTS];
    uint16_t max_output_id;
    uint16_t op_type;
    NodeFlags flags;
#if HAWAII
    struct Footprint {
        uint16_t value;
        uint8_t version;
    } footprint[2];
#endif
} Node;

static_assert(sizeof(Node) == NODE_NAME_LEN * 2 + 14 + NUM_INPUTS * 2 + HAWAII * 8, "Unexpected size for Node");

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
    uint16_t dummy;
    // uint8_t is not enough. For example, fully connected layer in MNIST has dims 256x1
    uint16_t dims[4];
    uint16_t scale;
    uint8_t param_flags;
    uint8_t extra_info[EXTRA_INFO_LEN];
    uint16_t parameter_info_idx; // must be the last member of this struct
} ParameterInfo;

static_assert(sizeof(ParameterInfo) == 28, "Unexpected size for ParameterInfo");

typedef struct SlotInfo {
#if INDIRECT_RECOVERY
    int8_t state_bit;
    uint8_t n_turning_points;
    uint16_t turning_points[TURNING_POINTS_LEN];
#endif
    int16_t user;
} SlotInfo;

typedef struct Model {
    uint16_t running;
    uint16_t run_counter;
    uint16_t layer_idx;
    SlotInfo slots_info[NUM_SLOTS];
    uint8_t dummy;
    uint8_t version; // must be the last field in this struct
} Model;

static_assert(sizeof(Model) == 8 + NUM_SLOTS * (2 + INDIRECT_RECOVERY * (2 + TURNING_POINTS_LEN * 2)), "Unexpected size for Model");

/**********************************
 *          Global data           *
 **********************************/
#define COUNTERS_LEN (MODEL_NODES_LEN+1)
struct Counters {
    uint32_t power_counters;
    uint32_t dma_invocations;
    uint32_t dma_bytes;

    uint32_t embedding;
    uint32_t stripping;
    uint32_t overflow_handling;

    uint32_t state_query;
    uint32_t table_updates;
    uint32_t table_preservation;
    uint32_t table_loading;

    uint32_t progress_seeking;
};

Counters *counters(uint16_t idx);
extern ParameterInfo intermediate_parameters_info_vm[MODEL_NODES_LEN];


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

/**********************************
 * Helpers for the model & nodes  *
 **********************************/
const uint8_t* get_param_base_pointer(const ParameterInfo *param, uint32_t *limit_p);
int16_t get_q15_param(Model* model, const ParameterInfo *param, uint16_t offset_in_word);
void put_q15_param(ParameterInfo *param, uint16_t offset_in_word, int16_t val);
int64_t get_int64_param(const ParameterInfo *param, size_t i);
uint16_t get_next_slot(Model *model, const ParameterInfo *param);
const ParameterInfo* get_parameter_info(uint16_t i);
const Node* get_node(size_t i);
const Node* get_node(const ParameterInfo* param);
SlotInfo * get_slot_info(Model* model, uint8_t i);
void my_memcpy_from_param(Model* model, void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);

/**********************************
 *       Operation handlers       *
 **********************************/
typedef void (*handler)(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node);
typedef void (*allocator)(Model *model, const ParameterInfo *input[], ParameterInfo *output, const Node* node);
// below are defined in ops.c
extern const handler handlers[];
extern const allocator allocators[];
