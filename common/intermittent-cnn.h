#pragma once

#include <cstdint>
#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"

struct ParameterInfo;
struct Model;

uint32_t job_index_to_offset(const ParameterInfo* output, uint16_t job_index);
uint32_t batch_start(uint32_t batch_end_offset);

int8_t get_state_bit(Model *model, uint8_t slot_id);

#if HAWAII || STATEFUL
static inline bool offset_has_state(uint16_t offset) {
    return offset % BATCH_SIZE == BATCH_SIZE - 1;
}
#elif JAPARI
static inline bool offset_has_state(uint16_t offset) {
    return offset % (BATCH_SIZE + 1) == BATCH_SIZE;
}
#else
static inline bool offset_has_state(uint16_t) {
    return false;
}
#endif

#if STATEFUL
static inline int8_t get_value_state_bit(int16_t val) {
    return (val >= 0) ? 1 : -1;
}
static inline void strip_state(int16_t* val) {
    // assuming input state bits are correct...
    // The following line is equivalient to: *val -= ((*val >= 0) ? 0x4000 : -0x4000));
    // I use bitwise operations to avoid branches
    *val -= (*val & 0x8000) + 0x4000;
}
#endif
#if JAPARI
static inline void check_footprint(int16_t val) {
    // -255 and 255 happens when only the first byte of a footprint is written
    MY_ASSERT(val == 0 || val == 1 || val == -1 || val == -255 || val == 255,
              "%d is not a valid footprint" NEWLINE, val);
}

static inline int8_t get_value_state_bit(int16_t val) {
    check_footprint(val);
    // 255 (0xff, 0x00 on little-endian systems) happens when the first byte of -1 (0xff, 0xff) is
    // written over 1 (0x01, 0x00), and it should be considered as -1 not completely written. In
    // other words, the state is still 1.
    return (val >= 0) ? 1 : -1;
}
#endif
int8_t param_state_bit(Model *model, const ParameterInfo *param, uint16_t offset);

uint32_t run_recovery(Model *model, ParameterInfo *output);
#if INDIRECT_RECOVERY
void flip_state_bit(Model *model, const ParameterInfo *output);
#endif
