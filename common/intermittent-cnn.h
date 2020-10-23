#pragma once

#include <stdint.h>
#include "data.h"
#include "my_debug.h"

extern uint16_t sample_idx;

struct ParameterInfo;
struct Model;
uint8_t run_cnn_tests(uint16_t n_samples);
#if STATEFUL_CNN
uint8_t get_state_bit(struct Model *model, uint8_t slot_id);
static inline uint8_t get_value_state_bit(int16_t val) {
    MY_ASSERT(-0x2000 <= val && val < 0x6000,
        "Unexpected embedded state in value %d" NEWLINE, val);
    return val >= 0x2000;
}
void flip_state_bit(struct Model *model, const ParameterInfo *output);
uint8_t param_state_bit(Model *model, const ParameterInfo *param, uint16_t offset);
uint32_t recovery_from_state_bits(struct Model *model, struct ParameterInfo *output);
#endif
