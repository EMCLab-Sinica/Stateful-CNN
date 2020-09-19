#pragma once

#include "data.h"
#include "platform.h"

struct Model;
struct ParameterInfo;
struct SlotInfo;

class ChunkHandler {
public:
    virtual void operator () (uint32_t output_offset, uint16_t output_chunk_len, uint8_t old_output_state_bit) const = 0;
};

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
uint16_t find_overflow_factor(struct Model *model, struct ParameterInfo *param);
void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale);
void iterate_chunks(Model *model, ParameterInfo *param, uint16_t start_offset, uint16_t len, const ChunkHandler& callback);
#if STATEFUL_CNN
void find_initial_state_bit(int16_t* p_offset, uint8_t* p_turning_point_idx, int16_t* p_next_turning_point, SlotInfo** p_slot_info, uint32_t initial_value_idx, Model* model, ParameterInfo* param);

#define check_next_turning_point(offset, turning_point_idx, next_turning_point, slot_info, value_idx) \
    if (next_turning_point > 0 && value_idx >= next_turning_point) { \
        check_next_turning_point_inner(&offset, &turning_point_idx, &next_turning_point, slot_info, value_idx); \
    }

void check_next_turning_point_inner(int16_t* p_offset, uint8_t* p_turning_point_idx, int16_t* p_next_turning_point, SlotInfo* slot_info, uint16_t value_idx);
#endif
