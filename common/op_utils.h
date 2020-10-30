#pragma once

#include <stdint.h>
#include "data.h"
#include "platform.h"

struct Model;
struct ParameterInfo;
struct SlotInfo;

class ChunkHandler {
public:
    virtual void handle_chunk(uint32_t output_offset, uint16_t output_chunk_len, uint8_t old_output_state_bit) const = 0;
};

class OutputChunkHandler : public ChunkHandler {
public:
    OutputChunkHandler(int16_t *_buffer) : buffer(_buffer) {}

    void handle_chunk(uint32_t offset, uint16_t real_chunk_len, uint8_t state_bit) const override;

private:
    int16_t *buffer;
};


extern int16_t lea_buffer[LEA_BUFFER_SIZE];
uint16_t find_max_multiplier(struct Model *model, const ParameterInfo *param, const int16_t* buffer);
void find_max_multiplier_vector(int16_t* vector, uint16_t len, uint16_t bound, int16_t val_offset, uint16_t* max_multiplier);
void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale);
void iterate_chunks(Model *model, const ParameterInfo *param, uint16_t start_offset, uint16_t len, const ChunkHandler& callback);
void determine_tile_c(ParameterInfo *param, const ParameterInfo *filter = nullptr);
#if STATEFUL_CNN
void find_initial_state_bit(int16_t* p_offset, uint8_t* p_turning_point_idx, int16_t* p_next_turning_point, SlotInfo** p_slot_info, uint32_t initial_value_idx, Model* model, const ParameterInfo* param);

#define check_next_turning_point(offset, turning_point_idx, next_turning_point, slot_info, value_idx) \
    if (next_turning_point > 0 && value_idx >= next_turning_point) { \
        check_next_turning_point_inner(&offset, &turning_point_idx, &next_turning_point, slot_info, value_idx); \
    }

void check_next_turning_point_inner(int16_t* p_offset, uint8_t* p_turning_point_idx, int16_t* p_next_turning_point, SlotInfo* slot_info, uint16_t value_idx);
#endif
