#pragma once

#include "platform.h"

struct Model;
struct ParameterInfo;

class ChunkHandler {
public:
    virtual void operator () (uint32_t output_offset, uint16_t output_chunk_len, uint8_t old_output_state_bit) const = 0;
};

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
uint16_t find_overflow_factor(struct Model *model, struct ParameterInfo *param);
void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale);
void iterate_chunks(Model *model, ParameterInfo *param, uint16_t start_offset, uint16_t len, const ChunkHandler& callback);
