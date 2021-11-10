#pragma once

#include <stdint.h>
#include "data.h"
#include "platform.h"

struct Model;
struct ParameterInfo;
struct SlotInfo;
struct ValueInfo;

typedef void (*ChunkHandler)(uint32_t output_offset, uint16_t output_chunk_len, int8_t old_output_state_bit, void* params);

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
int16_t upper_gauss(int16_t a, int16_t b);
uint16_t find_max_multiplier(struct Model *model, const ParameterInfo *param, int16_t* buffer = nullptr, uint16_t len = 0);
void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale);
void iterate_chunks(Model *model, const ParameterInfo *param, uint16_t start_offset, uint16_t len, const ChunkHandler& callback, void* params);
void determine_tile_c(ParameterInfo *param, const ParameterInfo* input, const ParameterInfo *filter = nullptr);

#if HAWAII
void hawaii_record_footprints(Model* model, uint16_t vector_len);
#endif

#if JAPARI
#define INPUT_BUFFER_WITH_FOOTPRINTS_LEN 256

extern int16_t input_buffer_with_footprints[INPUT_BUFFER_WITH_FOOTPRINTS_LEN];
int16_t extend_for_footprints(int16_t val, uint8_t force_aligned = 0);
uint8_t has_footprints(const ParameterInfo* cur_param);
#endif

#if INDIRECT_RECOVERY
const uint16_t INVALID_TURNING_POINT = static_cast<uint16_t>(-1);

void OutputChunkHandler(uint32_t offset, uint16_t real_chunk_len, int8_t state_bit, void* _params);
void find_initial_state_bit(int16_t* p_offset, uint8_t* p_turning_point_idx, uint16_t* p_next_turning_point, SlotInfo** p_slot_info, uint32_t initial_value_idx, Model* model, const ParameterInfo* param);

#define check_next_turning_point(offset, turning_point_idx, next_turning_point, slot_info, value_idx) \
    if (next_turning_point != INVALID_TURNING_POINT && value_idx >= next_turning_point) { \
        my_printf_debug("Checking next turning point after %d" NEWLINE, value_idx); \
        check_next_turning_point_inner(&offset, &turning_point_idx, &next_turning_point, slot_info, value_idx); \
    }

void check_next_turning_point_inner(int16_t* p_offset, uint8_t* p_turning_point_idx, uint16_t* p_next_turning_point, SlotInfo* slot_info, uint16_t value_idx);
#endif

void fix_first_unfinished_value_offset(const Model* model, uint32_t* p_first_unfinished_value_offset);
void make_buffer_aligned(int16_t** p_buffer);
float q15_to_float(int16_t val, const ValueInfo& val_info, uint8_t* p_use_prefix = nullptr);
void my_offset_q15_batched(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize);
