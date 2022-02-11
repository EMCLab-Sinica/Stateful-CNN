#pragma once

#include <cstdint>
#include "data.h"
#include "platform.h"

struct Model;
struct ParameterInfo;
struct SlotInfo;
struct ValueInfo;

typedef void (*ChunkHandler)(uint32_t output_offset, uint16_t output_chunk_len, int8_t old_output_state_bit, void* params);

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
int16_t upper_gauss(int16_t a, int16_t b);
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

struct OutputChunkHandlerParams {
    int16_t* buffer;
    uint16_t buffer_offset;
};
void OutputChunkHandler(uint32_t offset, uint16_t real_chunk_len, int8_t state_bit, void* _params);
void find_initial_state_bit(int16_t* p_offset, uint8_t* p_turning_point_idx, uint16_t* p_next_turning_point, SlotInfo** p_slot_info, uint32_t initial_value_idx, Model* model, const ParameterInfo* param);
void check_next_turning_point(int16_t& offset, uint8_t& turning_point_idx, uint16_t& next_turning_point, SlotInfo* slot_info, uint16_t value_idx);
#endif

void fix_first_unfinished_value_offset(const Model* model, uint32_t* p_first_unfinished_value_offset);
void make_buffer_aligned(int16_t** p_buffer);
float q15_to_float(int16_t val, const ValueInfo& val_info, uint8_t* p_use_prefix = nullptr, bool has_state = true);
void my_offset_q15_batched(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize, bool enforce_states = false);
#if INDIRECT_RECOVERY
uint16_t update_states(int16_t* buffer, uint16_t buffer_size, uint32_t offset, int16_t embedding_offset, uint16_t next_turning_point, bool enforce_states);
#endif
#if JAPARI
void move_weights(int16_t* filter_ptr, bool exact_tile, int16_t values_to_preserve, int16_t tile_width);
#endif
