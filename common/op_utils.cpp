#include "op_utils.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "cnn_common.h"

// Not using DSPLIB_DATA here as it does not work under C++ (?)
#ifdef __MSP430__
#pragma DATA_SECTION(".leaRAM")
#endif
int16_t lea_buffer[LEA_BUFFER_SIZE];

#if HAWAII
uint16_t hawaii_preserve_vector(Model* model, ParameterInfo* output, uint32_t output_offset, const int16_t* buffer, uint16_t vector_len) {
    my_memcpy_to_param(output, output_offset, buffer, vector_len * sizeof(int16_t), 0);
    for (int16_t non_recorded_jobs = vector_len; non_recorded_jobs >= 0; non_recorded_jobs -= BATCH_SIZE) {
        write_hawaii_layer_footprint(model->layer_idx, MIN_VAL(BATCH_SIZE, non_recorded_jobs));
    }
    return vector_len;
}
#endif

#if JAPARI
int16_t input_buffer_with_footprints[INPUT_BUFFER_WITH_FOOTPRINTS_LEN];

int16_t extend_for_footprints(int16_t val, uint8_t force_aligned) {
    if (force_aligned) {
        val = upper_gauss(val, BATCH_SIZE) * BATCH_SIZE;
    }
    return val + val / BATCH_SIZE;
}

uint8_t has_footprints(const ParameterInfo *cur_param) {
    return (cur_param->slot < NUM_SLOTS);
}
#endif

int16_t upper_gauss(int16_t a, int16_t b) {
    return (a + b - 1) / b;
}

#if INDIRECT_RECOVERY
void OutputChunkHandler(uint32_t offset, uint16_t real_chunk_len, uint8_t state_bit, void* _params) {
    int16_t* buffer = reinterpret_cast<int16_t*>(_params);
    int16_t* to_offset = buffer + offset;
#if STATEFUL
    if (!state_bit) {
        my_offset_q15_batched(to_offset, 0x4000, to_offset, real_chunk_len);
    }
#endif
#if JAPARI
    for (uint16_t idx = BATCH_SIZE; idx < real_chunk_len; idx += BATCH_SIZE + 1) {
        to_offset[idx] = (state_bit ? -1 : 1);
    }
#endif
}
#endif

struct MaxMultiplierChunkHandlerParams {
    Model *model;
    const ParameterInfo *param;
    // not declaring buffer as const to allow stripping states
    int16_t* buffer;
    uint16_t *max_multiplier;
};

static inline void reduce_max_multiplier(uint16_t* max_multiplier) {
    // XXX: a heuristic - works when 3 is too large and 2 is OK
    // as seen in Statefull/KWS
    if ((*max_multiplier) % 2) {
        (*max_multiplier)--;
    } else {
        (*max_multiplier) /= 2;
    }
}

void MaxMultiplierChunkHandler(uint32_t offset, uint16_t real_chunk_len, uint8_t state_bit, void* _params) {
    MaxMultiplierChunkHandlerParams* params = reinterpret_cast<MaxMultiplierChunkHandlerParams*>(_params);
#if !STATEFUL
    uint16_t bound = 32768;
#else
    // For stateful CNN, values should not reach 8192, or get_value_state_bit() is confused
    uint16_t bound = 8191;
#endif
    int16_t max_val, min_val;
    // Apparently TI's compiler does not handle multiplication between
    // int16_t and uint16_t correctly. Use unsigned everywhere to fix it.
    uint32_t u_max_val, u_min_val;
    uint16_t index;

    int16_t* cur_buffer;
    if (!params->buffer) {
        my_memcpy_from_param(params->model, lea_buffer, params->param, offset, real_chunk_len * sizeof(int16_t));
        cur_buffer = lea_buffer;
    } else {
        cur_buffer = params->buffer + offset;
    }

#if STATEFUL
    for (uint16_t idx = 0; idx < real_chunk_len; idx++) {
        int16_t val = cur_buffer[idx];
        if (get_value_state_bit(val)) {
            cur_buffer[idx] = val - 0x4000;
        }
    }
#endif

    dump_matrix_debug(cur_buffer, real_chunk_len, ValueInfo(params->param));

    my_max_q15(cur_buffer, real_chunk_len, &max_val, &index);
    my_printf_debug("Max value %d", max_val);
    my_printf_debug(" occurs at index %d" NEWLINE, index);
    u_max_val = abs(max_val);
    // use > instead of >= as the value may be exactly on the bound
    while (max_val && u_max_val * (*params->max_multiplier) > bound) {
        reduce_max_multiplier(params->max_multiplier);
    }

    my_min_q15(cur_buffer, real_chunk_len, &min_val, &index);
    my_printf_debug("Min value %d", min_val);
    my_printf_debug(" occurs at index %d" NEWLINE, index);
    u_min_val = abs(min_val);
    while (min_val && u_min_val * (*params->max_multiplier) > bound) {
        reduce_max_multiplier(params->max_multiplier);
    }
    my_printf_debug("Current max_multiplier=%d" NEWLINE, *params->max_multiplier);
}

uint16_t find_max_multiplier(Model *model, const ParameterInfo *param, int16_t* buffer, uint16_t len) {
    uint16_t max_multiplier = 0;
    if (!buffer && sample_idx == 0) {
        max_multiplier = read_max_multiplier(param);
        // all bytes are initialized as 0xff on NVM
        if (max_multiplier && max_multiplier != 0xffff) {
            return max_multiplier;
        }
    }
    max_multiplier = param->scale;

    MaxMultiplierChunkHandlerParams params({model, param, buffer, &max_multiplier});
    iterate_chunks(model, param, 0, len, MaxMultiplierChunkHandler, &params);

    my_printf_debug("max_multiplier=%d" NEWLINE, max_multiplier);

    MY_ASSERT(max_multiplier != 0);

    if (!buffer && sample_idx == 0) {
        write_max_multiplier(param, max_multiplier);
    }

    return max_multiplier;
}

void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale) {
    *shift = 0;
    while (scale >= 1) {
        scale /= 2;
        (*shift)++;
    }
    *scaleFract = scale * 32768;
}

void iterate_chunks(Model *model, const ParameterInfo *param, uint16_t start_offset, uint16_t len, const ChunkHandler& chunk_handler, void* params) {
    uint16_t params_len;
    if (!len) {
        params_len = param->params_len / sizeof(int16_t);
    } else {
        params_len = start_offset + len;
    }
    uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / 2 * 2);
    uint8_t state_bit = 0;

    uint16_t cur_chunk_len;
#if INDIRECT_RECOVERY
    dump_turning_points_debug(model, param);

    state_bit = get_state_bit(model, param->slot);
    uint8_t turning_point_idx = 0;
    uint16_t next_turning_point = INVALID_TURNING_POINT;
    SlotInfo *cur_slot_info = get_slot_info(model, param->slot);
    uint16_t n_turning_points = cur_slot_info ? cur_slot_info->n_turning_points : 0;
    uint8_t turning_point_found = 0;
    while (turning_point_idx < n_turning_points) {
        next_turning_point = cur_slot_info->turning_points[turning_point_idx];
        turning_point_idx++;
        if (next_turning_point != INVALID_TURNING_POINT && next_turning_point > start_offset) {
            turning_point_found = 1;
            break;
        }
        state_bit ^= 1;
    }
    if (!turning_point_found) {
        // no turning points not after start_offset found
        next_turning_point = INVALID_TURNING_POINT;
    }
#endif
    for (uint32_t offset = start_offset; offset < params_len; offset += cur_chunk_len) {
        cur_chunk_len = MIN_VAL(chunk_len, params_len - offset);
#if INDIRECT_RECOVERY
        uint8_t next_state_flipped = 0;
        // Use <= here as turning_point_idx is actually index for the _next_ turning point
        if (next_turning_point != INVALID_TURNING_POINT && turning_point_idx <= cur_slot_info->n_turning_points) {
            uint16_t chunk_len_before_turning_point = MIN_VAL(cur_chunk_len, next_turning_point - offset);
            if (chunk_len_before_turning_point != cur_chunk_len) {
                next_turning_point = cur_slot_info->turning_points[turning_point_idx];
                turning_point_idx++;
                next_state_flipped = 1;
            }
            cur_chunk_len = chunk_len_before_turning_point;
        }
#endif
        MY_ASSERT(cur_chunk_len != 0);
        chunk_handler(offset, cur_chunk_len, state_bit, params);
#if INDIRECT_RECOVERY
        if (next_state_flipped) {
            state_bit ^= 1;
        }
#endif
    }
}

#if INDIRECT_RECOVERY
void find_initial_state_bit(int16_t* p_offset, uint8_t* p_turning_point_idx, uint16_t* p_next_turning_point, SlotInfo** p_slot_info, uint32_t initial_value_idx, Model* model, const ParameterInfo* param) {
    my_printf_debug("Initialize next_turning_point from output offset %d" NEWLINE, initial_value_idx);
    *p_offset = get_state_bit(model, param->slot) ? 0x4000 : 0;
    *p_turning_point_idx = 0;
    *p_next_turning_point = INVALID_TURNING_POINT;
    *p_slot_info = get_slot_info(model, param->slot);
    uint8_t next_turning_point_found = 0;
    if (!(*p_slot_info)) {
        return;
    }
    while (*p_turning_point_idx < (*p_slot_info)->n_turning_points) {
        *p_next_turning_point = (*p_slot_info)->turning_points[*p_turning_point_idx];
        (*p_turning_point_idx)++;
        if (*p_next_turning_point != INVALID_TURNING_POINT && *p_next_turning_point > initial_value_idx) {
            next_turning_point_found = 1;
            break;
        }
        *p_offset ^= 0x4000;
    }
    if (!next_turning_point_found) {
        *p_next_turning_point = INVALID_TURNING_POINT;
    }
    my_printf_debug("next_turning_point = %d" NEWLINE, *p_next_turning_point);
}

void check_next_turning_point_inner(int16_t* p_offset, uint8_t* p_turning_point_idx, uint16_t* p_next_turning_point, SlotInfo* slot_info, uint16_t value_idx) {
    *p_offset ^= 0x4000;
    uint8_t next_turning_point_found = 0;
    while (*p_turning_point_idx < slot_info->n_turning_points) {
        *p_next_turning_point = slot_info->turning_points[*p_turning_point_idx];
        (*p_turning_point_idx)++;
        if (*p_next_turning_point != INVALID_TURNING_POINT && *p_next_turning_point >= value_idx) {
            next_turning_point_found = 1;
            break;
        }
        *p_offset ^= 0x4000;
    }
    if (!next_turning_point_found) {
        *p_next_turning_point = -1;
    }
    my_printf_debug("new offset=%d" NEWLINE, *p_offset);
}
#endif

void fix_first_unfinished_value_offset(const Model* model, uint32_t* p_first_unfinished_value_offset) {
#if !JAPARI
    if (BATCH_SIZE >= 2) {
        return;
    }
    // Force recovery from an even OFM index as most DSPLib function does not like odd dimensions
    if (*p_first_unfinished_value_offset % 2) {
        (*p_first_unfinished_value_offset)--;
#if HAWAII
        write_hawaii_layer_footprint(model->layer_idx, -1); // discard last job
#endif
    }
#endif
}

void make_buffer_aligned(int16_t** p_buffer) {
    if ((*p_buffer - lea_buffer) % 2) {
        (*p_buffer)++;
    }
}

float q15_to_float(int16_t val, const ValueInfo& val_info, uint8_t* p_use_prefix) {
#if STATEFUL
    if (val != -0x8000) {
        if (val < -0x2000) {
            // happens in the last value of each filter column (state embedding)
            val += 0x4000;
            MY_ASSERT(p_use_prefix != nullptr);
            *p_use_prefix = 1;
        }
        if (get_value_state_bit(val)) {
            // 2^15
            val -= 0x4000;
        }
    }
#endif
    return val_info.scale * static_cast<int32_t>(val) / 32768.0;
}

void my_offset_q15_batched(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize) {
    MY_ASSERT(pSrc == pDst);
    if (BATCH_SIZE == 1) {
        my_offset_q15(pSrc, offset, pDst, blockSize);
    } else {
        for (uint8_t val_idx = BATCH_SIZE - 1; val_idx < blockSize; val_idx += BATCH_SIZE) {
            pDst[val_idx] += offset;
        }
    }
}
