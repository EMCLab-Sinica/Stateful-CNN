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

void OutputChunkHandler::handle_chunk(uint32_t offset, uint16_t real_chunk_len, uint8_t state_bit) const {
    if (!state_bit) {
        int16_t* to_offset = buffer + offset;
        my_offset_q15(to_offset, 0x4000, to_offset, real_chunk_len);
    }
}

class MaxMultiplierChunkHandler : public ChunkHandler {
public:
    MaxMultiplierChunkHandler(Model *_model, const ParameterInfo *_param, const int16_t* _buffer, uint16_t *_max_multiplier)
        : model(_model), param(_param), buffer(_buffer), max_multiplier(_max_multiplier) {}

    void handle_chunk(uint32_t offset, uint16_t real_chunk_len, uint8_t state_bit) const override {
#if !STATEFUL_CNN
        uint16_t bound = 32768;
#else
        // For stateful CNN, values should not reach 8192, or get_value_state_bit() is confused
        uint16_t bound = 8191;
        int16_t val_offset = 0;
        if (!buffer) {
            // if the buffer is pre-filled (e.g., in GemmMerge), its state bits
            // should already be stripped, too
            val_offset = param_state_bit(model, param, offset) ? -16384 : 0;
        }
#endif
        int16_t max_val, min_val;
        // Apparently TI's compiler does not handle multiplication between
        // int16_t and uint16_t correctly. Use unsigned everywhere to fix it.
        uint32_t u_max_val, u_min_val;
        uint16_t index;

        const int16_t* cur_buffer;
        if (!buffer) {
            my_memcpy_from_param(model, lea_buffer, param, offset, real_chunk_len * sizeof(int16_t));
            cur_buffer = lea_buffer;
        } else {
            cur_buffer = buffer + offset;
        }

        dump_matrix_debug(cur_buffer, real_chunk_len, ValueInfo(param));

        my_max_q15(cur_buffer, real_chunk_len, &max_val, &index);
#if STATEFUL_CNN
        max_val += val_offset;
#endif
        my_printf_debug("Max value %d", max_val);
        my_printf_debug(" occurs at index %d" NEWLINE, index);
        u_max_val = abs(max_val);
        // use > instead of >= as the value may be exactly on the bound
        while (max_val && u_max_val * (*max_multiplier) > bound) {
            (*max_multiplier) /= 2;
        }

        my_min_q15(cur_buffer, real_chunk_len, &min_val, &index);
#if STATEFUL_CNN
        min_val += val_offset;
#endif
        my_printf_debug("Min value %d", min_val);
        my_printf_debug(" occurs at index %d" NEWLINE, index);
        u_min_val = abs(min_val);
        while (min_val && u_min_val * (*max_multiplier) > bound) {
            (*max_multiplier) /= 2;
        }
        my_printf_debug("Current max_multiplier=%d" NEWLINE, *max_multiplier);
    }
private:
    Model *model;
    const ParameterInfo *param;
    const int16_t* buffer;
    uint16_t *max_multiplier;
};

uint16_t find_max_multiplier(Model *model, const ParameterInfo *param, const int16_t* buffer, uint16_t len) {
    uint16_t max_multiplier = param->scale;

    iterate_chunks(model, param, 0, len, MaxMultiplierChunkHandler(model, param, buffer, &max_multiplier));

    my_printf_debug("max_multiplier=%d" NEWLINE, max_multiplier);

    MY_ASSERT(max_multiplier != 0);

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

void iterate_chunks(Model *model, const ParameterInfo *param, uint16_t start_offset, uint16_t len, const ChunkHandler& chunk_handler) {
    uint16_t params_len;
    if (!len) {
        params_len = param->params_len / sizeof(int16_t);
    } else {
        params_len = start_offset + len;
    }
    uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / 2 * 2);
    uint8_t state_bit = 0;

    uint16_t cur_chunk_len;
#if STATEFUL_CNN
    dump_turning_points_debug(model, param);

    state_bit = get_state_bit(model, param->slot);
    uint8_t turning_point_idx = 0;
    int16_t next_turning_point = -1;
    SlotInfo *cur_slot_info = get_slot_info(model, param->slot);
    uint16_t n_turning_points = cur_slot_info ? cur_slot_info->n_turning_points : 0;
    uint8_t turning_point_found = 0;
    while (turning_point_idx < n_turning_points) {
        next_turning_point = cur_slot_info->turning_points[turning_point_idx];
        turning_point_idx++;
        if (next_turning_point > start_offset) {
            turning_point_found = 1;
            break;
        }
        state_bit ^= 1;
    }
    if (!turning_point_found) {
        // no turning points not after start_offset found
        next_turning_point = -1;
    }
#endif
    for (uint32_t offset = start_offset; offset < params_len; offset += cur_chunk_len) {
        cur_chunk_len = MIN_VAL(chunk_len, params_len - offset);
#if STATEFUL_CNN
        uint8_t next_state_flipped = 0;
        // Use <= here as turning_point_idx is actually index for the _next_ turning point
        if (next_turning_point > 0 && turning_point_idx <= cur_slot_info->n_turning_points) {
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
        chunk_handler.handle_chunk(offset, cur_chunk_len, state_bit);
#if STATEFUL_CNN
        if (next_state_flipped) {
            state_bit ^= 1;
        }
#endif
    }
}

void determine_tile_c(ParameterInfo *param, const ParameterInfo *filter) {
    uint16_t OUTPUT_CHANNEL = param->dims[1];
    param->tile_c = MIN_VAL(DEFAULT_TILE_C, OUTPUT_CHANNEL);
}

#if STATEFUL_CNN
void find_initial_state_bit(int16_t* p_offset, uint8_t* p_turning_point_idx, int16_t* p_next_turning_point, SlotInfo** p_slot_info, uint32_t initial_value_idx, Model* model, const ParameterInfo* param) {
    *p_offset = get_state_bit(model, param->slot) ? 0x4000 : 0;
    *p_turning_point_idx = 0;
    *p_next_turning_point = -1;
    *p_slot_info = get_slot_info(model, param->slot);
    uint8_t next_turning_point_found = 0;
    if (!(*p_slot_info)) {
        return;
    }
    while (*p_turning_point_idx < (*p_slot_info)->n_turning_points) {
        *p_next_turning_point = (*p_slot_info)->turning_points[*p_turning_point_idx];
        (*p_turning_point_idx)++;
        if (*p_next_turning_point > static_cast<int16_t>(initial_value_idx)) {
            next_turning_point_found = 1;
            break;
        }
        *p_offset ^= 0x4000;
    }
    if (!next_turning_point_found) {
        *p_next_turning_point = -1;
    }
}

void check_next_turning_point_inner(int16_t* p_offset, uint8_t* p_turning_point_idx, int16_t* p_next_turning_point, SlotInfo* slot_info, uint16_t value_idx) {
    *p_offset ^= 0x4000;
    uint8_t next_turning_point_found = 0;
    while (*p_turning_point_idx < slot_info->n_turning_points) {
        *p_next_turning_point = slot_info->turning_points[*p_turning_point_idx];
        (*p_turning_point_idx)++;
        if (*p_next_turning_point >= value_idx) {
            next_turning_point_found = 1;
            break;
        }
        *p_offset ^= 0x4000;
    }
    if (!next_turning_point_found) {
        *p_next_turning_point = -1;
    }
}
#endif

