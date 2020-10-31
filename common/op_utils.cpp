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
        uint16_t bound = 8192;
        int16_t val_offset = param_state_bit(model, param, offset) ? -16384 : 0;
#endif
        if (!*max_multiplier) {
            *max_multiplier = bound;
        }

        int16_t max_val, min_val;
        uint16_t index;

        const int16_t* cur_buffer;
        if (!buffer) {
            my_memcpy_from_param(model, lea_buffer, param, offset, real_chunk_len * sizeof(int16_t));
            cur_buffer = lea_buffer;
        } else {
            cur_buffer = buffer + offset;
        }

        // dump_matrix(cur_buffer, real_chunk_len, ValueInfo(param));

        my_max_q15(cur_buffer, real_chunk_len, &max_val, &index);
#if STATEFUL_CNN
        max_val += val_offset;
#endif
        my_printf_debug("Max value %d", max_val);
        my_printf_debug(" occurs at index %d" NEWLINE, index);
        while (max_val && abs(max_val) * (*max_multiplier) >= bound) {
            (*max_multiplier) /= 2;
        }

        my_min_q15(cur_buffer, real_chunk_len, &min_val, &index);
#if STATEFUL_CNN
        min_val += val_offset;
#endif
        my_printf_debug("Min value %d", min_val);
        my_printf_debug(" occurs at index %d" NEWLINE, index);
        while (min_val && abs(min_val) * (*max_multiplier) >= bound) {
            (*max_multiplier) /= 2;
        }
        my_printf_debug("max_multiplier = %d" NEWLINE, *max_multiplier);
    }
private:
    Model *model;
    const ParameterInfo *param;
    const int16_t* buffer;
    uint16_t *max_multiplier;
};

uint16_t find_max_multiplier(Model *model, const ParameterInfo *param, const int16_t* buffer) {
    uint16_t max_multiplier = 0;

    iterate_chunks(model, param, 0, 0, MaxMultiplierChunkHandler(model, param, buffer, &max_multiplier));

    my_printf_debug("max_multiplier=%d" NEWLINE, max_multiplier);

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
    // TODO: determine these values automatically
    uint16_t CHANNEL = param->dims[1], H = param->dims[2];
    uint16_t kH = 0, INPUT_CHANNEL = 0;
    if (filter) {
        INPUT_CHANNEL = filter->dims[1];
        kH = filter->dims[2];
    }
    if (H == 14 && CHANNEL == 8) {
        param->tile_c = 3;
    } else if (H == 15 && CHANNEL == 64) {
        param->tile_c = 32;
    } else if (H == 7 && CHANNEL == 64 && kH == 3) {
        param->tile_c = 6;
    } else if (H == 7 && CHANNEL == 32 && INPUT_CHANNEL == 128 && kH == 1) {
        param->tile_c = 16;
    } else if (H == 7 && CHANNEL == 128 && INPUT_CHANNEL == 32 && kH == 1) {
        param->tile_c = 44;
    } else if (H == 7 && CHANNEL == 128 && INPUT_CHANNEL == 32 && kH == 3) {
        param->tile_c = 2;
    } else if (INPUT_CHANNEL == 256 && kH == 1) {
        param->tile_c = 4;
    }
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
