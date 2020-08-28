#pragma once

#include "platform.h"
#include "cnn_common.h"
#include "intermittent-cnn.h"

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
uint16_t find_overflow_factor(struct Model *model, struct ParameterInfo *param);
void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale);

template<typename T>
void iterate_chunks(Model *model, ParameterInfo *param, uint16_t start_offset, uint16_t len, T callback) {
    uint16_t params_len;
    if (!len) {
        params_len = param->params_len / sizeof(int16_t);
    } else {
        params_len = start_offset + len;
    }
    uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / 2 * 2);
    uint8_t state_bit = 0;

    uint16_t cur_chunk_len;
#ifdef WITH_PROGRESS_EMBEDDING
    dump_turning_points(param);

    state_bit = get_state_bit(model, param->slot);
    uint8_t turning_point_idx = 0;
    int16_t next_turning_point = -1;
    SlotInfo *cur_slot_info = get_slot_info(param->slot);
    while (turning_point_idx < cur_slot_info->n_turning_points) {
        next_turning_point = cur_slot_info->turning_points[turning_point_idx];
        turning_point_idx++;
        if (next_turning_point > start_offset) {
            break;
        }
        state_bit ^= 1;
    }
#endif
    for (uint32_t offset = start_offset; offset < params_len; offset += cur_chunk_len) {
        cur_chunk_len = MIN_VAL(chunk_len, params_len - offset);
#ifdef WITH_PROGRESS_EMBEDDING
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
        // TODO: sometimes cur_chunk_len becomes 0 - fix it
        // MY_ASSERT(cur_chunk_len);
        callback(offset, cur_chunk_len, state_bit);
#ifdef WITH_PROGRESS_EMBEDDING
        if (next_state_flipped) {
            state_bit ^= 1;
        }
#endif
    }
}
