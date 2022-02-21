#include <cstdint>
#include "my_debug.h"
#include "op_utils.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"
#include "cnn_common.h"
#include "platform.h"
#include "counters.h"

// Not using DSPLIB_DATA here as it does not work under C++ (?)
#ifdef __MSP430__
#pragma DATA_SECTION(".leaRAM")
#endif
int16_t lea_buffer[LEA_BUFFER_SIZE];

#if HAWAII
static int16_t non_recorded_jobs = 0;
void hawaii_record_footprints(Model* model, uint16_t vector_len) {
    non_recorded_jobs += vector_len;
    for (; non_recorded_jobs >= BATCH_SIZE; non_recorded_jobs -= BATCH_SIZE) {
        write_hawaii_layer_footprint(model->layer_idx, BATCH_SIZE);
    }
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
void OutputChunkHandler(uint32_t offset, uint16_t real_chunk_len, int8_t state_bit, void* _params) {
    OutputChunkHandlerParams* params = reinterpret_cast<OutputChunkHandlerParams*>(_params);
    int16_t* to_offset = params->buffer + (offset - params->buffer_offset);
#if STATEFUL
    my_offset_q15_batched(to_offset, -state_bit*0x4000, to_offset, real_chunk_len, true);
#endif
#if JAPARI
    for (uint16_t idx = BATCH_SIZE; idx < real_chunk_len; idx += BATCH_SIZE + 1) {
        to_offset[idx] = -state_bit;
    }
#endif
}
#endif

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
    start_cpu_counter(offsetof(Counters, state_query));
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
        state_bit = -state_bit;
    }
    if (!turning_point_found) {
        // no turning points not after start_offset found
        next_turning_point = INVALID_TURNING_POINT;
    }
    stop_cpu_counter();
#endif
    for (uint32_t offset = start_offset; offset < params_len; offset += cur_chunk_len) {
        cur_chunk_len = MIN_VAL(chunk_len, params_len - offset);
#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, state_query));
        uint8_t next_state_flipped = 0;
        // Use <= here as turning_point_idx is actually index for the _next_ turning point
        if (next_turning_point != INVALID_TURNING_POINT && turning_point_idx <= cur_slot_info->n_turning_points) {
            uint16_t chunk_len_before_turning_point = MIN_VAL(cur_chunk_len, next_turning_point - offset);
            if (offset + chunk_len >= next_turning_point) {
                next_turning_point = cur_slot_info->turning_points[turning_point_idx];
                turning_point_idx++;
                next_state_flipped = 1;
            }
            cur_chunk_len = chunk_len_before_turning_point;
        }
        stop_cpu_counter();
#endif
        MY_ASSERT(cur_chunk_len != 0);
        chunk_handler(offset, cur_chunk_len, state_bit, params);
#if INDIRECT_RECOVERY
        start_cpu_counter(offsetof(Counters, state_query));
        if (next_state_flipped) {
            state_bit = -state_bit;
        }
        stop_cpu_counter();
#endif
    }
}

#if INDIRECT_RECOVERY
void find_initial_state_bit(int16_t* p_offset, uint8_t* p_turning_point_idx, uint16_t* p_next_turning_point, SlotInfo** p_slot_info, uint32_t initial_value_idx, Model* model, const ParameterInfo* param) {
    start_cpu_counter(offsetof(Counters, state_query));
    my_printf_debug("Initialize next_turning_point from data offset %d" NEWLINE, initial_value_idx);
    *p_offset = get_state_bit(model, param->slot)*0x4000;
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
        *p_offset = -*p_offset;
    }
    if (!next_turning_point_found) {
        *p_next_turning_point = INVALID_TURNING_POINT;
    }
    my_printf_debug("next_turning_point = %d" NEWLINE, *p_next_turning_point);
    stop_cpu_counter();
}

void check_next_turning_point(int16_t& offset, uint8_t& turning_point_idx, uint16_t& next_turning_point, SlotInfo* slot_info, uint16_t value_idx) {
    start_cpu_counter(offsetof(Counters, state_query));
    uint8_t next_turning_point_found = 0;
    if (next_turning_point == INVALID_TURNING_POINT || value_idx < next_turning_point) {
        goto exit;
    }
    my_printf_debug("Checking next turning point after %d" NEWLINE, value_idx);
    offset = -offset;
    while (turning_point_idx < slot_info->n_turning_points) {
        next_turning_point = slot_info->turning_points[turning_point_idx];
        turning_point_idx++;
        if (next_turning_point != INVALID_TURNING_POINT && next_turning_point >= value_idx) {
            next_turning_point_found = 1;
            break;
        }
        offset = -offset;
    }
    if (!next_turning_point_found) {
        next_turning_point = static_cast<uint16_t>(-1);
    }
    my_printf_debug("new offset=%d" NEWLINE, offset);
exit:
    stop_cpu_counter();
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

float q15_to_float(int16_t val, const ValueInfo& val_info, uint8_t* p_use_prefix, bool has_state) {
#if STATEFUL
    if (has_state) {
        strip_state(&val);
    }
#endif
    return val_info.scale * static_cast<int32_t>(val) / 32768.0;
}

void my_offset_q15_batched(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize, bool enforce_states) {
    MY_ASSERT(pSrc == pDst);
    if (BATCH_SIZE == 1) {
        my_offset_q15(pSrc, offset, pDst, blockSize);
#if STATEFUL
        if (enforce_states) {
            uint16_t mask = offset - 0x4000;
            int16_t* end = pDst + blockSize;
            for (int16_t* ptr = pDst; ptr < end; ptr++) {
                *ptr = (*ptr & 0x7fff) | mask;
            }
        }
#endif
    } else {
        for (uint32_t val_idx = BATCH_SIZE - 1; val_idx < blockSize; val_idx += BATCH_SIZE) {
            pDst[val_idx] += offset;
#if STATEFUL
            if (enforce_states) {
                uint16_t mask = offset - 0x4000;
                pDst[val_idx] = (pDst[val_idx] & 0x7fff) | mask;
            }
#endif
        }
    }
}

#if INDIRECT_RECOVERY
uint16_t update_states(int16_t* buffer, uint16_t buffer_size, uint32_t offset, int16_t embedding_offset, uint16_t next_turning_point, bool enforce_states) {
    start_cpu_counter(offsetof(Counters, embedding));
    uint16_t buffer_size_first = MIN_VAL(next_turning_point - offset, buffer_size);
    MY_ASSERT(buffer_size_first <= buffer_size);
#if STATEFUL
    my_offset_q15_batched(buffer, -embedding_offset, buffer, buffer_size_first, enforce_states);
#else
    for (uint16_t j = BATCH_SIZE; j < buffer_size; j += BATCH_SIZE + 1) {
        buffer[j] = (-embedding_offset > 0) ? 1 : -1;
    }
#endif
    if (buffer_size_first != buffer_size) {
        int16_t* to_offset = buffer + buffer_size_first;
#if STATEFUL
        my_offset_q15_batched(to_offset, embedding_offset, to_offset, buffer_size - buffer_size_first, enforce_states);
#else
        for (uint16_t j = BATCH_SIZE; j < buffer_size - buffer_size_first; j += BATCH_SIZE + 1) {
            to_offset[j] = (embedding_offset > 0) ? 1 : -1;
        }
#endif
    }
    stop_cpu_counter();
    return buffer_size_first;
}
#endif

#if JAPARI
// https://tjsw.medium.com/86f06ac768da
template<uint8_t move_from, uint8_t batch_offset, std::enable_if_t<move_from < BATCH_SIZE>* = nullptr>
static inline void move_filter(int16_t*) {}

template<uint8_t move_from, uint8_t batch_offset, std::enable_if_t<move_from >= BATCH_SIZE>* = nullptr>
static inline void move_filter(int16_t* filter) {
    const uint8_t move_to = move_from/BATCH_SIZE*(BATCH_SIZE+1)+batch_offset;
    filter[move_to] = filter[move_from];
    move_filter<move_from-1, (batch_offset >= 1) ? (batch_offset-1) : (BATCH_SIZE-1)>(filter);
}

template<uint8_t offset>
static inline void clear_filter(int16_t* filter) {
    filter[offset] = 0;
    clear_filter<offset-(BATCH_SIZE+1)>(filter);
}

template<>
inline void clear_filter<BATCH_SIZE>(int16_t* filter) {
    filter[BATCH_SIZE] = 0;
}

void move_weights(int16_t* filter_ptr, bool exact_tile, int16_t values_to_preserve, int16_t tile_width) {
    // move loaded filters around to create zeros for footprint kernels
    start_cpu_counter(offsetof(Counters, embedding));
    if (!exact_tile) {
        int16_t move_offset = values_to_preserve - tile_width;
        int16_t cur_remaining = values_to_preserve % (BATCH_SIZE + 1);
        for (int16_t move_dest = values_to_preserve - 1; move_dest >= 0; move_dest--) {
            if (cur_remaining == 0) {
                filter_ptr[move_dest] = 0;
                move_offset--;
                cur_remaining = BATCH_SIZE;
                continue;
            }
            filter_ptr[move_dest] = filter_ptr[move_dest - move_offset];
            cur_remaining--;
        }
    } else {
        const uint8_t last_elem = OP_FILTERS-1;
        move_filter<last_elem, last_elem % BATCH_SIZE>(filter_ptr);
        clear_filter<last_elem/(BATCH_SIZE+1)*(BATCH_SIZE+1)+BATCH_SIZE>(filter_ptr);
    }
    stop_cpu_counter();
}
#endif
