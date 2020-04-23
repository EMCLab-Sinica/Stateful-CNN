#include <stdint.h>
#include <string.h>
#include "cnn_common.h"
#include "platform.h"
#include "debug.h"
#ifdef CY_PSOC_CREATOR_USED
#  include "config.h"
#  include "SharedDB.h"
#  ifdef CKPT
#    include "syscheckpoint.h"
#  endif
#endif

/* TODO: put them on Flash */

#if !defined(WITH_FAILURE_RESILIENT_OS) || defined(CKPT)
static uint8_t _intermediate_values[NUM_SLOTS * INTERMEDIATE_VALUES_SIZE];
uint8_t *intermediate_values(uint8_t slot_id, uint8_t will_write) {
    return _intermediate_values + slot_id * INTERMEDIATE_VALUES_SIZE;
}
// intermediate values are not managed by data manager for CKPT
void commit_intermediate_values(ParameterInfo *param) {};
#else
uint16_t curTaskID;
struct DBImage *DB;
uint8_t *intermediate_values(uint8_t slot_id, uint8_t will_write) {
    typedef void* (*accessor_type)(struct DBImage* DB, int taskID, int objID, uint16_t size);
    accessor_type accessor;
    if (will_write) {
        accessor = getWorking;
    } else {
        accessor = read;
    }
    switch (slot_id) {
        case 0:
            return accessor(DB, curTaskID, OBJ_INTERMEDIATE_VALUES_1, INTERMEDIATE_VALUES_SIZE);
        case 1:
            return accessor(DB, curTaskID, OBJ_INTERMEDIATE_VALUES_2, INTERMEDIATE_VALUES_SIZE);
        default:
            ERROR_OCCURRED();
    }
}
void commit_intermediate_values(ParameterInfo *param) {
    int objId;
    switch(param->slot) {
        case 0:
            objId = OBJ_INTERMEDIATE_VALUES_1;
            break;
        case 1:
            objId = OBJ_INTERMEDIATE_VALUES_2;
            break;
        default:
            ERROR_OCCURRED();
    }
    commit(DB, curTaskID, &objId, 1, INTERMEDIATE_VALUES_SIZE, 1);
}
#endif

Counters *counters() {
    return (Counters*)counters_data;
}

void setOutputValue(uint8_t value)
{
    my_printf_debug("Output set to %d" NEWLINE, value);
}

void vTimerHandler(void) {
    counters()->time_counters[counters()->counter_idx]++;
}

void my_memcpy(void* dest, const void* src, size_t n) {
    memcpy(dest, src, n);
}

void registerCheckpointing(uint8_t *addr, size_t len) {
#ifdef CKPT
    syscheckpoint_register(addr, len);
#endif
}
