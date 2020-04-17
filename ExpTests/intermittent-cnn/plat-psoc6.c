#include <stdint.h>
#include <string.h>
#include "cnn_common.h"
#include "platform.h"
#include "debug.h"
#ifdef CY_PSOC_CREATOR_USED
#  include "config.h"
#  ifdef CKPT
#    include "syscheckpoint.h"
#  endif
#endif

/* TODO: put them on Flash */

static uint8_t _intermediate_values[NUM_SLOTS * INTERMEDIATE_VALUES_SIZE];
uint8_t *intermediate_values() {
    return _intermediate_values;
}

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
