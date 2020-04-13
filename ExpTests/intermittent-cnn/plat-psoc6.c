#include <stdint.h>
#include "cnn_common.h"
#include "platform.h"
#include "debug.h"

/* TODO: put them on Flash */

static uint8_t _intermediate_values[NUM_SLOTS * INTERMEDIATE_VALUES_SIZE];
uint8_t *intermediate_values = _intermediate_values;

static uint16_t _counters[COUNTERS_LEN];
uint16_t *counters = _counters;

static uint16_t _power_counters[COUNTERS_LEN];
uint16_t *power_counters = _power_counters;

static uint8_t _counter_idx;
uint8_t *counter_idx = &_counter_idx;

void setOutputValue(uint8_t value)
{
    my_printf_debug("Output set to %d" NEWLINE, value);
}

void vTimerHandler(void) {
    counters[*counter_idx]++;
}
