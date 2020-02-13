#include <stdint.h>
#include "common.h"
/* on FRAM */

#pragma NOINIT(_intermediate_values)
static uint8_t _intermediate_values[NUM_SLOTS * INTERMEDIATE_VALUES_SIZE];
uint8_t *intermediate_values = _intermediate_values;
