#pragma once

#include <stdint.h>

#ifdef __MSP430__
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

#define MY_ASSERT(cond) if (!(cond)) { ERROR_OCCURRED(); }

void setOutputValue(uint8_t value);
void registerCheckpointing(uint8_t *addr, size_t len);
// similar to double buffering
uint8_t *intermediate_values(uint8_t slot_id, uint8_t will_write);
