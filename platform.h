#pragma once

#include <stdint.h>

#if defined(CY_TARGET_DEVICE)
#define CYPRESS
#endif

#ifdef __MSP430__
#  include "plat-msp430.h"
#elif defined(CYPRESS)
#  include "plat-psoc6.h"
#else
#  include "plat-linux.h"
#endif

void setOutputValue(uint8_t value);
void registerCheckpointing(uint8_t *addr, size_t len);
// similar to double buffering
uint8_t *intermediate_values(uint8_t slot_id, uint8_t will_write);
