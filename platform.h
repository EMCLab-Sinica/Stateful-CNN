#pragma once

#include <stdint.h>

#if defined(CY_TARGET_DEVICE) || defined(CY_PSOC_CREATOR_USED)
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
