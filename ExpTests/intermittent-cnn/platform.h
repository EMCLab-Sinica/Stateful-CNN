#pragma once

#include <stdint.h>

#ifdef __MSP430__
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

void setOutputValue(uint8_t value);
