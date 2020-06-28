#pragma once

#include <stdint.h>

#ifdef __MSP430__
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

#define MY_ASSERT(cond) if (!(cond)) { ERROR_OCCURRED(); }

void plat_print_results(void);
void setOutputValue(uint8_t value);
void registerCheckpointing(uint8_t *addr, size_t len);
// similar to double buffering
uint8_t *intermediate_values(void);
