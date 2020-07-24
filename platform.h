#pragma once

#include <stdint.h>
#include <stdlib.h>

#if defined(__MSP430__) || defined(__MSP432__)
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

#define MY_ASSERT(cond) if (!(cond)) { ERROR_OCCURRED(); }

[[ noreturn ]] void ERROR_OCCURRED(void);
void my_memcpy(void* dest, const void* src, size_t n);
void fill_int16(int16_t *dest, uint16_t n, int16_t val);
void plat_print_results(void);
void setOutputValue(uint8_t value);
void registerCheckpointing(uint8_t *addr, size_t len);
// similar to double buffering
uint8_t *intermediate_values(uint8_t slot_id);
