#pragma once

#include <stdint.h>
#include <stdlib.h>

#if defined(__MSP430__) || defined(__MSP432__)
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

[[ noreturn ]] void ERROR_OCCURRED(void);
void my_memcpy(void* dest, const void* src, size_t n);
void my_memcpy_to_param(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n);
void my_memcpy_from_param(void *dest, struct ParameterInfo *param, uint16_t offset_in_word, size_t n);
void plat_print_results(void);
void setOutputValue(uint8_t value);
void registerCheckpointing(uint8_t *addr, size_t len);
