#pragma once

#include <stdint.h>

#ifdef __MSP430__
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

void my_memcpy(void* dest, const void* src, size_t n);
