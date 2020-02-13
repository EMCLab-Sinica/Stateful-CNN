#pragma once

#ifdef __MSP430__
#  include <stddef.h>
void my_memcpy(void* dest, const void* src, size_t n);
#else
#  include <string.h>
#  define my_memcpy memcpy
#endif
