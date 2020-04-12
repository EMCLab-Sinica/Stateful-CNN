#pragma once

#include <string.h>
#include "cy_utils.h" // CY_ASSERT

#define ERROR_OCCURRED() do { CY_ASSERT(0); } while (0);

// much larger than conv needed
#define LEA_BUFFER_SIZE 16384

#define NVM_BYTE_ADDRESSABLE 0

#define USE_ALL_SAMPLES 1

void vTimerHandler(void);

static inline void my_memcpy(void* dest, const void* src, size_t n) {
    memcpy(dest, src, n);
}
