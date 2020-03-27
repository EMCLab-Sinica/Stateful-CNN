#pragma once

#include <string.h>
#include "cy_utils.h" // CY_ASSERT

#define ERROR_OCCURRED() do { CY_ASSERT(0); } while (0);

void vTimerHandler(void);

static inline void my_memcpy(void* dest, const void* src, size_t n) {
    memcpy(dest, src, n);
}
