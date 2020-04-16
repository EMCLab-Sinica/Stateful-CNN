#pragma once

#include <stddef.h> // size_t
// CY_ASSERT
#ifdef CY_PSOC_CREATOR_USED
#include <syslib/cy_syslib.h>
#else
#include <cy_syslib.h>
#endif

#define ERROR_OCCURRED() do { CY_ASSERT(0); } while (0);

// much larger than conv needed
#define LEA_BUFFER_SIZE 16384

#define NVM_BYTE_ADDRESSABLE 0

#define USE_ALL_SAMPLES 1

void vTimerHandler(void);

void my_memcpy(void* dest, const void* src, size_t n);
