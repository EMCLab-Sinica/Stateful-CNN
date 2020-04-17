#pragma once

#include <stddef.h> // size_t
#ifdef CY_PSOC_CREATOR_USED
#include <FreeRTOS.h>
#include <task.h>
#endif

#ifdef CY_PSOC_CREATOR_USED
#define ERROR_OCCURRED() do { vTaskSuspendAll(); while (1) {} } while (0);
#else
#define ERROR_OCCURRED() do { while (1) {} } while (0);
#endif

// larger than conv needed
#define LEA_BUFFER_SIZE 4096

#define NVM_BYTE_ADDRESSABLE 0

#define USE_ALL_SAMPLES 1

void vTimerHandler(void);

void my_memcpy(void* dest, const void* src, size_t n);
