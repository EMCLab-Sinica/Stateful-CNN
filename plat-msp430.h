#pragma once

#include <driverlib.h>
#include <msp430.h> /* __no_operation() */
#include <stdlib.h>

#define ERROR_OCCURRED() for (;;) { __no_operation(); }

#define TASK_FINISHED()

#define LEA_BUFFER_SIZE 1884 // (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)

#define USE_ALL_SAMPLES 0

void my_memcpy(void* dest, const void* src, size_t n);
