#pragma once

#include <stddef.h> // size_t

#define ERROR_OCCURRED() do { while (1) {} } while (0);

// larger than conv needed
#define LEA_BUFFER_SIZE 4096

#define USE_ALL_SAMPLES 1

void vTimerHandler(void);

void my_memcpy(void* dest, const void* src, size_t n);
