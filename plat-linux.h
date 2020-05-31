#pragma once

#include <inttypes.h>
#include <signal.h>

#define ERROR_OCCURRED() do { raise(SIGINT); } while (0);
#define TASK_FINISHED()

#define MEMCPY_DELAY_US 0

#define LEA_BUFFER_SIZE 1884

// USE_ALL_SAMPLES must be 1 as nvm.bin contains all samples
#define USE_ALL_SAMPLES 1

extern uint32_t *copied_size;

void my_memcpy(void* dest, const void* src, size_t n);
