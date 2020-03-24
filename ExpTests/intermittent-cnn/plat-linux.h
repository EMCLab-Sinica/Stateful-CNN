#pragma once

#include <stdio.h>
#include <inttypes.h>
#include <signal.h>
#include <string.h>
#include "debug.h"

#define ERROR_OCCURRED() do { raise(SIGINT); } while (0);

#define MEMCPY_DELAY_US 0

extern uint32_t *copied_size;

static inline void my_memcpy(void* dest, const void* src, size_t n) {
    *copied_size += n;
#if MEMCPY_DELAY_US
    usleep(MEMCPY_DELAY_US);
#endif
    my_printf_debug(__func__);
    my_printf_debug(" copied %d bytes" NEWLINE, (int)n);
    memcpy(dest, src, n);
}
