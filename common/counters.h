#pragma once

#include "my_debug.h"
#include "cnn_common.h"
#include <cstdint>

// Counter pointers have the form offsetof(Counter, field_name). I use offsetof() instead of
// pointers to member fields like https://stackoverflow.com/questions/670734/pointer-to-class-data-member
// as the latter involves pointer arithmetic and is slower for platforms with special pointer bitwidths (ex: MSP430)
#if ENABLE_COUNTERS
extern uint8_t current_counter;
extern uint8_t prev_counter;
const uint8_t INVALID_POINTER = 0xff;

static inline void add_counter(uint8_t counter, uint32_t value) {
    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(counters(get_model()->layer_idx)) + counter) += value;
}

static inline void start_cpu_counter(uint8_t mem_ptr) {
    MY_ASSERT(prev_counter == INVALID_POINTER, "There is already two counters - prev_counter=%d, current_counter=%d", prev_counter, current_counter);

    if (current_counter != INVALID_POINTER) {
        prev_counter = current_counter;
        add_counter(prev_counter, plat_stop_cpu_counter());
        my_printf_debug("Stopping outer CPU counter %d" NEWLINE, prev_counter);
    }
    my_printf_debug("Start CPU counter %d" NEWLINE, mem_ptr);
    current_counter = mem_ptr;
    plat_start_cpu_counter();
}

static inline void stop_cpu_counter(void) {
    MY_ASSERT(current_counter != INVALID_POINTER);

    my_printf_debug("Stop inner CPU counter %d" NEWLINE, current_counter);
    add_counter(current_counter, plat_stop_cpu_counter());
    if (prev_counter != INVALID_POINTER) {
        current_counter = prev_counter;
        my_printf_debug("Restarting outer CPU counter %d" NEWLINE, current_counter);
        plat_start_cpu_counter();
        prev_counter = INVALID_POINTER;
    } else {
        current_counter = INVALID_POINTER;
    }
}

void print_all_counters();

#else
#define start_cpu_counter(mem_ptr)
#define stop_cpu_counter()
#define print_all_counters()
#endif
