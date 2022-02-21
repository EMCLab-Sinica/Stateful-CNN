#pragma once

#include "my_debug.h"
#include "cnn_common.h"
#include <cstdint>

#define ENABLE_COUNTERS 0

#if ENABLE_COUNTERS
// pointer to member https://stackoverflow.com/questions/670734/pointer-to-class-data-member
extern uint8_t current_counter;
extern uint8_t prev_counter;
const uint8_t INVALID_POINTER = 0xff;

#if MY_DEBUG >= MY_DEBUG_VERBOSE
static size_t get_counter_offset(uint32_t Counters::* mem_ptr) {
    Counters dummy;
    return reinterpret_cast<size_t>(&(&dummy->*mem_ptr)) - reinterpret_cast<size_t>(&dummy);
}
#endif

static inline void add_counter(uint8_t counter, uint32_t value) {
    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(counters(get_model()->layer_idx)) + counter) += value;
}

static inline void start_cpu_counter(uint8_t mem_ptr) {
    MY_ASSERT(prev_counter == INVALID_POINTER);

    if (current_counter != INVALID_POINTER) {
        prev_counter = current_counter;
        add_counter(prev_counter, plat_stop_cpu_counter());
        my_printf_debug("Stopping outer CPU counter %lu" NEWLINE, get_counter_offset(prev_counter));
    }
    my_printf_debug("Start CPU counter %lu" NEWLINE, get_counter_offset(mem_ptr));
    current_counter = mem_ptr;
    plat_start_cpu_counter();
}

static inline void stop_cpu_counter(void) {
    MY_ASSERT(current_counter != INVALID_POINTER);

    my_printf_debug("Stop inner CPU counter %lu" NEWLINE, get_counter_offset(current_counter));
    add_counter(current_counter, plat_stop_cpu_counter());
    if (prev_counter != INVALID_POINTER) {
        current_counter = prev_counter;
        my_printf_debug("Restarting outer CPU counter %lu" NEWLINE, get_counter_offset(current_counter));
        plat_start_cpu_counter();
        prev_counter = INVALID_POINTER;
    } else {
        current_counter = INVALID_POINTER;
    }
}
#else
#define start_cpu_counter(mem_ptr)
#define stop_cpu_counter()
#endif
