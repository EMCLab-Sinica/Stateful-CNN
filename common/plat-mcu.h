#pragma once

#include "data.h"
#include "tools/ext_fram/extfram.h"
#include <stdint.h>

#define PLAT_LABELS_DATA_LEN 1

#ifdef __MSP430__
#include <DSPLib.h>
static inline void plat_start_cpu_counter(void) {
    msp_benchmarkStart(MSP_BENCHMARK_BASE, 1);
}

static inline uint32_t plat_stop_cpu_counter(void) {
    return msp_benchmarkStop(MSP_BENCHMARK_BASE);
}
#elif defined(__MSP432__)
#include <msp.h>
extern uint32_t last_cyccnt;
// following codes borrowed from https://stackoverflow.com/questions/32610019/arm-m4-instructions-per-cycle-ipc-counters
static inline void plat_start_cpu_counter(void) {
    DWT->CTRL |= 1;
    last_cyccnt = DWT->CYCCNT;
}

static inline uint32_t plat_stop_cpu_counter(void) {
    uint32_t ret = DWT->CYCCNT - last_cyccnt;
    DWT->CTRL &= 0XFFFFFFFE;
    return ret;
}
#else
#define plat_start_cpu_counter()
#define plat_stop_cpu_counter() 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

void IntermittentCNNTest(void);
void button_pushed(uint16_t button1_status, uint16_t button2_status);

#ifdef __cplusplus
}
#endif
