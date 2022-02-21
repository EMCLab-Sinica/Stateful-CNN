#pragma once

#include "data.h"
#include "Tools/ext_fram/extfram.h"
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
#endif

#ifdef __cplusplus
extern "C" {
#endif

void IntermittentCNNTest(void);
void button_pushed(uint16_t button1_status, uint16_t button2_status);

#ifdef __cplusplus
}
#endif
