#include "DSPLib.h"
#include "Tools/myuart.h"
#include <math.h>

#ifndef MSP_USE_LEA
#error "LEA not enabled!"
#endif

/* Input signal parameters */
#define FS                  8192
#define SAMPLES             200
#define SIGNAL_FREQUENCY    200
#define SIGNAL_AMPLITUDE    0.6
/* Constants */
#define PI                  3.1415926536
/* Input vector */
DSPLIB_DATA(input,4)
_q15 input[SAMPLES];
/* Result maximum value */
_q15 q15MaxVector;
/* Index of result */
uint16_t uint16MaxIndex;
/* Benchmark cycle counts */
volatile uint32_t cycleCount;
/* Function prototypes */
extern void initSignal(void);


void LEATest() {
    msp_status status;
    msp_max_q15_params maxParams;

    /* Disable WDT. */
    WDTCTL = WDTPW + WDTHOLD;
#ifdef __MSP430_HAS_PMM__
    /* Disable GPIO power-on default high-impedance mode for FRAM devices */
    PM5CTL0 &= ~LOCKLPM5;
#endif
    /* Initialize input signal */
    initSignal();

    while(1) {
        /* Initialize the parameter structure. */
        maxParams.length = SAMPLES;

        /* Invoke the msp_max_q15 API. */
        //msp_benchmarkStart(MSP_BENCHMARK_BASE, 1);
        status = msp_max_q15(&maxParams, input, &q15MaxVector, &uint16MaxIndex);
        //cycleCount = msp_benchmarkStop(MSP_BENCHMARK_BASE);
        msp_checkStatus(status);
        print2uart("LEATest: %d\n", q15MaxVector);
    }
}

void initSignal(void) {
    msp_status status;
    msp_sinusoid_q15_params sinParams;
    /* Generate Q15 input signal */
    sinParams.length = SAMPLES;
    sinParams.amplitude = _Q15(SIGNAL_AMPLITUDE);
    sinParams.cosOmega = _Q15(cosf(2*PI*SIGNAL_FREQUENCY/FS));
    sinParams.sinOmega = _Q15(sinf(2*PI*SIGNAL_FREQUENCY/FS));
    status = msp_sinusoid_q15(&sinParams, input);
    msp_checkStatus(status);
}
