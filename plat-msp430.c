#include <driverlib.h>
#ifdef __MSP430__
#include <msp430.h>
#include "main.h"
#elif defined(__MSP432__)
#include <msp432.h>
#endif
#include <stdint.h>
#include <string.h>
#include "cnn_common.h"
#include "platform.h"
#include "data.h"

/* on FRAM */

#define EXTERNAL_FRAM

#ifdef EXTERNAL_FRAM
#pragma DATA_SECTION(_intermediate_values, ".nvm")
// TODO
static uint8_t *_intermediate_values;
#else
static uint8_t _intermediate_values[INTERMEDIATE_VALUES_SIZE];
#endif
uint8_t *intermediate_values(void) {
    return _intermediate_values;
}

Counters *counters() {
    return (Counters*)counters_data;
}

#ifdef __MSP430__

#define MY_DMA_CHANNEL DMA_CHANNEL_0
static DMA_initParam dma_params = {
    .channelSelect = MY_DMA_CHANNEL,
};

#pragma vector=DMA_VECTOR
__interrupt void DMA_ISR(void)
{
    switch(__even_in_range(DMAIV,16))
    {
        case 0: break;
        case 2: break; // DMA0IFG = DMA Channel 0
        case 4: break; // DMA1IFG = DMA Channel 1
        case 6: break; // DMA2IFG = DMA Channel 2
        case 8: break; // DMA3IFG = DMA Channel 3
        case 10: break; // DMA4IFG = DMA Channel 4
        case 12: break; // DMA5IFG = DMA Channel 5
        case 14: break; // DMA6IFG = DMA Channel 6
        case 16: break; // DMA7IFG = DMA Channel 7
        default: break;
    }
}

#endif

#ifdef __MSP430__
#pragma vector=configTICK_VECTOR
__interrupt void vTimerHandler( void )
#elif defined(__MSP432__)
void TA0_N_IRQHandler(void)
#endif
{
    // one tick is configured as roughly 1 millisecond
    // See vApplicationSetupTimerInterrupt() in main.h and FreeRTOSConfig.h
    counters()->time_counters[counters()->counter_idx]++;
}

void setOutputValue(uint8_t value)
{
    if (value) {
        GPIO_setOutputHighOnPin(GPIO_PORT_P1, GPIO_PIN3);
    } else {
        GPIO_setOutputLowOnPin(GPIO_PORT_P1, GPIO_PIN3);
    }
}

void my_memcpy(void* dest, const void* src, size_t n) {
#ifdef __MSP430__
    DMA_init(&dma_params); // XXX: DMA not working without this
    DMA0SA = src;
    DMA0DA = dest;
    /* transfer size is in words (2 bytes) */
    DMA0SZ = n >> 1;
    // DMA_enableInterrupt(MY_DMA_CHANNEL);
    // _3 => increment
    DMA0CTL |= DMAEN + DMASRCINCR_3 + DMADSTINCR_3 + DMA_TRANSFER_BLOCK;
    DMA0CTL |= DMAREQ;
#elif defined(__MSP432__)
    // TODO: use DMA
    memcpy(dest, src, n);
#endif
}

void plat_print_results(void) {
}

_Noreturn void ERROR_OCCURRED(void) {
    for (;;) {
        __no_operation();
    }
}

#ifdef __MSP432__
// MSP430 intrinsic used by DSPLib
// http://downloads.ti.com/docs/esd/SLAU132/msp430-intrinsics-slau1321420.html
short __saturated_add_signed_short(short src1, short src2) {
    int sum = src1 + src2;
    if (sum > INT16_MAX) {
        sum = INT16_MAX;
    }
    if (sum < INT16_MIN) {
        sum = INT16_MIN;
    }
    return (short)sum;
}
#endif
