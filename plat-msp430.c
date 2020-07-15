#include <driverlib.h>
#ifdef __MSP430__
#include <msp430.h>
#include "main.h"
#elif defined(__MSP432__)
#include <msp432.h>
#endif
#include <stdint.h>
#include <string.h>
#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "platform.h"
#include "data.h"
#include "debug.h"

/* on FRAM */

//#define EXTERNAL_FRAM

#ifdef EXTERNAL_FRAM
// TODO
static uint8_t *_intermediate_values;
#else
#pragma DATA_SECTION(_intermediate_values, ".nvm2")
static uint8_t _intermediate_values[NUM_SLOTS * INTERMEDIATE_VALUES_SIZE];
#endif
uint8_t *intermediate_values(uint8_t slot_id) {
    return _intermediate_values + slot_id * INTERMEDIATE_VALUES_SIZE;
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

// broken if n has type size_t
void fill_int16(int16_t *dest, uint16_t n, int16_t val) {
#ifdef __MSP430__
    DMA_init(&dma_params);
    DMA0SA = &val;
    DMA0DA = dest;
    /* transfer size is in words (2 bytes) */
    DMA0SZ = n;
    // DMA_enableInterrupt(MY_DMA_CHANNEL);
    // _0 => unchanged
    // _3 => increment
    DMA0CTL |= DMAEN + DMASRCINCR_0 + DMADSTINCR_3 + DMA_TRANSFER_BLOCK;
    DMA0CTL |= DMAREQ;
#else
#error "TODO: implement fill_int16 for MSP432"
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

#pragma DATA_SECTION(myFirstTime, ".nvm")
static uint8_t myFirstTime;

#define DELAY_START_SECONDS 0

#if DELAY_START_SECONDS > 0
#pragma DATA_SECTION(myFirstTime, ".nvm")
static uint32_t delay_counter;
#endif

void IntermittentCNNTest() {
    Model *model = (Model*)model_data;

    if (myFirstTime != 1) {
#if DELAY_START_SECONDS > 0
        delay_counter = 0;
#endif

        for (uint8_t i = 0; i < COUNTERS_LEN; i++) {
            counters()->time_counters[i] = 0;
            counters()->power_counters[i] = 0;
        }

        myFirstTime = 1;
        model->run_counter = 0;
    }

#if DELAY_START_SECONDS > 0
    while (delay_counter < DELAY_START_SECONDS) {
        my_printf("%d" NEWLINE, delay_counter);
        delay_counter++;
        __delay_cycles(16E6);
    }
#endif

    if (!model->run_counter) {
        run_cnn_tests(1);
    }

    while (1) {
        __delay_cycles(16E6);
    }
}

void button_pushed(void) {
    static uint8_t push_counter = 0;
    // XXX: somehow the ISR for button is triggered immediately after recovery
    if (push_counter >= 1) {
        myFirstTime = 0;
    }
    push_counter++;
}
