#include <driverlib.h>
#include <stdint.h>
#include "FreeRTOSConfig.h"
#include "cnn_common.h"
#include "platform.h"

/* on FRAM */

#pragma DATA_SECTION(_intermediate_values, ".nvm")
static uint8_t _intermediate_values[INTERMEDIATE_VALUES_SIZE];
uint8_t *intermediate_values(void) {
    return _intermediate_values;
}

#pragma DATA_SECTION(_counters, ".nvm")
static Counters _counters;
Counters *counters() {
    return &_counters;
}

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

#pragma vector=configTICK_VECTOR
__interrupt void vTimerHandler( void )
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

#define MY_DMA_CHANNEL DMA_CHANNEL_0
static DMA_initParam dma_params = {
    .channelSelect = MY_DMA_CHANNEL,
};

void my_memcpy(void* dest, const void* src, size_t n) {
    DMA_init(&dma_params); // XXX: DMA not working without this
    DMA0SA = src;
    DMA0DA = dest;
    /* transfer size is in words (2 bytes) */
    DMA0SZ = n >> 1;
    // DMA_enableInterrupt(MY_DMA_CHANNEL);
    // _3 => increment
    DMA0CTL |= DMAEN + DMASRCINCR_3 + DMADSTINCR_3 + DMA_TRANSFER_BLOCK;
    DMA0CTL |= DMAREQ;
}
