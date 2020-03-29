#pragma once

#include <driverlib.h>
#include <msp430.h> /* __no_operation() */

#define ERROR_OCCURRED() for (;;) { __no_operation(); }
// _Pragma() is a C99 feature
// https://stackoverflow.com/a/3030312/3786245
#define ON_NVM(var_name) _Pragma("DATA_SECTION(" var_name ", \".map\")")

#define MY_DMA_CHANNEL DMA_CHANNEL_0
static DMA_initParam dma_params = {
    .channelSelect = MY_DMA_CHANNEL,
};

static inline void my_memcpy(void* dest, const void* src, size_t n) {
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
