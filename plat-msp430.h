#pragma once

#include <driverlib.h>
#include <msp430.h> /* __no_operation() */
#include <stdlib.h>

#define ERROR_OCCURRED() for (;;) { __no_operation(); }

#define LEA_BUFFER_SIZE 1884 // (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)

#define NVM_BYTE_ADDRESSABLE 1

#define USE_ALL_SAMPLES 0

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
