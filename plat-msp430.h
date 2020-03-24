#pragma once

#include <driverlib.h>
#include <msp430.h> /* __no_operation() */

#define ERROR_OCCURRED() for (;;) { __no_operation(); }
// _Pragma() is a C99 feature
// https://stackoverflow.com/a/3030312/3786245
#define ON_NVM(var_name) _Pragma("DATA_SECTION(var_name, \".map\")")

#define MY_DMA_CHANNEL DMA_CHANNEL_0
static DMA_initParam dma_params = {
    .channelSelect = MY_DMA_CHANNEL,
    .transferModeSelect = DMA_TRANSFER_BLOCK,
};

static inline void my_memcpy(void* dest, const void* src, size_t n) {
    DMA_init(&dma_params);
    DMA_setSrcAddress(MY_DMA_CHANNEL, (uint32_t)(src), DMA_DIRECTION_INCREMENT);
    DMA_setDstAddress(MY_DMA_CHANNEL, (uint32_t)(dest), DMA_DIRECTION_INCREMENT);
    /* transfer size is in words (2 bytes) */
    DMA_setTransferSize(MY_DMA_CHANNEL, (n) >> 1);
    // DMA_enableInterrupt(MY_DMA_CHANNEL);
    DMA_enableTransfers(MY_DMA_CHANNEL);
    DMA_startTransfer(MY_DMA_CHANNEL);
}

