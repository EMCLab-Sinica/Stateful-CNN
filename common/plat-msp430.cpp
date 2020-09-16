#include <driverlib.h>
#ifdef __MSP430__
#include <msp430.h>
#include "main.h"
#elif defined(__MSP432__)
#include <msp432.h>
#endif
#include <stdint.h>
#include <string.h>
#include "Tools/ext_fram/extfram.h"
#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "platform.h"
#include "data.h"
#include "my_debug.h"

/* on FRAM */

#define EXTERNAL_FRAM

#ifdef EXTERNAL_FRAM
static uint32_t intermediate_values_offset(uint8_t slot_id) {
    // Currently only intermediate values are on external FRAM, so starting
    // from FRAM address 0.
    return 0 + slot_id * INTERMEDIATE_VALUES_SIZE;
}
#else
#pragma DATA_SECTION(".nvm2")
static uint8_t _intermediate_values[NUM_SLOTS][INTERMEDIATE_VALUES_SIZE];
static uint8_t *intermediate_values(uint8_t slot_id) {
    return _intermediate_values[slot_id];
}
#endif

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
extern "C" void TA1_0_IRQHandler(void)
#endif
{
    // one tick is configured as roughly 1 millisecond
    // See vApplicationSetupTimerInterrupt() in main.h and FreeRTOSConfig.h
    counters()->time_counters[counters()->counter_idx]++;
#ifdef __MSP432__
    MAP_Timer_A_clearCaptureCompareInterrupt(TIMER_A1_BASE, TIMER_A_CAPTURECOMPARE_REGISTER_0);
#endif
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
    DMA_setSrcAddress(MY_DMA_CHANNEL, (uint32_t)src, DMA_DIRECTION_INCREMENT);
    DMA_setDstAddress(MY_DMA_CHANNEL, (uint32_t)dest, DMA_DIRECTION_INCREMENT);
    /* transfer size is in words (2 bytes) */
    DMA0SZ = n >> 1;
    // DMA_enableInterrupt(MY_DMA_CHANNEL);
    // _3 => increment
    DMA0CTL |= DMAEN + DMA_TRANSFER_BLOCK;
    DMA0CTL |= DMAREQ;
#elif defined(__MSP432__)
    MAP_DMA_enableModule();
    MAP_DMA_setControlBase(controlTable);
    MAP_DMA_setChannelControl(
        DMA_CH0_RESERVED0 | UDMA_PRI_SELECT, // Channel 0, PRImary channel
        // re-arbitrate after 1024 (maximum) items
        // an item is 16-bit
        UDMA_ARB_1024 | UDMA_SIZE_16 | UDMA_SRC_INC_16 | UDMA_DST_INC_16
    );
    // Use the first configurable DMA interrupt handler DMA_INT1_IRQHandler,
    // which is defined below (overriding weak symbol in startup*.c)
    MAP_DMA_assignInterrupt(DMA_INT1, 0);
    MAP_Interrupt_enableInterrupt(INT_DMA_INT1);
    MAP_Interrupt_disableSleepOnIsrExit();
    MAP_DMA_setChannelTransfer(
        DMA_CH0_RESERVED0 | UDMA_PRI_SELECT,
        UDMA_MODE_AUTO, // Set as auto mode with no need to retrigger after each arbitration
        const_cast<void*>(src), dest,
        n >> 1 // transfer size in items
    );
    curDMATransmitChannelNum = 0;
    MAP_DMA_enableChannel(0);
    MAP_DMA_requestSoftwareTransfer(0);
    while (MAP_DMA_isChannelEnabled(0)) {}
#endif
}

// XXX: consolidate common code between POSIX and MSP430
void my_memcpy_to_param(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n) {
#ifdef EXTERNAL_FRAM
    SPI_ADDR addr;
    addr.L = intermediate_values_offset(param->slot) + offset_in_word * sizeof(int16_t);
    SPI_WRITE(&addr, reinterpret_cast<const uint8_t*>(src), n);
#else
#error "TODO"
#endif
}

void my_memcpy_from_param(void *dest, struct ParameterInfo *param, uint16_t offset_in_word, size_t n) {
    if (param->slot >= SLOT_CONSTANTS_MIN) {
        uint32_t limit;
        const uint8_t *baseptr = get_param_base_pointer(param, &limit);
        uint32_t total_offset = param->params_offset + offset_in_word * sizeof(int16_t);
        MY_ASSERT(total_offset + n <= limit);
        my_memcpy(dest, baseptr + total_offset, n);
    } else {
#ifdef EXTERNAL_FRAM
        SPI_ADDR addr;
        addr.L = intermediate_values_offset(param->slot) + offset_in_word * sizeof(int16_t);
        SPI_READ(&addr, reinterpret_cast<uint8_t*>(dest), n);
#else
#error "TODO"
#endif
    }
}

void plat_print_results(void) {
}

[[ noreturn ]] void ERROR_OCCURRED(void) {
    for (;;) {
        __no_operation();
    }
}

#pragma DATA_SECTION(".nvm")
static uint8_t myFirstTime;

#define DELAY_START_SECONDS 0

#if DELAY_START_SECONDS > 0
#pragma DATA_SECTION(".nvm")
static uint32_t delay_counter;
#endif

void IntermittentCNNTest() {
    Model *model = (Model*)model_data;

#ifdef EXTERNAL_FRAM
    initSPI();
    // testSPI();
#endif

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

    SPI_ADDR addr;
    addr.L = intermediate_values_offset(0);
    SPI_FILL_Q15(&addr, 0, INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);

#if DELAY_START_SECONDS > 0
    while (delay_counter < DELAY_START_SECONDS) {
        my_printf("%d" NEWLINE, delay_counter);
        delay_counter++;
        __delay_cycles(16E6);
    }
#endif

    while (1) {
        run_cnn_tests(1);
    }
}

void button_pushed(uint16_t button1_status, uint16_t button2_status) {
    static uint8_t push_counter = 0;

    my_printf_debug("button1_status=%d button2_status=%d" NEWLINE, button1_status, button2_status);

    if (button1_status && button2_status) {
        // XXX: somehow interrupts for both buttons are triggered immediately after recovery
        return;
    }

    Model *model = (Model*)model_data;
    my_printf("%d" NEWLINE, model->run_counter);

    myFirstTime = 0;

    push_counter++;
}
