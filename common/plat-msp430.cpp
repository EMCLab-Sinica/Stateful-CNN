#include <driverlib.h>
#ifdef __MSP430__
#include <msp430.h>
#include "main.h"
#elif defined(__MSP432__)
#include <msp432.h>
#endif
#include <cstdint>
#include <string.h>
#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "platform.h"
#include "platform-private.h"
#include "data.h"
#include "my_debug.h"
#include "Tools/myuart.h"
#include "Tools/our_misc.h"
#include "Tools/dvfs.h"

static Counters counters_data;
Counters *counters() {
    return &counters_data;
}

#ifdef __MSP430__

#define MY_DMA_CHANNEL DMA_CHANNEL_0

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
    counters()->time_counters[model_vm.layer_idx]++;
#ifdef __MSP432__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast" // the macro TIMER_A1_BASE ends up with an old-style cast
    MAP_Timer_A_clearCaptureCompareInterrupt(TIMER_A1_BASE, TIMER_A_CAPTURECOMPARE_REGISTER_0);
#pragma GCC diagnostic pop
#endif
}

void my_memcpy(void* dest, const void* src, size_t n) {
#ifdef __MSP430__
    DMA0CTL = 0;

    DMACTL0 &= 0xFF00;
    // set DMA transfer trigger for channel 0
    DMACTL0 |= DMA0TSEL__DMAREQ;

    DMA_setSrcAddress(MY_DMA_CHANNEL, (uint32_t)src, DMA_DIRECTION_INCREMENT);
    DMA_setDstAddress(MY_DMA_CHANNEL, (uint32_t)dest, DMA_DIRECTION_INCREMENT);
    /* transfer size is in words (2 bytes) */
    DMA0SZ = n >> 1;
    DMA0CTL |= DMAEN + DMA_TRANSFER_BLOCK + DMA_SIZE_SRCWORD_DSTWORD;
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

void read_from_nvm(void* vm_buffer, uint32_t nvm_offset, size_t n) {
    SPI_ADDR addr;
    addr.L = nvm_offset;
    SPI_READ(&addr, reinterpret_cast<uint8_t*>(vm_buffer), n);
}

void write_to_nvm(const void* vm_buffer, uint32_t nvm_offset, size_t n, uint16_t timer_delay) {
    SPI_ADDR addr;
    addr.L = nvm_offset;
    check_nvm_write_address(nvm_offset, n);
    MY_ASSERT(n <= 1024);
    SPI_WRITE2(&addr, reinterpret_cast<const uint8_t*>(vm_buffer), n, timer_delay);
}

uint64_t get_nvm_writes(void) {
    return 0;
}

void my_erase() {
    eraseFRAM();
}

void copy_samples_data(void) {
    write_to_nvm_segmented(samples_data, SAMPLES_OFFSET, SAMPLES_DATA_LEN);
}

[[ noreturn ]] void ERROR_OCCURRED(void) {
    while (1);
}

#ifdef __MSP430__
#define GPIO_COUNTER_PORT GPIO_PORT_P8
#define GPIO_COUNTER_PIN GPIO_PIN0
#define GPIO_RESET_PORT GPIO_PORT_P5
#define GPIO_RESET_PIN GPIO_PIN7
#else
#define GPIO_COUNTER_PORT GPIO_PORT_P5
#define GPIO_COUNTER_PIN GPIO_PIN5
#define GPIO_RESET_PORT GPIO_PORT_P2
#define GPIO_RESET_PIN GPIO_PIN5
#endif

#define STABLE_POWER_ITERATIONS 10

void IntermittentCNNTest() {
    GPIO_setAsOutputPin(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
    GPIO_setOutputLowOnPin(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
    GPIO_setAsInputPinWithPullUpResistor(GPIO_RESET_PORT, GPIO_RESET_PIN);

    // sleep to wait for external FRAM
    // 5ms / (1/f)
    our_delay_cycles(5E-3 * getFrequency(FreqLevel));

    initSPI();
    if (testSPI() != 0) {
        // external FRAM failed to initialize - reset
        volatile uint16_t counter = 1000;
        // waiting some time seems to increase the possibility
        // of a successful FRAM initialization on next boot
        while (counter--);
        WDTCTL = 0;
    }

    Model* model = get_model();
    if (!GPIO_getInputPinValue(GPIO_RESET_PORT, GPIO_RESET_PIN)) {
        uartinit();

        my_printf(NEWLINE "run_counter = %d" NEWLINE, model->run_counter);

        first_run();

        // The first iteration takes longer as it needs to compute layer multipliers
        for (uint8_t idx = 0; idx < 1 + STABLE_POWER_ITERATIONS; idx++) {
            run_cnn_tests(1);
        }

        my_printf("Done testing run" NEWLINE);

        while (1);
    }

    while (1) {
        run_cnn_tests(1);
    }
}

void button_pushed(uint16_t button1_status, uint16_t button2_status) {
    my_printf_debug("button1_status=%d button2_status=%d" NEWLINE, button1_status, button2_status);
}

void notify_model_finished(void) {
    my_printf("." NEWLINE);
    GPIO_toggleOutputOnPin(GPIO_COUNTER_PORT, GPIO_COUNTER_PIN);
}
