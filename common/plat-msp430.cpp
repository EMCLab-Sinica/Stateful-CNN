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

/* external FRAM layout:
 * 0, +NUM_SLOTS * INTERMEDIATE_VALUES_SIZE: intermediate values
 * INTERMEDIATE_PARAMETERS_INFO_OFFSET, +INTERMEDIATE_PARAMETERS_INFO_DATA_LEN: intermediate parameters info
 * MODEL_OFFSET, +2 * sizeof(Model): two shadow copies of Model
 * FIRST_RUN_OFFSET, +sizeof(uint8_t): first run?
 */

#define INTERMEDIATE_PARAMETERS_INFO_OFFSET 0x70000
#define MODEL_OFFSET 0x72000
#define FIRST_RUN_OFFSET 0x72400

static_assert(INTERMEDIATE_PARAMETERS_INFO_OFFSET > NUM_SLOTS * INTERMEDIATE_VALUES_SIZE, "Incorrect external NVM layout");
static_assert(MODEL_OFFSET > INTERMEDIATE_PARAMETERS_INFO_OFFSET + INTERMEDIATE_PARAMETERS_INFO_DATA_LEN, "Incorrect external NVM layout");
static_assert(FIRST_RUN_OFFSET > MODEL_OFFSET + 2 * sizeof(Model), "Incorrect external NVM layout");

static uint32_t intermediate_values_offset(uint8_t slot_id) {
    return 0 + slot_id * INTERMEDIATE_VALUES_SIZE;
}

static uint32_t intermediate_parameters_info_addr(uint8_t i) {
    return INTERMEDIATE_PARAMETERS_INFO_OFFSET + i * sizeof(ParameterInfo);
}

static uint32_t model_addr(uint8_t i) {
    return MODEL_OFFSET + i * sizeof(Model);
}

static Counters counters_data;
Counters *counters() {
    return &counters_data;
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast" // the macro TIMER_A1_BASE ends up with an old-style cast
    MAP_Timer_A_clearCaptureCompareInterrupt(TIMER_A1_BASE, TIMER_A_CAPTURECOMPARE_REGISTER_0);
#pragma GCC diagnostic pop
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
    SPI_ADDR addr;
    addr.L = intermediate_values_offset(param->slot) + offset_in_word * sizeof(int16_t);
    SPI_WRITE(&addr, reinterpret_cast<const uint8_t*>(src), n);
}

void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n) {
    SPI_ADDR addr;
    addr.L = intermediate_values_offset(param->slot) + offset_in_word * sizeof(int16_t);
    SPI_READ(&addr, reinterpret_cast<uint8_t*>(dest), n);
}

ParameterInfo* get_intermediate_parameter_info(uint8_t i) {
    ParameterInfo* dst = intermediate_parameters_info_vm + i;
    SPI_ADDR addr;
    addr.L = intermediate_parameters_info_addr(i);
    SPI_READ(&addr, reinterpret_cast<uint8_t*>(dst), sizeof(ParameterInfo));
    return dst;
}

void commit_intermediate_parameter_info(uint8_t i) {
    SPI_ADDR addr;
    addr.L = intermediate_parameters_info_addr(i);
    const ParameterInfo* src = intermediate_parameters_info_vm + i;
    SPI_WRITE(&addr, reinterpret_cast<const uint8_t*>(src), sizeof(ParameterInfo));
}

static uint8_t get_newer_model_copy_id_extfram(void) {
    uint16_t version1, version2;
    SPI_ADDR addr;
    addr.L = model_addr(0) + offsetof(Model, version);
    SPI_READ(&addr, reinterpret_cast<uint8_t*>(&version1), sizeof(uint16_t));
    addr.L = model_addr(1) + offsetof(Model, version);
    SPI_READ(&addr, reinterpret_cast<uint8_t*>(&version2), sizeof(uint16_t));
    my_printf_debug("Versions of shadow Model copies: %d, %d" NEWLINE, version1, version2);
    return get_newer_model_copy_id(version1, version2);
}

Model* get_model(void) {
    Model *dst = &model_vm;

    uint8_t newer_model_copy_id = get_newer_model_copy_id_extfram();
    SPI_ADDR addr;
    addr.L = model_addr(newer_model_copy_id);
    SPI_READ(&addr, reinterpret_cast<uint8_t*>(dst), sizeof(Model));
    my_printf_debug("Using model copy %d, version %d" NEWLINE, newer_model_copy_id, dst->version);
    return dst;
}

void commit_model(void) {
    uint8_t newer_model_copy_id = get_newer_model_copy_id_extfram();
    uint8_t older_model_copy_id = newer_model_copy_id ^ 1;

    bump_model_version(&model_vm);

    SPI_ADDR addr;
    addr.L = model_addr(older_model_copy_id);
    SPI_WRITE(&addr, reinterpret_cast<uint8_t*>(&model_vm), sizeof(Model));
    my_printf_debug("Committing version %d to model copy %d" NEWLINE, model_vm.version, older_model_copy_id);
}

void plat_print_results(void) {
}

[[ noreturn ]] void ERROR_OCCURRED(void) {
    for (;;) {
        __no_operation();
    }
}

#define DELAY_START_SECONDS 0

#if DELAY_START_SECONDS > 0
#pragma DATA_SECTION(".nvm")
static uint32_t delay_counter;
#endif

void IntermittentCNNTest() {
    initSPI();
    // testSPI();

    uint8_t first_run = 0;
    SPI_ADDR addr;
    addr.L = FIRST_RUN_OFFSET;
    SPI_READ(&addr, &first_run, 1);

    if (first_run) {
        my_printf_debug("First run, resetting everything..." NEWLINE);
#if DELAY_START_SECONDS > 0
        delay_counter = 0;
#endif

#if STATEFUL_CNN
        addr.L = intermediate_values_offset(0);
        SPI_FILL_Q15(&addr, 0, INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);
#endif
        addr.L = intermediate_parameters_info_addr(0);
        SPI_WRITE(&addr, intermediate_parameters_info_data, INTERMEDIATE_PARAMETERS_INFO_DATA_LEN);
        addr.L = model_addr(0);
        SPI_WRITE(&addr, model_data, MODEL_DATA_LEN);
        addr.L = model_addr(1);
        SPI_WRITE(&addr, model_data, MODEL_DATA_LEN);

        get_model(); // refresh model_vm
        commit_model();

        first_run = 0;
        addr.L = FIRST_RUN_OFFSET;
        SPI_WRITE(&addr, &first_run, 1);
    }

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
    my_printf_debug("button1_status=%d button2_status=%d" NEWLINE, button1_status, button2_status);

    if (button1_status && button2_status) {
        // XXX: somehow interrupts for both buttons are triggered immediately after recovery
        return;
    }

    uint8_t first_run = 1;
    SPI_ADDR addr;
    addr.L = FIRST_RUN_OFFSET;
    SPI_WRITE(&addr, &first_run, 1);

    Model *model = get_model();
    my_printf("%d" NEWLINE, model->run_counter);
}
