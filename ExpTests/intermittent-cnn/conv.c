// disable debug code in DSPLib
//#define MSP_DISABLE_DIAGNOSTICS

#include <DSPLib.h>
#include "common.h"
#include "debug.h"
#include "op_handlers.h"

#ifdef __MSP430__
#include <FreeRTOS.h>
#include <croutine.h>
#define USE_CONCURRENT_CONV
#endif

#define configCONV_STACK_SIZE 100
#define NUM_TASKS 2

// TODO: make these adjustable on runtime
#define TILE_W 1
#define TILE_H 7

#ifdef USE_CONCURRENT_CONV
/* internal structure for msp_mac_q15() */
DSPLIB_DATA(msp_mac_params, 4)
MSP_LEA_MAC_PARAMS msp_mac_params[NUM_TASKS];
#endif

#define NUM_FILTERS 16
int16_t *input_buffer_addr[NUM_TASKS];
int16_t *next_input_buffer_addr;

typedef struct ConvTaskParams {
    ParameterInfo *conv_input;
    ParameterInfo *conv_filter;
    ParameterInfo *bias;
    ParameterInfo *output;
    uint16_t conv_idx;
    uint16_t output_h;
    uint16_t output_w;
    uint8_t first_filter;
    uint8_t output_h_offset;
} ConvTaskParams;

static ConvTaskParams arr_conv_params[NUM_TASKS];

static uint16_t arrH[NUM_TASKS], arrW[NUM_TASKS], arrkH[NUM_TASKS], arrkW[NUM_TASKS], arrCHANNEL[NUM_TASKS], arrOUTPUT_CHANNEL[NUM_TASKS];
static msp_mac_q15_params mac_params[NUM_TASKS];
static int16_t *filter_buffer_addr[NUM_FILTERS];  // filter index -> address
static int8_t cached_filter_idx[NUM_FILTERS];  // filter buffer id (0~filter_limit-1) -> filter index
static int8_t filter_buffer_id;
static uint8_t next_scheduled_task_idx = 0;
static uint8_t pending_filters[NUM_FILTERS];
static uint8_t pending_filter_idx = 0;

static inline int32_t *buffer_iq31_mac_results(uint8_t uxIndex) {
    return (int32_t*)(lea_buffer + LEA_BUFFER_SIZE) - (uxIndex + 1);
}

#ifdef USE_CONCURRENT_CONV
static uint32_t idleCounter = 0;

static void convTaskConcurrent(CoRoutineHandle_t xHandle, UBaseType_t uxIndex) {
    #include "conv_prologue.h"

    MSP_LEA_MAC_PARAMS *leaParams;
    //uint16_t interruptState;

    __bis_SR_register(GIE);

    crSTART(xHandle);

    for (;;) {

    #include "conv_pre.h"

    /* XXX: need more co-routines? */
    while(!msp_lea_ifg) {
        idleCounter++;
    }

    /* modified from DSPLib_1_30_00_02/source/vector/msp_mac_q15.c */
    // different tasks need different buffers for LEA params, or the program
    // counter goes to strange places (e.g., 0x18)
    leaParams = msp_mac_params + uxIndex;

    /* Set MSP_LEA_MAC_PARAMS structure. */
    leaParams->input2 = MSP_LEA_CONVERT_ADDRESS(filter_buffer_addr[conv_params->conv_idx]);
    leaParams->output = MSP_LEA_CONVERT_ADDRESS(buffer_iq31_mac_results(uxIndex));
    leaParams->vectorSize = mac_params[uxIndex].length;

    /* Load source arguments to LEA. */
    LEAPMS0 = MSP_LEA_CONVERT_ADDRESS(input_buffer_addr[uxIndex]);
    LEAPMS1 = MSP_LEA_CONVERT_ADDRESS(leaParams);

    // modified from DSPLib_1_30_00_02/include/DSPLib_lea.h

    /* Save interrupt state and disable interrupts. */
    //interruptState = __get_interrupt_state();
    //__disable_interrupt();

    /* Clear interrupt flag and invoke the command. */
    msp_lea_ifg = 0;
    /* Invoke the LEACMD__MAC command. */
    LEAPMCB = LEACMD__MAC | LEAITFLG1;

    /* Do not enter LPM0 so that CPU can do other work */
    __bis_SR_register(GIE);

    /* Restore original interrupt state. */
    //__set_interrupt_state(interruptState);

    crDELAY(xHandle, 0);

    /* after context switch of co-routines, variables on stack are lost */
    conv_params = &arr_conv_params[uxIndex];
    H = arrH[uxIndex];
    W = arrW[uxIndex];
    kH = arrkH[uxIndex];
    kW = arrkW[uxIndex];
    CHANNEL = arrCHANNEL[uxIndex];
    OUTPUT_CHANNEL = arrOUTPUT_CHANNEL[uxIndex];

    #include "conv_post.h"

    crDELAY(xHandle, 0);

    }

    crEND();
}

#else

static void convTask(unsigned short uxIndex) {
    #include "conv_prologue.h"

    #include "conv_pre.h"

    msp_status status = msp_mac_q15(&mac_params[uxIndex],
                                    input_buffer_addr[uxIndex],
                                    filter_buffer_addr[conv_params->conv_idx],
                                    buffer_iq31_mac_results(uxIndex));
    msp_checkStatus(status);

    #include "conv_post.h"
}

#endif

static inline void increment_task_idx(void) {
    next_scheduled_task_idx++;
    if (next_scheduled_task_idx == NUM_TASKS) {
        next_scheduled_task_idx = 0;
    }
}

static inline void schedule_tile(uint16_t idx, uint16_t output_h, uint16_t output_w, uint8_t tile_h, uint8_t tile_w, uint8_t first_filter) {
    for (uint8_t i = 0; i < tile_w; i++) {
        for (uint8_t j = 0; j < tile_h; j += NUM_TASKS) {
            uint8_t k = 0, original_next_scheduled_task_idx = next_scheduled_task_idx;
            for (; k < NUM_TASKS && j + k < tile_h; k++) {
                ConvTaskParams *conv_params = &arr_conv_params[next_scheduled_task_idx];
                conv_params->conv_idx = idx;
                conv_params->output_h = output_h + j + k;
                conv_params->output_w = output_w + i;
                conv_params->first_filter = first_filter;
                conv_params->output_h_offset = j + k;
                increment_task_idx();
            }
            next_scheduled_task_idx = original_next_scheduled_task_idx;
            for (uint8_t k2 = 0; k2 < k; k2++) {
#ifdef USE_CONCURRENT_CONV
                /* each co-routine runs as two parts:
                 * before and after LEA operations */
                vCoRoutineSchedule();
                vCoRoutineSchedule();
#else
                convTask(next_scheduled_task_idx);
                increment_task_idx();
#endif
            }
        }
    }
}

static inline void handle_conv_inner_loop(uint16_t n_conv, uint16_t output_h, uint16_t output_w, uint8_t tile_h, uint8_t tile_w) {
    uint8_t first_filter = 1;
    for (uint8_t idx = 0; idx < n_conv; idx++) {
        if (filter_buffer_addr[idx]) {
            schedule_tile(idx, output_h, output_w, tile_h, tile_w, first_filter);
            first_filter = 0;
        } else {
            pending_filters[pending_filter_idx] = idx;
            pending_filter_idx++;
        }
    }
    for (uint8_t idx = 0; idx < pending_filter_idx; idx++) {
        schedule_tile(pending_filters[idx], output_h, output_w, tile_h, tile_w, first_filter);
        first_filter = 0;
    }
    pending_filter_idx = 0;
}

uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output) {
    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
    my_printf_debug("Conv!" NEWLINE);

    msp_mac_q15_overflow_counter = 0;

#ifdef USE_CONCURRENT_CONV
    msp_lea_init();

    msp_lea_ifg = 1; // dummy

    static bool task_created = false;

    if (!task_created) {
        for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
            if (xCoRoutineCreate(convTaskConcurrent, 0, idx) != pdPASS) {
                // failed to create co-routines
                ERROR_OCCURRED();
            }
        }
        task_created = true;
    }
#endif

    if (get_param_bitwidth(conv_input) != 16 || get_param_bitwidth(conv_filter) != 16) {
        // incorrect bitwidth
        ERROR_OCCURRED();
    }
    /* original: input: N x C x H x W, filter: M x C x kW x kW
     * remapped: input: N x H x W x C, filter: M x kH x kW x C */
    const uint16_t H = conv_input->dims[1], W = conv_input->dims[2],
                   input_N = conv_filter->dims[0];
    /* XXX: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
    output->params_len = (uint16_t)(input_N * H * W * 2);
    output->bitwidth_and_flags = 16 << FLAG_SLOTS_WIDTH | get_next_slot(conv_input);
    output->dims[0] = 1;
    output->dims[1] = H;
    output->dims[2] = W;
    output->dims[3] = input_N;

    uint8_t ret = 0;

    uint32_t start, end;
    start = getElapsedMilliseconds();

    for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
        ConvTaskParams *conv_params = &arr_conv_params[idx];
        conv_params->conv_input = conv_input;
        conv_params->conv_filter = conv_filter;
        conv_params->bias = bias;
        conv_params->output = output;
        input_buffer_addr[idx] = NULL;
        next_input_buffer_addr = NULL;
    }

    for (uint8_t idx = 0; idx < NUM_FILTERS; idx++) {
        filter_buffer_addr[idx] = NULL;
        cached_filter_idx[idx] = -1;
    }
    filter_buffer_id = 0;

    for (uint16_t output_w = 0; output_w < W; output_w += TILE_W) {
        for (uint16_t output_h = 0; output_h < H; output_h += TILE_H) {
            handle_conv_inner_loop(input_N, output_h, output_w, TILE_H, TILE_W);
        }
    }

    end = getElapsedMilliseconds();
    counters[counter_idx] = end - start;
    counter_idx++;
#ifdef USE_CONCURRENT_CONV
    my_printf("idle for %l cycles" NEWLINE, idleCounter);
#endif

    my_printf_debug("handle_conv output" NEWLINE);
    dump_params(output);

    my_printf("msp_mac_q15_overflow_counter=%d" NEWLINE, msp_mac_q15_overflow_counter);

    return ret;
}
