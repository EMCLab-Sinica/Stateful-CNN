// disable debug code in DSPLib
//#define MSP_DISABLE_DIAGNOSTICS

#include <string.h>

#include <DSPLib.h>

#ifdef __MSP430__
#include <driverlib.h>
#include <FreeRTOS.h>
#include <croutine.h>
#include "Tools/my_timer.h"
#define USE_DMA 1
#else
#define USE_DMA 0
#endif

#include "ops.h"
#include "op_handlers.h"
#include "common.h" // for MY_NDEBUG

#define configCONV_STACK_SIZE 100
#define NUM_TASKS 2
#define USE_CONCURRENT_CONV_BY_DEFAULT 1
#define CACHED_FILTERS
#define CACHED_INPUTS
#define INPUTS_LEN 760

#ifdef __MSP430__
#pragma DATA_SECTION(lea_buffer, ".leaRAM")
#endif
union {
    // for conv
    struct {
        int16_t input[NUM_TASKS][INPUTS_LEN];
        int16_t filter[200];
        int32_t iq31_mac_result[NUM_TASKS];
#ifdef __MSP430__
        MSP_LEA_MAC_PARAMS params[NUM_TASKS];
#endif
    } conv;
    // for others
    struct {
        int16_t A[256];
        int16_t B[256];
        int16_t arrC[256]; // the term C is reserved for MSP430 :/
        int16_t temp[64];
    } general;
} lea_buffer;

#ifdef CACHED_FILTERS
int8_t cached_filter_index;
#endif
#ifdef CACHED_INPUTS
int16_t *input_buffer_addr[NUM_TASKS];
int8_t input_buffer_w[NUM_TASKS];
#endif

uint16_t counters[10];
uint8_t counter_idx = 0;

typedef struct ConvTaskParams {
    ParameterInfo *conv_input;
    ParameterInfo *conv_filter;
    ParameterInfo *bias;
    ParameterInfo *output;
    uint16_t conv_idx;
    uint16_t output_h;
    uint16_t output_w;
} ConvTaskParams;

static ConvTaskParams arr_conv_params[NUM_TASKS];

#if USE_DMA
#define MY_DMA_CHANNEL DMA_CHANNEL_0
static DMA_initParam dma_params = {
    .channelSelect = MY_DMA_CHANNEL,
    .transferModeSelect = DMA_TRANSFER_BLOCK,
};
#endif

static void my_memcpy(void* dest, const void* src, size_t n) {
#if !USE_DMA
    memcpy(dest, src, n);
#else
    DMA_init(&dma_params);
    DMA_setSrcAddress(MY_DMA_CHANNEL, (uint32_t)(src), DMA_DIRECTION_INCREMENT);
    DMA_setDstAddress(MY_DMA_CHANNEL, (uint32_t)(dest), DMA_DIRECTION_INCREMENT);
    /* transfer size is in words (2 bytes) */
    DMA_setTransferSize(MY_DMA_CHANNEL, (n) >> 1);
    // DMA_enableInterrupt(MY_DMA_CHANNEL);
    DMA_enableTransfers(MY_DMA_CHANNEL);
    DMA_startTransfer(MY_DMA_CHANNEL);
#endif
}

static uint16_t arrH[NUM_TASKS], arrW[NUM_TASKS], arrkH[NUM_TASKS], arrkW[NUM_TASKS], arrCHANNEL[NUM_TASKS], arrOUTPUT_CHANNEL[NUM_TASKS];
static msp_mac_q15_params mac_params[NUM_TASKS];

uint8_t use_concurrent_conv = USE_CONCURRENT_CONV_BY_DEFAULT;

#if __MSP430__
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
    leaParams = lea_buffer.conv.params + uxIndex;

    /* Set MSP_LEA_MAC_PARAMS structure. */
    leaParams->input2 = MSP_LEA_CONVERT_ADDRESS(lea_buffer.conv.filter);
    leaParams->output = MSP_LEA_CONVERT_ADDRESS(&lea_buffer.conv.iq31_mac_result[uxIndex]);
    leaParams->vectorSize = mac_params[uxIndex].length;

    /* Load source arguments to LEA. */
#ifdef CACHED_INPUTS
    LEAPMS0 = MSP_LEA_CONVERT_ADDRESS(input_buffer_addr[uxIndex]);
#else
    LEAPMS0 = MSP_LEA_CONVERT_ADDRESS(lea_buffer.conv.input[uxIndex]);
#endif
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
#endif

static void convTask(unsigned short uxIndex) {
    #include "conv_prologue.h"

    #include "conv_pre.h"

    msp_status status = msp_mac_q15(&mac_params[uxIndex],
#ifdef CACHED_INPUTS
                                    input_buffer_addr[uxIndex],
#else
                                    lea_buffer.conv.input[uxIndex],
#endif
                                    lea_buffer.conv.filter,
                                    &lea_buffer.conv.iq31_mac_result[uxIndex]);
    msp_checkStatus(status);

    #include "conv_post.h"
}

#ifndef __MSP430__
// defined in DSPLib_1_30_00_02/source/vector/msp_mac_q15.c
extern uint32_t msp_mac_q15_overflow_counter;
#endif

uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output) {
    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
#ifndef MY_NDEBUG
    my_printf("Conv!" NEWLINE);
#endif


#ifndef __MSP430__
    msp_mac_q15_overflow_counter = 0;
#endif

    // use_concurrent_conv can be configured for different
    // convolution neurons here.

#ifdef __MSP430__
    if (use_concurrent_conv) {
        msp_lea_init();

        msp_lea_ifg = 1; // dummy

        static bool task_created = false;

        if (!task_created) {
            for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
                if (xCoRoutineCreate(convTaskConcurrent, 0, idx) != pdPASS) {
                    my_printf("Failed to create co-routines." NEWLINE);
                    ERROR_OCCURRED();
                }
            }
            task_created = true;
        }
    }
#endif

    if (get_param_bitwidth(conv_input) != 16 || get_param_bitwidth(conv_filter) != 16) {
        my_printf("Error: incorrect bitwidth." NEWLINE);
        return 1;
    }
    /* original: input: N x C x H x W, filter: M x C x kW x kW
     * remapped: input: N x H x W x C, filter: M x kH x kW x C */
    /* TODO: really use remapped dimensions */
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

#ifdef __MSP430__
    uint32_t start, end;
    start = getTickCounter();
#endif
    for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
        ConvTaskParams *conv_params = &arr_conv_params[idx];
        conv_params->conv_input = conv_input;
        conv_params->conv_filter = conv_filter;
        conv_params->bias = bias;
        conv_params->output = output;
#ifdef CACHED_INPUTS
        input_buffer_addr[idx] = NULL;
        input_buffer_w[idx] = -1;
#endif
    }
#ifdef CACHED_FILTERS
    cached_filter_index = -1;
#endif

    for (uint16_t conv_idx = 0; conv_idx < input_N; conv_idx++) {
        //my_printf("conv_idx = %d" NEWLINE, conv_idx);
        for (uint16_t output_w = 0; output_w < W; output_w += NUM_TASKS) {
            for (uint16_t output_h = 0; output_h < H; output_h++) {
                for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
                    ConvTaskParams *conv_params = &arr_conv_params[idx];
                    conv_params->conv_idx = conv_idx;
                    conv_params->output_h = output_h;
                    conv_params->output_w = output_w + idx;
                }
                for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
#ifdef __MSP430__
                    if (use_concurrent_conv) {
                        /* each co-routine runs as two parts:
                         * before and after LEA operations */
                        vCoRoutineSchedule();
                        vCoRoutineSchedule();
                    } else
#endif
                    {
                        convTask(idx);
                    }
                }
            }
        }
    }
#ifdef __MSP430__
    end = getTickCounter();
    counters[counter_idx] = end - start;
    counter_idx++;
    my_printf("idle for %l cycles" NEWLINE, idleCounter);
#endif

#ifdef DUMP__CONV_PARAMS
    my_printf("handle_conv output" NEWLINE);
    dump_params(output);
#endif

#ifndef __MSP430__
    my_printf("msp_mac_q15_overflow_counter=%d" NEWLINE, msp_mac_q15_overflow_counter);
#endif

    return ret;
}

uint8_t handle_maxpool(const uint16_t stride, ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("MaxPool!" NEWLINE);
#endif

    /* XXX: add flags; assume no padding for now */
    ParameterInfo *data = input[0];

#ifdef DUMP_MAXPOOL_PARAMS
    my_printf("handle_maxpool input" NEWLINE);
    dump_params(data);
#endif

    const uint16_t channel = data->dims[3], H = data->dims[1], W = data->dims[2];
    output->params_len = data->params_len / (uint16_t)(stride * stride);
    output->bitwidth_and_flags = data->bitwidth_and_flags | get_next_slot(data);
    output->dims[0] = 1;
    output->dims[1] = H / stride;
    output->dims[2] = W / stride;
    output->dims[3] = channel;
    for (uint16_t c = 0; c < channel; c++) {
        for (uint16_t h = 0; h < H; h = (uint16_t)(h + stride)) {
            for (uint16_t w = 0; w < W; w = (uint16_t)(w + stride)) {
#ifdef DUMP_MAXPOOL_PARAMS
                my_printf("h=%d ", h);
                my_printf("w=%d ", w);
                my_printf("c=%d" NEWLINE, c);
#endif
                int16_t max_val = INT16_MIN;
                for (uint16_t sH = 0; sH < stride; sH++) {
                    for (uint16_t sW = 0; sW < stride; sW++) {
                        int16_t val = *get_q15_param(data, (size_t)((h+sH) * W * channel + (w+sW) * channel + c));
#ifdef DUMP_MAXPOOL_PARAMS
                        print_q15(val);
#endif
                        // XXX: use LEA?
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                size_t offset = (size_t)((h/stride) * (W/stride) * channel + (w/stride) * channel + c);
#ifdef DUMP_MAXPOOL_PARAMS
                my_printf("max=");
                print_q15(max_val);
                my_printf(NEWLINE "offset=%d" NEWLINE, (uint16_t)offset);
#endif
                *get_q15_param(output, offset) = max_val;
            }
        }
    }

#ifdef DUMP_MAXPOOL_PARAMS
    my_printf("handle_maxpool output" NEWLINE);
    dump_params(output);
#endif

    return 0;
}

// XXX: there should be a better way to encode the stride
uint8_t handle_maxpool_2(ParameterInfo *input[], ParameterInfo *output) {
    return handle_maxpool(2, input, output);
}

uint8_t handle_maxpool_3(ParameterInfo *input[], ParameterInfo *output) {
    return handle_maxpool(3, input, output);
}

uint8_t handle_add(ParameterInfo *input[], ParameterInfo *output) {
    /* Add: Y = X + W */
#ifndef MY_NDEBUG
    my_printf("Add!" NEWLINE);
#endif
    if (get_param_bitwidth(input[0]) != 16 || get_param_bitwidth(input[1]) != 16) {
        my_printf("Error: unsupported bitwidth" NEWLINE);
        return 1;
    }
    ParameterInfo *A = input[0], *B = input[1];
    output->params_len = input[0]->params_len;
    output->bitwidth_and_flags = input[0]->bitwidth_and_flags | get_next_slot(A);
    output->dims[0] = 1;
    output->dims[1] = A->dims[1];

    msp_add_q15_params params = { .length = A->dims[1] };

    my_memcpy(lea_buffer.general.A, get_q15_param(A, 0), output->params_len);
    my_memcpy(lea_buffer.general.B, get_q15_param(B, 0), output->params_len);
    msp_status status = msp_add_q15(&params, lea_buffer.general.A, lea_buffer.general.B, lea_buffer.general.A);
    msp_checkStatus(status);

    my_memcpy(get_q15_param(output, 0), lea_buffer.general.A, output->params_len);

    return 0;
}

uint8_t handle_matmul(ParameterInfo *input[], ParameterInfo *output) {
    ParameterInfo *A = input[0], *B = input[1];

#ifndef MY_NDEBUG

# ifdef DUMP_PARAMS
    my_printf("handle_matmul inputs" NEWLINE);
    // dump_params(A);
    my_printf("B" NEWLINE);
    dump_params(B);
# endif

    my_printf("MatMul! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);
#endif

    uint16_t output_len = (uint16_t)(A->dims[0] * B->dims[1]);
    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->params_len = (uint16_t)(output_len * 2);
    output->bitwidth_and_flags = 16 << FLAG_SLOTS_WIDTH | get_next_slot(A);

    if (A->dims[0] * A->dims[1] > 256) {
        my_printf("Matrix A too large!" NEWLINE);
        return 1;
    }

    /* Seems TI's debugger does not like alias of pointers :/ */
#define lea_buffer_matmul lea_buffer.general.arrC

    msp_fill_q15_params fill_params = {
        .length = 256,
        .value = 0,
    };
    msp_status status = msp_fill_q15(&fill_params, lea_buffer_matmul);
    msp_checkStatus(status);

    my_memcpy(lea_buffer.general.A, get_q15_param(A, 0), (uint16_t)(A->dims[0] * A->dims[1] * sizeof(uint16_t)));

    /* LEA wants addresses to be 4-aligned */
    uint16_t step = (uint16_t)((256 / B->dims[1]) / 4 * 4);
    for (uint16_t i = 0; i < B->dims[0]; i = (uint16_t)(i + step)) {
        msp_matrix_mpy_q15_params params;
        uint16_t current_width = (uint16_t)MIN_VAL(step, B->dims[0] - i);
        params.srcARows = A->dims[0];
        params.srcACols = current_width;
        params.srcBRows = current_width;
        params.srcBCols = B->dims[1];

        my_memcpy(lea_buffer.general.B, get_q15_param(B, (uint16_t)(i * B->dims[1])), (uint16_t)(current_width * B->dims[1] * sizeof(uint16_t)));

#ifdef DUMP_PARAMS
        my_printf("strip for A" NEWLINE);
        dump_matrix(lea_buffer.general.A + A->dims[0] * i, (size_t)(A->dims[0] * current_width));
        my_printf("B" NEWLINE);
        dump_matrix(lea_buffer.general.B, (size_t)(current_width * B->dims[1]));
#endif

        status = msp_matrix_mpy_q15(
            &params,
            lea_buffer.general.A + A->dims[0] * i,
            lea_buffer.general.B,
            lea_buffer.general.temp);
        msp_checkStatus(status);

#ifdef DUMP_PARAMS
        my_printf("temp" NEWLINE);
        dump_matrix(lea_buffer.general.temp, (size_t)(A->dims[0] * B->dims[1]));
#endif

        msp_add_q15_params params2 = { .length = output_len };
        status = msp_add_q15(&params2, lea_buffer_matmul, lea_buffer.general.temp, lea_buffer_matmul);
        msp_checkStatus(status);
    }
    my_memcpy(get_q15_param(output, 0), lea_buffer_matmul, output->params_len);

#undef lea_buffer_matmul

#ifdef DUMP_PARAMS
    my_printf("handle_matmul output" NEWLINE);
    dump_params(output);
#endif

    return 0;
}

uint8_t handle_relu(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("ReLu!" NEWLINE);
#endif
    ParameterInfo *X = input[0];
    memcpy(output, X, sizeof(ParameterInfo));
    /* XXX: use LEA? */
    uint16_t bitwidth = get_param_bitwidth(X);
    for (uint32_t i = 0; i < X->params_len / (bitwidth / 8); i++) {
        if (bitwidth == 16) {
            int16_t *ptr = get_q15_param(X, i);
            if (*ptr < 0) {
                *ptr = 0;
            }
        } else {
            my_printf("Error: unsupported bitwidth for ReLu." NEWLINE);
        }
    }
#ifdef DUMP_RELU_PARAMS
    dump_params(output);
#endif
    return 0;
}

uint8_t handle_reshape(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("Reshape!" NEWLINE);
#endif
    ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    if (get_param_bitwidth(shape) != 64) {
        my_printf("Error: unsupported shape format." NEWLINE);
        return 1;
    }
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
    }
#define lea_buffer_reshape lea_buffer.general.A
    /*
     * XXX: Here is an heuristic - no conv nodes after reshape, so remapping
     * NHWC back to NCHW.
     * */
    uint8_t do_nhwc2nchw = get_param_slot_id(data) != FLAG_SLOTS;
    if (do_nhwc2nchw) {
        // data are intermediate values
        int16_t *output_addr = get_q15_param(output, 0);
        my_memcpy(lea_buffer_reshape, output_addr, output->params_len);
        uint16_t NUM = data->dims[0], H = data->dims[1],
                 W = data->dims[2], CHANNEL = data->dims[3];
        for (uint16_t n = 0; n < NUM; n++) {
            for (uint16_t c = 0; c < CHANNEL; c++) {
                for (uint16_t h = 0; h < H; h++) {
                    for (uint16_t w = 0; w < W; w++) {
                        uint16_t old_idx = n * CHANNEL * H * W + c * H * W       + h * W       + w,
                                 new_idx = n * H * W * CHANNEL + h * W * CHANNEL + w * CHANNEL + c;
                        output_addr[new_idx] = lea_buffer_reshape[old_idx];
                    }
                }
            }
        }
    }
#undef lea_buffer_reshape

#ifdef DUMP_PARAMS
    if (do_nhwc2nchw) {
        my_printf("handle_reshape output" NEWLINE);
        dump_params(output);
    }
#endif

    return 0;
}

uint8_t handle_squeeze(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("Squeeze!" NEWLINE);
#endif
    ParameterInfo *data = input[0];
    /* XXX: add flags; assume squeeze all one-size axes */
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    for (uint8_t i = 0, j = 0; i < 4; i++) {
        if (input[0]->dims[i] != 1) {
            output->dims[j] = input[0]->dims[i];
            j++;
        }
    }
    return 0;
}

#ifdef __MSP430__

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
