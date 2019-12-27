#include <string.h>

#include <DSPLib.h>

#ifdef __MSP430__
#include <driverlib.h>
#include <FreeRTOS.h>
#include <croutine.h>
#include "Tools/my_timer.h"
#define USE_DMA 1
#define USE_CONCURRENT_CONV 1
#else
#define USE_DMA 0
#define USE_CONCURRENT_CONV 0
#endif

#include "ops.h"
#include "op_handlers.h"
#include "common.h" // for MY_NDEBUG

#define configCONV_STACK_SIZE 100
#define NUM_TASKS 2

#ifdef __MSP430__
#pragma DATA_SECTION(lea_buffer, ".leaRAM")
#endif
union {
    // for conv
    struct {
        int16_t input[NUM_TASKS][256];
        int16_t filter[NUM_TASKS][256];
        int32_t iq31_mac_result[NUM_TASKS];
    } conv;
    // for others
    struct {
        int16_t A[256];
        int16_t B[256];
        int16_t arrC[256]; // the term C is reserved for MSP430 :/
        int16_t temp[64];
    } general;
} lea_buffer;

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
#if USE_CONCURRENT_CONV
    // DMA_enableInterrupt(MY_DMA_CHANNEL);
#endif
    DMA_enableTransfers(MY_DMA_CHANNEL);
    DMA_startTransfer(MY_DMA_CHANNEL);
#endif
}


#if USE_CONCURRENT_CONV
static void convTask(CoRoutineHandle_t xHandle, UBaseType_t uxIndex) {
#else
static void convTask(unsigned short uxIndex) {
#endif
    /* put var declarations first to make the compiler happy */
    ConvTaskParams *conv_params;
    int16_t *input_addr;
    uint16_t buffer_size;
#if USE_CONCURRENT_CONV
    uint16_t interruptState;

    crSTART(xHandle);

    for (;;) {
#endif

    static uint16_t arrH[NUM_TASKS], arrW[NUM_TASKS], arrkH[NUM_TASKS], arrkW[NUM_TASKS], arrCHANNEL[NUM_TASKS];
    static msp_mac_q15_params mac_params[NUM_TASKS];
    static uint8_t truncated[NUM_TASKS];

    /* Cannot use C as a variable name here as C is a macro on MSP430 :( */
    uint16_t H, W, kH, kW, CHANNEL;
    conv_params = &arr_conv_params[uxIndex];

    arrH[uxIndex] = conv_params->conv_input->dims[2];
    arrW[uxIndex] = conv_params->conv_input->dims[3];
    arrkH[uxIndex] = conv_params->conv_filter->dims[2];
    arrkW[uxIndex] = conv_params->conv_filter->dims[3];
    arrCHANNEL[uxIndex] = conv_params->conv_filter->dims[1];

    H = arrH[uxIndex];
    W = arrW[uxIndex];
    kH = arrkH[uxIndex];
    kW = arrkW[uxIndex];
    CHANNEL = arrCHANNEL[uxIndex];

#define lea_buffer_input lea_buffer.conv.input[uxIndex]
#define lea_buffer_filter lea_buffer.conv.filter[uxIndex]

    /* MSP430 LEA requires length to be even */
    mac_params[uxIndex].length = (uint16_t)(CHANNEL * kH * kW / 2 * 2);
    truncated[uxIndex] = (mac_params[uxIndex].length != CHANNEL * kH * kW);
    buffer_size = (uint16_t)(sizeof(uint16_t) * mac_params[uxIndex].length);
    if (buffer_size > sizeof(lea_buffer_filter)) {
        my_printf("Error: buffer too small." NEWLINE);
        ERROR_OCCURRED();
    }

    /* copy filter data */
    /* TODO: cache it */
    my_memcpy(lea_buffer_filter,
              get_q15_param(conv_params->conv_filter, (size_t)(conv_params->conv_idx * CHANNEL * kH * kW)),
              buffer_size);

    /* copy input data, row by row */
    input_addr = get_q15_param(conv_params->conv_input, (size_t)((conv_params->output_h * W + conv_params->output_w) * CHANNEL));
    for (uint16_t h = 0; h < kH; h++) {
        size_t size = (size_t)(kW * CHANNEL);
        if (truncated[uxIndex] && h == kH - 1) {
            size--;
        }
        /* TODO: handle padding */
        my_memcpy(lea_buffer_input + h * kW * CHANNEL,  // dest
                  input_addr + h * W * CHANNEL,  // src
                  size * sizeof(uint16_t));  // size
    }

#ifdef DUMP_PARAMS
    my_printf("%d ", conv_params->output_h);
    my_printf("%d ", conv_params->output_w);
#endif

#if USE_CONCURRENT_CONV
    /* TODO: do context switch after msp_lea_doInvokeCommand */
    /* modified from DSPLib_1_30_00_02/source/vector/msp_mac_q15.c */
    MSP_LEA_MAC_PARAMS *leaParams;

    /* Allocate MSP_LEA_MAC_PARAMS structure. */
    leaParams = (MSP_LEA_MAC_PARAMS *)msp_lea_allocMemory(sizeof(MSP_LEA_MAC_PARAMS)/sizeof(uint32_t));

    /* Set MSP_LEA_MAC_PARAMS structure. */
    leaParams->input2 = MSP_LEA_CONVERT_ADDRESS(lea_buffer_filter);
    leaParams->output = MSP_LEA_CONVERT_ADDRESS(&lea_buffer.conv.iq31_mac_result[uxIndex]);
    leaParams->vectorSize = mac_params[uxIndex].length;

    /* Load source arguments to LEA. */
    LEAPMS0 = MSP_LEA_CONVERT_ADDRESS(lea_buffer_input);
    LEAPMS1 = MSP_LEA_CONVERT_ADDRESS(leaParams);

    // modified from DSPLib_1_30_00_02/include/DSPLib_lea.h

    /* Save interrupt state and disable interrupts. */
    interruptState = __get_interrupt_state();
    __disable_interrupt();

    /* Clear interrupt flag and invoke the command. */
    msp_lea_ifg = 0;
    /* Invoke the LEACMD__MAC command. */
    LEAPMCB = LEACMD__MAC | LEAITFLG1;

    /* Do not enter LPM0 so that CPU can do other work */
    __bis_SR_register(GIE);

    /* Restore original interrupt state. */
    __set_interrupt_state(interruptState);

    /* Free MSP_LEA_MAC_PARAMS structure. */
    msp_lea_freeMemory(sizeof(MSP_LEA_MAC_PARAMS)/sizeof(uint32_t));

    crDELAY(xHandle, 0);

    /* after context switch of co-routines, variables on stack are lost */
    conv_params = &arr_conv_params[uxIndex];
    H = arrH[uxIndex];
    W = arrW[uxIndex];
    kH = arrkH[uxIndex];
    kW = arrkW[uxIndex];
    CHANNEL = arrCHANNEL[uxIndex];
#else
    msp_status status = msp_mac_q15(&mac_params[uxIndex],
                                    lea_buffer_input, lea_buffer_filter,
                                    &lea_buffer.conv.iq31_mac_result[uxIndex]);
    msp_checkStatus(status);
#endif

    if (truncated[uxIndex]) {
        uint16_t last_idx = (uint16_t)(kH * kW - 1);
        lea_buffer.conv.iq31_mac_result[uxIndex] += (*get_q15_param(conv_params->conv_input, last_idx)) * (*get_q15_param(conv_params->conv_filter, last_idx)) * 2;
    }

    /* XXX: need more co-routines? */
    while(!msp_lea_ifg);

#ifdef DUMP_PARAMS
# ifdef __MSP430__
    my_printf("%l ", lea_buffer.conv.iq31_mac_result[uxIndex]);
# else
    my_printf("%ld ", lea_buffer.conv.iq31_mac_result[uxIndex]);
# endif
    if (conv_params->output_w % 4 == 0) {
        my_printf(NEWLINE);
    }
#endif
    {
    int16_t q15_mac_result = iq31_to_q15(&lea_buffer.conv.iq31_mac_result[uxIndex]);
    q15_mac_result = (int16_t)(q15_mac_result + *get_q15_param(conv_params->bias, conv_params->conv_idx));

    int16_t *output_data = get_q15_param(conv_params->output, 0);
    output_data[conv_params->conv_idx * H * W + conv_params->output_h * W + conv_params->output_w] = q15_mac_result;
    }

#undef lea_buffer_filter
#undef lea_buffer_input

#if USE_CONCURRENT_CONV
    crDELAY(xHandle, 0);

    }

    crEND();
#endif
}

uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("Conv!" NEWLINE);
#endif

#if USE_CONCURRENT_CONV
    msp_lea_init();

    static bool task_created = false;

    if (!task_created) {
        for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
            if (xCoRoutineCreate(convTask, 0, idx) != pdPASS) {
                my_printf("Failed to create co-routines." NEWLINE);
                ERROR_OCCURRED();
            }
        }
        task_created = true;
    }
#endif

    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
    if (conv_input->bitwidth_and_flags >> 1 != 16 || conv_filter->bitwidth_and_flags >> 1 != 16) {
        my_printf("Error: incorrect bitwidth." NEWLINE);
        return 1;
    }
    /* original: input: N x C x H x W, filter: M x C x kW x kW
     * remapped: input: N x H x W x C, filter: M x kH x kW x C */
    /* TODO: really use remapped dimensions */
    const uint16_t H = conv_input->dims[2], W = conv_input->dims[3],
                   input_N = conv_filter->dims[0];
    /* TODO: add flags; assume auto_pad=SAME_UPPER, stride=(1, 1), dilation=(1, 1) for now */
    output->params_len = (uint16_t)(input_N * H * W * 2);
    output->bitwidth_and_flags = 16 << 1 | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = input_N;
    output->dims[2] = H;
    output->dims[3] = W;

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
    }

    for (uint16_t conv_idx = 0; conv_idx < input_N; conv_idx++) {
        //my_printf("conv_idx = %d" NEWLINE, conv_idx);
        for (uint16_t output_h = 0; output_h < H; output_h++) {
            for (uint16_t output_w = 0; output_w < W; output_w += NUM_TASKS) {
                for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
                    ConvTaskParams *conv_params = &arr_conv_params[idx];
                    conv_params->conv_idx = conv_idx;
                    conv_params->output_h = output_h;
                    conv_params->output_w = output_w + idx;
                }
                for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
#if USE_CONCURRENT_CONV
                    /* each co-routine runs as two parts:
                     * before and after LEA operations */
                    vCoRoutineSchedule();
                    vCoRoutineSchedule();
#else
                    convTask(idx);
#endif
                }
            }
        }
    }
#ifdef __MSP430__
    end = getTickCounter();
    counters[counter_idx] = end - start;
    counter_idx++;
#endif

#ifndef MY_NDEBUG
    my_printf("handle_conv output" NEWLINE);
#endif

    return ret;
}

uint8_t handle_maxpool(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("MaxPool!" NEWLINE);
#endif
    /* TODO: add flags; assume stripe=2, no padding for now */
    const uint16_t stride = 2; // for less type conversions
    ParameterInfo *data = input[0];
    output->params_len = data->params_len / (uint16_t)(stride * stride);
    output->bitwidth_and_flags = data->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
    output->dims[0] = 1;
    output->dims[1] = data->dims[1];
    output->dims[2] = data->dims[2] / stride;
    output->dims[3] = data->dims[3] / stride;
    const uint16_t channel = data->dims[1], H = data->dims[2], W = data->dims[3];
    msp_max_q15_params params = { .length = 4 };
    int16_t max_val;
    uint16_t index;
#define lea_buffer_maxpool lea_buffer.general.A
    for (uint16_t i = 0; i < channel; i++) {
        for (uint16_t j = 0; j < H; j = (uint16_t)(j + stride)) {
            for (uint16_t k = 0; k < W; k = (uint16_t)(k + stride)) {
                lea_buffer_maxpool[0] = *get_q15_param(data, (size_t)(i * H * W + j     * W + k    ));
                lea_buffer_maxpool[1] = *get_q15_param(data, (size_t)(i * H * W + j     * W + (k+1)));
                lea_buffer_maxpool[2] = *get_q15_param(data, (size_t)(i * H * W + (j+1) * W + k    ));
                lea_buffer_maxpool[3] = *get_q15_param(data, (size_t)(i * H * W + (j+1) * W + (k+1)));
                msp_status status = msp_max_q15(&params, lea_buffer_maxpool, &max_val, &index);
                msp_checkStatus(status);
                *get_q15_param(output, (size_t)(i * H * W + j * W + k)) = max_val;
            }
        }
    }
#undef lea_buffer_maxpool

#ifndef MY_NDEBUG
    my_printf("handle_maxpool output" NEWLINE);
    dump_params(output);
#endif

    return 0;
}

uint8_t handle_add(ParameterInfo *input[], ParameterInfo *output) {
    /* Add: Y = X + W */
#ifndef MY_NDEBUG
    my_printf("Add!" NEWLINE);
#endif
    if (input[0]->bitwidth_and_flags >> 1 != 16 || input[1]->bitwidth_and_flags >> 1 != 16) {
        my_printf("Error: unsupported bitwidth" NEWLINE);
        return 1;
    }
    ParameterInfo *A = input[0], *B = input[1];
    output->params_len = input[0]->params_len;
    output->bitwidth_and_flags = input[0]->bitwidth_and_flags | FLAG_INTERMEDIATE_VALUES;
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
    my_printf("handle_matmul inputs" NEWLINE);
    dump_params(A);
    dump_params(B);

    my_printf("MatMul! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);
#endif

    uint16_t output_len = (uint16_t)(A->dims[0] * B->dims[1]);
    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->params_len = (uint16_t)(output_len * 2);
    output->bitwidth_and_flags = 16 << 1 | FLAG_INTERMEDIATE_VALUES;

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

    my_memcpy(lea_buffer.general.A, get_q15_param(A, 0), (uint16_t)(A->dims[0] * A->dims[1]));

    /* LEA wants addresses to be 4-aligned */
    uint16_t step = (uint16_t)((256 / B->dims[1]) / 4 * 4);
    for (uint16_t i = 0; i < B->dims[0]; i = (uint16_t)(i + step)) {
        msp_matrix_mpy_q15_params params;
        uint16_t current_width = (uint16_t)MIN_VAL(step, B->dims[0] - i);
        params.srcARows = A->dims[0];
        params.srcACols = current_width;
        params.srcBRows = current_width;
        params.srcBCols = B->dims[1];

        my_memcpy(lea_buffer.general.B, get_q15_param(B, (uint16_t)(i * B->dims[1])), (uint16_t)(current_width * B->dims[1]));

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

#ifndef MY_NDEBUG
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
    /* TODO: use LEA? */
    uint16_t bitwidth = X->bitwidth_and_flags >> 1;
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
#ifndef MY_NDEBUG
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
    if (shape->bitwidth_and_flags >> 1 != 64) {
        my_printf("Error: unsupported shape format." NEWLINE);
        return 1;
    }
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
    }
    return 0;
}

uint8_t handle_squeeze(ParameterInfo *input[], ParameterInfo *output) {
#ifndef MY_NDEBUG
    my_printf("Squeeze!" NEWLINE);
#endif
    ParameterInfo *data = input[0];
    /* TODO: add flags; assume squeeze all one-size axes */
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
