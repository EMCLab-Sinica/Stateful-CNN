// disable debug code in DSPLib
//#define MSP_DISABLE_DIAGNOSTICS

#include <DSPLib.h>

#ifdef __MSP430__
#include <FreeRTOS.h>
#include <croutine.h>
#define USE_CONCURRENT_CONV
#endif

#include "ops.h"
#include "op_handlers.h"
#include "common.h"
#include "debug.h"
#include "platform.h"

#define configCONV_STACK_SIZE 100
#define NUM_TASKS 2
#define CACHED_FILTERS
#define CACHED_INPUTS
#define INPUTS_LEN 760
#define LEA_BUFFER_SIZE 1024

DSPLIB_DATA(lea_buffer, 4)
int16_t lea_buffer[LEA_BUFFER_SIZE];

#ifdef USE_CONCURRENT_CONV
/* internal structure for msp_mac_q15() */
DSPLIB_DATA(msp_mac_params, 4)
MSP_LEA_MAC_PARAMS msp_mac_params[NUM_TASKS];
#endif

#ifdef CACHED_FILTERS
#define NUM_FILTERS 16
#endif
#ifdef CACHED_INPUTS
int16_t *input_buffer_addr[NUM_TASKS];
int16_t *next_input_buffer_addr;
int8_t input_buffer_w;
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

static uint16_t arrH[NUM_TASKS], arrW[NUM_TASKS], arrkH[NUM_TASKS], arrkW[NUM_TASKS], arrCHANNEL[NUM_TASKS], arrOUTPUT_CHANNEL[NUM_TASKS];
static msp_mac_q15_params mac_params[NUM_TASKS];
static int16_t *filter_buffer_addr[NUM_FILTERS];

static inline int32_t *buffer_iq31_mac_results(uint8_t uxIndex) {
    return (int32_t*)(lea_buffer + 1024) - (uxIndex + 1);
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
#ifdef CACHED_INPUTS
    LEAPMS0 = MSP_LEA_CONVERT_ADDRESS(input_buffer_addr[uxIndex]);
#else
    LEAPMS0 = MSP_LEA_CONVERT_ADDRESS(lea_buffer);
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

#else

static void convTask(unsigned short uxIndex) {
    #include "conv_prologue.h"

    #include "conv_pre.h"

    msp_status status = msp_mac_q15(&mac_params[uxIndex],
#ifdef CACHED_INPUTS
                                    input_buffer_addr[uxIndex],
#else
                                    lea_buffer,
#endif
                                    filter_buffer_addr[conv_params->conv_idx],
                                    buffer_iq31_mac_results(uxIndex));
    msp_checkStatus(status);

    #include "conv_post.h"
}

#endif

// defined in DSPLib_1_30_00_02/source/vector/msp_mac_q15.c
extern uint32_t msp_mac_q15_overflow_counter;

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
#ifdef CACHED_INPUTS
        input_buffer_addr[idx] = NULL;
        next_input_buffer_addr = NULL;
        input_buffer_w = -1;
#endif
    }

    for (uint8_t idx = 0; idx < NUM_FILTERS; idx++) {
        filter_buffer_addr[idx] = NULL;
    }

    for (uint16_t conv_idx = 0; conv_idx < input_N; conv_idx++) {
        //my_printf_debug("conv_idx = %d" NEWLINE, conv_idx);
        for (uint16_t output_w = 0; output_w < W; output_w++) {
            for (uint16_t output_h = 0; output_h < H; output_h += NUM_TASKS) {
                for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
                    ConvTaskParams *conv_params = &arr_conv_params[idx];
                    conv_params->conv_idx = conv_idx;
                    conv_params->output_h = output_h + idx;
                    conv_params->output_w = output_w;
                }
                for (uint8_t idx = 0; idx < NUM_TASKS; idx++) {
#ifdef USE_CONCURRENT_CONV
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

uint8_t handle_maxpool(const uint16_t stride, ParameterInfo *input[], ParameterInfo *output) {
    my_printf_debug("MaxPool!" NEWLINE);

    /* XXX: add flags; assume no padding for now */
    ParameterInfo *data = input[0];

    my_printf_debug("handle_maxpool input" NEWLINE);
    dump_params(data);

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
                my_printf_debug("h=%d ", h);
                my_printf_debug("w=%d ", w);
                my_printf_debug("c=%d" NEWLINE, c);

                int16_t max_val = INT16_MIN;
                for (uint16_t sH = 0; sH < stride; sH++) {
                    for (uint16_t sW = 0; sW < stride; sW++) {
                        int16_t val = *get_q15_param(data, (size_t)((h+sH) * W * channel + (w+sW) * channel + c));
                        print_q15_debug(val);
                        // XXX: use LEA?
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                size_t offset = (size_t)((h/stride) * (W/stride) * channel + (w/stride) * channel + c);
                my_printf_debug("max=");
                print_q15_debug(max_val);
                my_printf_debug(NEWLINE "offset=%d" NEWLINE, (uint16_t)offset);
                *get_q15_param(output, offset) = max_val;
            }
        }
    }

    my_printf_debug("handle_maxpool output" NEWLINE);
    dump_params(output);

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
    my_printf_debug("Add!" NEWLINE);

    if (get_param_bitwidth(input[0]) != 16 || get_param_bitwidth(input[1]) != 16) {
        // unsupported bitwidth
        ERROR_OCCURRED();
    }
    ParameterInfo *A = input[0], *B = input[1];
    output->params_len = input[0]->params_len;
    output->bitwidth_and_flags = input[0]->bitwidth_and_flags | get_next_slot(A);
    output->dims[0] = 1;
    output->dims[1] = A->dims[1];

    msp_add_q15_params params = { .length = A->dims[1] };

    int16_t *buffer_a = lea_buffer,
            *buffer_b = lea_buffer + output->params_len / sizeof(int16_t);
    my_memcpy(buffer_a, get_q15_param(A, 0), output->params_len);
    my_memcpy(buffer_b, get_q15_param(B, 0), output->params_len);
    msp_status status = msp_add_q15(&params, buffer_a, buffer_b, buffer_a);
    msp_checkStatus(status);

    my_memcpy(get_q15_param(output, 0), buffer_a, output->params_len);

    return 0;
}

uint8_t handle_matmul(ParameterInfo *input[], ParameterInfo *output) {
    ParameterInfo *A = input[0], *B = input[1];

    my_printf_debug("handle_matmul inputs" NEWLINE);
    // dump_params(A);
    my_printf_debug("B" NEWLINE);
    dump_params(B);
    my_printf_debug("MatMul! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    uint16_t output_len = (uint16_t)(A->dims[0] * B->dims[1]);
    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->params_len = (uint16_t)(output_len * 2);
    output->bitwidth_and_flags = 16 << FLAG_SLOTS_WIDTH | get_next_slot(A);

    if (A->dims[0] * A->dims[1] > 256) {
        // Matrix A too large!
        ERROR_OCCURRED();
    }

    int16_t *buffer_a = lea_buffer,
            *buffer_temp = buffer_a + A->dims[0] * A->dims[1],
            *buffer_matmul = buffer_temp + A->dims[0] * B->dims[1],
            *buffer_b = buffer_matmul + A->dims[0] * B->dims[1];

    msp_fill_q15_params fill_params = {
        .length = 256,
        .value = 0,
    };
    msp_status status = msp_fill_q15(&fill_params, buffer_matmul);
    msp_checkStatus(status);

    my_memcpy(buffer_a, get_q15_param(A, 0), (uint16_t)(A->dims[0] * A->dims[1] * sizeof(uint16_t)));

    /* LEA wants addresses to be 4-aligned */
    uint16_t step = (uint16_t)((256 / B->dims[1]) / 4 * 4);
    for (uint16_t i = 0; i < B->dims[0]; i = (uint16_t)(i + step)) {
        msp_matrix_mpy_q15_params params;
        uint16_t current_width = (uint16_t)MIN_VAL(step, B->dims[0] - i);
        params.srcARows = A->dims[0];
        params.srcACols = current_width;
        params.srcBRows = current_width;
        params.srcBCols = B->dims[1];

        my_memcpy(buffer_b, get_q15_param(B, (uint16_t)(i * B->dims[1])), (uint16_t)(current_width * B->dims[1] * sizeof(uint16_t)));

        my_printf_debug("strip for A" NEWLINE);
        dump_matrix(buffer_a + A->dims[0] * i, (size_t)(A->dims[0] * current_width));
        my_printf_debug("B" NEWLINE);
        dump_matrix(buffer_b, (size_t)(current_width * B->dims[1]));

        status = msp_matrix_mpy_q15(
            &params,
            buffer_a + A->dims[0] * i,
            buffer_b,
            buffer_temp);
        msp_checkStatus(status);

        my_printf_debug("temp" NEWLINE);
        dump_matrix(buffer_temp, (size_t)(A->dims[0] * B->dims[1]));

        msp_add_q15_params params2 = { .length = output_len };
        status = msp_add_q15(&params2, buffer_matmul, buffer_temp, buffer_matmul);
        msp_checkStatus(status);
    }
    my_memcpy(get_q15_param(output, 0), buffer_matmul, output->params_len);

    my_printf_debug("handle_matmul output" NEWLINE);
    dump_params(output);

    return 0;
}

uint8_t handle_relu(ParameterInfo *input[], ParameterInfo *output) {
    my_printf_debug("ReLu!" NEWLINE);

    ParameterInfo *X = input[0];
    my_memcpy(output, X, sizeof(ParameterInfo));
    /* XXX: use LEA? */
    uint16_t bitwidth = get_param_bitwidth(X);
    for (uint32_t i = 0; i < X->params_len / (bitwidth / 8); i++) {
        if (bitwidth == 16) {
            int16_t *ptr = get_q15_param(X, i);
            if (*ptr < 0) {
                *ptr = 0;
            }
        } else {
            // unsupported bitwidth for ReLu
            ERROR_OCCURRED();
        }
    }
    dump_params(output);
    return 0;
}

uint8_t handle_reshape(ParameterInfo *input[], ParameterInfo *output) {
    my_printf_debug("Reshape!" NEWLINE);

    ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth_and_flags = data->bitwidth_and_flags;
    if (get_param_bitwidth(shape) != 64) {
        // unsupported shape format
        ERROR_OCCURRED();
    }
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
    }
#define lea_buffer_reshape lea_buffer
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

    if (do_nhwc2nchw) {
        my_printf_debug("handle_reshape output" NEWLINE);
        dump_params(output);
    }

    return 0;
}

uint8_t handle_squeeze(ParameterInfo *input[], ParameterInfo *output) {
    my_printf_debug("Squeeze!" NEWLINE);

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
