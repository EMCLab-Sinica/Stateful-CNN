// disable debug code in DSPLib
//#define MSP_DISABLE_DIAGNOSTICS

#include <DSPLib.h>
#include "common.h"
#include "debug.h"
#include "op_handlers.h"
#include "ops.h"

#ifdef __MSP430__
#include <FreeRTOS.h>
#endif

#define configCONV_STACK_SIZE 100

// TODO: make these adjustable on runtime
#define TILE_W 1

int16_t *input_buffer_addr;
int16_t *next_input_buffer_addr;

#define CONV_TASK_FLAG_PROCESSED_FILTERS_BASE 2
typedef struct ConvTaskParams {
    ParameterInfo *conv_input;
    ParameterInfo *conv_filter;
    ParameterInfo *bias;
    ParameterInfo *output;
    // Keep the order of the following 3 fields consistent with OpExtraData.conv
    uint16_t conv_idx;
    uint16_t output_h;
    uint16_t output_w;
    uint16_t flags;
    uint8_t do_reinitialize_input;
    uint8_t tile_h;
    uint16_t starting_output_h;
    uint16_t starting_output_h_offset;
    OpExtraData *extra_data;
} ConvTaskParams;

static ConvTaskParams conv_params;
static struct {
    uint16_t dest_offset;
    uint8_t filter_limit;
    uint8_t truncated;
    uint16_t OUTPUT_CHANNEL;
    uint16_t W_by_OUTPUT_CHANNEL;
} global_conv_params;

static msp_mac_q15_params mac_params;
static int16_t *filter_buffer_addr[NUM_FILTERS];  // filter index -> address
static int8_t cached_filter_idx[NUM_FILTERS];  // filter buffer id (0~filter_limit-1) -> filter index
static int8_t filter_buffer_id;
static uint8_t pending_filters[NUM_FILTERS];
static uint8_t pending_filter_idx = 0;

int32_t *iq31_mac_results = (int32_t*)(lea_buffer + LEA_BUFFER_SIZE) - 1;

static void convTask(void) {
    #include "conv_prologue.h"

    #include "conv_pre.h"

    msp_status status = msp_mac_q15(&mac_params,
                                    input_buffer_addr,
                                    filter_buffer_addr[conv_params.conv_idx],
                                    iq31_mac_results);
    msp_checkStatus(status);

    #include "conv_post.h"
}

static inline void schedule_tile(uint16_t idx, uint16_t output_h, uint16_t output_w, uint8_t tile_h, uint8_t tile_w, uint8_t processed_filters, uint16_t H, uint16_t W, uint16_t old_output_h_offset) {
    OpExtraData *extra_data = conv_params.extra_data;
    extra_data->current_filter = idx;
    uint8_t cur_output_h_offset = processed_filters ? 0 : extra_data->output_h_offset;
    my_printf_debug("cur_output_h_offset = %d" NEWLINE, cur_output_h_offset);
    conv_params.conv_idx = idx;
    conv_params.starting_output_h = output_h + old_output_h_offset;
    conv_params.starting_output_h_offset = cur_output_h_offset;
    conv_params.tile_h = tile_h;
    for (uint8_t i = 0; i < MIN_VAL(tile_w, W - output_w); i++) {
        for (uint8_t j = cur_output_h_offset; j < MIN_VAL(tile_h, H - output_h); j++) {
            extra_data->output_h_offset = j;
            conv_params.flags &= 0xff00;
            conv_params.output_h = output_h + j;
            conv_params.output_w = output_w + i;
            conv_params.flags |= processed_filters * CONV_TASK_FLAG_PROCESSED_FILTERS_BASE;
            my_printf_debug("j = %d" NEWLINE, j);
            conv_params.do_reinitialize_input = (j == cur_output_h_offset);
            convTask();
        }
    }
    extra_data->output_h_offset = 0;
    extra_data->processed_filters[idx] = 1;
}

static inline void handle_conv_inner_loop(uint16_t n_conv, uint16_t output_h, uint16_t output_w, uint8_t tile_h, uint8_t tile_w, uint16_t H, uint16_t W) {
    uint8_t scheduled_filters = 0;
    OpExtraData *extra_data = conv_params.extra_data;
    uint16_t old_output_h_offset = extra_data->output_h_offset;
    if (extra_data->current_filter) {
        schedule_tile(extra_data->current_filter, output_h, output_w, tile_h, tile_w, scheduled_filters, H, W, old_output_h_offset);
        scheduled_filters++;
    }
    for (uint8_t idx = 0; idx < n_conv; idx++) {
        if (extra_data->processed_filters[idx]) {
            my_printf_debug("Skipping processed filter %d" NEWLINE, idx);
            continue;
        }
        if (filter_buffer_addr[idx]) {
            schedule_tile(idx, output_h, output_w, tile_h, tile_w, scheduled_filters, H, W, old_output_h_offset);
            scheduled_filters++;
        } else {
            my_printf_debug("Filter %d is not cached, append it to the pending list" NEWLINE, idx);
            pending_filters[pending_filter_idx] = idx;
            pending_filter_idx++;
        }
    }
    for (uint8_t idx = 0; idx < pending_filter_idx; idx++) {
        uint8_t filter_idx = pending_filters[idx];
        schedule_tile(filter_idx, output_h, output_w, tile_h, tile_w, scheduled_filters, H, W, old_output_h_offset);
        my_printf_debug("Mark filter %d as processed" NEWLINE, filter_idx);
        scheduled_filters++;
    }
    pending_filter_idx = 0;
    for (uint8_t idx = 0; idx < n_conv; idx++) {
        extra_data->processed_filters[idx] = 0;
    }
}

uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags) {
    ParameterInfo *conv_input = input[0], *conv_filter = input[1], *bias = input[2];
    my_printf_debug("Conv!" NEWLINE);

    msp_mac_q15_overflow_counter = 0;

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

    conv_params.conv_input = conv_input;
    conv_params.conv_filter = conv_filter;
    conv_params.bias = bias;
    conv_params.output = output;
    conv_params.extra_data = extra_data;
    conv_params.flags = flags << 8;
    input_buffer_addr = NULL;
    next_input_buffer_addr = NULL;

    if (!extra_data->conv_running) {
        extra_data->conv_idx = extra_data->output_h = extra_data->output_h_offset = extra_data->output_w = 0;
        for (uint8_t idx = 0; idx < NUM_FILTERS; idx++) {
            extra_data->processed_filters[idx] = 0;
        }
        extra_data->current_filter = 0;
        extra_data->conv_running = 1;
    }

    for (uint8_t idx = 0; idx < NUM_FILTERS; idx++) {
        filter_buffer_addr[idx] = NULL;
        cached_filter_idx[idx] = -1;
    }
    filter_buffer_id = 0;

    uint8_t tile_h = 1; // fallback value
    if (H == 14) {
        tile_h = 6;
    } else if (H == 28) {
        tile_h = 28;
    }

    uint16_t kH = conv_filter->dims[1],
             kW = conv_filter->dims[2],
             CHANNEL = conv_filter->dims[3];
    global_conv_params.dest_offset = kW * CHANNEL;
    global_conv_params.OUTPUT_CHANNEL = conv_filter->dims[0];
    global_conv_params.W_by_OUTPUT_CHANNEL = W * global_conv_params.OUTPUT_CHANNEL;

    /* MSP430 LEA requires length to be even */
    mac_params.length = (uint16_t)(CHANNEL * kH * kW / 2 * 2);
    global_conv_params.truncated = (mac_params.length != CHANNEL * kH * kW);
    if (global_conv_params.truncated) {
        // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
        // also odd, so dummy values are needed between slices to make
        // addresses even.
        // a dummy value for each slice (kW * CHANNEL q15 values)
        mac_params.length += kH + 1;
        global_conv_params.dest_offset++;
    }

    global_conv_params.filter_limit = MIN_VAL(
        conv_filter->dims[0],
        (LEA_BUFFER_SIZE - 4 - global_conv_params.dest_offset * (kH + tile_h - 1)) / (global_conv_params.dest_offset * kH)
    );

    my_printf_debug("filter_limit: %d" NEWLINE, global_conv_params.filter_limit);

    uint16_t starting_w = extra_data->output_w,
             starting_h = extra_data->output_h;
    if (starting_w >= W || starting_h >= H) {
        ERROR_OCCURRED();
    }
    for (uint16_t output_w = starting_w; output_w < W; output_w += TILE_W) {
        extra_data->output_w = output_w;
        for (uint16_t output_h = (output_w == starting_w ? starting_h : 0); output_h < H; output_h += tile_h) {
            extra_data->output_h = output_h;
            handle_conv_inner_loop(input_N, output_h, output_w, tile_h, TILE_W, H, W);
        }
    }

    my_printf_debug("handle_conv output" NEWLINE);
    dump_params(output);

    my_printf_debug("msp_mac_q15_overflow_counter=%d" NEWLINE, msp_mac_q15_overflow_counter);

    extra_data->conv_running = 0;

    return ret;
}
