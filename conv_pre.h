{
    conv_params = &arr_conv_params[uxIndex];

    arrH[uxIndex] = conv_params->conv_input->dims[1];
    arrW[uxIndex] = conv_params->conv_input->dims[2];
    arrkH[uxIndex] = conv_params->conv_filter->dims[1];
    arrkW[uxIndex] = conv_params->conv_filter->dims[2];
    arrCHANNEL[uxIndex] = conv_params->conv_filter->dims[3];
    arrOUTPUT_CHANNEL[uxIndex] = conv_params->conv_filter->dims[0];

    H = arrH[uxIndex];
    W = arrW[uxIndex];
    kH = arrkH[uxIndex];
    kW = arrkW[uxIndex];
    CHANNEL = arrCHANNEL[uxIndex];
    OUTPUT_CHANNEL = arrOUTPUT_CHANNEL[uxIndex];

    int16_t dest_offset = kW * CHANNEL;

    /* MSP430 LEA requires length to be even */
    mac_params[uxIndex].length = (uint16_t)(CHANNEL * kH * kW / 2 * 2);
    uint8_t truncated = (mac_params[uxIndex].length != CHANNEL * kH * kW);
    if (truncated) {
#ifdef CACHED_INPUTS
        // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
        // also odd, so dummy values are needed between slices to make
        // addresses even.
        // a dummy value for each slice (kW * CHANNEL q15 values)
        mac_params[uxIndex].length += kH + 1;
        dest_offset++;
#else
        // 1 for the truncated value, another dummy
        mac_params[uxIndex].length += 2;
#endif
    }
    buffer_size = (uint16_t)(sizeof(uint16_t) * mac_params[uxIndex].length);
    if (buffer_size > sizeof(lea_buffer.conv.filter)) {
        my_printf("Error: buffer too small." NEWLINE);
        ERROR_OCCURRED();
    }

    /* copy filter data */
#ifdef CACHED_FILTERS
    if (cached_filter_index != conv_params->conv_idx) {
#endif
        filter_addr = get_q15_param(
            conv_params->conv_filter,
            (size_t)(conv_params->conv_idx * CHANNEL * kH * kW));
#ifdef CACHED_INPUTS
        if (truncated) {
            int16_t *filter_buffer_addr = lea_buffer.conv.filter;
            for (uint16_t h = 0; h < kH; h++) {
                memcpy(filter_buffer_addr, filter_addr, kW * CHANNEL * sizeof(int16_t));
                filter_buffer_addr += dest_offset;
                filter_addr += kW * CHANNEL;
            }
        } else {
#endif
            my_memcpy(lea_buffer.conv.filter,
                      filter_addr,
                      buffer_size);
            if (truncated) {
                // dummy value
                lea_buffer.conv.filter[buffer_size / sizeof(int16_t) - 1] = 0;
            }
#ifdef CACHED_INPUTS
        }
#endif
#ifdef CACHED_FILTERS
        cached_filter_index = conv_params->conv_idx;
    }
#endif

    int8_t field_size = (int8_t)((kH - 1) / 2);

    /* copy input data, row by row */

    input_addr = get_q15_param(
        conv_params->conv_input,
        (size_t)(CHANNEL * (
                (conv_params->output_h) * W +
                (conv_params->output_w)
        )));

    /* int32_t instead of int16_t as TI's compiler cannot handle negative
     * offsets correctly. The expression `input_addr + (int16_t)(-2)` is
     * compiled as:
     * 1. -2 is represented as 0x00FFFE (general registers are 24-bit long).
     *    Assume this value is stored in R11.
     * 2. RLAM.A #1,R11  # multiply by 2 to transform the offset for int16_t
     *    to the difference of addresses.
     * In step 2, R11 becomes 0x01FFFC, while it should be -4, or 0x00FFFC,
     * and thus the resultant address is offset by 0x10000.
     */
    int32_t w_start = int16_max(-field_size,    -conv_params->output_w),
            w_end   = int16_min( field_size, W-1-conv_params->output_w);
    int16_t *src = NULL,
            *dest,
            *dest_initial = lea_buffer.conv.input[uxIndex];
    int16_t src_offset = W * CHANNEL;
    uint8_t input_buffer_reinitialized = 1;
#ifndef CACHED_INPUTS
    dest = dest_initial;
#else
    dest = input_buffer_addr[uxIndex];
    if (dest && dest + kH * dest_offset < dest_initial + INPUTS_LEN
             && input_buffer_w[uxIndex] == conv_params->output_w) {
        input_buffer_reinitialized = 0;
    }
#endif

    int32_t h_start,
            h_end = int16_min(field_size, H-1-conv_params->output_h);
    if (input_buffer_reinitialized) {
#ifdef DUMP_CONV_PARAMS
        my_printf("Reinitialize input buffer" NEWLINE);
#endif
        msp_fill_q15_params fill_params = {
#ifdef CACHED_INPUTS
            .length = INPUTS_LEN,
#else
            .length = (uint16_t)((kH * kW * CHANNEL+1)/2*2),
#endif
            .value = 0,
        };
        msp_status status = msp_fill_q15(&fill_params, lea_buffer.conv.input[uxIndex]);
        msp_checkStatus(status);

#ifdef CACHED_INPUTS
        dest = input_buffer_addr[uxIndex] = dest_initial;
        input_buffer_w[uxIndex] = conv_params->output_w;
#endif

        h_start = int16_max(-field_size, -conv_params->output_h);
    } else {
        h_start = field_size;
    }

    dest += (h_start + field_size) * dest_offset + (w_start + field_size) * CHANNEL;

#ifdef DUMP_CONV_PARAMS
    my_printf("h_start=%d ", h_start);
    my_printf("h_end=%d" NEWLINE, h_end);
#endif

    size_t size = (size_t)((w_end-w_start+1) * CHANNEL * sizeof(uint16_t)); // in bytes
    if (h_start <= h_end) {
        src = input_addr + (h_start * W + w_start) * CHANNEL;
#if defined(CACHED_INPUTS) && defined(DUMP_CONV_PARAMS)
        my_printf("Copying row to lea_buffer.conv.input[%d] + %d" NEWLINE,
                  uxIndex, (int)(dest - lea_buffer.conv.input[uxIndex]));
#endif
        for (int32_t h = h_start; h <= h_end; h++) {
            my_memcpy(dest, src, size);
            src += src_offset;
            dest += dest_offset;
        }
    }
}
