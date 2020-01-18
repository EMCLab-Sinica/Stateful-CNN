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

    /* MSP430 LEA requires length to be even */
    mac_params[uxIndex].length = (uint16_t)(CHANNEL * kH * kW / 2 * 2);
    truncated[uxIndex] = (mac_params[uxIndex].length != CHANNEL * kH * kW);
    if (truncated[uxIndex]) {
        // 1 for the truncated value, another dummy
        mac_params[uxIndex].length = (uint16_t)(mac_params[uxIndex].length + 2);
    }
    buffer_size = (uint16_t)(sizeof(uint16_t) * mac_params[uxIndex].length);
    if (buffer_size > sizeof(lea_buffer.conv.filter[uxIndex])) {
        my_printf("Error: buffer too small." NEWLINE);
        ERROR_OCCURRED();
    }

    /* copy filter data */
    /* TODO: cache it */
    filter_addr = arr_filter_addr[uxIndex] = get_q15_param(
        conv_params->conv_filter,
        (size_t)(conv_params->conv_idx * CHANNEL * kH * kW));
    my_memcpy(lea_buffer.conv.filter[uxIndex],
              filter_addr,
              buffer_size);
    if (truncated[uxIndex]) {
        // dummy value
        lea_buffer.conv.filter[uxIndex][buffer_size] = 0;
    }

    int8_t field_size = (int8_t)((kH - 1) / 2);

    /* copy input data, row by row */

    msp_fill_q15_params fill_params = {
        .length = (uint16_t)((kH * kW * CHANNEL+1)/2*2),
        .value = 0,
    };
    msp_status status = msp_fill_q15(&fill_params, lea_buffer.conv.input[uxIndex]);
    msp_checkStatus(status);

    input_addr = arr_input_addr[uxIndex] = get_q15_param(
        conv_params->conv_input,
        (size_t)(CHANNEL * (
                (conv_params->output_h) * W +
                (conv_params->output_w)
        )));

    int16_t h_start = (int16_t)int_max(-field_size,    -conv_params->output_h),
            h_end   = (int16_t)int_min( field_size, H-1-conv_params->output_h);
#ifdef DUMP_PARAMS
    if (dump_conv_params) {
        my_printf("h_start=%d ", h_start);
        my_printf("h_end=%d" NEWLINE, h_end);
    }
#endif
    for (int16_t h = h_start; h <= h_end; h++) {
        int16_t w_start = (int16_t)int_max(-field_size,    -conv_params->output_w),
                w_end   = (int16_t)int_min( field_size, W-1-conv_params->output_w);
        size_t size = (size_t)((w_end-w_start+1) * CHANNEL); // in WORD
        int16_t *src = input_addr + (h * W + w_start) * CHANNEL;
#ifdef DUMP_PARAMS
        if (dump_conv_params && conv_params->output_h == 0 && conv_params->output_w == 0) {
            dump_matrix(src, size);
        }
#endif
        /* TODO: handle padding */
        my_memcpy(lea_buffer.conv.input[uxIndex] + ((h + field_size) * kW + (w_start + field_size)) * CHANNEL,  // dest
                  src, // src
                  size * sizeof(uint16_t));  // size
    }
}
