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

    /* MSP430 LEA requires length to be even */
    mac_params[uxIndex].length = (uint16_t)(CHANNEL * kH * kW / 2 * 2);
    truncated[uxIndex] = (mac_params[uxIndex].length != CHANNEL * kH * kW);
    buffer_size = (uint16_t)(sizeof(uint16_t) * mac_params[uxIndex].length);
    if (buffer_size > sizeof(lea_buffer.conv.filter[uxIndex])) {
        my_printf("Error: buffer too small." NEWLINE);
        ERROR_OCCURRED();
    }

    /* copy filter data */
    /* TODO: cache it */
    my_memcpy(lea_buffer.conv.filter[uxIndex],
              get_q15_param(conv_params->conv_filter, (size_t)(conv_params->conv_idx * CHANNEL * kH * kW)),
              buffer_size);

    /* copy input data, row by row */
    input_addr = get_q15_param(conv_params->conv_input, (size_t)((conv_params->output_h * W + conv_params->output_w) * CHANNEL));
    for (uint16_t h = 0; h < kH; h++) {
        size_t size = (size_t)(kW * CHANNEL);
        if (truncated[uxIndex] && h == kH - 1) {
            size--;
        }
#ifdef DUMP_PARAMS
        if (dump_conv_params && conv_params->output_h == 0 && conv_params->output_w == 0) {
            dump_matrix(input_addr + h * W * CHANNEL, size);
        }
#endif
        /* TODO: handle padding */
        my_memcpy(lea_buffer.conv.input[uxIndex] + h * kW * CHANNEL,  // dest
                  input_addr + h * W * CHANNEL,  // src
                  size * sizeof(uint16_t));  // size
    }
