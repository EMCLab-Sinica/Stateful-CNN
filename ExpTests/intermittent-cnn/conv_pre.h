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
        // when CHANNEL * kH * kW is odd, CHANNEL * kW (dest_offset) is
        // also odd, so dummy values are needed between slices to make
        // addresses even.
        // a dummy value for each slice (kW * CHANNEL q15 values)
        mac_params[uxIndex].length += kH + 1;
        dest_offset++;
    }
    buffer_size = sizeof(int16_t) * mac_params[uxIndex].length;

    uint8_t filter_limit = MIN_VAL(NUM_FILTERS, (LEA_BUFFER_SIZE - 4 - dest_offset * (kH + TILE_H - 1)) / (dest_offset * kH));

    /* copy filter data */
    if (!filter_buffer_addr[conv_params->conv_idx]) {
        filter_addr = get_q15_param(
            conv_params->conv_filter,
            (size_t)(conv_params->conv_idx * CHANNEL * kH * kW));
        int16_t filter_offset = kH * dest_offset;
        filter_buffer_addr[conv_params->conv_idx] = (int16_t*)buffer_iq31_mac_results(NUM_TASKS - 1) - filter_offset * (filter_buffer_id + 1);

        if (truncated) {
            int16_t *current_filter_buffer_addr = filter_buffer_addr[conv_params->conv_idx];
            for (uint16_t h = 0; h < kH; h++) {
                my_memcpy(current_filter_buffer_addr, filter_addr, kW * CHANNEL * sizeof(int16_t));
                current_filter_buffer_addr += dest_offset;
                filter_addr += kW * CHANNEL;
            }
        } else {
            my_memcpy(filter_buffer_addr[conv_params->conv_idx],
                      filter_addr,
                      buffer_size);
            if (truncated) {
                // dummy value
                filter_buffer_addr[conv_params->conv_idx][buffer_size / sizeof(int16_t) - 1] = 0;
            }
        }
        if (cached_filter_idx[filter_buffer_id] >= 0) {
            filter_buffer_addr[cached_filter_idx[filter_buffer_id]] = NULL;
        }
        cached_filter_idx[filter_buffer_id] = conv_params->conv_idx;
        filter_buffer_id++;
        if (filter_buffer_id == filter_limit) {
            filter_buffer_id = 0;
        }
    }

    if (conv_params->first_filter) {
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
                *dest;
        int16_t src_offset = W * CHANNEL;
        uint16_t inputs_len = LEA_BUFFER_SIZE - 4 - filter_limit * kH * dest_offset;
        if (!conv_params->output_h_offset) {
            next_input_buffer_addr = lea_buffer;
        }

        dest = input_buffer_addr[uxIndex] = next_input_buffer_addr;

        int32_t h_start,
                h_end = int16_min(field_size, H-1-conv_params->output_h);

        if (!conv_params->output_h_offset) {
            my_printf_debug("Reinitialize input buffer" NEWLINE);

            msp_fill_q15_params fill_params = {
                .length = inputs_len,
                .value = 0,
            };
            msp_status status = msp_fill_q15(&fill_params, lea_buffer);
            msp_checkStatus(status);

            h_start = int16_max(-field_size, -conv_params->output_h);
        } else {
            h_start = field_size;
        }

        dest += (h_start + field_size) * dest_offset + (w_start + field_size) * CHANNEL;

        my_printf_debug("h_start=%d ", h_start);
        my_printf_debug("h_end=%d" NEWLINE, h_end);

        size_t size = (size_t)((w_end-w_start+1) * CHANNEL * sizeof(uint16_t)); // in bytes
        if (h_start <= h_end) {
            src = input_addr + (h_start * W + w_start) * CHANNEL;
            my_printf_debug("Copying row to lea_buffer + %d" NEWLINE,
                            (int)(dest - lea_buffer));
            for (int32_t h = h_start; h <= h_end; h++) {
                my_memcpy(dest, src, size);
                src += src_offset;
                dest += dest_offset;
            }
        }
    } else{
        if (!conv_params->output_h_offset) {
            next_input_buffer_addr = lea_buffer;
        }
        input_buffer_addr[uxIndex] = next_input_buffer_addr;
    }

    /* XXX: assume stride=1 */
    next_input_buffer_addr += dest_offset; // dest_offset already calibrated for truncation
    my_printf_debug("Increment next_input_buffer_addr" NEWLINE);
    my_printf_debug("next_input_buffer_addr = lea_buffer + %d" NEWLINE, (int)(next_input_buffer_addr - lea_buffer));
    if (next_input_buffer_addr > lea_buffer + LEA_BUFFER_SIZE) {
        ERROR_OCCURRED();
    }
}
