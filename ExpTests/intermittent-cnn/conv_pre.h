{
    /* copy filter data */
    if (!filter_buffer_addr[conv_params.conv_idx]) {
        kH = conv_params.conv_filter->dims[1];
        kW = conv_params.conv_filter->dims[2];
        CHANNEL = conv_params.conv_filter->dims[3];
        filter_addr = get_q15_param(
            conv_params.conv_filter,
            (size_t)(conv_params.conv_idx * CHANNEL * kH * kW));
        uint16_t buffer_size = sizeof(int16_t) * mac_params.length;
        int16_t filter_offset = kH * global_conv_params.dest_offset;
        filter_buffer_addr[conv_params.conv_idx] = (int16_t*)iq31_mac_results - filter_offset * (filter_buffer_id + 1);

        if (global_conv_params.truncated) {
            int16_t *current_filter_buffer_addr = filter_buffer_addr[conv_params.conv_idx];
            for (uint16_t h = 0; h < kH; h++) {
                my_memcpy(current_filter_buffer_addr, filter_addr, kW * CHANNEL * sizeof(int16_t));
                current_filter_buffer_addr += global_conv_params.dest_offset;
                filter_addr += kW * CHANNEL;
            }
        } else {
            my_memcpy(filter_buffer_addr[conv_params.conv_idx],
                      filter_addr,
                      buffer_size);
            if (global_conv_params.truncated) {
                // dummy value
                filter_buffer_addr[conv_params.conv_idx][buffer_size / sizeof(int16_t) - 1] = 0;
            }
        }
        if (cached_filter_idx[filter_buffer_id] >= 0) {
            filter_buffer_addr[cached_filter_idx[filter_buffer_id]] = NULL;
        }
        cached_filter_idx[filter_buffer_id] = conv_params.conv_idx;
        filter_buffer_id++;
        if (filter_buffer_id == global_conv_params.filter_limit) {
            filter_buffer_id = 0;
        }
    }

    uint8_t scheduled_filters = (conv_params.flags & 0x00ff) >> 1;
    my_printf_debug("scheduled_filters = %d" NEWLINE, scheduled_filters);
    my_printf_debug("conv_params.output_h = %d" NEWLINE, conv_params.output_h);
    my_printf_debug("conv_params.starting_output_h = %d" NEWLINE, conv_params.starting_output_h);
    if (scheduled_filters == 0 || (scheduled_filters == 1 && conv_params.output_h < conv_params.starting_output_h)) {
        kH = conv_params.conv_filter->dims[1];
        W = conv_params.conv_input->dims[2];
        CHANNEL = conv_params.conv_filter->dims[3];
        int8_t field_size = (int8_t)((kH - 1) / 2);

        /* copy input data, row by row */

        input_addr = get_q15_param(
            conv_params.conv_input,
            (size_t)(CHANNEL * (
                    (conv_params.output_h) * W +
                    (conv_params.output_w)
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
        int32_t w_start = int16_max(-field_size,    -conv_params.output_w),
                w_end   = int16_min( field_size, W-1-conv_params.output_w);
        int16_t *src = NULL,
                *dest;
        int16_t src_offset = W * CHANNEL;
        uint16_t inputs_len = LEA_BUFFER_SIZE - 4 - global_conv_params.filter_limit * kH * global_conv_params.dest_offset;
        if (conv_params.do_reinitialize_input) {
            next_input_buffer_addr = lea_buffer + conv_params.starting_output_h_offset * global_conv_params.dest_offset;
        }

        dest = input_buffer_addr = next_input_buffer_addr;

        H = conv_params.conv_input->dims[1];
        int32_t h_start,
                h_end = int16_min(field_size, H-1-conv_params.output_h);

        if (conv_params.do_reinitialize_input) {
            my_printf_debug("Reinitialize input buffer" NEWLINE);

            if (scheduled_filters == 0) {
                msp_fill_q15_params fill_params = {
                    .length = inputs_len,
                    .value = 0,
                };
                msp_status status = msp_fill_q15(&fill_params, lea_buffer);
                msp_checkStatus(status);
            }

            h_start = int16_max(-field_size, -conv_params.output_h);
        } else {
            h_start = field_size;
        }

        dest += (h_start + field_size) * global_conv_params.dest_offset + (w_start + field_size) * CHANNEL;

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
                dest += global_conv_params.dest_offset;
            }
        }
    } else{
        if (conv_params.do_reinitialize_input) {
            next_input_buffer_addr = lea_buffer;
        }
        input_buffer_addr = next_input_buffer_addr;
    }

    /* XXX: assume stride=1 */
    next_input_buffer_addr += global_conv_params.dest_offset; // dest_offset already calibrated for truncation
    my_printf_debug("Increment next_input_buffer_addr" NEWLINE);
    my_printf_debug("next_input_buffer_addr = lea_buffer + %d" NEWLINE, (int)(next_input_buffer_addr - lea_buffer));
    if (next_input_buffer_addr > lea_buffer + LEA_BUFFER_SIZE) {
        ERROR_OCCURRED();
    }
}
