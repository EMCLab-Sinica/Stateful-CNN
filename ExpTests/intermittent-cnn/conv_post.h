#ifdef DUMP_PARAMS
    if (dump_conv_params) {
        my_printf("uxIndex=%d ", uxIndex);
        my_printf("conv_idx=%d ", conv_params->conv_idx);
        my_printf("output_h=%d ", conv_params->output_h);
        my_printf("output_w=%d" NEWLINE, conv_params->output_w);

        my_printf("input" NEWLINE);
        dump_matrix(lea_buffer.conv.input[uxIndex], mac_params[uxIndex].length);
        my_printf("filter" NEWLINE);
        dump_matrix(lea_buffer.conv.filter[uxIndex], mac_params[uxIndex].length);
# ifdef __MSP430__
        my_printf("iq31_mac_result=%l" NEWLINE, lea_buffer.conv.iq31_mac_result[uxIndex]);
# else
        my_printf("iq31_mac_result=%d" NEWLINE, lea_buffer.conv.iq31_mac_result[uxIndex]);
# endif
    }
#endif

    if (truncated[uxIndex]) {
        uint16_t last_idx = (uint16_t)(kH * kW - 1);
        lea_buffer.conv.iq31_mac_result[uxIndex] += (*get_q15_param(conv_params->conv_input, last_idx)) * (*get_q15_param(conv_params->conv_filter, last_idx)) * 2;
    }

    {
        int16_t q15_mac_result = iq31_to_q15(&lea_buffer.conv.iq31_mac_result[uxIndex]);
        q15_mac_result = (int16_t)(q15_mac_result + *get_q15_param(conv_params->bias, conv_params->conv_idx));

        int16_t *output_data = get_q15_param(conv_params->output, 0);
        output_data[conv_params->conv_idx * H * W + conv_params->output_h * W + conv_params->output_w] = q15_mac_result;
    }

