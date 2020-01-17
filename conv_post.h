{
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

        my_printf("iq31_mac_result=");
        print_iq31(lea_buffer.conv.iq31_mac_result[uxIndex]);
        my_printf(NEWLINE);
    }
#endif

    int16_t q15_mac_result = iq31_to_q15(lea_buffer.conv.iq31_mac_result[uxIndex]);
    q15_mac_result = (int16_t)(q15_mac_result + *get_q15_param(conv_params->bias, conv_params->conv_idx));

#ifdef DUMP_PARAMS
    if (dump_conv_params) {
        my_printf("after adding bias OFM value=");
        print_q15(q15_mac_result);
        my_printf(NEWLINE);
    }
#endif

    int16_t *output_data = get_q15_param(conv_params->output, 0);
    size_t offset = (size_t)(conv_params->output_h * W * OUTPUT_CHANNEL + conv_params->output_w * OUTPUT_CHANNEL + conv_params->conv_idx);
#ifdef DUMP_PARAMS
    if (dump_conv_params) {
        my_printf("offset of output_data=%ld" NEWLINE, offset);
    }
#endif
    output_data[offset] = q15_mac_result;
}

