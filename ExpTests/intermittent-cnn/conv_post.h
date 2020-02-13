{
    /* START dump data */
    my_printf_debug("uxIndex=%d ", uxIndex);
    my_printf_debug("conv_idx=%d ", conv_params->conv_idx);
    my_printf_debug("output_h=%d ", conv_params->output_h);
    my_printf_debug("output_w=%d" NEWLINE, conv_params->output_w);

#ifdef CACHED_INPUTS
    my_printf_debug("input_buffer_addr = lea_buffer.conv.input + %d" NEWLINE, (int)(input_buffer_addr[uxIndex] - lea_buffer.conv.input));
#endif
    my_printf_debug("input" NEWLINE);
#ifdef CACHED_INPUTS
    dump_matrix(input_buffer_addr[uxIndex], mac_params[uxIndex].length);
#else
    dump_matrix(lea_buffer.conv.input, mac_params[uxIndex].length);
#endif
    my_printf_debug("filter" NEWLINE);
    dump_matrix(lea_buffer.conv.filter, mac_params[uxIndex].length);

    my_printf_debug("iq31_mac_result=");
    print_iq31_debug(lea_buffer.conv.iq31_mac_result[uxIndex]);
    my_printf_debug(NEWLINE);
    /* END dump data */

    int16_t q15_mac_result = iq31_to_q15(lea_buffer.conv.iq31_mac_result[uxIndex]);
    q15_mac_result += *get_q15_param(conv_params->bias, conv_params->conv_idx);

    my_printf_debug("after adding bias OFM value=");
    print_q15_debug(q15_mac_result);
    my_printf_debug(NEWLINE);

    int16_t *output_data = get_q15_param(conv_params->output, 0);
    size_t offset = (size_t)(conv_params->output_h * W * OUTPUT_CHANNEL + conv_params->output_w * OUTPUT_CHANNEL + conv_params->conv_idx);
    my_printf_debug("offset of output_data=%" PRIsize_t NEWLINE, offset);
    output_data[offset] = q15_mac_result;
}
