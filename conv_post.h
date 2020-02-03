{
#ifdef DUMP_CONV_PARAMS
    my_printf("uxIndex=%d ", uxIndex);
    my_printf("conv_idx=%d ", conv_params->conv_idx);
    my_printf("output_h=%d ", conv_params->output_h);
    my_printf("output_w=%d" NEWLINE, conv_params->output_w);

#ifdef CACHED_INPUTS
    my_printf("input_buffer_addr = lea_buffer.conv.input[%d] + %d" NEWLINE, uxIndex, (int)(input_buffer_addr[uxIndex] - lea_buffer.conv.input[uxIndex]));
#endif
    my_printf("input" NEWLINE);
#ifdef CACHED_INPUTS
    dump_matrix(input_buffer_addr[uxIndex], mac_params[uxIndex].length);
#else
    dump_matrix(lea_buffer.conv.input[uxIndex], mac_params[uxIndex].length);
#endif
    my_printf("filter" NEWLINE);
    dump_matrix(lea_buffer.conv.filter, mac_params[uxIndex].length);

    my_printf("iq31_mac_result=");
    print_iq31(lea_buffer.conv.iq31_mac_result[uxIndex]);
    my_printf(NEWLINE);
#endif

    int16_t q15_mac_result = iq31_to_q15(lea_buffer.conv.iq31_mac_result[uxIndex]);
    q15_mac_result += *get_q15_param(conv_params->bias, conv_params->conv_idx);

#ifdef DUMP_CONV_PARAMS
    my_printf("after adding bias OFM value=");
    print_q15(q15_mac_result);
    my_printf(NEWLINE);
#endif

    int16_t *output_data = get_q15_param(conv_params->output, 0);
    size_t offset = (size_t)(conv_params->output_h * W * OUTPUT_CHANNEL + conv_params->output_w * OUTPUT_CHANNEL + conv_params->conv_idx);
#ifdef DUMP_CONV_PARAMS
# ifdef __MSP430__
    my_printf("offset of output_data=%l" NEWLINE, offset);
# else
    my_printf("offset of output_data=%ld" NEWLINE, offset);
# endif
#endif
    output_data[offset] = q15_mac_result;

#ifdef CACHED_INPUTS
    /* XXX: assume stride=1 and offset-by-1-row in two consecutive tasks */
    int16_t dest_offset = kW * CHANNEL;
    input_buffer_addr[uxIndex] += dest_offset;
    if (dest_offset % 2) {
        input_buffer_addr[uxIndex]++;
    }
#endif
}

