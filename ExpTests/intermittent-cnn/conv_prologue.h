/* put var declarations first to make the compiler happy */
ConvTaskParams *conv_params;
int16_t *input_addr, *filter_addr;
uint16_t buffer_size;
/* Cannot use C as a variable name here as C is a macro on MSP430 :( */
uint16_t H, W, kH, kW, CHANNEL, OUTPUT_CHANNEL;
