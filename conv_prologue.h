/* put var declarations first to make the compiler happy */
int16_t *input_addr, *filter_addr;
/* Cannot use C as a variable name here as C is a macro on MSP430 :( */
uint16_t H, W, kH, kW, CHANNEL;
