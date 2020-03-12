#include "ops.h"

uint8_t expected_inputs_len[] = {2, 3, 2, 1, 1, 2, 1, };

handler handlers[] = {
	handle_add,
	handle_conv,
	handle_matmul,
	handle_maxpool,
	handle_relu,
	handle_reshape,
	handle_squeeze,
};
