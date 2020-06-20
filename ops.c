#include "cnn_common.h"

#include "ops.h"

uint8_t expected_inputs_len[] = {2, 2, 3, 1, 1, 2, 1, 1, 2, 1, 1, 1, };

uint8_t inplace_update[] = {0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, };

handler handlers[] = {
	handle_add,
	handle_concat,
	handle_conv,
	handle_dropout,
	handle_globalaveragepool,
	handle_matmul,
	handle_maxpool,
	handle_relu,
	handle_reshape,
	handle_softmax,
	handle_squeeze,
	handle_transpose,
};
