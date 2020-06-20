#include "cnn_common.h"

#include "ops.h"

uint8_t expected_inputs_len[] = {2, 2, 3, 1, 1, 2, 1, 1, 2, 1, 1, 1, };

uint8_t inplace_update[] = {0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, };

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
allocator allocators[] = {
	alloc_add,
	alloc_concat,
	alloc_conv,
	alloc_dropout,
	alloc_globalaveragepool,
	alloc_matmul,
	alloc_maxpool,
	alloc_relu,
	alloc_reshape,
	alloc_softmax,
	alloc_squeeze,
	alloc_transpose,
};
uint32_t alloc_dropout(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags)
{
	UNUSED(flags);
	my_memcpy(output, input[0], sizeof(struct ParameterInfo));
	return 0;
}
uint32_t alloc_globalaveragepool(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags)
{
	UNUSED(flags);
	my_memcpy(output, input[0], sizeof(struct ParameterInfo));
	return 0;
}
uint32_t alloc_relu(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags)
{
	UNUSED(flags);
	my_memcpy(output, input[0], sizeof(struct ParameterInfo));
	return 0;
}
uint32_t alloc_reshape(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags)
{
	UNUSED(flags);
	my_memcpy(output, input[0], sizeof(struct ParameterInfo));
	return 0;
}
uint32_t alloc_softmax(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags)
{
	UNUSED(flags);
	my_memcpy(output, input[0], sizeof(struct ParameterInfo));
	return 0;
}
uint32_t alloc_squeeze(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags)
{
	UNUSED(flags);
	my_memcpy(output, input[0], sizeof(struct ParameterInfo));
	return 0;
}
uint32_t alloc_transpose(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags)
{
	UNUSED(flags);
	my_memcpy(output, input[0], sizeof(struct ParameterInfo));
	return 0;
}
