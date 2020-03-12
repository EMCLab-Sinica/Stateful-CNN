#pragma once

#include "common.h"

#define Add 0
#define Conv 1
#define MatMul 2
#define MaxPool 3
#define Relu 4
#define Reshape 5
#define Squeeze 6
uint8_t handle_add(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags);
uint8_t handle_conv(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags);
uint8_t handle_matmul(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags);
uint8_t handle_maxpool(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags);
uint8_t handle_relu(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags);
uint8_t handle_reshape(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags);
uint8_t handle_squeeze(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags);
#define CONV_ACTIVATIONS_RELU 1
#define RELU_MERGED 1
