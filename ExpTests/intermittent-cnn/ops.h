#pragma once

#include "common.h"

#define Add 0
#define Conv 1
#define MatMul 2
#define MaxPool 3
#define Relu 4
#define Reshape 5
#define Squeeze 6
void handle_add(ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
void handle_conv(ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
void handle_matmul(ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
void handle_maxpool(ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
void handle_relu(ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
void handle_reshape(ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
void handle_squeeze(ParameterInfo *input[], ParameterInfo *output, uint16_t flags);
#define CONV_BIAS_MERGED 1
#define TRANSPOSED 2
