#pragma once

struct ParameterInfo;

#define Add 0
#define Conv 1
#define MatMul 2
#define MaxPool 3
#define Relu 4
#define Reshape 5
#define Squeeze 6
void handle_add(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_conv(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_matmul(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_maxpool(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_relu(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_reshape(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_squeeze(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
#define CONV_BIAS_MERGED 1
#define TRANSPOSED 2
