#pragma once

struct ParameterInfo;

#define Add 0
#define Concat 1
#define Conv 2
#define Dropout 3
#define GlobalAveragePool 4
#define MatMul 5
#define MaxPool 6
#define Relu 7
#define Reshape 8
#define Softmax 9
#define Squeeze 10
#define Transpose 11
void handle_add(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_add(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_concat(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_concat(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_conv(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_conv(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_dropout(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_dropout(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_globalaveragepool(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_globalaveragepool(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_matmul(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_matmul(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_maxpool(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_maxpool(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_relu(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_relu(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_reshape(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_reshape(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_softmax(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_softmax(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_squeeze(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_squeeze(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_transpose(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
uint32_t alloc_transpose(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
#define AUTO_PAD_VALID 1
#define NHWC2NCHW 2
