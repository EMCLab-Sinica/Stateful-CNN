#pragma once

struct ParameterInfo;

#define Add 0
#define Concat 1
#define Conv 2
#define ConvMerge 3
#define Dropout 4
#define GlobalAveragePool 5
#define MatMul 6
#define MaxPool 7
#define Relu 8
#define Reshape 9
#define Softmax 10
#define Squeeze 11
#define Transpose 12
void handle_add(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_add(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_concat(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_concat(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_conv(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_conv(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_convmerge(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_convmerge(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_dropout(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_dropout(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_globalaveragepool(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_globalaveragepool(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_matmul(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_matmul(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_maxpool(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_maxpool(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_relu(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_relu(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_reshape(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_reshape(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_softmax(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_softmax(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_squeeze(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_squeeze(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_transpose(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void alloc_transpose(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
#define AUTO_PAD_VALID 1
#define NHWC2NCHW 2
#define TRANSPOSED 4
