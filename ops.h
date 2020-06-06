#pragma once

struct ParameterInfo;

#define Add 0
#define Conv 1
#define MatMul 2
#define MaxPool 3
#define Relu 4
#define Reshape 5
#define Squeeze 6
void handle_add(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_conv(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_matmul(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_maxpool(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_relu(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_reshape(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
void handle_squeeze(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);
