#pragma once

#include "platform.h"

struct ParameterInfo;
struct Model;
struct msp_scale_q15_params;

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
uint16_t find_overflow_factor(struct Model *model, struct ParameterInfo *param);
void float_to_scale_params(struct msp_scale_q15_params *scale_params, float scale);
