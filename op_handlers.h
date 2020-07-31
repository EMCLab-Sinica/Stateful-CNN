#pragma once

#include "platform.h"

struct ParameterInfo;
struct Model;

extern int16_t lea_buffer[LEA_BUFFER_SIZE];
uint16_t find_overflow_factor(struct Model *model, struct ParameterInfo *param);
void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale);
