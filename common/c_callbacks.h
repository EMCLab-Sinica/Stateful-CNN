#pragma once

#include <stdint.h>
#include <stddef.h>

struct ParameterInfo;
typedef void (*data_preservation_func)(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n, uint16_t timer_delay);
