#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h> // sprintf()
#include "platform.h"

#define MY_NDEBUG

#if defined(__MSP430__) || defined(__MSP432__)
#  include "Tools/myuart.h"
#  define my_printf print2uart_new
#  define my_flush()
#  define NEWLINE "\r\n"
#else
#  include <stdio.h>
#  define my_printf printf
#  define my_flush() do { fflush(stdout); } while (0);
#  define NEWLINE "\n"
#endif

#define MY_ASSERT(cond) if (!(cond)) { my_printf("Assertion failed at %s:%d" NEWLINE, __FILE__, __LINE__); ERROR_OCCURRED(); }

struct ParameterInfo;
struct Model;
struct Node;

struct ValueInfo {
    ValueInfo(ParameterInfo *cur_param, Model *model = nullptr);
    ValueInfo() {}

    uint16_t scale;
#ifdef WITH_PROGRESS_EMBEDDING
    uint8_t state;
#endif
};

extern uint8_t dump_integer;

void dump_value(struct Model *model, struct ParameterInfo *cur_param, size_t offset);
void dump_matrix(const int16_t *mat, size_t len, const ValueInfo& val_info);
void dump_matrix(ParameterInfo *param, uint16_t offset, uint16_t len, const ValueInfo& val_info);
void dump_params(struct Model *model, struct ParameterInfo *cur_param);
void dump_params_nhwc(struct Model *model, struct ParameterInfo *cur_param, size_t offset);
void dump_model(struct Model *model, struct Node *nodes);

#ifndef MY_NDEBUG

void dump_matrix2(int16_t *mat, size_t rows, size_t cols, const ValueInfo& val_info);
#define dump_value_debug dump_value
#define dump_matrix_debug dump_matrix
#define dump_params_debug dump_params
#define dump_params_nhwc_debug dump_params_nhwc
#define dump_model_debug dump_model
#define my_printf_debug my_printf

#else

#define dump_matrix2(...)
#define dump_value_debug(...)
#define dump_matrix_debug(...)
#define dump_params_debug(...)
#define dump_params_nhwc_debug(...)
#define dump_model_debug(...)
#define my_printf_debug(...)

#endif

