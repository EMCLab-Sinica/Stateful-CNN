#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h> // sprintf()
#include "platform.h"

#define MY_NDEBUG
#define DUMP_INTEGERS

#if defined(__MSP430__) || defined(__MSP432__)
#  include "Tools/myuart.h"
#  define PRId32 "L" // see print2uart() in Tools/myuart.c
#  define PRId64 "L"
#  define PRIsize_t "l"
#  define my_printf print2uart
#  define my_flush()
#  define NEWLINE "\r\n"
#else
#  include <stdio.h>
#  include <inttypes.h> // for PRId32
#  define PRIsize_t "zu"
#  define my_printf printf
#  define my_flush() do { fflush(stdout); } while (0);
#  define NEWLINE "\n"
#endif

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

void dump_value(struct Model *model, struct ParameterInfo *cur_param, size_t offset);

#ifndef MY_NDEBUG

void dump_params(struct Model *model, struct ParameterInfo *cur_param);
void dump_params_nhwc(struct Model *model, struct ParameterInfo *cur_param, size_t offset);
void dump_matrix(int16_t *mat, size_t len, const ValueInfo& val_info);
void dump_matrix2(int16_t *mat, size_t rows, size_t cols, const ValueInfo& val_info);
#define dump_value_debug dump_value
#define my_printf_debug my_printf

#else

#define dump_params(model, cur_param)
#define dump_params_nhwc(model, cur_param, offset)
#define dump_matrix(...)
#define dump_matrix2(...)
#define dump_value_debug(...)
#define my_printf_debug(...)

#endif

void dump_model(struct Model *model, struct Node *nodes);
