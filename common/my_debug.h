#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h> // sprintf()
#include "data.h"
#include "platform.h"

// 0: silent, assertion disabled
// 1: normal
// 2: verbose
#ifndef MY_DEBUG
#define MY_DEBUG 0
#endif

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

#if MY_DEBUG >= 1
template<typename ...Args>
void my_printf_wrapper(Args... args) {
    my_printf(args...);
}

template<>
void my_printf_wrapper();

template<typename ...Args>
void my_assert_impl(const char *file, uint16_t line, uint8_t cond, Args... args) {
    if (!cond) {
        my_printf("Assertion failed at %s:%d" NEWLINE, file, line);
        my_printf_wrapper(args...);
        ERROR_OCCURRED();
    }
}

#define MY_ASSERT(...) my_assert_impl(__FILE__, __LINE__, __VA_ARGS__)
#else
#define MY_ASSERT(...)
#endif

struct ParameterInfo;
struct Model;
struct Node;

struct ValueInfo {
    ValueInfo(const ParameterInfo *cur_param, Model *model = nullptr);
    ValueInfo() = delete;

    uint16_t scale;
};

extern uint8_t dump_integer;

void dump_value(struct Model *model, const ParameterInfo *cur_param, size_t offset);
void dump_matrix(const int16_t *mat, size_t len, const ValueInfo& val_info);
void dump_matrix(Model* model, ParameterInfo *param, uint16_t offset, uint16_t len, const ValueInfo& val_info);
void dump_matrix2(int16_t *mat, size_t rows, size_t cols, const ValueInfo& val_info);
void dump_params(struct Model *model, const ParameterInfo *cur_param);
void dump_params_nhwc(struct Model *model, const ParameterInfo *cur_param);
void dump_model(struct Model *model);
#if STATEFUL
void dump_turning_points(Model *model, const ParameterInfo *output);
#endif

#if MY_DEBUG >= 2

#define dump_value_debug dump_value
#define dump_matrix_debug dump_matrix
#define dump_matrix2_debug dump_matrix2
#define dump_params_debug dump_params
#define dump_params_nhwc_debug dump_params_nhwc
#define dump_model_debug dump_model
#define my_printf_debug my_printf
#if STATEFUL
#define dump_turning_points_debug dump_turning_points
#endif

#else

#define dump_value_debug(...)
#define dump_matrix_debug(...)
#define dump_matrix2_debug(...)
#define dump_params_debug(...)
#define dump_params_nhwc_debug(...)
#define dump_model_debug(...)
#define my_printf_debug(...)
#if STATEFUL
#define dump_turning_points_debug(...)
#endif

#endif

