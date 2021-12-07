#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h> // sprintf()
#include "data.h"
#include "platform.h"

#define MY_DEBUG_NO_ASSERT 0
#define MY_DEBUG_NORMAL 1
#define MY_DEBUG_LAYERS 2
#define MY_DEBUG_VERBOSE 3

#ifndef MY_DEBUG
#define MY_DEBUG MY_DEBUG_NO_ASSERT
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

#if MY_DEBUG >= MY_DEBUG_NORMAL
#define MY_ASSERT(...) my_assert_impl(__FILE__, __LINE__, __VA_ARGS__)
#else
#define MY_ASSERT(...)
#endif
// for checks that need to run when MY_DEBUG == 0
#define MY_ASSERT_ALWAYS(...) my_assert_impl(__FILE__, __LINE__, __VA_ARGS__)

struct ParameterInfo;
struct Model;
struct Node;

struct ValueInfo {
    ValueInfo(const ParameterInfo *cur_param, Model *model = nullptr);
    ValueInfo() = delete;

    uint16_t scale;
};

extern uint8_t dump_integer;

void dump_value(struct Model *model, const ParameterInfo *cur_param, size_t offset, bool has_state = true);
void dump_matrix(const int16_t *mat, size_t len, const ValueInfo& val_info, bool has_state = true);
void dump_matrix(const int16_t *mat, size_t rows, size_t cols, const ValueInfo& val_info, bool has_state = true);
void dump_params(struct Model *model, const ParameterInfo *cur_param);
void dump_params_nhwc(struct Model *model, const ParameterInfo *cur_param);
void dump_model(struct Model *model);
void dump_turning_points(Model *model, const ParameterInfo *output);
void compare_vm_nvm_impl(int16_t* vm_data, Model* model, const ParameterInfo* output, uint16_t output_offset, uint16_t blockSize);
void check_nvm_write_address_impl(uint32_t nvm_offset, size_t n);

#if MY_DEBUG >= MY_DEBUG_VERBOSE

#define dump_value_debug dump_value
#define dump_matrix_debug dump_matrix
#define dump_model_debug dump_model
#define my_printf_debug my_printf
#define dump_turning_points_debug dump_turning_points

#else

#define dump_value_debug(...)
#define dump_matrix_debug(...)
#define dump_matrix_debug(...)
#define dump_model_debug(...)
#define my_printf_debug(...)
#define dump_turning_points_debug(...)

#endif

#if MY_DEBUG >= MY_DEBUG_LAYERS

#define dump_params_debug dump_params
#define dump_params_nhwc_debug dump_params_nhwc

#else

#define dump_params_debug(...)
#define dump_params_nhwc_debug(...)

#endif

#if MY_DEBUG >= MY_DEBUG_NORMAL

#define compare_vm_nvm compare_vm_nvm_impl
#define check_nvm_write_address check_nvm_write_address_impl

#else

#define compare_vm_nvm(...)
#define check_nvm_write_address(...)

#endif
