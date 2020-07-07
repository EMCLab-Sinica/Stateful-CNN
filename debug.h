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
#  define NEWLINE "\r\n"
#else
#  include <stdio.h>
#  include <inttypes.h> // for PRId32
#  define PRIsize_t "zu"
#  define my_printf printf
#  define NEWLINE "\n"
#endif

void print_q15(int16_t val);
void print_iq31(int32_t val);

struct ParameterInfo;
struct Model;
struct Node;

void dump_matrix(int16_t *mat, size_t len);

#ifndef MY_NDEBUG

void dump_params(struct ParameterInfo *cur_param);
void dump_params_nhwc(struct ParameterInfo *cur_param, size_t offset);
void dump_matrix2(int16_t *mat, size_t rows, size_t cols);
#define print_q15_debug print_q15
#define print_iq31_debug print_iq31
#define my_printf_debug my_printf

#else

#define dump_params(cur_param)
#define dump_params_nhwc(cur_param, offset)
#define dump_matrix2(mat, rows, cols)
#define print_q15_debug(val)
#define print_iq31_debug(val)
#define my_printf_debug(...)

#endif

void dump_model(struct Model *model, struct Node *nodes);
