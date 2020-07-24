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

// XXX: get rid of `state` variables if WITH_PROGRESS_EMBEDDING is not defined?
void print_q15(int16_t val, uint16_t scale, uint8_t state);
void print_iq31(int32_t val, uint16_t scale);

struct ParameterInfo;
struct Model;
struct Node;

void dump_value(struct Model *model, struct ParameterInfo *cur_param, size_t offset);

#ifndef MY_NDEBUG

void dump_params(struct Model *model, struct ParameterInfo *cur_param);
void dump_params_nhwc(struct Model *model, struct ParameterInfo *cur_param, size_t offset);
void dump_matrix(int16_t *mat, size_t len, uint16_t scale, uint8_t state);
void dump_matrix2(int16_t *mat, size_t rows, size_t cols, uint16_t scale, uint8_t state);
#define print_q15_debug print_q15
#define print_iq31_debug print_iq31
#define my_printf_debug my_printf

#else

#define dump_params(model, cur_param)
#define dump_params_nhwc(model, cur_param, offset)
#define dump_matrix(mat, len, scale, state)
#define dump_matrix2(mat, rows, cols, scale, state)
#define print_q15_debug(val, scale, state)
#define print_iq31_debug(val, scale)
#define my_printf_debug(...)

#endif

void dump_model(struct Model *model, struct Node *nodes);
