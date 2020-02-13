#pragma once

#include <stdint.h>
#include <stddef.h>

#define MY_NDEBUG
#define DUMP_INTEGERS

#if defined(__linux__)
#  include <stdio.h>
#  include <inttypes.h> // for PRId32
#  define PRIsize_t "zu"
#  define my_printf printf
#  define NEWLINE "\n"
#elif defined(__MSP430__)
#  include "Tools/myuart.h"
#  define PRId32 "L" // see print2uart() in Tools/myuart.c
#  define PRIsize_t "l"
#  define my_printf print2uart
#  define NEWLINE "\r\n"
#endif

void print_q15(int16_t val);
void print_iq31(int32_t val);

#ifndef MY_NDEBUG

struct _ParameterInfo;
void dump_params(struct _ParameterInfo *cur_param);
void dump_matrix(int16_t *mat, size_t len);
void dump_model(void);
#define print_q15_debug print_q15
#define print_iq31_debug print_iq31
#define my_printf_debug my_printf

#else

#define dump_params(cur_param)
#define dump_matrix(mat, len)
#define dump_model()
#define print_q15_debug(val)
#define print_iq31_debug(val)
#define my_printf_debug(...)

#endif
