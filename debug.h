#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h> // sprintf()
#include "platform.h"

#define MY_NDEBUG
#define DUMP_INTEGERS

#if defined(__MSP430__)
#  include "Tools/myuart.h"
#  define PRId32 "L" // see print2uart() in Tools/myuart.c
#  define PRIsize_t "l"
#  define my_printf print2uart
#  define NEWLINE "\r\n"
#elif defined(CYPRESS)
#  include <inttypes.h> // for PRId32
#  define PRIsize_t "zu"
#  ifdef CY_PSOC_CREATOR_USED
#    if (CY_CPU_CORTEX_M0P)  /* core is Cortex-M0+ */
#      include "UARTM0.h"
#    else /* core is Cortex-M4 */
#      include "UARTM4.h"
#    endif
#    define my_printf(format, ...) { \
        char uartString[100]; \
        sprintf(uartString, format, ##__VA_ARGS__); \
        Uprintf(uartString); \
    }
#  else
#    include <stdio.h>
#    define my_printf printf
#  endif
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
typedef struct Model Model;
typedef struct Node Node;

#ifndef MY_NDEBUG

void dump_params(struct ParameterInfo *cur_param);
void dump_matrix(int16_t *mat, size_t len);
void dump_matrix2(int16_t *mat, size_t rows, size_t cols);
#define print_q15_debug print_q15
#define print_iq31_debug print_iq31
#define my_printf_debug my_printf

#else

#define dump_params(cur_param)
#define dump_matrix(mat, len)
#define dump_matrix2(mat, rows, cols)
#define print_q15_debug(val)
#define print_iq31_debug(val)
#define my_printf_debug(...)

#endif

void dump_model(Model *model, Node *nodes);
