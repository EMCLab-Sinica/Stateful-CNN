#define MY_NDEBUG
#define DUMP_INTEGERS

#ifndef MY_NDEBUG

#include "common.h"

void dump_params(ParameterInfo *cur_param);
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
