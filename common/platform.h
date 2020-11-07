#pragma once

#include <stdint.h>
#include <stdlib.h>

#if defined(__MSP430__) || defined(__MSP432__)
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

/* external FRAM layout:
 * 0, +NUM_SLOTS * INTERMEDIATE_VALUES_SIZE: intermediate values
 * INTERMEDIATE_PARAMETERS_INFO_OFFSET, +INTERMEDIATE_PARAMETERS_INFO_DATA_LEN: intermediate parameters info
 * MODEL_OFFSET, +2 * sizeof(Model): two shadow copies of Model
 * FIRST_RUN_OFFSET, +sizeof(uint8_t): first run?
 */

#define INTERMEDIATE_PARAMETERS_INFO_OFFSET (NVM_SIZE - 0x10000)
#define MODEL_OFFSET (NVM_SIZE - 0x8000)
#define FIRST_RUN_OFFSET (NVM_SIZE - 0x7600)
#define COUNTERS_OFFSET (NVM_SIZE - 0x7400)

[[ noreturn ]] void ERROR_OCCURRED(void);
void my_memcpy(void* dest, const void* src, size_t n);
void my_memcpy_to_param(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n);
void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);
ParameterInfo* get_intermediate_parameter_info(uint8_t i);
void commit_intermediate_parameter_info(uint8_t i);
Model* get_model(void);
void commit_model(void);
void first_run(void);
void plat_print_results(void);
