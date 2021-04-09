#pragma once

#include <stdint.h>
#include <stdlib.h>

#if defined(__MSP430__) || defined(__MSP432__)
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct ParameterInfo;
struct Model;
extern uint8_t dma_counter_enabled;

[[ noreturn ]] void ERROR_OCCURRED(void);
void my_memcpy(void* dest, const void* src, size_t n);
void my_memcpy_to_param(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n, uint16_t timer_delay);
void my_memcpy_from_intermediate_values(void *dest, const struct ParameterInfo *param, uint16_t offset_in_word, size_t n);
void read_from_samples(void *dest, uint16_t offset_in_word, size_t n);
void check_nvm_write_address(uint32_t nvm_offset, size_t n);
struct ParameterInfo* get_intermediate_parameter_info(uint8_t i);
void commit_intermediate_parameter_info(uint8_t i);
struct Model* get_model(void);
void commit_model(void);
uint16_t read_max_multiplier(const struct ParameterInfo* param);
void write_max_multiplier(const struct ParameterInfo* param, uint16_t max_multiplier);
void first_run(void);
void notify_model_finished(void);
uint64_t get_nvm_writes(void);
#if HAWAII
void write_hawaii_layer_footprint(uint16_t layer_idx, int16_t n_jobs);
uint16_t read_hawaii_layer_footprint(uint16_t layer_idx);
void reset_hawaii_layer_footprint(uint16_t layer_idx);
#endif

#ifdef __cplusplus
}
#endif
