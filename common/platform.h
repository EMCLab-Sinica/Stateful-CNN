#pragma once

#include <cstdint>
#include <cstdlib>

#if defined(__MSP430__) || defined(__MSP432__)
#  include "plat-msp430.h"
#else
#  include "plat-linux.h"
#endif

struct ParameterInfo;
struct Model;
struct Counters;
extern uint8_t dma_counter_enabled;

[[ noreturn ]] void ERROR_OCCURRED(void);
void my_memcpy(void* dest, const void* src, size_t n);
void my_memcpy_to_param(ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n, uint16_t timer_delay);
void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n);
void my_memcpy_from_parameters(void *dest, const ParameterInfo *param, uint16_t offset_in_bytes, size_t n);
void read_from_samples(void *dest, uint16_t offset_in_word, size_t n);
ParameterInfo* get_intermediate_parameter_info(uint8_t i);
void commit_intermediate_parameter_info(uint8_t i);
Model* get_model(void);
Model* load_model_from_nvm(void);
void commit_model(void);
void first_run(void);
void notify_model_finished(void);
uint64_t get_nvm_writes(void);
#if HAWAII
void write_hawaii_layer_footprint(uint16_t layer_idx, int16_t n_jobs);
uint16_t read_hawaii_layer_footprint(uint16_t layer_idx);
void reset_hawaii_layer_footprint(uint16_t layer_idx);
#endif
void start_cpu_counter(void);
// pointer to member https://stackoverflow.com/questions/670734/pointer-to-class-data-member
void stop_cpu_counter(uint32_t Counters::* mem_ptr);
