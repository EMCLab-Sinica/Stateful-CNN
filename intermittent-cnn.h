#pragma once

#include <stdint.h>

struct ParameterInfo;
struct Model;
void print_results(struct Model *model, struct ParameterInfo *output_node);
void set_sample_index(struct Model *model, uint8_t index);
int run_model(struct Model *model, int8_t *ansptr, struct ParameterInfo **output_node_ptr);
void run_cnn_tests(uint16_t n_samples);
uint8_t get_state_bit(struct Model *model, uint8_t slot_id);
uint8_t get_value_state_bit(int16_t val);
void flip_state_bit(struct Model *model, uint8_t slot_id);
uint32_t recovery_from_state_bits(struct Model *model, struct ParameterInfo *output);
