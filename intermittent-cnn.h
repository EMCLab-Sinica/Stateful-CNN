#pragma once

#include <stdint.h>

void print_results(void);
void init_pointers(void);
void set_sample_index(uint8_t index);
int run_model(int8_t *ansptr);
void run_cnn_tests(uint16_t n_samples);
