#pragma once

#include <stdint.h>

struct ParameterInfo;
struct Model;
struct DBImage;
void print_results(struct Model *model, struct ParameterInfo *output_node);
void set_sample_index(struct Model *model, uint8_t index);
int run_model(struct DBImage *DB, struct Model *model, int8_t *ansptr, struct ParameterInfo **output_node_ptr);
void run_cnn_tests(uint16_t n_samples);
