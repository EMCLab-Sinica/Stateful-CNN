
#pragma once
#include <stdint.h>

#define SCALE 50
#define NUM_SLOTS 2
#define INTERMEDIATE_VALUES_SIZE 65536

extern uint8_t *inputs_data;
#define INPUTS_DATA_LEN 36

extern uint8_t *parameters_data;
#define PARAMETERS_DATA_LEN 12420

extern uint8_t *samples_data;
#define SAMPLES_DATA_LEN 62720

extern uint8_t *model_data;
#define MODEL_DATA_LEN 492

extern uint8_t *labels_data;
#define LABELS_DATA_LEN 40
