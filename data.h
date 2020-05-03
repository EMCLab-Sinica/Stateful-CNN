
#pragma once
#include <stdint.h>
#include "platform.h"

#define SCALE 50
#define NUM_SLOTS 2
#define INTERMEDIATE_VALUES_SIZE 13000u
#define COUNTERS_LEN 64

extern uint8_t *parameters_data;
#define PARAMETERS_DATA_LEN 12420

extern uint8_t *model_data;
#define MODEL_DATA_LEN 528

extern uint8_t *counters_data;
#define COUNTERS_DATA_LEN 258

#if USE_ALL_SAMPLES

extern uint8_t *samples_data;
#define SAMPLES_DATA_LEN 31360

extern uint8_t *labels_data;
#define LABELS_DATA_LEN 20

#else

extern uint8_t *samples_data;
#define SAMPLES_DATA_LEN 1568

extern uint8_t *labels_data;
#define LABELS_DATA_LEN 1

#endif
