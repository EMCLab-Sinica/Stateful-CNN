#pragma once

#include "data.h"
#include <stdint.h>

#define PLAT_LABELS_DATA_LEN 1

#ifdef __cplusplus
extern "C" {
#endif

void IntermittentCNNTest(void);
void button_pushed(uint16_t button1_status, uint16_t button2_status);

#ifdef __cplusplus
}
#endif
