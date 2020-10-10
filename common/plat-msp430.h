#pragma once

#include "data.h"
#include <stdint.h>

#define LEA_BUFFER_SIZE 1884 // (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)

#define PLAT_LABELS_DATA_LEN 1

#ifdef __cplusplus
extern "C" {
#endif

void IntermittentCNNTest(void);
void button_pushed(uint16_t button1_status, uint16_t button2_status);

#ifdef __cplusplus
}
#endif
