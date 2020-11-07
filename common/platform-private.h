#pragma once

#include <stdint.h>
#include <stdlib.h>

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

void read_from_nvm(void* vm_buffer, uint32_t nvm_offset, size_t n);
void write_to_nvm(const void* vm_buffer, uint32_t nvm_offset, size_t n);
void my_erase(uint32_t nvm_offset, size_t n);
