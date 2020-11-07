#pragma once

#include <stdint.h>
#include <stdlib.h>

void read_from_nvm(void* vm_buffer, uint32_t nvm_offset, size_t n);
void write_to_nvm(const void* vm_buffer, uint32_t nvm_offset, size_t n);
void my_erase(uint32_t nvm_offset, size_t n);
