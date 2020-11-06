#pragma once

#include <stdint.h>

// somehow MSP432 does not work with pState with length 2048
#define ARM_PSTATE_LEN 1024

void my_add_q15(const int16_t *pSrcA, const int16_t *pSrcB, int16_t *pDst, uint32_t blockSize);
void my_fill_q15(int16_t value, int16_t *pDst, uint32_t blockSize);
void my_offset_q15(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize);
void my_matrix_mpy_q15(uint16_t A_rows, uint16_t A_cols, uint16_t B_rows, uint16_t B_cols, int16_t *pSrcA, int16_t *pSrcB, int16_t *pDst, uint8_t B_transposed);
void my_max_q15(const int16_t *pSrc, uint32_t blockSize, int16_t *pResult, uint16_t *pIndex);
void my_min_q15(const int16_t *pSrc, uint32_t blockSize, int16_t *pResult, uint16_t *pIndex);
void my_scale_q15(const int16_t *pSrc, int16_t scaleFract, uint8_t shift, int16_t *pDst, uint32_t blockSize);
