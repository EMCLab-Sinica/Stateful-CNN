#pragma once

#include <cstdint>
#include <cstdlib>
struct ParameterInfo;

void my_add_q15(const int16_t *pSrcA, const int16_t *pSrcB, int16_t *pDst, uint32_t blockSize);
void my_fill_q15(int16_t value, int16_t *pDst, uint32_t blockSize);
void my_offset_q15(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize);
void my_matrix_mpy_q15(uint16_t A_rows, uint16_t A_cols, uint16_t B_rows, uint16_t B_cols, int16_t *pSrcA, int16_t *pSrcB, int16_t *pDst,
                       ParameterInfo *param, uint16_t offset_in_word, size_t values_to_preserve);
void my_vector_mult_q15(const int16_t *pSrcA, const int16_t *pSrcB, int16_t *pDst, uint32_t blockSize);
void my_max_q15(const int16_t *pSrc, uint32_t blockSize, int16_t *pResult, uint16_t *pIndex);
void my_min_q15(const int16_t *pSrc, uint32_t blockSize, int16_t *pResult, uint16_t *pIndex);
void my_scale_q15(const int16_t *pSrc, int16_t scaleFract, uint8_t shift, int16_t *pDst, uint32_t blockSize);
void my_interleave_q15(const int16_t *pSrc, uint16_t channel, uint16_t numChannels, int16_t *pDst, uint32_t blockSize);
void my_deinterleave_q15(const int16_t *pSrc, uint16_t channel, uint16_t numChannels, int16_t *pDst, uint32_t blockSize);
int16_t padding_for_lea(int16_t val);
void check_buffer_address(const int16_t* addr, uint32_t blockSize);
