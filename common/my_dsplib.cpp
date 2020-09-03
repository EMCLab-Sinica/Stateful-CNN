#ifndef USE_ARM_CMSIS
#include <DSPLib.h>
#else
#include <arm_math.h>
#endif

#include "my_dsplib.h"
#include "platform.h"
#include "debug.h"

void my_add_q15(const int16_t *pSrcA, const int16_t *pSrcB, int16_t *pDst, uint32_t blockSize) {
#ifndef USE_ARM_CMSIS
    msp_add_q15_params add_params;
    add_params.length = blockSize;
    msp_status status = msp_add_q15(&add_params, pSrcA, pSrcB, pDst);
    msp_checkStatus(status);
#else
    arm_add_q15(pSrcA, pSrcB, pDst, blockSize);
#endif
}

void my_fill_q15(int16_t value, int16_t *pDst, uint32_t blockSize) {
#ifndef USE_ARM_CMSIS
    msp_fill_q15_params fill_params;
    fill_params.length = blockSize;
    fill_params.value = value;
    msp_status status = msp_fill_q15(&fill_params, pDst);
    msp_checkStatus(status);
#else
    arm_fill_q15(value, pDst, blockSize);
#endif
}

void my_offset_q15(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize) {
#ifndef USE_ARM_CMSIS
    msp_offset_q15_params offset_params;
    offset_params.length = blockSize;
    offset_params.offset = offset;
    msp_status status = msp_offset_q15(&offset_params, pSrc, pDst);
    msp_checkStatus(status);
#else
    arm_offset_q15(pSrc, offset, pDst, blockSize);
#endif
}

void my_max_q15(const int16_t *pSrc, uint32_t blockSize, int16_t *pResult, uint16_t *pIndex) {
#ifndef USE_ARM_CMSIS
    msp_max_q15_params max_params;
    max_params.length = blockSize;
    msp_status status = msp_max_q15(&max_params, pSrc, pResult, pIndex);
    msp_checkStatus(status);
#else
    uint32_t pIndex_u32;
    arm_max_q15(pSrc, blockSize, pResult, &pIndex_u32);
    *pIndex = pIndex_u32;
#endif
}

void my_min_q15(const int16_t *pSrc, uint32_t blockSize, int16_t *pResult, uint16_t *pIndex) {
#ifndef USE_ARM_CMSIS
    msp_min_q15_params min_params;
    min_params.length = blockSize;
    msp_status status = msp_min_q15(&min_params, pSrc, pResult, pIndex);
    msp_checkStatus(status);
#else
    uint32_t pIndex_u32;
    arm_min_q15(pSrc, blockSize, pResult, &pIndex_u32);
    *pIndex = pIndex_u32;
#endif
}

void my_matrix_mpy_q15(uint16_t A_rows, uint16_t A_cols, uint16_t B_rows, uint16_t B_cols, int16_t *pSrcA, int16_t *pSrcB, int16_t *pDst, uint8_t B_transposed) {
#ifndef USE_ARM_CMSIS
    // XXX: LEA doc requires all matrix dimensions to be even, while LEA
    // appears to still give correct results when srcARows is odd
    // srcBCols should really be even, though
    // http://e2e.ti.com/support/microcontrollers/msp430/f/166/t/716353?MSP430FR5992-MSP-DSPLib-msp-matrix-mpy-q15
    MY_ASSERT((A_cols & 1) || (B_cols & 1) == 0);
    msp_matrix_mpy_q15_params matrix_mpy_params;
    matrix_mpy_params.srcARows = A_rows;
    matrix_mpy_params.srcACols = A_cols;
    matrix_mpy_params.srcBRows = B_rows;
    matrix_mpy_params.srcBCols = B_cols;
    msp_status status = msp_matrix_mpy_q15(&matrix_mpy_params, pSrcA, pSrcB, pDst);
    msp_checkStatus(status);
#else
    arm_matrix_instance_q15 A, B, C;
    arm_mat_init_q15(&A, A_rows, A_cols, pSrcA);
    arm_mat_init_q15(&B, B_rows, B_cols, pSrcB);
    arm_mat_init_q15(&C, A_rows, B_cols, pDst);
    arm_status status;
    if (B_transposed) {
        status = arm_mat_mult_fast_q15(&A, &B, &C, NULL);
    } else {
        int16_t pState[1024];
        MY_ASSERT(B_rows * B_cols < 1024);
        status = arm_mat_mult_fast_q15(&A, &B, &C, pState);
    }
    MY_ASSERT(status == ARM_MATH_SUCCESS);
#endif
}

void my_scale_q15(const int16_t *pSrc, int16_t scaleFract, uint8_t shift, int16_t *pDst, uint32_t blockSize) {
#ifndef USE_ARM_CMSIS
    msp_scale_q15_params scale_params;
    scale_params.length = blockSize;
    scale_params.scale = scaleFract;
    scale_params.shift = shift;
    msp_status status = msp_scale_q15(&scale_params, pSrc, pDst);
    msp_checkStatus(status);
#else
    arm_scale_q15(pSrc, scaleFract, shift, pDst, blockSize);
#endif
}

