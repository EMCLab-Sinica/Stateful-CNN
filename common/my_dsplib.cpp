#ifndef USE_ARM_CMSIS
#include <DSPLib.h>
#else
#include <arm_math.h>
#endif

#include "my_dsplib.h"
#include "platform.h"
#include "my_debug.h"
#include "op_utils.h"

#ifndef USE_ARM_CMSIS
#define my_checkStatus(status) MY_ASSERT(status == MSP_SUCCESS, "Error from TI-DSPLib: %d" NEWLINE, status)
#endif

static inline void check_buffer_address(const int16_t* addr) {
    MY_ASSERT(addr >= lea_buffer && addr < lea_buffer + LEA_BUFFER_SIZE);
    MY_ASSERT((addr - lea_buffer) % 2 == 0);
}

void my_add_q15(const int16_t *pSrcA, const int16_t *pSrcB, int16_t *pDst, uint32_t blockSize) {
    check_buffer_address(pSrcA);
    check_buffer_address(pSrcB);
    check_buffer_address(pDst);
#ifndef USE_ARM_CMSIS
    uint32_t blockSizeForLEA = blockSize / 2 * 2;
    if (blockSizeForLEA) {
        msp_add_q15_params add_params;
        add_params.length = blockSizeForLEA;
        msp_status status = msp_add_q15(&add_params, pSrcA, pSrcB, pDst);
        my_checkStatus(status);
    }
    if (blockSize % 2) {
        pDst[blockSize - 1] = pSrcA[blockSize - 1] + pSrcB[blockSize - 1];
    }
#else
    arm_add_q15(pSrcA, pSrcB, pDst, blockSize);
#endif
}

void my_fill_q15(int16_t value, int16_t *pDst, uint32_t blockSize) {
    check_buffer_address(pDst);
#ifndef USE_ARM_CMSIS
    uint32_t blockSizeForLEA = blockSize / 2 * 2;
    if (blockSizeForLEA) {
        msp_fill_q15_params fill_params;
        fill_params.length = blockSizeForLEA;
        fill_params.value = value;
        msp_status status = msp_fill_q15(&fill_params, pDst);
        my_checkStatus(status);
    }
    if (blockSize % 2) {
        pDst[blockSize - 1] = value;
    }
#else
    arm_fill_q15(value, pDst, blockSize);
#endif
}

void my_offset_q15(const int16_t *pSrc, int16_t offset, int16_t *pDst, uint32_t blockSize) {
#ifndef USE_ARM_CMSIS
    // XXX: the alignment adjustment code in this function only supports pSrc == pDst
    MY_ASSERT(pSrc == pDst);
    // if pSrc is not 4-byte aligned...
    if (reinterpret_cast<uint64_t>(pSrc) & 3) {
        *pDst = *pSrc + offset;
        pSrc++;
        pDst++;
        MY_ASSERT(blockSize); // avoid overflow in the next line
        blockSize--;
    }
    check_buffer_address(pSrc);
    check_buffer_address(pDst);
    // LEA does not like zero-sized blocks
    uint16_t block_size_for_lea = blockSize / 2 * 2;
    if (block_size_for_lea) {
        msp_offset_q15_params offset_params;
        offset_params.length = block_size_for_lea;
        offset_params.offset = offset;
        msp_status status = msp_offset_q15(&offset_params, pSrc, pDst);
        my_checkStatus(status);
    }
    if (blockSize % 2) {
        pDst[blockSize - 1] = pSrc[blockSize - 1] + offset;
    }
#else
    arm_offset_q15(pSrc, offset, pDst, blockSize);
#endif
}

void my_max_q15(const int16_t *pSrc, uint32_t blockSize, int16_t *pResult, uint16_t *pIndex) {
    uint8_t unaligned = 0;
    if ((pSrc - lea_buffer) % 2) {
        unaligned = 1;
        pSrc++;
        MY_ASSERT(blockSize > 0);
        blockSize--;
    }
#ifndef USE_ARM_CMSIS
    uint32_t blockSizeForLEA = blockSize / 2 * 2;
    if (blockSizeForLEA) {
        msp_max_q15_params max_params;
        max_params.length = blockSizeForLEA;
        msp_status status = msp_max_q15(&max_params, pSrc, pResult, pIndex);
        my_checkStatus(status);
    }
    if (blockSize % 2) {
        if (*pResult < pSrc[blockSize - 1]) {
            *pResult = pSrc[blockSize - 1];
            *pIndex = blockSize - 1;
        };
    }
#else
    uint32_t pIndex_u32;
    arm_max_q15(pSrc, blockSize, pResult, &pIndex_u32);
    *pIndex = pIndex_u32;
#endif
    if (unaligned) {
        int16_t candidate = *(pSrc - 1); // -1 as pSrc was +1
        if (*pResult > candidate) {
            (*pIndex)++;
        } else {
            *pIndex = 0;
            *pResult = candidate;
        }
    }
}

void my_min_q15(const int16_t *pSrc, uint32_t blockSize, int16_t *pResult, uint16_t *pIndex) {
    uint8_t unaligned = 0;
    if ((pSrc - lea_buffer) % 2) {
        unaligned = 1;
        pSrc++;
        MY_ASSERT(blockSize > 0);
        blockSize--;
    }
#ifndef USE_ARM_CMSIS
    uint32_t blockSizeForLEA = blockSize / 2 * 2;
    if (blockSizeForLEA) {
        msp_min_q15_params min_params;
        min_params.length = blockSizeForLEA;
        msp_status status = msp_min_q15(&min_params, pSrc, pResult, pIndex);
        my_checkStatus(status);
    }
    if (blockSize % 2) {
        if (*pResult > pSrc[blockSize - 1]) {
            *pResult = pSrc[blockSize - 1];
            *pIndex = blockSize - 1;
        };
    }
#else
    uint32_t pIndex_u32;
    arm_min_q15(pSrc, blockSize, pResult, &pIndex_u32);
    *pIndex = pIndex_u32;
#endif
    if (unaligned) {
        int16_t candidate = *(pSrc - 1); // -1 as pSrc was +1
        if (*pResult < candidate) {
            (*pIndex)++;
        } else {
            *pIndex = 0;
            *pResult = candidate;
        }
    }
}

#ifdef USE_ARM_CMSIS
static int16_t pState[ARM_PSTATE_LEN];
#endif

void my_matrix_mpy_q15(uint16_t A_rows, uint16_t A_cols, uint16_t B_rows, uint16_t B_cols, int16_t *pSrcA, int16_t *pSrcB, int16_t *pDst) {
    // XXX: LEA doc requires all matrix dimensions to be even, while LEA
    // appears to still give correct results when srcARows is odd
    // srcBCols should really be even, though
    // http://e2e.ti.com/support/microcontrollers/msp430/f/166/t/716353?MSP430FR5992-MSP-DSPLib-msp-matrix-mpy-q15
    MY_ASSERT((A_cols & 1) || (B_cols & 1) == 0);
    MY_ASSERT(B_rows * B_cols <= ARM_PSTATE_LEN);
    MY_ASSERT(A_cols == B_rows);
    check_buffer_address(pSrcA);
    check_buffer_address(pSrcB);
#ifndef USE_ARM_CMSIS
    msp_matrix_mpy_q15_params matrix_mpy_params;
    matrix_mpy_params.srcARows = A_rows;
    matrix_mpy_params.srcACols = A_cols;
    matrix_mpy_params.srcBRows = B_rows;
    matrix_mpy_params.srcBCols = B_cols;
    msp_status status = msp_matrix_mpy_q15(&matrix_mpy_params, pSrcA, pSrcB, pDst);
    my_checkStatus(status);
#else
    arm_matrix_instance_q15 A, B, C;
    arm_mat_init_q15(&A, A_rows, A_cols, pSrcA);
    arm_mat_init_q15(&B, B_rows, B_cols, pSrcB);
    arm_mat_init_q15(&C, A_rows, B_cols, pDst);
    arm_status status;
    status = arm_mat_mult_fast_q15(&A, &B, &C, pState);
    MY_ASSERT(status == ARM_MATH_SUCCESS);
#endif
}

void my_scale_q15(const int16_t *pSrc, int16_t scaleFract, uint8_t shift, int16_t *pDst, uint32_t blockSize) {
#ifndef USE_ARM_CMSIS
    uint32_t blockSizeForLEA = blockSize / 2 * 2;
    if (blockSizeForLEA) {
        msp_scale_q15_params scale_params;
        scale_params.length = blockSizeForLEA;
        scale_params.scale = scaleFract;
        scale_params.shift = shift;
        msp_status status = msp_scale_q15(&scale_params, pSrc, pDst);
        my_checkStatus(status);
    }
    if (blockSize % 2) {
        pDst[blockSize - 1] = (pSrc[blockSize - 1] * scaleFract) >> (15 - shift);
    }
#else
    arm_scale_q15(pSrc, scaleFract, shift, pDst, blockSize);
#endif
}

void my_interleave_q15(const int16_t *pSrc, uint16_t channel, uint16_t numChannels, int16_t *pDst, uint32_t blockSize) {
    MY_ASSERT(channel < numChannels);
    // XXX: not using LEA here as pSrc and/or pDst is often unaligned
    // CMSIS does not have interleave (yet)
    for (uint32_t idx = 0; idx < blockSize; idx++) {
        *(pDst + channel) = *pSrc;
        pSrc++;
        pDst += numChannels;
    }
}

void my_deinterleave_q15(const int16_t *pSrc, uint16_t channel, uint16_t numChannels, int16_t *pDst, uint32_t blockSize) {
    // XXX: not using LEA here as I didn't allocate LEA memory for inputs with footprints
    for (uint32_t idx = 0; idx < blockSize; idx++) {
        *pDst = *(pSrc + channel);
        pSrc += numChannels;
        pDst++;
    }
}
