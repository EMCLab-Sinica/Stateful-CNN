cmake_minimum_required(VERSION 2.10)

project(intermittent-cnn)

option(USE_ARM_CMSIS "Use DSP from ARM CMSIS pack" OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wstrict-prototypes -Wshadow -g -O0")
# XXX: for packed struct Model. Is it possible to turn on this warning?
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-address-of-packed-member")

set(DSPLIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/TI-DSPLib)
set(ARM_CMSIS_DSP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ARM-CMSIS)

add_library(fake-msp430sdk
    fake-msp430sdk/msp430.c
)
target_include_directories(fake-msp430sdk
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/fake-msp430sdk
)

add_library(dsplib
    ${DSPLIB_PATH}/source/matrix/msp_matrix_mpy_q15.c
    ${DSPLIB_PATH}/source/vector/msp_add_q15.c
    ${DSPLIB_PATH}/source/utility/msp_interleave_q15.c
    ${DSPLIB_PATH}/source/utility/msp_fill_q15.c
)
target_include_directories(dsplib
    SYSTEM PUBLIC
        ${DSPLIB_PATH}/include
)
target_link_libraries(dsplib fake-msp430sdk)

add_library(arm_cmsis_dsp
    ${ARM_CMSIS_DSP_PATH}/DSP/Source/MatrixFunctions/arm_mat_init_q15.c
    ${ARM_CMSIS_DSP_PATH}/DSP/Source/MatrixFunctions/arm_mat_mult_fast_q15.c
    ${ARM_CMSIS_DSP_PATH}/DSP/Source/SupportFunctions/arm_fill_q15.c
)
target_include_directories(arm_cmsis_dsp
    PUBLIC
        ${ARM_CMSIS_DSP_PATH}/Include
        ${ARM_CMSIS_DSP_PATH}/DSP/Include
)
target_compile_definitions(arm_cmsis_dsp
    PRIVATE
        ARM_MATH_MATRIX_CHECK
)

add_executable(intermittent-cnn
    intermittent-cnn.c
    ops.c
    op_handlers.c
    conv.c
    common.c
    debug.c
    plat-linux.c
)
target_link_libraries(intermittent-cnn dsplib arm_cmsis_dsp fake-msp430sdk)

if (USE_ARM_CMSIS)
    target_compile_definitions(intermittent-cnn
        PRIVATE
            USE_ARM_CMSIS
    )
endif (USE_ARM_CMSIS)
