cmake_minimum_required(VERSION 2.8)

project(intermittent-cnn)

option(USE_ARM_CMSIS "Use DSP from ARM CMSIS pack" OFF)

# Too many parameters are unused if some features are not enabled
set(COMMON_FLAGS "-Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter -g -O0 -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} -Wstrict-prototypes -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -std=c++11")

set(DSPLIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/TI-DSPLib)
set(ARM_CMSIS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ARM-CMSIS)

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
    ${DSPLIB_PATH}/source/vector/msp_offset_q15.c
    ${DSPLIB_PATH}/source/vector/msp_max_q15.c
    ${DSPLIB_PATH}/source/vector/msp_min_q15.c
    ${DSPLIB_PATH}/source/vector/msp_scale_q15.c
    ${DSPLIB_PATH}/source/utility/msp_interleave_q15.c
    ${DSPLIB_PATH}/source/utility/msp_fill_q15.c
)
target_include_directories(dsplib
    SYSTEM PUBLIC
        ${DSPLIB_PATH}/include
)
target_link_libraries(dsplib fake-msp430sdk)

add_library(arm_cmsis_dsp
    ${ARM_CMSIS_PATH}/DSP/Source/MatrixFunctions/arm_mat_init_q15.c
    ${ARM_CMSIS_PATH}/DSP/Source/MatrixFunctions/arm_mat_mult_fast_q15.c
    ${ARM_CMSIS_PATH}/DSP/Source/SupportFunctions/arm_fill_q15.c
)
target_include_directories(arm_cmsis_dsp
    PUBLIC
        ${ARM_CMSIS_PATH}/Core/Include
        ${ARM_CMSIS_PATH}/DSP/Include
)
target_compile_definitions(arm_cmsis_dsp
    PRIVATE
        ARM_MATH_MATRIX_CHECK
)

add_executable(intermittent-cnn
    intermittent-cnn.cpp
    data.cpp
    op_handlers.cpp
    conv.cpp
    cnn_common.cpp
    debug.cpp
    plat-linux.cpp
)
target_link_libraries(intermittent-cnn dsplib arm_cmsis_dsp fake-msp430sdk)

target_compile_definitions(intermittent-cnn
    PRIVATE
    MY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if (USE_ARM_CMSIS)
    target_compile_definitions(intermittent-cnn
        PRIVATE
            USE_ARM_CMSIS
    )
endif (USE_ARM_CMSIS)
