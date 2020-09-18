cmake_minimum_required(VERSION 2.8)

project(intermittent-cnn)

option(USE_ARM_CMSIS "Use DSP from ARM CMSIS pack" OFF)
set(MY_DEBUG "1" CACHE STRING "Local debug flag. See my_debug.h for details.")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Too many parameters are unused if some features are not enabled
set(COMMON_FLAGS "-Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter -g -O0 -pedantic")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} -Wstrict-prototypes -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -std=c++11")

if (USE_ARM_CMSIS)
    set(ARM_CMSIS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ARM-CMSIS)
else (USE_ARM_CMSIS)
    set(DSPLIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/TI-DSPLib)
endif (USE_ARM_CMSIS)
set(COMMON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_library(fake-msp430sdk
    fake-msp430sdk/msp430.c
)
target_include_directories(fake-msp430sdk
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/fake-msp430sdk
)

if (USE_ARM_CMSIS)
    # Keep this list in sync with ARM-CMSIS/create_symlinks.sh
    add_library(arm_cmsis_dsp
        ${ARM_CMSIS_PATH}/DSP/Source/BasicMathFunctions/arm_add_q15.c
        ${ARM_CMSIS_PATH}/DSP/Source/BasicMathFunctions/arm_offset_q15.c
        ${ARM_CMSIS_PATH}/DSP/Source/BasicMathFunctions/arm_scale_q15.c
        ${ARM_CMSIS_PATH}/DSP/Source/MatrixFunctions/arm_mat_init_q15.c
        ${ARM_CMSIS_PATH}/DSP/Source/MatrixFunctions/arm_mat_mult_fast_q15.c
        ${ARM_CMSIS_PATH}/DSP/Source/StatisticsFunctions/arm_max_q15.c
        ${ARM_CMSIS_PATH}/DSP/Source/StatisticsFunctions/arm_min_q15.c
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
else (USE_ARM_CMSIS)
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
endif (USE_ARM_CMSIS)

add_executable(intermittent-cnn
    ${COMMON_SRC_PATH}/intermittent-cnn.cpp
    ${COMMON_SRC_PATH}/data.cpp
    ${COMMON_SRC_PATH}/op_handlers.cpp
    ${COMMON_SRC_PATH}/conv.cpp
    ${COMMON_SRC_PATH}/cnn_common.cpp
    ${COMMON_SRC_PATH}/my_debug.cpp
    ${COMMON_SRC_PATH}/plat-linux.cpp
    ${COMMON_SRC_PATH}/my_dsplib.cpp
)
target_link_libraries(intermittent-cnn fake-msp430sdk)

target_compile_definitions(intermittent-cnn
    PRIVATE
        MY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        POSIX_BUILD
        MY_DEBUG=${MY_DEBUG}
)

if (USE_ARM_CMSIS)
    target_compile_definitions(intermittent-cnn
        PRIVATE
            USE_ARM_CMSIS
    )
    target_link_libraries(intermittent-cnn arm_cmsis_dsp)
else (USE_ARM_CMSIS)
    target_link_libraries(intermittent-cnn dsplib)
endif (USE_ARM_CMSIS)
