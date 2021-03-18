cmake_minimum_required(VERSION 2.8.12)

project(intermittent-cnn)

set(MY_DEBUG "1" CACHE STRING "Local debug flag. See my_debug.h for details.")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Too many parameters are unused if some features are not enabled
set(COMMON_FLAGS "-Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter -pedantic")
if (${MY_DEBUG} GREATER 1)
    set(COMMON_FLAGS "${COMMON_FLAGS} -g -O0")
else ()
    set(COMMON_FLAGS "${COMMON_FLAGS} -O3")
endif ()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} -Wstrict-prototypes -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -std=c++11 -Wold-style-cast")

set(COMMON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/common)


file(GLOB CCS_ROOT $ENV{HOME}/ti/ccs*)
if (NOT "${CCS_ROOT}" STREQUAL "")
    message(STATUS "Found CCS at ${CCS_ROOT}")
    set (BUILD_MSP432 ON)
else ()
    message(WARNING "No CCS found. Not building plat-msp432.")
    set (BUILD_MSP432 OFF)
endif ()

# Keep this list in sync with ARM-CMSIS/create_symlinks.sh
set(ARM_CMSIS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ARM-CMSIS)
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

set(DSPLIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/TI-DSPLib)
add_library(dsplib
    ${DSPLIB_PATH}/source/matrix/msp_matrix_mpy_q15.c
    ${DSPLIB_PATH}/source/vector/msp_add_q15.c
    ${DSPLIB_PATH}/source/vector/msp_offset_q15.c
    ${DSPLIB_PATH}/source/vector/msp_max_q15.c
    ${DSPLIB_PATH}/source/vector/msp_min_q15.c
    ${DSPLIB_PATH}/source/vector/msp_scale_q15.c
    ${DSPLIB_PATH}/source/utility/msp_deinterleave_q15.c
    ${DSPLIB_PATH}/source/utility/msp_interleave_q15.c
    ${DSPLIB_PATH}/source/utility/msp_fill_q15.c
)
target_include_directories(dsplib
    SYSTEM PUBLIC
        ${DSPLIB_PATH}/include
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/fake-msp430sdk
)

add_executable(intermittent-cnn
    ${COMMON_SRC_PATH}/intermittent-cnn.cpp
    ${COMMON_SRC_PATH}/data.cpp
    ${COMMON_SRC_PATH}/op_handlers.cpp
    ${COMMON_SRC_PATH}/op_utils.cpp
    ${COMMON_SRC_PATH}/conv.cpp
    ${COMMON_SRC_PATH}/gemm.cpp
    ${COMMON_SRC_PATH}/pooling.cpp
    ${COMMON_SRC_PATH}/cnn_common.cpp
    ${COMMON_SRC_PATH}/my_debug.cpp
    ${COMMON_SRC_PATH}/plat-linux.cpp
    ${COMMON_SRC_PATH}/platform.cpp
    ${COMMON_SRC_PATH}/my_dsplib.cpp
)
target_link_libraries(intermittent-cnn)

target_compile_definitions(intermittent-cnn
    PRIVATE
        MY_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        POSIX_BUILD
        MY_DEBUG=${MY_DEBUG}
)

target_link_libraries(intermittent-cnn arm_cmsis_dsp dsplib)

# Below is not actually used for the build on PC. I added it here so that
# clangd can identify platform-dependent codes

if (BUILD_MSP432)
    add_library(plat-msp432 ${COMMON_SRC_PATH}/plat-msp430.cpp)
    target_include_directories(plat-msp432
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/msp432
    )
    target_include_directories(plat-msp432
        SYSTEM PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/msp432/driverlib/MSP432P4xx
            ${CCS_ROOT}/ccs/ccs_base/arm/include
            $<TARGET_PROPERTY:arm_cmsis_dsp,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(plat-msp432
        PRIVATE
            __MSP432__
            __MSP432P401R__
    )
endif ()
