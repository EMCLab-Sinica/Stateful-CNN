cmake_minimum_required(VERSION 2.8.12)

project(intermittent-cnn)

set(MY_DEBUG "1" CACHE STRING "Local debug flag. See my_debug.h for details.")
option(ENABLE_COVERAGE "Enable coverage via gcov." OFF)
option(USE_PROTOBUF "Use Protobuf to save results" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Too many parameters are unused if some features are not enabled
set(COMMON_FLAGS "-Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter -pedantic")
if (${MY_DEBUG} GREATER 1)
    set(COMMON_FLAGS "${COMMON_FLAGS} -g3 -O0")
else ()
    set(COMMON_FLAGS "${COMMON_FLAGS} -O3")
endif ()
if (ENABLE_COVERAGE)
    set(COMMON_FLAGS "${COMMON_FLAGS} --coverage -fprofile-dir=gcov/%p")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif ()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} -Wstrict-prototypes -std=c11")
# Don't use something that require C++17 or newer, which are not supported by MSP-CGT yet
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -std=c++14 -Wold-style-cast")

if (USE_PROTOBUF)
    find_package(Protobuf REQUIRED)
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/exp/model_output.proto)
    protobuf_generate_python(PROTO_PY ${CMAKE_CURRENT_SOURCE_DIR}/exp/model_output.proto)
    add_custom_target(model_output_pb2 ALL DEPENDS ${PROTO_PY})
endif ()

set(COMMON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/common)


file(GLOB CCS_ARM_ROOT $ENV{HOME}/ti/ccs*/ccs/ccs_base/arm)
if (NOT "${CCS_ARM_ROOT}" STREQUAL "")
    message(STATUS "Found CCS ARM root at ${CCS_ARM_ROOT}")
    option (BUILD_MSP432 "Build for MSP432" ON)
else ()
    message(WARNING "No CCS ARM root found. Not building plat-mcu.")
    option (BUILD_MSP432 "Build for MSP432" OFF)
endif ()

# Keep this list in sync with ARM-CMSIS/sync-cmsis.py
set(ARM_CMSIS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ARM-CMSIS/CMSIS)
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
    SYSTEM PUBLIC
        ${ARM_CMSIS_PATH}/Core/Include
        ${ARM_CMSIS_PATH}/DSP/Include
    PRIVATE
        ${COMMON_SRC_PATH}
        ${CMAKE_BINARY_DIR}
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
        ${CMAKE_CURRENT_SOURCE_DIR}/msp430-compat
    PRIVATE
        ${COMMON_SRC_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
)

set (intermittent_cnn_SOURCES
    ${COMMON_SRC_PATH}/intermittent-cnn.cpp
    ${COMMON_SRC_PATH}/op_handlers.cpp
    ${COMMON_SRC_PATH}/op_utils.cpp
    ${COMMON_SRC_PATH}/conv.cpp
    ${COMMON_SRC_PATH}/counters.cpp
    ${COMMON_SRC_PATH}/gemm.cpp
    ${COMMON_SRC_PATH}/pooling.cpp
    ${COMMON_SRC_PATH}/cnn_common.cpp
    ${COMMON_SRC_PATH}/my_debug.cpp
    ${COMMON_SRC_PATH}/plat-pc.cpp
    ${COMMON_SRC_PATH}/platform.cpp
    ${COMMON_SRC_PATH}/my_dsplib.cpp
    ${CMAKE_BINARY_DIR}/data.cpp
)
if (USE_PROTOBUF)
    list(APPEND intermittent_cnn_SOURCES ${PROTO_SRCS})
endif ()
add_executable(intermittent-cnn ${intermittent_cnn_SOURCES})

target_compile_definitions(intermittent-cnn
    PRIVATE
        POSIX_BUILD
        MY_DEBUG=${MY_DEBUG}
)

target_include_directories(intermittent-cnn
    PRIVATE
        ${COMMON_SRC_PATH}
        ${CMAKE_BINARY_DIR}
)

target_link_libraries(intermittent-cnn arm_cmsis_dsp dsplib)

if (USE_PROTOBUF)
    target_compile_definitions(intermittent-cnn
        PRIVATE
            USE_PROTOBUF
    )
    target_include_directories(intermittent-cnn
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}  # for *.pb.h files
    )
    target_link_libraries(intermittent-cnn protobuf::libprotobuf)
endif ()

# Below is not actually used for the build on PC. I added it here so that
# clangd can identify platform-dependent codes

if (BUILD_MSP432)
    add_library(plat-mcu ${COMMON_SRC_PATH}/plat-mcu.cpp)
    target_include_directories(plat-mcu
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/msp432
    )
    target_include_directories(plat-mcu
        SYSTEM PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/msp432/driverlib/MSP432P4xx
            ${CMAKE_BINARY_DIR}
            ${CCS_ARM_ROOT}/include
            $<TARGET_PROPERTY:arm_cmsis_dsp,INCLUDE_DIRECTORIES>
    )
    target_compile_definitions(plat-mcu
        PRIVATE
            __MSP432__
            __MSP432P401R__
    )
endif ()
