cmake_minimum_required(VERSION 2.10)

project(intermittent-cnn)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wstrict-prototypes -Wshadow -g -O0")

set(TOPDIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(FREERTOS_DIR ${TOPDIR}/FreeRTOS_Source)

add_library(freertos
    ${FREERTOS_DIR}/list.c
)
target_include_directories(freertos
    SYSTEM PUBLIC
        ${TOPDIR}
        ${FREERTOS_DIR}/include
        ${FREERTOS_DIR}/portable/GCC-Linux
)

set(DSPLIB_PATH ${TOPDIR}/DSPLib_1_30_00_02)

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
    ${DSPLIB_PATH}/source/vector/msp_mac_q15.c
    ${DSPLIB_PATH}/source/vector/msp_max_q15.c
    ${DSPLIB_PATH}/source/utility/msp_fill_q15.c
)
target_include_directories(dsplib
    SYSTEM PUBLIC
        ${DSPLIB_PATH}/include
)
target_link_libraries(dsplib fake-msp430sdk)

add_executable(test-list
    test-list.c
)
target_link_libraries(test-list freertos)

add_executable(intermittent-cnn
    intermittent-cnn.c
    ops.c
    op_handlers.c
    conv.c
    common.c
    debug.c
    plat-linux.c
)
target_link_libraries(intermittent-cnn dsplib fake-msp430sdk)
