cmake_minimum_required(VERSION 2.10)

project(intermittent-cnn)

set(TOPDIR ../../)
set(FREERTOS_DIR ${TOPDIR}/FreeRTOS_Source)

add_library(freertos
    ${FREERTOS_DIR}/list.c
)
target_include_directories(freertos
    PUBLIC
        ${TOPDIR}
        ${FREERTOS_DIR}/include
        ${FREERTOS_DIR}/portable/GCC-Linux
)

add_executable(test-list
    test-list.c
)
target_link_libraries(test-list freertos)
