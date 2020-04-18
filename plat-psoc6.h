#pragma once

#include <stddef.h> // size_t
#ifdef WITH_FREERTOS
#include <FreeRTOS.h>
#include <task.h>
#endif

#ifdef WITH_FREERTOS
#define ERROR_OCCURRED() do { vTaskSuspendAll(); while (1) {} } while (0);
#else
#define ERROR_OCCURRED() do { while (1) {} } while (0);
#endif

// larger than conv needed
#define LEA_BUFFER_SIZE 4096

#define NVM_BYTE_ADDRESSABLE 0

#define USE_ALL_SAMPLES 1

void vTimerHandler(void);

void my_memcpy(void* dest, const void* src, size_t n);

#ifdef WITH_FREERTOS
// Not using vTaskDelete() as it does not release heap space
// for TCB and stack, leading to exhaustion of the heap
#define TASK_FINISHED()                                     \
    do {                                                    \
        vTaskPrioritySet(NULL, tskIDLE_PRIORITY);           \
        taskYIELD();                                        \
        my_printf("%s still running!" NEWLINE, __func__);   \
        ERROR_OCCURRED();                                   \
    } while (0);
#endif
