#ifdef POSIX_BUILD

#define _POSIX_C_SOURCE 1 // for kill()

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "debug.h"
#include "platform.h"
#include "data.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <getopt.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ptrace.h>

#define MEMCPY_DELAY_US 0

/* data on NVM, made persistent via mmap() with a file */
uint8_t *nvm;
const uint8_t *parameters_data, *parameters2_data, *samples_data, *labels_data;
uint8_t *model_data, *parameters_info_data;
uint16_t dma_invocations[COUNTERS_LEN];
uint16_t dma_bytes[COUNTERS_LEN];

static uint8_t *intermediate_values(uint8_t slot_id) {
    return nvm + slot_id * INTERMEDIATE_VALUES_SIZE;
}

Counters *counters() {
    return (Counters*)(labels_data + PLAT_LABELS_DATA_LEN);
}

int main(int argc, char* argv[]) {
    int ret = 0, opt_ch, read_only = 0, n_samples = 0;

    while((opt_ch = getopt(argc, argv, "fr")) != -1) {
        switch (opt_ch) {
            case 'r':
                read_only = 1;
                break;
            case 'f':
                dump_integer = 0;
                break;
            default:
                printf("Usage: %s [-r] [n_samples]\n", argv[0]);
                return 1;
        }
    }
    if (argv[optind]) {
        n_samples = atoi(argv[optind]);
    }

    chdir(MY_SOURCE_DIR);

    int nvm_fd = open("nvm.bin", O_RDWR);
    nvm = (uint8_t*)mmap(NULL, NVM_SIZE, PROT_READ|PROT_WRITE, read_only ? MAP_PRIVATE : MAP_SHARED, nvm_fd, 0);
    if (nvm == MAP_FAILED) {
        perror("mmap() failed");
        goto exit;
    }
    // Keep the order consistent with `outputs` in transform.py
    parameters_data = intermediate_values(NUM_SLOTS);
    parameters2_data = parameters_data + PARAMETERS_DATA_LEN;
    samples_data = parameters2_data + PARAMETERS2_DATA_LEN;
    model_data = const_cast<uint8_t*>(samples_data + PLAT_SAMPLES_DATA_LEN);
    parameters_info_data = model_data + MODEL_DATA_LEN;
    labels_data = parameters_info_data + PARAMETERS_INFO_DATA_LEN;

#ifdef USE_ARM_CMSIS
    my_printf("Use DSP from ARM CMSIS pack" NEWLINE);
#else
    my_printf("Use TI DSPLib" NEWLINE);
#endif

    ret = run_cnn_tests(n_samples);

    for (uint16_t counter_idx = 0; counter_idx < COUNTERS_LEN; counter_idx++) {
        dma_invocations[counter_idx] = 0;
        dma_bytes[counter_idx] = 0;
    }

exit:
    close(nvm_fd);
    return ret;
}

void plat_reset_model(void) {
}

void plat_print_results(void) {
    my_printf(NEWLINE "DMA invocations:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 8d", dma_invocations[i]);
    }
    my_printf(NEWLINE "DMA bytes:" NEWLINE);
    for (uint8_t i = 0; i < counters()->counter_idx; i++) {
        my_printf("% 8d", dma_bytes[i]);
    }
}

void setOutputValue(uint8_t) {}

void my_memcpy(void* dest, const void* src, size_t n) {
    uint16_t counter_idx = counters()->counter_idx;
    dma_invocations[counter_idx]++;
    dma_bytes[counter_idx] += n;
#if MEMCPY_DELAY_US
    usleep(MEMCPY_DELAY_US);
#endif
    // my_printf_debug("%s copied %zu bytes" NEWLINE, __func__, n);
    memcpy(dest, src, n);
}

void my_memcpy_to_param(struct ParameterInfo *param, uint16_t offset_in_word, const void *src, size_t n) {
    MY_ASSERT(param->bitwidth == 16);
    MY_ASSERT(param->slot < SLOT_CONSTANTS_MIN);
    uint32_t total_offset = param->params_offset + offset_in_word * sizeof(int16_t);
    MY_ASSERT(total_offset + n <= INTERMEDIATE_VALUES_SIZE);
    uint8_t *baseptr = intermediate_values(param->slot);
    uint8_t *dest = baseptr + total_offset;
    my_memcpy(dest, src, n);
}

void my_memcpy_from_param(void *dest, struct ParameterInfo *param, uint16_t offset_in_word, size_t n) {
    if (param->slot >= SLOT_CONSTANTS_MIN) {
        uint32_t limit;
        const uint8_t *baseptr = get_param_base_pointer(param, &limit);
        uint32_t total_offset = param->params_offset + offset_in_word * sizeof(int16_t);
        MY_ASSERT(total_offset + n <= limit);
        my_memcpy(dest, baseptr + total_offset, n);
    } else {
        my_memcpy(dest, intermediate_values(param->slot) + offset_in_word * sizeof(int16_t), n);
    }
}

void fill_int16(uint8_t slot, uint16_t offset, uint16_t n, int16_t val) {
    MY_ASSERT(slot < SLOT_CONSTANTS_MIN);
    int16_t *dest = reinterpret_cast<int16_t*>(intermediate_values(slot)) + offset;
    for (size_t idx = 0; idx < n; idx++) {
        dest[idx] = val;
    }
}

[[ noreturn ]] void ERROR_OCCURRED(void) {
    if (ptrace(PTRACE_TRACEME, 0, NULL, 0) == -1) {
        // Let the debugger break
        kill(getpid(), SIGINT);
    }
    // give up otherwise
    exit(1);
}

#endif // POSIX_BUILD
