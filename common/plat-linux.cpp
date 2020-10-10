#ifdef POSIX_BUILD

#define _POSIX_C_SOURCE 1 // for kill()

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "my_debug.h"
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
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ptrace.h>

#define MEMCPY_DELAY_US 0

/* data on NVM, made persistent via mmap() with a file */
uint8_t *nvm;
uint8_t *counters_data;
uint8_t *intermediate_parameters_info_data_nvm, *model_data_nvm;
uint16_t dma_invocations[COUNTERS_LEN];
uint16_t dma_bytes[COUNTERS_LEN];

const uint32_t intermediate_parameters_info_data_nvm_offset = NVM_SIZE - 0x10000;
const uint32_t model_data_nvm_offset = NVM_SIZE - 0x8000;
const uint32_t counters_data_offset = NVM_SIZE - 0x7800;

static_assert(intermediate_parameters_info_data_nvm_offset > NUM_SLOTS * INTERMEDIATE_VALUES_SIZE, "Incorrect NVM layout");
static_assert(model_data_nvm_offset > intermediate_parameters_info_data_nvm_offset + INTERMEDIATE_PARAMETERS_INFO_DATA_LEN, "Incorrect NVM layout");
static_assert(counters_data_offset > model_data_nvm_offset + 2 * MODEL_DATA_LEN, "Incorrect NVM layout");

static uint8_t *intermediate_values(uint8_t slot_id) {
    return nvm + slot_id * INTERMEDIATE_VALUES_SIZE;
}

Counters *counters() {
    return reinterpret_cast<Counters*>(counters_data);
}

int main(int argc, char* argv[]) {
    int ret = 0, opt_ch, button_pushed = 0, read_only = 0, n_samples = 0;
    Model *model;
    int nvm_fd = -1, samples_fd = -1;

    while((opt_ch = getopt(argc, argv, "bfr")) != -1) {
        switch (opt_ch) {
            case 'b':
                button_pushed = 1;
                break;
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

    struct stat stat_buf;
    if (stat("nvm.bin", &stat_buf) != 0) {
        if (errno != ENOENT) {
            perror("Checking nvm.bin failed");
            goto exit;
        }
        nvm_fd = open("nvm.bin", O_RDWR|O_CREAT, 0600);
        ftruncate(nvm_fd, NVM_SIZE);
    } else {
        nvm_fd = open("nvm.bin", O_RDWR);
    }
    nvm = reinterpret_cast<uint8_t*>(mmap(NULL, NVM_SIZE, PROT_READ|PROT_WRITE, read_only ? MAP_PRIVATE : MAP_SHARED, nvm_fd, 0));
    if (nvm == MAP_FAILED) {
        perror("mmap() failed");
        goto exit;
    }

    samples_fd = open("samples.bin", O_RDONLY);
    samples_data = reinterpret_cast<uint8_t*>(mmap(NULL, SAMPLE_SIZE * N_SAMPLES, PROT_READ, MAP_PRIVATE, samples_fd, 0));
    if (samples_data == MAP_FAILED) {
        perror("mmap() for samples failed");
        goto exit;
    }

    intermediate_parameters_info_data_nvm = nvm + intermediate_parameters_info_data_nvm_offset;
    model_data_nvm = nvm + model_data_nvm_offset;
    counters_data = nvm + counters_data_offset;

#ifdef USE_ARM_CMSIS
    my_printf("Use DSP from ARM CMSIS pack" NEWLINE);
#else
    my_printf("Use TI DSPLib" NEWLINE);
#endif

    model = get_model();

    // emulating button_pushed - treating as a fresh run
    if (button_pushed) {
        model->version = 0;
    }

    if (!model->version) {
        // the first time
#if STATEFUL_CNN
        memset(intermediate_values(0), 0, INTERMEDIATE_VALUES_SIZE * NUM_SLOTS);
#endif
        my_memcpy(intermediate_parameters_info_data_nvm, intermediate_parameters_info_data, INTERMEDIATE_PARAMETERS_INFO_DATA_LEN);
        my_memcpy(model_data_nvm, model_data, MODEL_DATA_LEN);
        my_memcpy(model_data_nvm + sizeof(Model), model_data, MODEL_DATA_LEN);

        get_model(); // refresh model_vm
        commit_model();
    }

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
    // Not using memcpy here so that it is more likely that power fails during
    // memcpy, which is the case for external FRAM
    uint8_t *dest_u = reinterpret_cast<uint8_t*>(dest);
    const uint8_t *src_u = reinterpret_cast<const uint8_t*>(src);
    for (size_t idx = 0; idx < n; idx++) {
        dest_u[idx] = src_u[idx];
    }
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

void my_memcpy_from_intermediate_values(void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n) {
    my_memcpy(dest, intermediate_values(param->slot) + offset_in_word * sizeof(int16_t), n);
}

ParameterInfo* get_intermediate_parameter_info(uint8_t i) {
    ParameterInfo* dst = intermediate_parameters_info_vm + i;
    const ParameterInfo* src = reinterpret_cast<ParameterInfo*>(intermediate_parameters_info_data_nvm) + i;
    MY_ASSERT(src->parameter_info_idx == i + N_INPUT);
    my_memcpy(dst, src, sizeof(ParameterInfo));
    my_printf_debug("Load intermediate parameter info %d from NVM" NEWLINE, i);
    return dst;
}

void commit_intermediate_parameter_info(uint8_t i) {
    ParameterInfo* dst = reinterpret_cast<ParameterInfo*>(intermediate_parameters_info_data_nvm) + i;
    const ParameterInfo* src = intermediate_parameters_info_vm + i;
    MY_ASSERT(src->parameter_info_idx == i + N_INPUT);
    my_memcpy(dst, src, sizeof(ParameterInfo));
    my_printf_debug("Committing intermediate parameter info %d to NVM" NEWLINE, i);
}

Model* get_model(void) {
    Model *dst = &model_vm;
    Model *model_nvm = reinterpret_cast<Model*>(model_data_nvm);
    uint8_t newer_model_copy_id = get_newer_model_copy_id(model_nvm->version, (model_nvm + 1)->version);
    my_memcpy(dst, model_nvm + newer_model_copy_id, sizeof(Model));
    my_printf_debug("Using model copy %d, version %d" NEWLINE, newer_model_copy_id, dst->version);
    return dst;
}

void commit_model(void) {
    Model *model_nvm = reinterpret_cast<Model*>(model_data_nvm);
    uint8_t newer_model_copy_id = get_newer_model_copy_id(model_nvm->version, (model_nvm + 1)->version);
    uint8_t older_model_copy_id = newer_model_copy_id ^ 1;
    bump_model_version(&model_vm);
    my_memcpy(model_nvm + older_model_copy_id, &model_vm, sizeof(Model));
    my_printf_debug("Committing version %d to model copy %d" NEWLINE, model_vm.version, older_model_copy_id);
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
