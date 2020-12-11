#ifdef POSIX_BUILD

#define _POSIX_C_SOURCE 1 // for kill()

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "my_debug.h"
#include "platform.h"
#include "platform-private.h"
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
#include <fstream>

/* data on NVM, made persistent via mmap() with a file */
uint8_t *nvm;
uint32_t shutdown_counter = UINT32_MAX;

Counters *counters() {
    return reinterpret_cast<Counters*>(nvm + COUNTERS_OFFSET);
}

int main(int argc, char* argv[]) {
    int ret = 0, opt_ch, button_pushed = 0, read_only = 0, n_samples = 0;
    Model *model;
    int nvm_fd = -1;

    while((opt_ch = getopt(argc, argv, "bfrc:")) != -1) {
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
            case 'c':
                shutdown_counter = atol(optarg);
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
        first_run();
    }

    ret = run_cnn_tests(n_samples);

    for (uint16_t counter_idx = 0; counter_idx < COUNTERS_LEN; counter_idx++) {
        counters()->dma_invocations[counter_idx] = 0;
        counters()->dma_bytes[counter_idx] = 0;
    }

exit:
    close(nvm_fd);
    return ret;
}

[[ noreturn ]] static void exit_with_status(uint8_t exit_code) {
    if (ptrace(PTRACE_TRACEME, 0, NULL, 0) == -1) {
        // Let the debugger break
        kill(getpid(), SIGINT);
    }
    // give up otherwise
    exit(exit_code);
}

void my_memcpy_ex(void* dest, const void* src, size_t n, uint8_t write_to_nvm) {
    if (!dma_counter_enabled) {
        memcpy(dest, src, n);
        return;
    }

    Model* model = &model_vm;
    counters()->dma_invocations[model->layer_idx]++;
    counters()->dma_bytes[model->layer_idx] += n;
    // my_printf_debug("%s copied %zu bytes" NEWLINE, __func__, n);
    // Not using memcpy here so that it is more likely that power fails during
    // memcpy, which is the case for external FRAM
    uint8_t *dest_u = reinterpret_cast<uint8_t*>(dest);
    const uint8_t *src_u = reinterpret_cast<const uint8_t*>(src);
    for (size_t idx = 0; idx < n; idx++) {
        dest_u[idx] = src_u[idx];
        if (write_to_nvm) {
            my_printf_debug("Writing to NVM offset %ld" NEWLINE, dest_u + idx - nvm);
            shutdown_counter--;
            if (!shutdown_counter) {
                exit_with_status(2);
            }
        }
    }
}

void my_memcpy(void* dest, const void* src, size_t n) {
    my_memcpy_ex(dest, src, n, 0);
}

void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t n) {
    my_memcpy_ex(vm_buffer, nvm + nvm_offset, n, 0);
}

void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t n) {
    check_nvm_write_address(nvm_offset, n);
    my_memcpy_ex(nvm + nvm_offset, vm_buffer, n, 1);
}

void my_erase() {
    // initializing as 0xff to match the ext_fram library
    memset(nvm, 0xff, NVM_SIZE);
}

void copy_samples_data(void) {
    std::ifstream samples_file("samples.bin");
    const uint16_t samples_buflen = 1024;
    char samples_buffer[samples_buflen];
    uint32_t samples_offset = SAMPLES_OFFSET;
    while (true) {
        samples_file.read(samples_buffer, samples_buflen);
        int16_t read_len = samples_file.gcount();
        write_to_nvm(samples_buffer, samples_offset, read_len);
        samples_offset += read_len;
        my_printf_debug("Copied %d bytes of samples data" NEWLINE, read_len);
        if (read_len < samples_buflen) {
            break;
        }
    }
}

[[ noreturn ]] void ERROR_OCCURRED(void) {
    exit_with_status(1);
}

#endif // POSIX_BUILD
