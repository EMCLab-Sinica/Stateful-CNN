#ifdef POSIX_BUILD

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "counters.h"
#include "my_debug.h"
#include "platform.h"
#include "data.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include <memory>
#ifdef USE_PROTOBUF
#include "model_output.pb.h"
#endif

/* data on NVM, made persistent via mmap() with a file */
uint8_t *nvm;
static uint32_t shutdown_counter = UINT32_MAX;
static std::ofstream out_file;

#if ENABLE_COUNTERS
Counters counters_data[2][COUNTERS_LEN];
uint8_t counters_cur_copy_id = 0;
uint32_t total_jobs = 0;
#endif

#ifdef USE_PROTOBUF
static void save_model_output_data() {
    model_output_data->SerializeToOstream(&out_file);
}
#endif

int main(int argc, char* argv[]) {
    int ret = 0, opt_ch, button_pushed = 0, read_only = 0, n_samples = 0;
    Model *model;
    int nvm_fd = -1;

    while((opt_ch = getopt(argc, argv, "bfrc:s:")) != -1) {
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
            case 's':
#ifdef USE_PROTOBUF
                out_file.open(optarg);
                break;
#else
                my_printf("Cannot save outputs as protobuf support is not compiled." NEWLINE);
                return 1;
#endif
            default:
                my_printf("Usage: %s [-r] [n_samples]" NEWLINE, argv[0]);
                return 1;
        }
    }
    if (argv[optind]) {
        n_samples = atoi(argv[optind]);
    }

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

#if USE_ARM_CMSIS
    my_printf_debug("Use DSP from ARM CMSIS pack" NEWLINE);
#else
    my_printf_debug("Use TI DSPLib" NEWLINE);
#endif

#ifdef USE_PROTOBUF
    if (out_file.is_open()) {
        model_output_data = std::make_unique<ModelOutput>();
        std::atexit(save_model_output_data);
    }
#endif

    model = load_model_from_nvm();

    // emulating button_pushed - treating as a fresh run
    if (button_pushed) {
        model->version = 0;
    }

    if (!model->version) {
        // the first time
        first_run();
    }

    ret = run_cnn_tests(n_samples);

    print_all_counters();

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
#if ENABLE_COUNTERS
    counters()->dma_invocations++;
    counters()->dma_bytes += n;
#endif
    // Not using memcpy here so that it is more likely that power fails during
    // memcpy, which is the case for external FRAM
    uint8_t *dest_u = reinterpret_cast<uint8_t*>(dest);
    const uint8_t *src_u = reinterpret_cast<const uint8_t*>(src);
    for (size_t idx = 0; idx < n; idx++) {
        dest_u[idx] = src_u[idx];
        if (write_to_nvm) {
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

void my_memcpy_from_parameters(void *dest, const ParameterInfo *param, uint32_t offset_in_bytes, size_t n) {
    MY_ASSERT(offset_in_bytes + n <= PARAMETERS_DATA_LEN);
    my_memcpy(dest, parameters_data + param->params_offset + offset_in_bytes, n);
}

void read_from_nvm(void *vm_buffer, uint32_t nvm_offset, size_t n) {
    MY_ASSERT(n <= 1024);
    my_memcpy_ex(vm_buffer, nvm + nvm_offset, n, 0);
}

void write_to_nvm(const void *vm_buffer, uint32_t nvm_offset, size_t n, uint16_t timer_delay) {
    MY_ASSERT(n <= 1024);
    check_nvm_write_address(nvm_offset, n);
    my_memcpy_ex(nvm + nvm_offset, vm_buffer, n, 1);
}

void my_erase() {
    memset(nvm, 0, NVM_SIZE);
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

void notify_model_finished(void) {}

[[ noreturn ]] void ERROR_OCCURRED(void) {
    exit_with_status(1);
}

#endif // POSIX_BUILD
