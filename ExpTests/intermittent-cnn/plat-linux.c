#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "debug.h"
#include "platform.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>

#define NVM_SIZE 256*1024

/* data on NVM, made persistent via mmap() with a file */
uint8_t *intermediate_values;
uint8_t *inputs_data, *parameters_data, *samples_data, *model_data, *labels_data;
uint16_t *counters;
uint16_t *power_counters;
uint8_t *counter_idx;

uint32_t *copied_size;

void sig_handler(int sig_no) {
    if (sig_no == SIGALRM) {
        counters[*counter_idx]++;
    }
}

int main(int argc, char* argv[]) {
    int nvm_fd, ret = 0;
    uint8_t *nvm;

    nvm_fd = open("nvm.bin", O_RDWR);
    nvm = mmap(NULL, NVM_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, nvm_fd, 0);
    if (nvm == MAP_FAILED) {
        perror("mmap() failed");
        goto exit;
    }
    intermediate_values = nvm;
    // Keep the order consistent with `outputs` in transform.py
    inputs_data = nvm + NUM_SLOTS * INTERMEDIATE_VALUES_SIZE;
    parameters_data = inputs_data + INPUTS_DATA_LEN;
    samples_data = parameters_data + PARAMETERS_DATA_LEN;
    model_data = samples_data + SAMPLES_DATA_LEN;
    labels_data = model_data + MODEL_DATA_LEN;
    copied_size = (uint32_t*)(labels_data + LABELS_DATA_LEN);
    counters = (uint16_t*)(copied_size + 1);
    power_counters = counters + COUNTERS_LEN;
    counter_idx = (uint8_t*)(power_counters + COUNTERS_LEN);

    struct itimerval interval;
    interval.it_value.tv_sec = interval.it_interval.tv_sec = 0;
    interval.it_value.tv_usec = interval.it_interval.tv_usec = 1000;
    setitimer(ITIMER_REAL, &interval, NULL);
    signal(SIGALRM, sig_handler);

#ifdef USE_ARM_CMSIS
    my_printf("Use DSP from ARM CMSIS pack" NEWLINE);
#else
    my_printf("Use TI DSPLib" NEWLINE);
#endif

    init_pointers();
    if (argc >= 3) {
        printf("Usage: %s [n_samples]\n", argv[0]);
        ret = 1;
    } else if (argc == 2) {
        run_cnn_tests(atoi(argv[1]));
    } else {
        run_cnn_tests(0);
    }

exit:
    close(nvm_fd);
    my_printf("Copied size: %" PRId32 NEWLINE, *copied_size);
    *copied_size = 0;
    return ret;
}

void plat_reset_model(void) {
    *copied_size = 0;
}

void setOutputValue(uint8_t value) {
    my_printf("Output set to %d" NEWLINE, value);
}

void my_memcpy(void* dest, const void* src, size_t n) {
    *copied_size += n;
#if MEMCPY_DELAY_US
    usleep(MEMCPY_DELAY_US);
#endif
    my_printf_debug(__func__);
    my_printf_debug(" copied %d bytes" NEWLINE, (int)n);
    memcpy(dest, src, n);
}
