#include "intermittent-cnn.h"
#include "common.h"
#include "debug.h"
#include "platform.h"
#include <DSPLib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>

#define NVM_SIZE 256*1024

/* data on NVM, made persistent via mmap() with a file */
uint8_t *intermediate_values;
uint8_t *inputs_data, *parameters_data, *model_data;

uint32_t *copied_size;
static uint32_t memcpy_delay_us = 0;

void run_tests(char *filename) {
    uint8_t label, predicted;
    FILE *test_file = fopen(filename, "r");
    uint32_t correct = 0, total = 0;
    while (!feof(test_file)) {
        fscanf(test_file, "|labels ");
        for (uint8_t i = 0; i < 10; i++) {
            int j;
            int ret = fscanf(test_file, "%d", &j);
            if (ret != 1) {
                fprintf(stderr, "fscanf returns %d, pos = %ld\n", ret, ftell(test_file));
                ERROR_OCCURRED();
            }
            if (j == 1) {
                label = i;
                // not break here, so that remaining numbers are consumed
            }
        }
        fscanf(test_file, " |features ");
        for (uint16_t i = 0; i < 28*28; i++) {
            int j;
            int ret = fscanf(test_file, "%d", &j);
            if (ret != 1) {
                fprintf(stderr, "fscanf returns %d, pos = %ld\n", ret, ftell(test_file));
                ERROR_OCCURRED();
            }
            ((int16_t*)parameters_data)[i] = _Q15(1.0 * j / 256 / SCALE);
        }
        fscanf(test_file, "\n");
        my_printf_debug("Test %d\n", total);
        run_model(&predicted);
        total++;
        if (label == predicted) {
            correct++;
        }
        my_printf_debug("%d %d\n", label, predicted);
    }
    my_printf("correct=%d total=%d rate=%f\n", correct, total, 1.0*correct/total);
    fclose(test_file);
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
    model_data = parameters_data + PARAMETERS_DATA_LEN;
    copied_size = (uint32_t*)(model_data + MODEL_DATA_LEN);

    if (argc >= 3) {
        printf("Usage: %s [test filename]\n", argv[0]);
        ret = 1;
    } else if (argc == 2) {
        run_tests(argv[1]);
    } else {
        ret = run_model(NULL);
    }

exit:
    close(nvm_fd);
    my_printf("Copied size: %" PRId32 NEWLINE, *copied_size);
    *copied_size = 0;
    return ret;
}

uint32_t getElapsedMilliseconds() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}


void my_memcpy(void* dest, const void* src, size_t n) {
    *copied_size += n;
    if (memcpy_delay_us) {
        usleep(memcpy_delay_us);
    }
    my_printf_debug(__func__);
    my_printf_debug(" copied %d bytes" NEWLINE, (int)n);
    memcpy(dest, src, n);
}

void plat_reset_model(void) {
    *copied_size = 0;
}
