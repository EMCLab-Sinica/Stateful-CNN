#include "intermittent-cnn.h"
#include "data.h"
#include "common.h"
#include <stdint.h>
#include <stdio.h>
#include <DSPLib.h>

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
        printf("Test %d\n", total);
        run_model(&predicted);
        total++;
        if (label == predicted) {
            correct++;
        }
#ifndef MY_NDEBUG
        printf("%d %d\n", label, predicted);
#endif
        reset_model();
    }
    printf("correct=%d total=%d rate=%f\n", correct, total, 1.0*correct/total);
    fclose(test_file);
}

int main(int argc, char* argv[]) {
    if (argc >= 3) {
        printf("Usage: %s [test filename]\n", argv[0]);
        return 1;
    } else if (argc == 2) {
        run_tests(argv[1]);
    } else {
        return run_model(NULL);
    }
}
