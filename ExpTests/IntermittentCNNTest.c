#include "intermittent-cnn.h"
#include "common.h"
#include "data.h"
#include "debug.h"

#pragma NOINIT(delay_counter)
static uint32_t delay_counter;

#pragma DATA_SECTION(myFirstTime, ".map")
static uint8_t myFirstTime;

void IntermittentCNNTest() {
    init_pointers();

    if (myFirstTime != 1) {
        delay_counter = 0;

        for (uint8_t i = 0; i < COUNTERS_LEN; i++) {
            counters[i] = 0;
            power_counters[i] = 0;
        }

        myFirstTime = 1;
    }

    while (delay_counter < 10) {
        my_printf("%d" NEWLINE, delay_counter);
        delay_counter++;
        __delay_cycles(16E6);
    }

    if (!model->run_counter) {
        run_model(NULL);
    }

    while (1) {
        print_results();
        __delay_cycles(16E6);
    }
}
