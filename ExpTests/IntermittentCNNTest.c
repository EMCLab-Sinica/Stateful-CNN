#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "data.h"
#include "debug.h"

#pragma NOINIT(delay_counter)
static uint32_t delay_counter;

#pragma DATA_SECTION(myFirstTime, ".map")
static uint8_t myFirstTime;

#define DELAY_START_SECONDS 0

void IntermittentCNNTest() {
    Model *model = (Model*)model_data;

    if (myFirstTime != 1) {
        delay_counter = 0;

        for (uint8_t i = 0; i < COUNTERS_LEN; i++) {
            counters()->time_counters[i] = 0;
            counters()->power_counters[i] = 0;
        }

        myFirstTime = 1;
        model->run_counter = 0;
    }

    while (delay_counter < DELAY_START_SECONDS) {
        my_printf("%d" NEWLINE, delay_counter);
        delay_counter++;
        __delay_cycles(16E6);
    }

    if (!model->run_counter) {
        run_cnn_tests(1);
    }

    while (1) {
        __delay_cycles(16E6);
    }
}

void button_pushed(void) {
    static uint8_t push_counter = 0;
    // XXX: somehow the ISR for button is triggered immediately after recovery
    if (push_counter >= 1) {
        myFirstTime = 0;
    }
    push_counter++;
}
