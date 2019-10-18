#include "intermittent-cnn.h"
#include "common.h"
#include "data.h"

void IntermittentCNNTest() {
    while (1) {
        if (run_model() != 0) {
            while (1) {
                __no_operation();
            }
        }
    }
}
