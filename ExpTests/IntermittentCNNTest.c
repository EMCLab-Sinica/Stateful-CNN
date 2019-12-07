#include "intermittent-cnn.h"
#include "common.h"
#include "data.h"

void IntermittentCNNTest() {
    run_model();
    while (1) {
        __no_operation();
    }
}
