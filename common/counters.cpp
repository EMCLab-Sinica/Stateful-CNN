#include <cinttypes>
#include <cstring>
#include "cnn_common.h"
#include "counters.h"
#include "platform.h"

#if ENABLE_COUNTERS
uint8_t current_counter = INVALID_POINTER;
uint8_t prev_counter = INVALID_POINTER;

Counters *counters() {
#if ENABLE_PER_LAYER_COUNTERS
    return counters_data + model_vm.layer_idx;
#else
    return counters_data;
#endif
}

template<uint32_t Counters::* MemPtr>
static uint32_t print_counters() {
    uint32_t total = 0;
    for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
        total += counters_data[i].*MemPtr;
#if ENABLE_PER_LAYER_COUNTERS
        my_printf("%12" PRIu32, counters_data[i].*MemPtr);
#else
        break;
#endif
    }
    my_printf(" total=%12" PRIu32, total);
    return total;
}

void print_all_counters() {
#if ENABLE_DEMO_COUNTERS
    return;
#endif

    my_printf("op types:                ");
#if ENABLE_PER_LAYER_COUNTERS
    for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
        my_printf("% 12d", get_node(i)->op_type);
    }
#endif
    uint32_t total_dma_bytes = 0, total_macs = 0, total_overhead = 0;
    my_printf(NEWLINE "Power counters:          "); print_counters<&Counters::power_counters>();
    my_printf(NEWLINE "DMA invocations:         "); print_counters<&Counters::dma_invocations>();
    my_printf(NEWLINE "DMA bytes:               "); total_dma_bytes = print_counters<&Counters::dma_bytes>();
    my_printf(NEWLINE "MACs:                    "); total_macs = print_counters<&Counters::macs>();
    // state-embedding overheads
    my_printf(NEWLINE "Embeddings:              "); total_overhead += print_counters<&Counters::embedding>();
    my_printf(NEWLINE "Strippings:              "); total_overhead += print_counters<&Counters::stripping>();
    my_printf(NEWLINE "Overflow handling:       "); total_overhead += print_counters<&Counters::overflow_handling>();
    // state-assignment overheads
    my_printf(NEWLINE "State queries:           "); total_overhead += print_counters<&Counters::state_query>();
    my_printf(NEWLINE "Table updates:           "); total_overhead += print_counters<&Counters::table_updates>();
    my_printf(NEWLINE "Table preservation:      "); total_overhead += print_counters<&Counters::table_preservation>();
    my_printf(NEWLINE "Table loading:           "); total_overhead += print_counters<&Counters::table_loading>();
    // recovery overheads
    my_printf(NEWLINE "Progress seeking:        "); total_overhead += print_counters<&Counters::progress_seeking>();
    // misc
    my_printf(NEWLINE "Memory layout:           "); total_overhead += print_counters<&Counters::memory_layout>();
    my_printf(NEWLINE "Job preservation:        "); total_overhead += print_counters<&Counters::job_preservation>();
#if JAPARI
    my_printf(NEWLINE "Footprint preservation:  "); total_overhead += print_counters<&Counters::footprint_preservation>();
    my_printf(NEWLINE "Data loading:            "); total_overhead += print_counters<&Counters::data_loading>();
#endif

    my_printf(NEWLINE "Total DMA bytes: %d", total_dma_bytes);
    my_printf(NEWLINE "Total MACs: %d", total_macs);
    my_printf(NEWLINE "Total overhead: %" PRIu32, total_overhead);
    my_printf(NEWLINE "run_counter: %d" NEWLINE, get_model()->run_counter);
}

void reset_counters() {
#if ENABLE_COUNTERS
    memset(counters_data, 0, sizeof(Counters) * COUNTERS_LEN);
#endif
}

void report_progress() {
#if ENABLE_DEMO_COUNTERS
    static uint8_t last_progress = 0;

    Model* model = get_model();
    if (!model->n_jobs) {
        return;
    }
    uint32_t cur_jobs = counters()->job_preservation / 2;
    uint8_t cur_progress = 100 * cur_jobs / model->n_jobs;
    // report only when the percentage is changed to avoid high UART overheads
    if (cur_progress != last_progress) {
        my_printf("P,%d,%d,%d" NEWLINE, cur_progress,
                  counters()->job_preservation/1024, counters()->footprint_preservation/1024);
        last_progress = cur_progress;
    }
#endif
}

#endif
