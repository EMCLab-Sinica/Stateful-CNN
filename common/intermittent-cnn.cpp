#include <cstdint>
#include <cstring>
#include <cinttypes> // for PRId32
#include <cmath>

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

uint16_t sample_idx;

static void handle_node(Model *model, uint16_t node_idx) {
    const Node *cur_node = get_node(node_idx);
#if MY_DEBUG >= MY_DEBUG_LAYERS
    my_printf("Current node: %d, ", node_idx);
    my_printf("name = %.*s, ", NODE_NAME_LEN, cur_node->name);
    my_printf("op_type = %d" NEWLINE, cur_node->op_type);
#endif

    int16_t input_id[3];
    const ParameterInfo *input[3];
    for (uint16_t j = 0; j < cur_node->inputs_len; j++) {
        input_id[j] = cur_node->inputs[j];
        my_printf_debug("input_id[%d] = %d" NEWLINE, j, input_id[j]);
        input[j] = get_parameter_info(input_id[j]);
        // dump_params(input[j]);
    }
    my_printf_debug(NEWLINE);

    /* Allocate an ParameterInfo for output. Details are filled by
     * individual operation handlers */
    ParameterInfo *output = get_intermediate_parameter_info(node_idx);
    my_memcpy(output, input[0], sizeof(ParameterInfo) - sizeof(uint16_t)); // don't overwrite parameter_info_idx
    output->params_offset = 0;
    allocators[cur_node->op_type](model, input, output, cur_node);
    my_printf_debug("Needed mem = %d" NEWLINE, output->params_len);
    MY_ASSERT(output->params_len < INTERMEDIATE_VALUES_SIZE);

#if STATEFUL
    my_printf_debug("Old output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif
    handlers[cur_node->op_type](model, input, output, cur_node);
    // For some operations (e.g., ConvMerge), scale is determined in the handlers
    my_printf_debug("Output scale = %d" NEWLINE, output->scale);
    MY_ASSERT(output->scale > 0);  // fail when overflow
#if STATEFUL
    my_printf_debug("New output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif

    MY_ASSERT(output->bitwidth);

    commit_intermediate_parameter_info(node_idx);

    if (node_idx == MODEL_NODES_LEN - 1) {
        model->running = 0;
        model->run_counter++;
    }
}

#if MY_DEBUG >= MY_DEBUG_NORMAL
const float first_sample_outputs[] = FIRST_SAMPLE_OUTPUTS;
#endif

static void run_model(int8_t *ansptr, const ParameterInfo **output_node_ptr) {
    my_printf_debug("N_INPUT = %d" NEWLINE, N_INPUT);

    Model *model = get_model();
    if (!model->running) {
        // reset model
        model->layer_idx = 0;
        for (uint8_t idx = 0; idx < NUM_SLOTS; idx++) {
            SlotInfo *cur_slot_info = get_slot_info(model, idx);
            cur_slot_info->user = -1;
        }
#if HAWAII
        for (uint16_t node_idx = 0; node_idx < MODEL_NODES_LEN; node_idx++) {
            reset_hawaii_layer_footprint(node_idx);
        }
#endif
        model->running = 1;
        commit_model();
    }

    dump_model_debug(model);

    for (uint16_t node_idx = model->layer_idx; node_idx < MODEL_NODES_LEN; node_idx++) {
        handle_node(model, node_idx);
        model->layer_idx++;

        commit_model();

        dump_model_debug(model);
    }

    // the parameter info for the last node should also be refreshed when MY_DEBUG == 0
    // Otherwise, the model is not correctly re-initialized in some cases
    const ParameterInfo *output_node = get_parameter_info(MODEL_NODES_LEN + N_INPUT - 1);
    if (output_node_ptr) {
        *output_node_ptr = output_node;
    }
#if MY_DEBUG >= MY_DEBUG_NORMAL
    int16_t max = INT16_MIN;
    uint16_t u_ans;
    uint8_t ans_len = sizeof(first_sample_outputs) / sizeof(float);
#if JAPARI
    ans_len = extend_for_footprints(ans_len);
#endif
    uint8_t buffer_len = MIN_VAL(output_node->dims[1], ans_len);
    my_memcpy_from_param(model, lea_buffer, output_node, 0, buffer_len * sizeof(int16_t));

#if STATEFUL
    for (uint8_t idx = BATCH_SIZE - 1; idx < buffer_len; idx += BATCH_SIZE) {
        strip_state(lea_buffer + idx);
    }
#endif

    if (sample_idx == 0) {
        float output_max = 0;
        for (uint8_t buffer_idx = 0; buffer_idx < ans_len; buffer_idx++) {
            output_max = MAX_VAL(std::fabs(first_sample_outputs[buffer_idx]), output_max);
        }
        for (uint8_t buffer_idx = 0, ofm_idx = 0; buffer_idx < buffer_len; buffer_idx++) {
            int16_t got_q15 = lea_buffer[buffer_idx];
#if JAPARI
            if (offset_has_state(buffer_idx)) {
                check_footprint(got_q15);
            } else
#endif
            {
                float got_real = q15_to_float(got_q15, ValueInfo(output_node), nullptr, false);
                float expected = first_sample_outputs[ofm_idx];
                float error = fabs((got_real - expected) / output_max);
                // Errors in CIFAR-10/Stateful are quite large...
                MY_ASSERT(error <= 0.1,
                          "Value error too large at index %d: got=%f, expected=%f" NEWLINE, buffer_idx, got_real, expected);
                ofm_idx++;
            }
        }
    }

    my_max_q15(lea_buffer, buffer_len, &max, &u_ans);
#if JAPARI
    u_ans = u_ans / (BATCH_SIZE + 1) * BATCH_SIZE + u_ans % (BATCH_SIZE + 1);
#endif
    *ansptr = u_ans;
#endif
}

template<uint32_t Counters::* MemPtr>
static uint32_t print_counters() {
    uint32_t total = 0;
    for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
        total += counters(i)->*MemPtr;
#if ENABLE_PER_LAYER_COUNTERS
        my_printf("%12" PRIu32, counters(i)->*MemPtr);
#else
        break;
#endif
    }
    my_printf(" total=%12" PRIu32, total);
    return total;
}


void print_all_counters() {
#if ENABLE_COUNTERS
    my_printf("op types:            ");
#if ENABLE_PER_LAYER_COUNTERS
    for (uint16_t i = 0; i < MODEL_NODES_LEN; i++) {
        my_printf("% 12d", get_node(i)->op_type);
    }
#endif
    uint32_t total_dma_bytes = 0, total_macs = 0, total_overhead = 0;
    my_printf(NEWLINE "Power counters:      "); print_counters<&Counters::power_counters>();
    my_printf(NEWLINE "DMA invocations:     "); print_counters<&Counters::dma_invocations>();
    my_printf(NEWLINE "DMA bytes:           "); total_dma_bytes = print_counters<&Counters::dma_bytes>();
    my_printf(NEWLINE "MACs:                "); total_macs = print_counters<&Counters::macs>();
    // state-embedding overheads
    my_printf(NEWLINE "Embeddings:          "); total_overhead += print_counters<&Counters::embedding>();
    my_printf(NEWLINE "Strippings:          "); total_overhead += print_counters<&Counters::stripping>();
    my_printf(NEWLINE "Overflow handling:   "); total_overhead += print_counters<&Counters::overflow_handling>();
    // state-assignment overheads
    my_printf(NEWLINE "State queries:       "); total_overhead += print_counters<&Counters::state_query>();
    my_printf(NEWLINE "Table updates:       "); total_overhead += print_counters<&Counters::table_updates>();
    my_printf(NEWLINE "Table preservation:  "); total_overhead += print_counters<&Counters::table_preservation>();
    my_printf(NEWLINE "Table loading:       "); total_overhead += print_counters<&Counters::table_loading>();
    // recovery overheads
    my_printf(NEWLINE "Progress seeking:    "); total_overhead += print_counters<&Counters::progress_seeking>();
    // misc
    my_printf(NEWLINE "Memory layout:       "); total_overhead += print_counters<&Counters::memory_layout>();
#if JAPARI
    my_printf(NEWLINE "Data preservation:   "); total_overhead += print_counters<&Counters::preservation>();
    my_printf(NEWLINE "Data loading:        "); total_overhead += print_counters<&Counters::data_loading>();
#endif

    my_printf(NEWLINE "Total DMA bytes: %d", total_dma_bytes);
    my_printf(NEWLINE "Total MACs: %d", total_macs);
    my_printf(NEWLINE "Total overhead: %" PRIu32, total_overhead);
    my_printf(NEWLINE "run_counter: %d" NEWLINE, get_model()->run_counter);

    my_printf("NVM writes: %ld" NEWLINE, get_nvm_writes());
#endif
}

uint8_t run_cnn_tests(uint16_t n_samples) {
    int8_t predicted = -1;
    const ParameterInfo *output_node;
#if MY_DEBUG >= MY_DEBUG_NORMAL
    int8_t label = -1;
    uint32_t correct = 0, total = 0;
    if (!n_samples) {
        n_samples = PLAT_LABELS_DATA_LEN;
    }
    const uint8_t *labels = labels_data;
#endif
    for (uint16_t i = 0; i < n_samples; i++) {
        sample_idx = i;
        run_model(&predicted, &output_node);
#if MY_DEBUG >= MY_DEBUG_NORMAL
        label = labels[i];
        total++;
        if (label == predicted) {
            correct++;
        }
        if (i % 100 == 99) {
            my_printf("Sample %d finished" NEWLINE, sample_idx);
            // stdout is not flushed at \n if it is not a terminal
            my_flush();
        }
        my_printf_debug("idx=%d label=%d predicted=%d correct=%d" NEWLINE, i, label, predicted, label == predicted);
#endif
    }
#if MY_DEBUG >= MY_DEBUG_NORMAL
    if (n_samples == 1) {
        dump_params(get_model(), output_node);
    }
    my_printf("correct=%" PRId32 " ", correct);
    my_printf("total=%" PRId32 " ", total);
    my_printf("rate=%f" NEWLINE, 1.0*correct/total);

    // Allow only 1% of accuracy drop
    if (N_SAMPLES == N_ALL_SAMPLES && correct < (FP32_ACCURACY - 0.01) * total) {
        return 1;
    }
#endif
    return 0;
}


#if INDIRECT_RECOVERY
static void check_feature_map_states(Model *model, const ParameterInfo* output, uint32_t first_unfinished_job_index, uint32_t len, const char* func) {
#if MY_DEBUG >= MY_DEBUG_NORMAL
    my_printf_debug("Running check_feature_map_states..." NEWLINE);
#if 0
    for (uint32_t idx = 0; idx < len; idx++) {
        my_printf_debug("% 6d ", get_q15_param(model, output, idx));
        if (idx % 16 == 15) {
            my_printf_debug(NEWLINE);
        }
    }
#endif
    for (uint32_t idx = 0; ; idx++) {
        uint32_t offset = job_index_to_offset(output, idx);
        if (offset >= len) {
            break;
        }
        int16_t val = get_q15_param(model, output, offset);
        int8_t cur_state_bit = param_state_bit(model, output, offset);
        if (idx < first_unfinished_job_index) {
            cur_state_bit = -cur_state_bit;
        }
        MY_ASSERT(get_value_state_bit(val) == cur_state_bit,
            "Value %d at job index %d (offset %" PRIu32 ") does not have expected state bit %d" NEWLINE, val, idx, offset, cur_state_bit);
    }
#endif
}
#endif

#if STATEFUL
static uint8_t value_finished(Model* model, const ParameterInfo* output, uint32_t job_index) {
    uint32_t offset = job_index_to_offset(output, job_index);
    int16_t val = get_q15_param(model, output, offset);
    uint8_t ret = (get_value_state_bit(val) != param_state_bit(model, output, offset));
    my_printf_debug("Value %d at job index %d (offset %" PRIu32 ") indicates %s" NEWLINE, val, job_index, offset, ret ? "finished" : "unfinished");
    return ret;
}

#endif

#if INDIRECT_RECOVERY

void flip_state_bit(Model *model, const ParameterInfo *output) {
#if JAPARI
    MY_ASSERT(has_footprints(output));
#endif
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    // XXX: better way than copying the array?
#if JAPARI
    // abandon output features smaller than a batch
    uint16_t new_turning_point = (output->params_len / 2) / (BATCH_SIZE + 1) * (BATCH_SIZE + 1);
#else
    uint16_t new_turning_point = (output->params_len / 2) / BATCH_SIZE * BATCH_SIZE;
#endif
    my_printf_debug("New turning point=%d" NEWLINE, new_turning_point);
    uint8_t new_turning_point_inserted = 0;
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        if (new_turning_point < cur_slot_info->turning_points[idx]) {
            uint8_t new_turning_point_idx = idx;
            cur_slot_info->n_turning_points++;
            MY_ASSERT(cur_slot_info->n_turning_points <= TURNING_POINTS_LEN);
            for (uint8_t idx2 = cur_slot_info->n_turning_points - 1; idx2 > new_turning_point_idx; idx2--) {
                cur_slot_info->turning_points[idx2] = cur_slot_info->turning_points[idx2 - 1];
            }
            cur_slot_info->turning_points[new_turning_point_idx] = new_turning_point;
            new_turning_point_inserted = 1;
            break;
        } else if (new_turning_point == cur_slot_info->turning_points[idx]) {
            cur_slot_info->n_turning_points--;
            for (uint8_t idx2 = idx; idx2 < cur_slot_info->n_turning_points; idx2++) {
                cur_slot_info->turning_points[idx2] = cur_slot_info->turning_points[idx2 + 1];
            }
            new_turning_point_inserted = 1;
            break;
        }
    }
    if (!new_turning_point_inserted) {
        cur_slot_info->n_turning_points++;
        cur_slot_info->turning_points[cur_slot_info->n_turning_points - 1] = new_turning_point;
    }

    dump_turning_points_debug(model, output);

    cur_slot_info->state_bit = -cur_slot_info->state_bit;

    // Use first_unfinished_job_index = 0 here as all values finished and the initial state bit is flipped above
    check_feature_map_states(model, output, 0, output->params_len / sizeof(int16_t), __func__);
}

int8_t get_state_bit(Model *model, uint8_t slot_id) {
    switch (slot_id) {
        case SLOT_PARAMETERS:
        case SLOT_TEST_SET:
            return 0;
        default:
            return get_slot_info(model, slot_id)->state_bit;
    }
}

int8_t param_state_bit(Model *model, const ParameterInfo *param, uint16_t offset) {
    int8_t ret = get_state_bit(model, param->slot);
    SlotInfo *cur_slot_info = get_slot_info(model, param->slot);
    if (!cur_slot_info) {
        return 0;
    }
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        if (offset >= cur_slot_info->turning_points[idx]) {
            ret = -ret;
        } else {
            break;
        }
    }
    return ret;
}

#endif

#if HAWAII
uint32_t run_recovery(Model* model, ParameterInfo*) {
    uint32_t footprint = read_hawaii_layer_footprint(model->layer_idx);
    return footprint / BATCH_SIZE;
}
#endif

#if JAPARI
static uint8_t value_finished(Model* model, const ParameterInfo* output, uint32_t job_index) {
    uint32_t offset = job_index_to_offset(output, job_index);
    int16_t val = get_q15_param(model, output, offset);
    int16_t expected_footprint = -param_state_bit(model, output, offset);
    check_footprint(val);
    uint8_t ret = (val == expected_footprint);
    my_printf_debug("Footprint %d (expected %d) at job index %d (offset %" PRIu32 ") indicates %s" NEWLINE, val, expected_footprint, job_index, offset, ret ? "finished" : "unfinished");
    return ret;
}
#endif

uint32_t job_index_to_offset(const ParameterInfo *output, uint16_t job_index) {
#if STATEFUL
    if (job_index >= output->params_len / sizeof(int16_t)) {
        return job_index;
    }
#endif
#if JAPARI
    if (job_index >= output->params_len / sizeof(int16_t) / (BATCH_SIZE + 1)) {
        return job_index * (BATCH_SIZE + 1) + BATCH_SIZE;
    }
#endif

    const Node* node = get_node(output);
#ifdef OpConv
    uint8_t is_conv = (node->op_type == OpConv);
#else
    uint8_t is_conv = 0;
#endif

#if !JAPARI
    if (!is_conv) {
        return (job_index + 1) * BATCH_SIZE - 1;
    }
#else
    if (!is_conv) {
        if (node->op_type == OpRelu) {
            uint16_t OUTPUT_CHANNEL = output->dims[1];
            if (OUTPUT_CHANNEL % (BATCH_SIZE + 1) != 0) {
                uint8_t jobs_in_a_tile = OUTPUT_CHANNEL / (BATCH_SIZE + 1);
                return job_index / jobs_in_a_tile * OUTPUT_CHANNEL + job_index % jobs_in_a_tile * (BATCH_SIZE + 1) + BATCH_SIZE;
            }
        }
        return (job_index + 1) * (BATCH_SIZE + 1) - 1;
    }
#endif

    /* BEGIN constants */
    uint16_t input_tile_len, input_tile_jobs, jobs_in_a_filter_tile, jobs_in_an_op, output_tile_c, OUTPUT_CHANNEL;
    output_tile_c = node->flags.extra.conv.output_tile_c;
    OUTPUT_CHANNEL = output->dims[1];

#if !INDIRECT_RECOVERY
    // not taking this shortcut for approaches that use indirect recovery as
    // output padding is used in those approaches
    if (output_tile_c == OUTPUT_CHANNEL) {
        return job_index * BATCH_SIZE + BATCH_SIZE - 1;
    }
#endif

    uint16_t OUTPUT_H = output->dims[2], OUTPUT_W = output->dims[3];
    input_tile_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;
#if JAPARI
    input_tile_jobs = input_tile_len / (BATCH_SIZE + 1);
#else
    input_tile_jobs = input_tile_len / BATCH_SIZE;
#endif
    output_tile_c = upper_gauss(output_tile_c, BATCH_SIZE) * BATCH_SIZE;
    jobs_in_a_filter_tile = OUTPUT_H * OUTPUT_W * output_tile_c / BATCH_SIZE;
    jobs_in_an_op = output_tile_c / BATCH_SIZE;
    // TODO: handle cases where the following condition is not met
    MY_ASSERT(output_tile_c % BATCH_SIZE == 0);
#if JAPARI
    output_tile_c = extend_for_footprints(output_tile_c);
#endif
    /* END constants */

    uint8_t input_tile_c_index = job_index / input_tile_jobs;
    job_index = job_index % input_tile_jobs;
    uint16_t channel_offset = job_index / jobs_in_a_filter_tile * output_tile_c;
    job_index %= jobs_in_a_filter_tile;
    uint32_t offset = input_tile_c_index * input_tile_len +
                      channel_offset;

    if (jobs_in_an_op) {
        // an op contains at least a batch
        offset += OUTPUT_CHANNEL * (job_index / jobs_in_an_op);
#if !JAPARI
        offset += (job_index % jobs_in_an_op + 1) * BATCH_SIZE - 1;
#else
        offset += (job_index % jobs_in_an_op + 1) * (BATCH_SIZE + 1) - 1;
#endif
    } else {
        // TODO
        ERROR_OCCURRED();
    }
    return offset;
}

uint32_t batch_start(uint32_t batch_end_offset) {
#if JAPARI
    return batch_end_offset - BATCH_SIZE;
#else
    return batch_end_offset - (BATCH_SIZE - 1);
#endif
}

#if INDIRECT_RECOVERY

static uint8_t after_recovery = 1;

uint32_t run_recovery(Model *model, ParameterInfo *output) {
    if (!after_recovery) {
        return 0;
    }

    // recovery from state bits
    uint32_t end_job_index = output->params_len / 2;
#if JAPARI
    end_job_index /= (BATCH_SIZE + 1);
#endif
    my_printf_debug("end_job_index = %d" NEWLINE, end_job_index);
    uint32_t cur_begin_job_index = 0;
    uint32_t cur_end_job_index = end_job_index;
    uint32_t first_unfinished_job_index = 0;

    my_printf_debug("new_output_state_bit for first value = %d" NEWLINE, -param_state_bit(model, output, 0));
    dump_turning_points_debug(model, output);

    while (1) {
        if (cur_end_job_index - cur_begin_job_index <= 1) {
            if (!value_finished(model, output, cur_begin_job_index)) {
                first_unfinished_job_index = 0;
            } else if (!value_finished(model, output, cur_end_job_index)) {
                first_unfinished_job_index = cur_end_job_index;
            } else if (cur_end_job_index == end_job_index) {
                // all values finished - power failure just before the state
                // bit for the output is flipped
                first_unfinished_job_index = end_job_index;
            } else {
                MY_ASSERT(false);
            }
            break;
        }
        uint32_t middle_job_index = cur_begin_job_index + (cur_end_job_index - cur_begin_job_index) / 2;
        if (value_finished(model, output, middle_job_index)) {
            cur_begin_job_index = middle_job_index;
        } else {
            cur_end_job_index = middle_job_index;
        }
        my_printf_debug(
            "job_index of begin = %" PRId32 ", job_index of end = %" PRId32 NEWLINE,
            cur_begin_job_index, cur_end_job_index
        );
    }

    my_printf_debug("first_unfinished_job_index = %d" NEWLINE, first_unfinished_job_index);

    if (!after_recovery) {
        MY_ASSERT(first_unfinished_job_index == 0);
    } else {
        after_recovery = 0;
    }

    check_feature_map_states(model, output, first_unfinished_job_index, output->params_len / 2, __func__);

    return first_unfinished_job_index;
}
#endif
