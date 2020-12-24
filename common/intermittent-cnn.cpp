#include <stdint.h>
#include <string.h>
#include <inttypes.h> // for PRId32
#include <math.h>

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"

uint16_t sample_idx;

static void handle_node(Model *model, uint16_t node_idx) {
    my_printf_debug("Current node: %d, ", node_idx);

    /* schedule it */
    const Node *cur_node = get_node(node_idx);
    my_printf_debug("name = %.*s, ", NODE_NAME_LEN, cur_node->name);
    my_printf_debug("op_type = %d" NEWLINE, cur_node->op_type);

    int16_t input_id[3];
    const ParameterInfo *input[3];
    MY_ASSERT(cur_node->inputs_len == expected_inputs_len[cur_node->op_type]);
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
    allocators[cur_node->op_type](model, input, output, &cur_node->flags);
    my_printf_debug("Needed mem = %d" NEWLINE, output->params_len);
    MY_ASSERT(output->params_len < INTERMEDIATE_VALUES_SIZE);
    if (output->slot == SLOT_INTERMEDIATE_VALUES) {
        my_printf_debug("New params_offset = %d" NEWLINE, output->params_offset);
    }

#if STATEFUL
    my_printf_debug("Old output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif
    handlers[cur_node->op_type](model, input, output, &cur_node->flags);
    // For some operations (e.g., ConvMerge), scale is determined in the handlers
    my_printf_debug("Output scale = %d" NEWLINE, output->scale);
    MY_ASSERT(output->scale > 0);  // fail when overflow
#if STATEFUL
    my_printf_debug("New output state bit=%d" NEWLINE, get_state_bit(model, output->slot));
#endif

#if MY_DEBUG >= 1
    my_printf_debug("output dims: ");
    uint8_t has_dims = 0;
    for (uint8_t j = 0; j < 4; j++) {
        if (output->dims[j]) {
            has_dims = 1;
            my_printf_debug("%d, ", output->dims[j]);
        }
    }
    my_printf_debug(NEWLINE);
    MY_ASSERT(has_dims);
    MY_ASSERT(output->bitwidth);
#endif

    commit_intermediate_parameter_info(node_idx);

    if (node_idx == MODEL_NODES_LEN - 1) {
        model->running = 0;
        model->run_counter++;
        notify_model_finished();
    }
}

float first_sample_outputs[] = FIRST_SAMPLE_OUTPUTS;

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

    counters()->power_counters[model->layer_idx]++;

    dump_model_debug(model);

    for (uint16_t node_idx = model->layer_idx; node_idx < MODEL_NODES_LEN; node_idx++) {
        handle_node(model, node_idx);
        model->layer_idx++;

        commit_model();

        dump_model_debug(model);
    }

    /* XXX: is the last node always the output node? */
    const ParameterInfo *output_node = get_parameter_info(MODEL_NODES_LEN + N_INPUT - 1);
    if (output_node_ptr) {
        *output_node_ptr = output_node;
    }
    int16_t max = INT16_MIN;
    uint16_t u_ans;
    uint8_t buffer_len = MIN_VAL(output_node->dims[1], sizeof(first_sample_outputs) / sizeof(float));
    my_memcpy_from_param(model, lea_buffer, output_node, 0, buffer_len * sizeof(int16_t));

#if MY_DEBUG >= 1
    if (sample_idx == 0) {
        for (uint8_t buffer_idx = 0, ofm_idx = 0; buffer_idx < buffer_len; buffer_idx++) {
            int16_t got_q15 = lea_buffer[buffer_idx];
#if JAPARI
            if (buffer_idx % (BATCH_SIZE + 1) == BATCH_SIZE) {
                check_footprint(got_q15);
            } else
#endif
            {
                float got_real = q15_to_float(got_q15, ValueInfo(output_node));
                float expected = first_sample_outputs[ofm_idx];
                float error = fabs((got_real - expected) / expected);
                // Errors in CIFAR-10/Stateful are quite large...
                MY_ASSERT(error <= 0.25,
                          "Value error too large at index %d: got=%f, expected=%f" NEWLINE, buffer_idx, got_real, expected);
                ofm_idx++;
            }
        }
    }
#endif

    my_max_q15(lea_buffer, buffer_len, &max, &u_ans);
#if JAPARI
    u_ans = u_ans / (BATCH_SIZE + 1) * BATCH_SIZE + u_ans % (BATCH_SIZE + 1);
#endif
    *ansptr = u_ans;
}

#if MY_DEBUG >= 1
static void print_results(const ParameterInfo *output_node) {
    Model *model = get_model();

    dump_params(model, output_node);

    my_printf("op types:       ");
    for (uint8_t i = 0; i < MODEL_NODES_LEN; i++) {
        my_printf("% 8d", get_node(i)->op_type);
        if (i % 16 == 15) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE "ticks:          ");
    for (uint8_t i = 0; i < MODEL_NODES_LEN; i++) {
        my_printf("% 8d", counters()->time_counters[i]);
        if (i % 16 == 15) {
            my_printf(NEWLINE);
        }
    }
#if NON_VOLATILE_COUNTERS
    my_printf(NEWLINE "power counters: ");
    for (uint8_t i = 0; i < MODEL_NODES_LEN; i++) {
        my_printf("% 8d", counters()->power_counters[i]);
        if (i % 16 == 15) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE "DMA invocations:");
    for (uint8_t i = 0; i < MODEL_NODES_LEN; i++) {
        my_printf("% 8d", counters()->dma_invocations[i]);
    }
    my_printf(NEWLINE "DMA bytes:      ");
    uint32_t total_dma_bytes = 0;
    for (uint8_t i = 0; i < MODEL_NODES_LEN; i++) {
        total_dma_bytes += counters()->dma_bytes[i];
        my_printf("% 8d", counters()->dma_bytes[i]);
    }
    my_printf(NEWLINE "Total DMA bytes: %d", total_dma_bytes);
#endif
    my_printf(NEWLINE "run_counter: %d" NEWLINE, model->run_counter);
}
#endif

uint8_t run_cnn_tests(uint16_t n_samples) {
    int8_t label = -1, predicted = -1;
    uint32_t correct = 0, total = 0;
    if (!n_samples) {
        n_samples = PLAT_LABELS_DATA_LEN;
    }
    const ParameterInfo *output_node;
    const uint8_t *labels = labels_data;
    for (uint16_t i = 0; i < n_samples; i++) {
        sample_idx = i;
        label = labels[i];
        run_model(&predicted, &output_node);
        total++;
        if (label == predicted) {
            correct++;
        }
#if MY_DEBUG >= 1
        if (i % 100 == 0) {
            my_printf("Sample %d finished" NEWLINE, sample_idx);
            // stdout is not flushed at \n if it is not a terminal
            my_flush();
        }
        my_printf_debug("idx=%d label=%d predicted=%d correct=%d" NEWLINE, i, label, predicted, label == predicted);
#endif
    }
#if MY_DEBUG >= 1
    if (n_samples == 1) {
        print_results(output_node);
    }
    my_printf("correct=%" PRId32 " ", correct);
    my_printf("total=%" PRId32 " ", total);
    my_printf("rate=%f" NEWLINE, 1.0*correct/total);
#endif

    // Allow only 1% of accuracy drop
    if (N_SAMPLES == N_ALL_SAMPLES && correct < (FP32_ACCURACY - 0.01) * total) {
        return 1;
    }
    return 0;
}


#if INDIRECT_RECOVERY
static void check_feature_map_states(Model *model, const ParameterInfo* output, uint32_t first_unfinished_job_index, uint32_t len, const char* func) {
#if MY_DEBUG >= 1
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
        uint8_t cur_state_bit = param_state_bit(model, output, offset);
        if (idx < first_unfinished_job_index) {
            cur_state_bit ^= 1;
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

void flip_state_bit(Model *model, const ParameterInfo *output) {
#if INDIRECT_RECOVERY

#if JAPARI
    MY_ASSERT(has_footprints(output));
#endif
    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
    // XXX: better way than copying the array?
    uint16_t new_turning_point = output->params_len / 2;
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

    cur_slot_info->state_bit ^= 1;

    // Use first_unfinished_job_index = 0 here as all values finished and the initial state bit is flipped above
    check_feature_map_states(model, output, 0, output->params_len / sizeof(int16_t), __func__);

#endif // INDIRECT_RECOVERY
}

#if INDIRECT_RECOVERY

uint8_t get_state_bit(Model *model, uint8_t slot_id) {
    switch (slot_id) {
        case SLOT_PARAMETERS:
        case SLOT_TEST_SET:
            return 0;
        default:
            return get_slot_info(model, slot_id)->state_bit;
    }
}

uint8_t param_state_bit(Model *model, const ParameterInfo *param, uint16_t offset) {
    uint8_t ret = get_state_bit(model, param->slot);
    SlotInfo *cur_slot_info = get_slot_info(model, param->slot);
    if (!cur_slot_info) {
        return 0;
    }
    for (uint8_t idx = 0; idx < cur_slot_info->n_turning_points; idx++) {
        if (offset >= cur_slot_info->turning_points[idx]) {
            ret = ret ^ 1;
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
    int16_t expected_footprint = (param_state_bit(model, output, offset) ? -1 : 1);
    check_footprint(val);
    uint8_t ret = (val == expected_footprint);
    my_printf_debug("Footprint %d (expected %d) at job index %d (offset %" PRIu32 ") indicates %s" NEWLINE, val, expected_footprint, job_index, offset, ret ? "finished" : "unfinished");
    return ret;
}
#endif

uint32_t job_index_to_offset(const ParameterInfo* output, uint16_t job_index) {
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
#if !JAPARI
    if (node->op_type != Conv) {
        return (job_index + 1) * BATCH_SIZE - 1;
    }
#else
    if (node->op_type != Conv) {
        if (node->op_type == Relu) {
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
    OUTPUT_CHANNEL = output->dims[1];
    uint16_t OUTPUT_H = output->dims[2], OUTPUT_W = output->dims[3];
    input_tile_len = OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W;
#if JAPARI
    input_tile_jobs = input_tile_len / (BATCH_SIZE + 1);
#else
    input_tile_jobs = input_tile_len / BATCH_SIZE;
#endif
    output_tile_c = upper_gauss(node->flags.extra.conv.output_tile_c, BATCH_SIZE) * BATCH_SIZE;
    jobs_in_a_filter_tile = OUTPUT_H * OUTPUT_W * output_tile_c / BATCH_SIZE;
    jobs_in_an_op = output_tile_c / BATCH_SIZE;
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

    my_printf_debug("new_output_state_bit for first value = %d" NEWLINE, param_state_bit(model, output, 0) ^ 1);
    dump_turning_points_debug(model, output);

    while (1) {
#if 0
        dump_matrix_debug(model, output, cur_begin_offset, cur_end_offset - cur_begin_offset, ValueInfo(output));
#endif
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
