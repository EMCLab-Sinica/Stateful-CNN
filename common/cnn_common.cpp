#include <cinttypes> // for PRId32
#include <cmath>
#include <cstdint>

#include "cnn_common.h"
#include "data.h"
#include "intermittent-cnn.h"
#include "my_debug.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"

ParameterInfo intermediate_parameters_info_vm[MODEL_NODES_LEN];
uint16_t sample_idx;

const ParameterInfo* get_parameter_info(uint16_t i) {
    if (i < N_INPUT) {
        return reinterpret_cast<const ParameterInfo*>(model_parameters_info_data) + i;
    } else {
        return get_intermediate_parameter_info(i - N_INPUT);
    }
}

const Node* get_node(size_t i) {
    return reinterpret_cast<const Node*>(nodes_data) + i;
}

const Node* get_node(const ParameterInfo* param) {
    return get_node(param->parameter_info_idx - N_INPUT);
}

SlotInfo* get_slot_info(Model* model, uint8_t i) {
    if (i < NUM_SLOTS) {
        return model->slots_info + i;
    } else {
        return nullptr;
    }
}

int16_t get_q15_param(Model* model, const ParameterInfo *param, uint16_t i) {
    MY_ASSERT(param->bitwidth == 16);
    if (param->slot == SLOT_TEST_SET) {
        int16_t ret;
        read_from_samples(&ret, i, sizeof(int16_t));
        return ret;
    } else if (param->slot == SLOT_PARAMETERS) {
        int16_t ret;
        my_memcpy_from_parameters(&ret, param, i * sizeof(int16_t), sizeof(int16_t));
        return ret;
    } else {
        int16_t ret;
        my_memcpy_from_param(model, &ret, param, i, sizeof(int16_t));
        return ret;
    }
}

void put_q15_param(ParameterInfo *param, uint16_t i, int16_t val) {
    my_memcpy_to_param(param, i, &val, sizeof(int16_t), 0);
}

int64_t get_int64_param(const ParameterInfo *param, size_t i) {
    MY_ASSERT(param->bitwidth == 64);
    MY_ASSERT(param->slot == SLOT_PARAMETERS);
    int64_t ret;
    my_memcpy_from_parameters(&ret, param, i * sizeof(int64_t), sizeof(int64_t));
    return ret;

}

uint16_t get_next_slot(Model *model, const ParameterInfo *param) {
    uint16_t slot_id = param->slot;
    /* pick the next unused slot */
    uint16_t next_slot_id = slot_id;
    uint8_t cycle_count = 0;
    while (1) {
        next_slot_id++;
        // Fail if the loop has run a cycle
        if (next_slot_id >= NUM_SLOTS) {
            next_slot_id = 0;
            cycle_count++;
            MY_ASSERT(cycle_count <= 1);
        }
        int16_t slot_user_id = get_slot_info(model, next_slot_id)->user;
        if (slot_user_id < 0) {
            break;
        }
        // previously allocated, most likely in a previous power cycle
        if (slot_user_id == model->layer_idx) {
            break;
        }
        const Node *slot_user = get_node(slot_user_id);
        if (slot_user->max_output_id < model->layer_idx) {
            break;
        }
        // The recorded slot user is not the actual user. This happens when Concat
        // uses a new slot for scaled IFM. The old slot is actually used by nobody
        // and available for allocation.
        if (get_parameter_info(N_INPUT + slot_user_id)->slot != next_slot_id) {
            break;
        }
    }
    my_printf_debug("next_slot_id = %d" NEWLINE, next_slot_id);
    get_slot_info(model, next_slot_id)->user = model->layer_idx;
    return next_slot_id;
}

void my_memcpy_from_param(Model* model, void *dest, const ParameterInfo *param, uint16_t offset_in_word, size_t n) {
    if (param->slot == SLOT_TEST_SET) {
        read_from_samples(dest, offset_in_word, n);
    } else if (param->slot == SLOT_PARAMETERS) {
        my_memcpy_from_parameters(dest, param, offset_in_word * sizeof(int16_t), n);
    } else {
        my_memcpy_from_intermediate_values(dest, param, offset_in_word, n);
    }
}

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

#if ENABLE_DEMO_COUNTERS
    my_printf("CMD,P,%d" NEWLINE, 100 * node_idx / MODEL_NODES_LEN);
#endif
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
