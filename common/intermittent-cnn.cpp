#include <cstdint>
#include <cstring>
#include <cinttypes> // for PRId32

#include "intermittent-cnn.h"
#include "cnn_common.h"
#include "counters.h"
#include "data.h"
#include "my_debug.h"
#include "op_utils.h"
#include "platform.h"

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
