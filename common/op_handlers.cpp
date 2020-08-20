#include "cnn_common.h"
#include "op_handlers.h"
#include "debug.h"
#include "platform.h"
#include "conv.h"
#include "intermittent-cnn.h"
#include "my_dsplib.h"

// Not using DSPLIB_DATA here as it does not work under C++ (?)
#ifdef __MSP430__
#pragma DATA_SECTION(".leaRAM")
#endif
int16_t lea_buffer[LEA_BUFFER_SIZE];

#define RESHAPE_AUTO_DIM (uint16_t)(-1)

void alloc_maxpool(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    uint16_t stride = flags & 0x0f;

    ParameterInfo *data = input[0];

    const uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t new_H = H / stride;
    uint16_t new_W = W / stride;

    output->params_len = new_H * new_W * CHANNEL * sizeof(int16_t);
    output->slot = get_next_slot(model, data);
    output->dims[0] = 1;
    output->dims[1] = CHANNEL;
    output->dims[2] = new_H;
    output->dims[3] = new_W;
}

static int16_t maxpool_patch(uint16_t output_h, uint16_t output_w, uint16_t c, uint16_t flags, ParameterInfo *data, ParameterInfo *output, Model *model) {
    const uint16_t CHANNEL = data->dims[1], W = data->dims[3];
    uint16_t stride = flags & 0x0f;
    uint16_t kernel_size = (flags & 0xf0) >> 4;
#ifdef WITH_PROGRESS_EMBEDDING
    uint8_t input_state_bit = get_state_bit(model, data->slot);
    uint8_t old_output_state_bit = get_state_bit(model, output->slot);
#endif

    int16_t offset_h, offset_w;
    offset_h = W * CHANNEL;
    offset_w = CHANNEL;

    my_printf_debug("output_h=%d ", output_h);
    my_printf_debug("output_w=%d ", output_w);
    my_printf_debug("c=%d" NEWLINE, c);

    int16_t max_val = INT16_MIN;
#ifndef MY_NDEBUG
    uint16_t max_val_offset;
#endif
    for (uint16_t sH = 0; sH < kernel_size; sH++) {
        for (uint16_t sW = 0; sW < kernel_size; sW++) {
            uint16_t val_offset = (output_h*stride+sH) * offset_h + (output_w*stride+sW) * offset_w + c;
            int16_t val = get_q15_param(data, val_offset);
#ifdef WITH_PROGRESS_EMBEDDING
            if (input_state_bit) {
                val -= 0x4000;
            }
#endif
            dump_value_debug(model, data, val_offset);
            // XXX: use LEA?
            if (val > max_val) {
                max_val = val;
#ifndef MY_NDEBUG
                max_val_offset = val_offset;
#endif
            }
        }
    }
#ifdef WITH_PROGRESS_EMBEDDING
    if (!old_output_state_bit) {
        max_val += 0x4000;
    }
#endif
    // need a space as dump_value does not append spaces when DUMP_INTEGERS is not defined
    my_printf_debug(" max=");
    dump_value_debug(model, data, max_val_offset);
    return max_val;
}

void handle_maxpool(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t flags) {
    my_printf_debug("MaxPool!" NEWLINE);

    uint16_t stride = flags & 0x0f;
    uint8_t need_nhwc2nchw = ((flags & 0xff00) >> 8 == NHWC2NCHW);

    /* XXX: add flags; assume no padding for now */
    ParameterInfo *data = input[0];

    my_printf_debug("handle_maxpool input" NEWLINE);
    dump_params_debug(model, data);

    const uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t new_H = H / stride;
    uint16_t new_W = W / stride;

    determine_tile_c(output);
    uint16_t tile_c = output->tile_c;
    my_printf_debug("tile_c = %d" NEWLINE, tile_c);

#ifdef WITH_PROGRESS_EMBEDDING
    uint32_t first_unfinished_value_offset = recovery_from_state_bits(model, output);
    uint16_t initial_n, initial_c, initial_h, initial_w;
    initial_n = first_unfinished_value_offset / (new_H * new_W * tile_c);

    my_printf_debug("initial_n = %d" NEWLINE, initial_n);
#endif

    uint16_t tile_c_offset = 0;
#ifdef WITH_PROGRESS_EMBEDDING
    tile_c_offset = initial_n * tile_c;
#endif
    for (; tile_c_offset < CHANNEL; tile_c_offset += tile_c) {
        uint16_t real_tile_c = MIN_VAL(tile_c, CHANNEL - tile_c_offset);
        uint16_t output_offset = tile_c_offset * new_H * new_W;
        if (!need_nhwc2nchw) {
            // NHWC
            uint16_t output_h = 0;
#ifdef WITH_PROGRESS_EMBEDDING
            initial_c = first_unfinished_value_offset % real_tile_c;
            first_unfinished_value_offset /= real_tile_c;
            initial_w = first_unfinished_value_offset % new_W;
            first_unfinished_value_offset /= new_W;
            initial_h = first_unfinished_value_offset % new_H;

            my_printf_debug("initial_h = %d" NEWLINE, initial_h);
            my_printf_debug("initial_w = %d" NEWLINE, initial_w);
            my_printf_debug("initial_c = %d" NEWLINE, initial_c);

            if (tile_c_offset == initial_n * tile_c) {
                output_h = initial_h;
                output_offset += initial_h * new_W * real_tile_c;
            }
#endif
            for (; output_h < new_H; output_h++) {
                uint16_t output_w = 0;
#ifdef WITH_PROGRESS_EMBEDDING
                if (tile_c_offset == initial_n * tile_c && output_h == initial_h) {
                    output_w = initial_w;
                    output_offset += initial_w * real_tile_c;
                }
#endif
                for (; output_w < new_W; output_w++) {
                    uint16_t c = 0;
#ifdef WITH_PROGRESS_EMBEDDING
                    if (tile_c_offset == initial_n * tile_c && output_h == initial_h && output_w == initial_w) {
                        c = initial_c;
                        output_offset += initial_c;
                    }
#endif
                    for (; c < real_tile_c; c++) {
                        int16_t max_val = maxpool_patch(output_h, output_w, c + tile_c_offset, flags, data, output, model);
                        my_printf_debug(NEWLINE "offset=%d" NEWLINE, output_offset);
                        put_q15_param(output, output_offset, max_val);
                        output_offset++;
                    }
                }
            }
        } else {
            // NCHW
            uint16_t c = 0;
#ifdef WITH_PROGRESS_EMBEDDING
            initial_w = first_unfinished_value_offset % new_W;
            first_unfinished_value_offset /= new_W;
            initial_h = first_unfinished_value_offset % new_H;
            first_unfinished_value_offset /= new_H;
            initial_c = first_unfinished_value_offset % real_tile_c;

            my_printf_debug("initial_h = %d" NEWLINE, initial_h);
            my_printf_debug("initial_w = %d" NEWLINE, initial_w);
            my_printf_debug("initial_c = %d" NEWLINE, initial_c);

            if (tile_c_offset == initial_n * tile_c) {
                c = initial_c;
                output_offset += initial_c * new_H * new_W;
            }
#endif
            for (; c < real_tile_c; c++) {
                uint16_t output_h = 0;
#ifdef WITH_PROGRESS_EMBEDDING
                if (tile_c_offset == initial_n * tile_c && c == initial_c) {
                    output_h = initial_h;
                    output_offset += initial_h * new_W;
                }
#endif
                for (; output_h < new_H; output_h++) {
                    uint16_t output_w = 0;
#ifdef WITH_PROGRESS_EMBEDDING
                    if (tile_c_offset == initial_n * tile_c && c == initial_c && output_h == initial_h) {
                        output_w = initial_w;
                        output_offset += initial_w;
                    }
#endif
                    for (; output_w < new_W; output_w++) {
                        int16_t max_val = maxpool_patch(output_h, output_w, c + tile_c_offset, flags, data, output, model);
                        my_printf_debug(NEWLINE "offset=%d" NEWLINE, output_offset);
                        put_q15_param(output, output_offset, max_val);
                        output_offset++;
                    }
                }
            }
        }
    }

#ifdef WITH_PROGRESS_EMBEDDING
    flip_state_bit(model, output);
#endif

    my_printf_debug("handle_maxpool output" NEWLINE);
    if (!need_nhwc2nchw) {
        dump_params_nhwc_debug(model, output, 0);
    } else if (tile_c == CHANNEL) {
        dump_params_debug(model, output);
    }
}

void alloc_add(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    ParameterInfo *A = input[0], *B = input[1];
    MY_ASSERT(A->bitwidth == 16 && B->bitwidth == 16);

    output->slot = get_next_slot(model, A);
}

void handle_add(Model*, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    /* Add: Y = X + W */
    my_printf_debug("Add!" NEWLINE);

    ParameterInfo *A = input[0], *B = input[1];

    uint16_t vector_size = A->dims[1];

    int16_t *buffer_a = lea_buffer,
            *buffer_b = lea_buffer + output->params_len / sizeof(int16_t);
    my_memcpy_from_param(buffer_a, A, 0, output->params_len);
    my_memcpy_from_param(buffer_b, B, 0, output->params_len);
    int16_t scaleFract;
    uint8_t shift;
    if (A->scale > B->scale) {
        float_to_scale_params(&scaleFract, &shift, 1.0f * B->scale / A->scale);
        my_scale_q15(buffer_b, scaleFract, shift, buffer_b, vector_size);
    } else if (B->scale > A->scale) {
        float_to_scale_params(&scaleFract, &shift, 1.0f * A->scale / B->scale);
        my_scale_q15(buffer_a, scaleFract, shift, buffer_a, vector_size);
    }
    my_add_q15(buffer_a, buffer_b, buffer_a, vector_size);

    my_memcpy_to_param(output, 0, buffer_a, output->params_len);
}

void alloc_matmul(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    ParameterInfo *A = input[0], *B = input[1];

    uint16_t output_len = A->dims[0] * B->dims[1];

    output->dims[0] = A->dims[0];
    output->dims[1] = B->dims[1];
    output->params_len = output_len * sizeof(int16_t);
    output->bitwidth = 16;
    output->slot = get_next_slot(model, A);
    output->scale = A->scale * B->scale;
}

void handle_matmul(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    ParameterInfo *A = input[0], *B = input[1];

    my_printf_debug("handle_matmul inputs" NEWLINE);
    // dump_params_debug(model, A);
    my_printf_debug("B" NEWLINE);
    dump_params_debug(model, B);
    my_printf_debug("MatMul! A: (%dx%d), B: (%dx%d)" NEWLINE,
              A->dims[0], A->dims[1], B->dims[0], B->dims[1]);

    MY_ASSERT(A->dims[0] * A->dims[1] <= 256);

    int16_t A_len = A->dims[0] * A->dims[1];

    int16_t *buffer_a = lea_buffer,
            *buffer_temp = buffer_a + A_len,
            *buffer_matmul = buffer_temp + A->dims[0] * B->dims[1],
            *buffer_b = buffer_matmul + A->dims[0] * B->dims[1];

    my_fill_q15(0, buffer_matmul, 256);

    my_memcpy_from_param(buffer_a, A, 0, A->dims[0] * A->dims[1] * sizeof(uint16_t));

#ifdef WITH_PROGRESS_EMBEDDING
    if (get_state_bit(model, A->slot)) {
        for (uint16_t idx = 0; idx < A_len; idx++) {
            buffer_a[idx] -= 0x4000;
        }
    }
#endif

    /* LEA wants addresses to be 4-aligned */
    uint16_t step = (uint16_t)((256 / B->dims[1]) / 4 * 4);
    for (uint16_t i = 0; i < B->dims[0]; i = (uint16_t)(i + step)) {
        uint16_t current_width = (uint16_t)MIN_VAL(step, B->dims[0] - i);

        my_memcpy_from_param(buffer_b,
                  B, i * B->dims[1],
                  current_width * B->dims[1] * sizeof(uint16_t));

        my_printf_debug("strip for A" NEWLINE);
        dump_matrix_debug(buffer_a + A->dims[0] * i, (size_t)(A->dims[0] * current_width), ValueInfo(A, model));
        my_printf_debug("B" NEWLINE);
        dump_matrix_debug(buffer_b, (size_t)(current_width * B->dims[1]), ValueInfo(B, model));

        my_matrix_mpy_q15(A->dims[0], current_width, current_width, B->dims[1], buffer_a + A->dims[0] * i, buffer_b, buffer_temp, 0);

        my_printf_debug("temp" NEWLINE);
        dump_matrix_debug(buffer_temp, (size_t)(A->dims[0] * B->dims[1]), ValueInfo(output, model));

        my_add_q15(buffer_matmul, buffer_temp, buffer_matmul, output->params_len / sizeof(int16_t));
    }
    my_memcpy_to_param(output, 0, buffer_matmul, output->params_len);

    my_printf_debug("handle_matmul output" NEWLINE);
    dump_params_debug(model, output);

#ifdef WITH_PROGRESS_EMBEDDING
    flip_state_bit(model, output);
#endif
}

void alloc_relu(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    ParameterInfo *data = input[0];
    output->slot = get_next_slot(model, data);
    output->flags &= ~TRANSPOSED;
}

void handle_relu(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    my_printf_debug("ReLu!" NEWLINE);

    ParameterInfo *X = input[0];
    my_printf_debug("handle_relu input" NEWLINE);
    dump_params_nhwc_debug(model, X, 0);

    uint16_t CHANNEL = X->dims[1];

    /* XXX: use LEA? */
    uint16_t bitwidth = X->bitwidth;
    MY_ASSERT(bitwidth == 16);
    int16_t data_len = X->params_len / (bitwidth / 8);

    int16_t threshold = 0, offset = 0;
#ifdef WITH_PROGRESS_EMBEDDING
    if (get_state_bit(model, X->slot)) {
        threshold = 0x4000;
        offset = -0x4000;
    }
    if (!get_state_bit(model, output->slot)) {
        offset += 0x4000;
    }
#endif

    my_printf_debug("threshold = %d" NEWLINE, threshold);
    my_printf_debug("offset = %d" NEWLINE, offset);

    uint16_t data_offset = 0;
    uint16_t output_offset = 0;
#ifdef WITH_PROGRESS_EMBEDDING
    uint32_t first_unfinished_value_offset = recovery_from_state_bits(model, output);
    data_offset += first_unfinished_value_offset;
    output_offset += first_unfinished_value_offset;
#endif

    if (X->flags & TRANSPOSED) {
        // input is in NWHC
        // TODO: state-aware recovery
        uint16_t H = X->dims[2], W = X->dims[3];
        uint16_t output_h = 0, output_w = 0, c = 0;
#ifdef WITH_PROGRESS_EMBEDDING
        output_h = first_unfinished_value_offset / (W * CHANNEL);
        first_unfinished_value_offset %= (W * CHANNEL);
        output_w = first_unfinished_value_offset / CHANNEL;
        c = first_unfinished_value_offset % CHANNEL;
        my_printf_debug("initial output_h = %d, ", output_h);
        my_printf_debug("initial output_w = %d, ", output_w);
        my_printf_debug("initial c = %d" NEWLINE, c);
#endif
        for (; output_h < H; output_h++) {
            for (; output_w < W; output_w++) {
                for (; c < CHANNEL; c++) {
                    int16_t input_tile_c_index = c / X->tile_c;
                    int16_t input_tile_c_offset = c % X->tile_c;
                    uint16_t cur_input_tile_c = MIN_VAL(X->tile_c, CHANNEL - input_tile_c_index * X->tile_c);
                    int16_t val_offset = input_tile_c_index * W * H * X->tile_c + output_w * H * cur_input_tile_c + output_h * cur_input_tile_c + input_tile_c_offset;
                    int16_t val = get_q15_param(X, val_offset);
                    output_offset = output_h * W * CHANNEL + output_w * CHANNEL + c;
                    put_q15_param(output, output_offset, MAX_VAL(val, threshold) + offset);
                    my_printf_debug(
                        "output_h = %d, output_w = %d, c = %d, offset = %d, input val = %d" NEWLINE,
                        output_h, output_w, c, val_offset, val);
                    my_printf_debug("output_offset = %d" NEWLINE, output_offset);
                }
                c = 0;
            }
            output_w = 0;
        }
    } else {
        uint16_t i = 0;
#ifdef WITH_PROGRESS_EMBEDDING
        i = first_unfinished_value_offset;
#endif
        for (; i < data_len; i++) {
            put_q15_param(output, output_offset, MAX_VAL(get_q15_param(X, data_offset), threshold) + offset);
            data_offset++;
            output_offset++;
        }
    }

    output->tile_c = CHANNEL;

#ifdef WITH_PROGRESS_EMBEDDING
    flip_state_bit(model, output);
#endif

    my_printf_debug("handle_relu output" NEWLINE);
    dump_params_nhwc_debug(model, output, 0);
}

void handle_reshape(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    my_printf_debug("Reshape!" NEWLINE);

    ParameterInfo *data = input[0], *shape = input[1];
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth = data->bitwidth;
    output->slot = data->slot;
    get_slot_info(output->slot)->user = model->layer_idx;
    MY_ASSERT(shape->bitwidth == 64);
    /*
     * At most one dimension of the new shape can be -1. In this case, the
     * value is inferred from the size of the tensor and the remaining
     * dimensions.
     *
     * A dimension could also be 0, in which case the actual dimension value
     * is unchanged (i.e. taken from the input tensor).
     * */
    uint32_t new_len = 1;
    for (uint8_t i = 0; i < 4 && i < shape->dims[0]; i++) {
        output->dims[i] = (uint16_t)get_int64_param(shape, i);
        if (!output->dims[i]) {
            output->dims[i] = data->dims[i];
        }
        if (output->dims[i] != RESHAPE_AUTO_DIM) {
            new_len *= output->dims[i];
        }
    }
    for (uint8_t i = shape->dims[0]; i < 4; i++) {
        output->dims[i] = 0;
    }
    uint16_t inferred_dim = output->params_len / sizeof(int16_t);
    int8_t auto_idx = -1;
    for (uint8_t i = 0; i < 4; i++) {
        if (output->dims[i] != RESHAPE_AUTO_DIM && output->dims[i] != 0) {
            inferred_dim /= output->dims[i];
        } else if (output->dims[i] == RESHAPE_AUTO_DIM) {
            auto_idx = i;
        }
    }
    if (auto_idx != -1) {
        output->dims[auto_idx] = inferred_dim;
        new_len *= inferred_dim;
    }
    MY_ASSERT(new_len * sizeof(int16_t) == output->params_len)
}

void handle_squeeze(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    my_printf_debug("Squeeze!" NEWLINE);

    ParameterInfo *data = input[0];
    /* XXX: add flags; assume squeeze all one-size axes */
    output->params_offset = data->params_offset;
    output->params_len = data->params_len;
    output->bitwidth = data->bitwidth;
    output->slot = data->slot;
    get_slot_info(output->slot)->user = model->layer_idx;
    for (uint8_t i = 0, j = 0; i < 4; i++) {
        if (input[0]->dims[i] != 1) {
            output->dims[j] = input[0]->dims[i];
            j++;
        }
    }
}

template<typename T>
static void iterate_chunks(ParameterInfo *param, T callback) {
    uint16_t params_len = param->params_len / sizeof(int16_t);
    uint16_t chunk_len = LIMIT_DMA_SIZE((LEA_BUFFER_SIZE - 1) / 2 * 2);

    uint16_t cur_chunk_len;
#ifdef WITH_PROGRESS_EMBEDDING
    uint8_t turning_point_idx = 0;
    int16_t next_turning_point = -1;
    SlotInfo *cur_slot_info = get_slot_info(param->slot);
    if (turning_point_idx < cur_slot_info->n_turning_points) {
        next_turning_point = cur_slot_info->turning_points[turning_point_idx];
    }
#endif
    for (uint32_t offset = 0; offset < params_len; offset += cur_chunk_len) {
        cur_chunk_len = MIN_VAL(chunk_len, params_len - offset);
#ifdef WITH_PROGRESS_EMBEDDING
        if (next_turning_point > 0) {
            uint16_t chunk_len_before_turning_point = MIN_VAL(cur_chunk_len, next_turning_point - offset);
            if (chunk_len_before_turning_point != cur_chunk_len) {
                turning_point_idx++;
                next_turning_point = cur_slot_info->turning_points[turning_point_idx];
            }
            cur_chunk_len = chunk_len_before_turning_point;
        }
#endif
        callback(offset, cur_chunk_len);
    }
}

void alloc_concat(Model *, ParameterInfo *[], ParameterInfo*, uint16_t) {
}

void handle_concat(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    my_printf_debug("Concat!" NEWLINE);

    ParameterInfo *A = input[0], *B = input[1];
    // XXX: assume concatenating 2 tensors at the CHANNEL dimension and they
    // have the same number of channels.
    MY_ASSERT(A->dims[1] == B->dims[1]);
    output->tile_c = A->dims[1];
    output->dims[1] *= 2;
    output->flags |= SEPARATE_TILING;

    float scale;
    ParameterInfo *scaled = NULL;
    // The one with smaller `scale` (with larger values) is scaled down
    if (A->scale < B->scale) {
        scale = 1.0f * A->scale / B->scale;
        scaled = A;
        output->scale = A->scale = B->scale;
    } else if (A->scale > B->scale) {
        scale = 1.0f * B->scale / A->scale;
        scaled = B;
        output->scale = B->scale = A->scale;
    }
    if (scaled) {
#ifdef WITH_PROGRESS_EMBEDDING
        uint8_t orig_slot = scaled->slot;
#endif
        uint8_t new_slot = get_next_slot(model, scaled);
#ifdef WITH_PROGRESS_EMBEDDING
        uint8_t old_output_state_bit = get_state_bit(model, new_slot);
#endif
        ParameterInfo tmp_param;
        my_memcpy(&tmp_param, scaled, sizeof(struct ParameterInfo));
        tmp_param.slot = new_slot;

        iterate_chunks(scaled, [&] (uint32_t offset, uint16_t real_chunk_len) {
            my_memcpy_from_param(lea_buffer, scaled, offset, real_chunk_len * sizeof(int16_t));
#ifdef WITH_PROGRESS_EMBEDDING
            my_offset_q15(lea_buffer, get_slot_info(orig_slot)->state_bit ? -0x4000 : 0, lea_buffer, real_chunk_len);
#endif
            my_scale_q15(lea_buffer, scale * 32768, 0, lea_buffer, real_chunk_len);
#ifdef WITH_PROGRESS_EMBEDDING
            my_offset_q15(lea_buffer, old_output_state_bit ? 0 : 0x4000, lea_buffer, real_chunk_len);
#endif
            my_memcpy_to_param(&tmp_param, offset, lea_buffer, real_chunk_len * sizeof(int16_t));
        });

        // XXX: touching nodes is dirty :(
        Node *nodes = (Node*)(model + 1);
        nodes[get_slot_info(output->slot)->user].max_output_id |= MAX_OUTPUT_ID_INVALID; // no longer used
        scaled->slot = new_slot;
#ifdef WITH_PROGRESS_EMBEDDING
        flip_state_bit(model, scaled);
#endif
    }

    // saving slots here as it might be changed during the downscaling loop above
    output->extra_info[0] = A->parameter_info_idx;
    output->extra_info[1] = B->parameter_info_idx;
    output->slot = A->slot;

    dump_params_nhwc_debug(model, A, 0);
    dump_params_nhwc_debug(model, B, 0);
}

void handle_dropout(Model*, ParameterInfo*[], ParameterInfo*, uint16_t) {
    ERROR_OCCURRED();
}

void alloc_globalaveragepool(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    ParameterInfo *data = input[0];

    MY_ASSERT(data->dims[0] == 1);
    uint16_t output_len = data->dims[1];

    output->dims[0] = output->dims[2] = output->dims[3] = 1;
    output->dims[1] = data->dims[1];
    output->params_len = output_len * sizeof(int16_t);
    output->bitwidth = 16;
    output->slot = get_next_slot(model, data);
}

void handle_globalaveragepool(Model *model, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    my_printf_debug("GlobalAveragePool!" NEWLINE);

    ParameterInfo *data = input[0];
    uint16_t CHANNEL = data->dims[1], H = data->dims[2], W = data->dims[3];
    uint16_t len = H * W;
    for (uint16_t c = 0; c < CHANNEL; c++) {
        uint32_t total = 0;
        for (uint16_t h = 0; h < H; h++) {
            for (uint16_t w = 0; w < W; w++) {
                // Input is from Conv, which uses NHWC
                total += get_q15_param(data, h * W * CHANNEL + w * CHANNEL + c);
            }
        }
        put_q15_param(output, c, total / len);
    }

    dump_params_debug(model, output);
}

void handle_softmax(Model*, ParameterInfo*[], ParameterInfo*, uint16_t) {
    // Do nothing - softmax does not change the relative order of values.
    // Just let run_model determine the max value
}

void handle_transpose(Model*, ParameterInfo *input[], ParameterInfo *output, uint16_t) {
    my_printf_debug("Transpose!" NEWLINE);

    ParameterInfo *X = input[0];
    // not actually transpose data as we happen to need NHWC
    // XXX: assume NHWC -> NCHW
    output->dims[1] = X->dims[3];
    output->dims[2] = X->dims[1];
    output->dims[3] = X->dims[2];
}

uint16_t find_overflow_factor(Model *model, ParameterInfo *param) {
    uint16_t overflow_factor = 1;

    iterate_chunks(param, [&] (uint32_t offset, uint16_t real_chunk_len) {
#ifndef WITH_PROGRESS_EMBEDDING
        int16_t min_bound = -32768 / SCALE;
        int16_t max_bound = 32767 / SCALE;
#else
        int16_t min_bound = -8192 / SCALE;
        int16_t max_bound = 8191 / SCALE;
        int16_t val_offset = param_state_bit(model, param, offset) ? -16384 : 0;
#endif
        int16_t max_val, min_val;
        uint16_t index;

        my_memcpy_from_param(lea_buffer, param, offset, real_chunk_len * sizeof(int16_t));

        // dump_matrix(lea_buffer, real_chunk_len, ValueInfo(param));

        my_max_q15(lea_buffer, real_chunk_len, &max_val, &index);
#ifdef WITH_PROGRESS_EMBEDDING
        max_val += val_offset;
#endif
        my_printf_debug("Max value %d", max_val);
        my_printf_debug(" occurs at index %d" NEWLINE, index);
        while (max_val && max_val >= max_bound * overflow_factor) {
            overflow_factor *= 2;
        }

        my_min_q15(lea_buffer, real_chunk_len, &min_val, &index);
#ifdef WITH_PROGRESS_EMBEDDING
        min_val += val_offset;
#endif
        my_printf_debug("Min value %d", min_val);
        my_printf_debug(" occurs at index %d" NEWLINE, index);
        while (min_val && min_val <= min_bound * overflow_factor) {
            overflow_factor *= 2;
        }
    });

    my_printf_debug("Overflow factor = %d" NEWLINE, overflow_factor);

    return overflow_factor;
}

void float_to_scale_params(int16_t *scaleFract, uint8_t *shift, float scale) {
    *shift = 0;
    while (scale >= 1) {
        scale /= 2;
        (*shift)++;
    }
    *scaleFract = scale * 32768;
}
