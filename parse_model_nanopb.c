#include <stdio.h>
#include <stdlib.h>
#include "pb_encode.h"
#include "pb_decode.h"
#include "utils.h"
#include "onnx-nanopb.pb.h"

int main(void) {
    bool status;

    uint8_t *buf = malloc(MAX_MSG_SIZE);
    size_t msg_len = read_buffer(MAX_MSG_SIZE, buf);

    onnx_ModelProto model = onnx_ModelProto_init_zero;

    pb_istream_t stream = pb_istream_from_buffer(buf, msg_len);

    status = pb_decode(&stream, onnx_ModelProto_fields, &model);

    if (!status) {
        printf("Decoding failed: %s\n", PB_GET_ERROR(&stream));
        return 1;
    }

    /* XXX: how to handle pb_callback_t for repeated nodes */

    free(buf);

    return 0;
}

