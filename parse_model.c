#include <stdio.h>
#include <stdlib.h>
#include "external/onnx.proto3.pb-c.h"
#include "utils.h"

int main(void) {
  Onnx__ModelProto *msg;
  size_t i, j;

  uint8_t *buf = malloc(MAX_MSG_SIZE);
  size_t msg_len = read_buffer(MAX_MSG_SIZE, buf);

  if (!msg_len) {
    fprintf(stderr, "no data read\n");
    return 1;
  }

  msg = onnx__model_proto__unpack(NULL, msg_len, buf);
  if (msg == NULL) {
    fprintf(stderr, "error unpacking incoming message\n");
    return 1;
  }

  Onnx__GraphProto *graph = msg->graph;
  printf("Inputs:\n");
  for (i = 0; i < graph->n_input; i++) {
      printf("%s\n", graph->input[i]->name);
  }
  printf("\nNodes:\n");
  for (i = 0; i < graph->n_node; i++) {
      Onnx__NodeProto *n = graph->node[i];
      printf("name = %s, op_type = %s\n", n->name, n->op_type);
      printf("\tNode inputs:\n");
      for (j = 0; j < n->n_input; j++) {
          printf("\t\t%s\n", n->input[j]);
      }
      printf("\tNode outputs:\n");
      for (j = 0; j < n->n_output; j++) {
          printf("\t\t%s\n", n->output[j]);
      }
  }

  onnx__model_proto__free_unpacked(msg, NULL);
  free(buf);

  return 0;
}
