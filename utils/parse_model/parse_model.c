#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "external/onnx.proto3.pb-c.h"
#include "utils.h"

static void handle_value_info(Onnx__ValueInfoProto *value_info) {
    size_t i;

    Onnx__TensorShapeProto *shape = value_info->type->tensor_type->shape;
    if (!shape || !shape->n_dim) {
        return;
    }
    printf("name = %s, shape = (", value_info->name);
    for (i = 0; i < shape->n_dim; i++) {
        Onnx__TensorShapeProto__Dimension *d = shape->dim[i];
        switch (d->value_case) {
            case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
                printf("%" PRId64, d->dim_value);
                break;
            case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
                printf("%s", d->dim_param);
                break;
            default:
                break;
        }
        if (i != shape->n_dim - 1) {
            printf(", ");
        }
    }
    printf(")\n");
}

static void handle_tensor(Onnx__TensorProto *tensor) {
    size_t i;

    printf("name=%s, dims=[", tensor->name);
    for (i = 0; i < tensor->n_dims; i++) {
        printf("%" PRId64, tensor->dims[i]);
        if (i != tensor->n_dims - 1) {
            printf(", ");
        }
    }
    printf("], ");

    switch (tensor->data_type) {
        case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
            printf("type=INT64");
            break;
        default:
            printf("(Tensor data type %d not implemented)", tensor->data_type);
            break;
    }
}

int main(void) {
  Onnx__ModelProto *msg;
  size_t i, j, k;

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

  for (i = 0; i < msg->n_opset_import; i++) {
      Onnx__OperatorSetIdProto *opset_import = msg->opset_import[i];
      if (!strlen(opset_import->domain)) {
          printf("%s", "ai.onnx");
      } else {
          printf("%s", opset_import->domain);
      }
      printf(" v%" PRId64 "\n", opset_import->version);
      break;
  }

  Onnx__GraphProto *graph = msg->graph;
  printf("\nInputs:\n");
  for (i = 0; i < graph->n_input; i++) {
      handle_value_info(graph->input[i]);
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
      printf("\tNode attributes:\n");
      for (j = 0; j < n->n_attribute; j++) {
          Onnx__AttributeProto *attr = n->attribute[j];
              printf("\t\t%s = ", attr->name);
          switch (attr->type) {
              case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT:
                  printf("FLOAT:%f", attr->f);
                  break;
              case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT:
                  printf("INT:%" PRId64, attr->i);
                  break;
              case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR:
                  printf("TENSOR:");
                  handle_tensor(attr->t);
                  break;
              case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING:
                  printf("STRING:%s", attr->s.data);
                  break;
              case ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS:
                  printf("INTS:[");
                  for (k = 0; k < attr->n_ints; k++) {
                      printf("%" PRId64, attr->ints[k]);
                      if (k != attr->n_ints - 1) {
                          printf(", ");
                      }
                  }
                  printf("]");
                  break;
              default:
                  printf("(Attribute type %d not implemented)", attr->type);
                  break;
          }
          printf("\n");
      }
  }
  printf("\nInitializers:\n");
  for (i = 0; i < graph->n_initializer; i++) {
      handle_tensor(graph->initializer[i]);
      printf("\n");
  }
  printf("\nValue info:\n");
  for (i = 0; i < graph->n_value_info; i++) {
      handle_value_info(graph->value_info[i]);
  }

  onnx__model_proto__free_unpacked(msg, NULL);
  free(buf);

  return 0;
}
