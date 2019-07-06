#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    uint16_t inputs_len;
    int16_t *inputs;
} Node;

typedef struct {
    uint16_t nodes_len;
    uint16_t n_input;
    Node *nodes;
} Model;

Model model;

void load_model(int fd) {
    uint16_t i, j;

    read(fd, &(model.nodes_len), sizeof(model.nodes_len));
    model.nodes = malloc(model.nodes_len * sizeof(Node));

    read(fd, &(model.n_input), sizeof(model.n_input));

    for (i = 0; i < model.nodes_len; i++) {
        Node *cur_node = &(model.nodes[i]);
        read(fd, &(cur_node->inputs_len), sizeof(uint16_t));
        cur_node->inputs = malloc(cur_node->inputs_len * sizeof(int16_t));

        for (j = 0; j < cur_node->inputs_len; j++) {
            read(fd, &(cur_node->inputs[j]), sizeof(int16_t));
        }
    }
}

void free_model(void) {
    uint16_t i;

    for (i = 0; i < model.nodes_len; i++) {
        free(model.nodes[i].inputs);
    }

    free(model.nodes);
}

void dump_model(void) {
    uint16_t i, j;
    for (i = 0; i < model.nodes_len; i++) {
        Node *cur_node = &(model.nodes[i]);
        printf("(");
        for (j = 0; j < cur_node->inputs_len; j++) {
            printf("%d", cur_node->inputs[j]);
            if (j != cur_node->inputs_len - 1) {
                printf(", ");
            }
        }
        printf(") ");
    }
    printf("\n");
}

int main (void) {
    int fd = open("model.bin", O_RDONLY);

    uint16_t cur_group[16] = { 0 };
    uint8_t grp_index = 0;

    uint16_t i, j, k;

    uint16_t next_node_idx = 1;

    load_model(fd);

    printf("model.n_input = %d\n", model.n_input);

    /* initialize - the first node must have no inputs as
     * ONNX already sort nodes topologically */
    cur_group[0] = 0;
    grp_index = 1;

    while (next_node_idx < model.nodes_len) {
        for (i = next_node_idx; i < model.nodes_len; i++) {
            Node *cur_node = &(model.nodes[i]);
            uint8_t no_inputs = 1;
            for (j = 0; j < cur_node->inputs_len; j++) {
                if (cur_node->inputs[j] >= model.n_input) {
                    no_inputs = 0;
                }
            }
            if (no_inputs) {
                printf("Node %d has not inputs.\n", i);
                next_node_idx = i + 1;
                if (grp_index < 16) {
                    cur_group[grp_index] = i;
                    grp_index++;
                } else {
                    break;
                }
            }
        }

        printf("Current group: ");
        for (i = 0; i < grp_index; i++) {
            printf("%d ", cur_group[i]);
        }
        printf("\n");

        for (i = 0; i < model.nodes_len; i++) {
            Node *cur_node = &(model.nodes[i]);
            for (j = 0; j < cur_node->inputs_len; j++) {
                for (k = 0; k < grp_index; k++) {
                    if (cur_node->inputs[j] == cur_group[k] + model.n_input) {
                        cur_node->inputs[j] = -1;
                    }
                }
            }
        }

        grp_index = 0;
        memset(cur_group, 0, sizeof(cur_group));

        dump_model();
    }

    free_model();

    close(fd);

    return 0;
}
