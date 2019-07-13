#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define NEWLINE "\n"

typedef struct {
    uint16_t inputs_len;
    uint16_t inputs_offset;
    uint16_t scheduled;  /* 16 bits for aligned memory */
} Node;

typedef struct {
    uint16_t nodes_len;
    uint16_t n_input;
} Model;

Model *model;
uint16_t *inputs;

static inline Node* model_node(size_t i) {
    return (Node*)(model + 1) + i;
}

static inline int16_t* node_input(Node *node, size_t i) {
    return (int16_t*)((uint8_t*)inputs + node->inputs_offset) + i;
}

static void dump_model(void) {
#ifndef NDEBUG
    uint16_t i, j;
    for (i = 0; i < model->nodes_len; i++) {
        Node *cur_node = model_node(i);
        printf("(");
        for (j = 0; j < cur_node->inputs_len; j++) {
            printf("%d", *node_input(cur_node, j));
            if (j != cur_node->inputs_len - 1) {
                printf(", ");
            }
        }
        printf(") ");
    }
    printf(NEWLINE);
#endif
}

static int map_files(void) {
    int map_size;
    int fd_model = -1, fd_inputs = -1;
    int ret = 0;

    fd_model = open("model.bin", O_RDONLY);
    fd_inputs = open("inputs.bin", O_RDONLY);
    if (fd_model == -1 || fd_inputs == -1) {
        perror("open files failed");
        ret = -1;
        goto error;
    }

    map_size = 4 * getpagesize();
    model = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd_model, 0);
    inputs = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd_inputs, 0);

    if (model == MAP_FAILED || inputs == MAP_FAILED) {
        perror("mmap failed");
        ret = -1;
    }

error:
    if (fd_model != -1) {
        close(fd_model);
    }
    if (fd_inputs != -1) {
        close(fd_inputs);
    }

    return ret;
}

int main (void) {
    uint16_t cur_group[16] = { 0 };
    uint8_t grp_index = 0;
    uint16_t group_last_item;

    uint16_t i, j, k;

    uint16_t next_node_idx = 1;

    if (map_files() != 0) {
        return 1;
    }

    printf("model->n_input = %d" NEWLINE, model->n_input);

    /* initialize - the first node must have no inputs as
     * ONNX already sort nodes topologically */
    cur_group[0] = 0;
    grp_index = 1;

    dump_model();

    while (next_node_idx < model->nodes_len) {
        for (i = next_node_idx; i < model->nodes_len; i++) {
            Node *cur_node = model_node(i);
            uint8_t no_inputs = 1;

            if (cur_node->scheduled) {
                continue;
            }

            for (j = 0; j < cur_node->inputs_len; j++) {
                if (*node_input(cur_node, j) >= model->n_input) {
                    no_inputs = 0;
                }
            }
            if (no_inputs) {
#ifndef NDEBUG
                printf("Node %d has no inputs." NEWLINE, i);
#endif
                cur_group[grp_index] = i;
                grp_index++;
                next_node_idx = i + 1;
                if (grp_index == 16) {
                    break;
                }
            }
        }

        if (!grp_index) {
            printf("Error!" NEWLINE);
            break;
        }

        if (grp_index < 16) {
            next_node_idx = 0;
        }

        printf("Current group: ");
        for (i = 0; i < grp_index; i++) {
            printf("%d ", cur_group[i]);
            /* schedule it */
            model_node(cur_group[i])->scheduled = 1;
        }
        printf(" - %d element(s)." NEWLINE, grp_index);

        group_last_item = cur_group[grp_index - 1];

        if (group_last_item == model->nodes_len - 1) {
            break;
        }

        for (i = cur_group[0]; i < model->nodes_len; i++) {
            Node *cur_node = model_node(i);
            for (j = 0; j < cur_node->inputs_len; j++) {
                if (*node_input(cur_node, j) > group_last_item + model->n_input) {
                    break;
                }
                for (k = 0; k < grp_index; k++) {
                    if (*node_input(cur_node, j) == cur_group[k] + model->n_input) {
                        *node_input(cur_node, j) = -1;
                    }
                }
            }
        }

        grp_index = 0;
        memset(cur_group, 0, sizeof(cur_group));

        dump_model();
    }

    return 0;
}
