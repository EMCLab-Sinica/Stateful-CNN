#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main (void) {
  int fd = open("inputs_table.bin", O_RDONLY);

  uint16_t inputs_table_len;
  read(fd, &inputs_table_len, sizeof(inputs_table_len));
  uint16_t **inputs_table = malloc(inputs_table_len * sizeof(uint16_t*));

  for (uint16_t i = 0; i < inputs_table_len; i++) {
    uint16_t inputs_len;
    read(fd, &inputs_len, sizeof(inputs_len));
    inputs_table[i] = malloc(inputs_len * sizeof(uint16_t));

    for (uint16_t j = 0; j < inputs_len; j++) {
        read(fd, &inputs_table[i][j], sizeof(uint16_t));
        printf("%u ", inputs_table[i][j]);
    }
    printf("\n");

    free(inputs_table[i]);
  }

  free(inputs_table);
  close(fd);

  return 0;
}
