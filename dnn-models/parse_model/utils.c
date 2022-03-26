#include <stdint.h>
#include <stdio.h>
#include "utils.h"

size_t read_buffer(unsigned max_length, uint8_t *out) {
  size_t cur_len = 0;
  size_t nread;
  while ((nread = fread(out + cur_len, 1, max_length - cur_len, stdin)) != 0) {
    cur_len += nread;
    if (cur_len == max_length) {
      fprintf(stderr, "max message length exceeded\n");
      return 0;
    }
  }
  return cur_len;
}
