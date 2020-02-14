CPPFLAGS = -I .
DEBUG = 0
CFLAGS = -std=c99 -Wall -Wextra -Wstrict-prototypes -Wshadow -MMD -g -O0

UNAME_S := $(shell uname -s)

DSPLIB_SRC_PATH = ../../DSPLib_1_30_00_02/source

SRCS = \
    $(DSPLIB_SRC_PATH)/matrix/msp_matrix_mpy_q15.c \
    $(DSPLIB_SRC_PATH)/vector/msp_add_q15.c \
    $(DSPLIB_SRC_PATH)/vector/msp_mac_q15.c \
    $(DSPLIB_SRC_PATH)/vector/msp_max_q15.c \
    $(DSPLIB_SRC_PATH)/utility/msp_fill_q15.c \
    fake-msp430sdk/msp430.c \
    intermittent-cnn.c \
    ops.c \
    op_handlers.c \
    common.c \
    data.c \
    debug.c

ifeq ($(UNAME_S),Linux)
    SRCS += plat-linux.c
endif

# https://stackoverflow.com/a/15360191/3786245
vpath %.c $(sort $(dir $(SRCS)))

OBJS = $(addprefix out/,$(patsubst %.c, %.o, $(notdir $(SRCS))))
# http://wen00072.github.io/blog/2014/03/06/makefile-header-file-dependency-issues/
DEPS = $(patsubst %.o, %.d, $(OBJS))

MODEL := $(DATA_PATH)/models/mnist/model_optimized.onnx
IMAGE := $(DATA_PATH)/example3.png
INPUT_DATA_FILES = data.c data.h ops.c ops.py ops.h inputs.bin model.bin parameters.bin
DATA_FILES = $(INPUT_DATA_FILES) nvm.bin

all: out/intermittent-cnn nvm.bin

$(OBJS): $(INPUT_DATA_FILES)
$(OBJS): out/%.o: %.c
	mkdir -p out && $(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

ops.py ops.c ops.h: gen_ops.py
	python $<

out/intermittent-cnn: $(OBJS)
	$(CC) $^ -o $@

out/intermittent-cnn: CPPFLAGS += -isystem ../../DSPLib_1_30_00_02/include -I fake-msp430sdk

data.c data.h: bin2c.py model.bin
	python bin2c.py

inputs.bin model.bin parameters.bin: transform.py ops.py
	python transform.py $(MODEL) $(IMAGE)

nvm.bin:
	dd if=/dev/zero of=nvm.bin bs=1024 count=256 # 256KB

clean:
	rm -rvf intermittent-cnn out __pycache__ $(DATA_FILES)

-include $(DEPS)

.PHONY: all clean
