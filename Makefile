CPPFLAGS = -I .
DEBUG = 0
CFLAGS = -std=c99 -Wall -Wextra -Wstrict-prototypes -Wconversion -Wshadow -MMD -g -O0

UNAME_S := $(shell uname -s)

DSPLIB_SRC_PATH = ../../DSPLib_1_30_00_02/source

OBJS = \
    $(DSPLIB_SRC_PATH)/matrix/msp_matrix_mpy_q15.o \
    $(DSPLIB_SRC_PATH)/vector/msp_add_q15.o \
    $(DSPLIB_SRC_PATH)/vector/msp_mac_q15.o \
    $(DSPLIB_SRC_PATH)/vector/msp_max_q15.o \
    $(DSPLIB_SRC_PATH)/utility/msp_fill_q15.o \
    fake-msp430sdk/msp430.o \
    ops.o \
    op_handlers.o \
    common.o \
    data.o

ifeq ($(UNAME_S),Linux)
    OBJS += plat-linux.o
endif
# http://wen00072.github.io/blog/2014/03/06/makefile-header-file-dependency-issues/
DEPS = $(patsubst %.o, %.d, $(OBJS))

MODEL := $(DATA_PATH)/models/mnist/model_optimized.onnx
IMAGE := $(DATA_PATH)/example3.png
DATA_FILES = data.c data.h ops.c ops.py ops.h inputs.bin model.bin parameters.bin

all: intermittent-cnn

data_files: $(DATA_FILES)

ops.py ops.c ops.h: gen_ops.py
	python $<

intermittent-cnn: $(OBJS)
intermittent-cnn: CPPFLAGS += -isystem ../../DSPLib_1_30_00_02/include -I fake-msp430sdk

data.c data.h: bin2c.py model.bin
	python bin2c.py

inputs.bin model.bin parameters.bin: transform.py ops.py
	python transform.py $(MODEL) $(IMAGE)

clean:
	rm -rf intermittent-cnn $(OBJS) $(DEPS) __pycache__ $(DATA_FILES)

-include $(DEPS)

.PHONY: all clean data_files
