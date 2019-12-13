CPPFLAGS = -I .
DEBUG = 0
CFLAGS = -std=c99 -Wall -Wextra -Wstrict-prototypes -Wconversion -Wshadow
ifeq ($(DEBUG),1)
    CFLAGS += -g -O0
else
    CFLAGS += -O3
    CPPFLAGS += -DNDEBUG
endif

DSPLIB_SRC_PATH = DSPLib_1_30_00_02/source

DSPLIB_OBJS = \
    $(DSPLIB_SRC_PATH)/matrix/msp_matrix_mpy_q15.o \
    $(DSPLIB_SRC_PATH)/vector/msp_add_q15.o \
    $(DSPLIB_SRC_PATH)/vector/msp_mac_q15.o \
    $(DSPLIB_SRC_PATH)/vector/msp_max_q15.o \
    fake-msp430sdk/msp430.o

UNAME_S := $(shell uname -s)

MODEL := ../../models/mnist/model_optimized.onnx
IMAGE := ../../example3.png

all: intermittent-cnn

ops.c: ops.h gen_ops.py

ops.h: gen_ops.py
	python $<

intermittent-cnn: $(DSPLIB_OBJS) ops.o op_handlers.o common.o data.o
intermittent-cnn: CPPFLAGS += -isystem DSPLib_1_30_00_02/include -I fake-msp430sdk

ifeq ($(UNAME_S),Linux)
    intermittent-cnn: plat-linux.o
endif

data.c data.h: bin2c.py model.bin
	python bin2c.py

model.bin: transform.py
	python transform.py $(MODEL) $(IMAGE)

clean:
	rm -rf intermittent-cnn $(DSPLIB_OBJS) *.o __pycache__ ops.* *.bin data.*

.PHONY: all clean
