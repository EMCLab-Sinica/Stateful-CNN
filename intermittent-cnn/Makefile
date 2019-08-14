DSPLIB_SRC_PATH = DSPLib_1_30_00_02/source

DSPLIB_OBJS = \
    $(DSPLIB_SRC_PATH)/vector/msp_mac_q15.o \
    $(DSPLIB_SRC_PATH)/vector/msp_max_q15.o \
    fake-msp430sdk/msp430.o

all: intermittent-cnn

ops.c: ops.h gen_ops.py

ops.h: gen_ops.py
	python $<

intermittent-cnn: $(DSPLIB_OBJS) ops.o op_handlers.o common.o
intermittent-cnn: CPPFLAGS += -isystem DSPLib_1_30_00_02/include -I fake-msp430sdk

clean: clean-common
	rm -rf intermittent-cnn $(DSPLIB_OBJS) ops.*

include ../common.mk

.PHONY: all clean
