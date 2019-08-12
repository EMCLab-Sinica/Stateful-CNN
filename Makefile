CPPFLAGS = -I .
DEBUG = 0
CFLAGS = -std=c99 -Wall -Wextra -Wstrict-prototypes -Wconversion
ifeq ($(DEBUG),1)
    CFLAGS += -g -O0
else
    CFLAGS += -O3
    CPPFLAGS += -DNDEBUG
endif
PROGS = main parse_model

DSPLIB_SRC_PATH = DSPLib_1_30_00_02/source

DSPLIB_OBJS = \
    $(DSPLIB_SRC_PATH)/utility/msp_iq31_to_q15.o \
    $(DSPLIB_SRC_PATH)/utility/msp_deinterleave_q15.o \
    $(DSPLIB_SRC_PATH)/vector/msp_mac_q15.o \
    fake-msp430sdk/msp430.o

all: $(PROGS)

external/onnx.proto3.pb-c.c : external/onnx.proto3
	protoc-c $< --c_out=.

main: $(DSPLIB_OBJS) ops.o op_handlers.o common.o
main: CPPFLAGS += -isystem DSPLib_1_30_00_02/include -I fake-msp430sdk

ops.c: ops.h gen_ops.py

ops.h: gen_ops.py
	python $<

parse_model: external/onnx.proto3.pb-c.o utils.o
parse_model: LDFLAGS += -lprotobuf-c

clean:
	rm -rf $(PROGS) *.o $(DSPLIB_OBJS) *.pb.* external/*.pb-c.* *.dSYM __pycache__ ops.*

.PHONY: all clean
