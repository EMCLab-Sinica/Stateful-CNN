CPPFLAGS = -I .
DEBUG = 0
CFLAGS = -std=gnu89 -Wall -Wextra -Wstrict-prototypes
ifeq ($(DEBUG),1)
    CFLAGS += -g -O0
else
    CFLAGS += -O3
    CPPFLAGS += -DNDEBUG
endif
PROGS = main parse_model

DSPLIB_SRC_PATH = DSPLib_1_30_00_02/source

DSPLIB_OBJS = \
    $(DSPLIB_SRC_PATH)/vector/msp_mpy_q15.o \
    fake-msp430sdk/msp430.o

all: $(PROGS)

external/onnx.proto3.pb-c.c : external/onnx.proto3
	protoc-c $< --c_out=.

main: $(DSPLIB_OBJS)
main: CPPFLAGS += -I DSPLib_1_30_00_02/include -I fake-msp430sdk

parse_model: external/onnx.proto3.pb-c.o utils.o
parse_model: LDFLAGS += -lprotobuf-c

clean:
	rm -rf $(PROGS) *.o $(DSPLIB_OBJS) *.pb.* external/*.pb-c.* *.dSYM

.PHONY: all clean
