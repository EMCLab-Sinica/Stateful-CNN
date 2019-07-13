CPPFLAGS = -I .
DEBUG = 0
CFLAGS = -std=gnu89 -Dinline= -Wall -Wextra -Wstrict-prototypes
ifeq ($(DEBUG),1)
    CFLAGS += -g -O0
else
    CFLAGS += -O3
    CPPFLAGS += -DNDEBUG
endif
PROGS = main parse_model

all: $(PROGS)

external/onnx.proto3.pb-c.c : external/onnx.proto3
	protoc-c $< --c_out=.

parse_model: external/onnx.proto3.pb-c.o utils.o
parse_model: CPPFLAGS += `pkg-config --cflags libprotobuf-c`
parse_model: LDFLAGS += `pkg-config --libs libprotobuf-c`

clean:
	git submodule foreach git clean -dfx
	rm -rf $(PROGS) *.o *.pb.* external/*.pb-c.* model.bin *.dSYM

.PHONY: all clean
