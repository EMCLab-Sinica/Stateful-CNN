CPPFLAGS = -I . `pkg-config --cflags libprotobuf-c`
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

parse_model: external/onnx.proto3.pb-c.o utils.o protobuf-c/protobuf-c/protobuf-c.o

clean:
	git submodule foreach git clean -dfx
	rm -rf $(PROGS) *.o *.pb.* external/*.pb-c.* model.bin *.dSYM

.PHONY: all clean
