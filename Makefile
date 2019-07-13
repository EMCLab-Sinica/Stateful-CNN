CPPFLAGS = -I . `pkg-config --cflags libprotobuf-c`
DEBUG = 0
CFLAGS = -std=gnu89 -Dinline= -Wall -Wextra -Wstrict-prototypes
ifeq ($(DEBUG),1)
    CFLAGS += -g -O0
else
    CFLAGS += -O3
    CPPFLAGS += -DNDEBUG
endif
PROGS = main parse_model parse_model_nanopb

all: $(PROGS)

external/onnx.proto3.pb-c.c : external/onnx.proto3
	protoc-c $< --c_out=.

# Remove a field not supported by nanopb
onnx-nanopb.proto3: external/onnx.proto3
	sed 's|string dim_param|//string dim_param|' $< > $@

onnx-nanopb.pb.c: onnx-nanopb.proto3
	protoc --plugin=protoc-gen-nanopb=nanopb/generator/protoc-gen-nanopb $^ --nanopb_out=.

parse_model: external/onnx.proto3.pb-c.o utils.o protobuf-c/protobuf-c/protobuf-c.o

parse_model_nanopb: CPPFLAGS+=-I protobuf-c

parse_model_nanopb: onnx-nanopb.pb.o utils.o nanopb/pb_decode.o nanopb/pb_encode.o nanopb/pb_common.o

parse_model_nanopb: CPPFLAGS+=-I nanopb

clean:
	git submodule foreach git clean -dfx
	rm -rf $(PROGS) *.o *.pb.* external/*.pb-c.* onnx-nanopb.proto3 model.bin *.dSYM

.PHONY: all clean
