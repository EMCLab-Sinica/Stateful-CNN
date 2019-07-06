CPPFLAGS = -I . `pkg-config --cflags libprotobuf-c`
CFLAGS = -std=gnu89 -Dinline= -g -Wall -Wextra -Wstrict-prototypes
PROGS = main parse_model parse_model_nanopb

all: $(PROGS)

onnx/onnx/onnx.proto3.pb-c.c : onnx/onnx/onnx.proto3
	protoc-c $< --c_out=.

# Remove a field not supported by nanopb
onnx-nanopb.proto3: onnx/onnx/onnx.proto3
	sed 's|string dim_param|//string dim_param|' $< > $@

onnx-nanopb.pb.c: onnx-nanopb.proto3
	protoc --plugin=protoc-gen-nanopb=nanopb/generator/protoc-gen-nanopb $^ --nanopb_out=.

parse_model: onnx/onnx/onnx.proto3.pb-c.o utils.o protobuf-c/protobuf-c/protobuf-c.o

parse_model_nanopb: CPPFLAGS+=-I protobuf-c

parse_model_nanopb: onnx-nanopb.pb.o utils.o nanopb/pb_decode.o nanopb/pb_encode.o nanopb/pb_common.o

parse_model_nanopb: CPPFLAGS+=-I nanopb

clean:
	git submodule foreach git clean -dfx
	rm -rf $(PROGS) *.o *.pb.* onnx-nanopb.proto3 model.bin *.dSYM

.PHONY: all clean
