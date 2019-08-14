all: parse_model

external/onnx.proto3.pb-c.c : external/onnx.proto3
	protoc-c $< --c_out=.

parse_model: external/onnx.proto3.pb-c.o utils.o
parse_model: LDFLAGS += -lprotobuf-c

clean: clean-common
	rm -rf parse_model *.pb.* external/*.pb-c.*

include ../common.mk

.PHONY: all clean
