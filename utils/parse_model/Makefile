CPPFLAGS = -I .
CFLAGS = -g -Wall -Wextra

all: parse_model

external/onnx.proto3.pb-c.c : external/onnx.proto3
	protoc-c $< --c_out=.

parse_model: external/onnx.proto3.pb-c.o utils.o
parse_model: LDFLAGS += -lprotobuf-c

clean:
	rm -rf parse_model *.o *.pb.* external/*.pb-c.*

.PHONY: all clean
