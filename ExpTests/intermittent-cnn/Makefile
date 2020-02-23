MODEL := $(DATA_PATH)/models/mnist/model_optimized.onnx
IMAGE := $(DATA_PATH)/example3.png
INPUT_DATA_FILES = data.c data.h ops.c ops.py ops.h
DATA_FILES = $(INPUT_DATA_FILES) nvm.bin

all: out nvm.bin

ops.py ops.c ops.h: gen_ops.py
	python $<

nvm.bin data.c data.h: transform.py ops.py
	python transform.py $(MODEL) $(IMAGE)

clean:
	rm -rvf __pycache__ $(DATA_FILES)

out: $(INPUT_DATA_FILES)
	mkdir -p out && cd out && cmake .. && make

.PHONY: all out clean
