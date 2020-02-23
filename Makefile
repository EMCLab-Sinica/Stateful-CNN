MODEL := $(DATA_PATH)/models/mnist/model_optimized.onnx
IMAGE := $(DATA_PATH)/example3.png
INPUT_DATA_FILES = data.c data.h ops.c ops.py ops.h inputs.bin model.bin parameters.bin
DATA_FILES = $(INPUT_DATA_FILES) nvm.bin

all: out nvm.bin

ops.py ops.c ops.h: gen_ops.py
	python $<

data.c data.h: bin2c.py model.bin
	python bin2c.py

inputs.bin model.bin parameters.bin: transform.py ops.py
	python transform.py $(MODEL) $(IMAGE)

nvm.bin:
	dd if=/dev/zero of=nvm.bin bs=1024 count=256 # 256KB

clean:
	rm -rvf __pycache__ $(DATA_FILES)

out: $(INPUT_DATA_FILES)
	mkdir -p out && cd out && cmake .. && make

.PHONY: all out clean
