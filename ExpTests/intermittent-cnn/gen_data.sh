DATA_PATH=../../../intermittent-cnn
MODEL=$DATA_PATH/models/mnist/model_optimized.onnx
IMAGE=$DATA_PATH/example3.png

python gen_ops.py
python transform.py --with-progress-embedding $MODEL $IMAGE
