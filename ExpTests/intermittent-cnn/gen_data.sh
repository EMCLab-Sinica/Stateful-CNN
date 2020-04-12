DATA_PATH=../../../intermittent-cnn
MODEL=$DATA_PATH/models/mnist/model_optimized.onnx
INPUT_FILE=$DATA_PATH/MNIST/Test-28x28_cntk_text.txt
#INPUT_FILE=$DATA_PATH/example3.png

python gen_ops.py
# TODO: add --with-progress-embedding back after fixing the performance regression
python transform.py $MODEL $INPUT_FILE
