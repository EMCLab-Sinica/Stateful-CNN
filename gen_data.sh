# https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx
MODEL=data/mnist-8.onnx
INPUT_FILE=data/Test-28x28_cntk_text.txt
#INPUT_FILE=data/example3.png

python optimize_model.py $MODEL
python gen_ops.py
# TODO: add --with-progress-embedding back after fixing the performance regression
python transform.py --with-progress-embedding ${MODEL/.onnx/_optimized.onnx} $INPUT_FILE
