# https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx
# NEED_OPTIMIZE: onnx optimizer cannot handle squeezenet_cifar10.onnx, so skip it

MODEL=data/mnist-8.onnx
INPUT_FILE=data/Test-28x28_cntk_text.txt
NEED_OPTIMIZE=1
#MODEL=data/squeezenet_cifar10.onnx
#INPUT_FILE=data/cifar10-test_batch
#NEED_OPTIMIZE=0

if [[ "$NEED_OPTIMIZE" = 1 ]] ; then
    python optimize_model.py $MODEL
    MODEL=${MODEL/.onnx/_optimized.onnx}
fi
python gen_ops.py
python transform.py $MODEL $INPUT_FILE
