import os.path
import sys

# argv[1] should be path to https://github.com/zshancock/SqueezeNet_vs_CIFAR10
sys.path.append(sys.argv[1])

from squeezenet_architecture import SqueezeNet
import onnx
import keras2onnx

model = SqueezeNet()
model.load_weights(os.path.join(sys.argv[1], 'models', 'squeeze_net.h5'))

onnx_model = keras2onnx.convert_keras(model, model.name)

onnx.save(onnx_model, 'squeezenet_cifar10.onnx')
