import os.path
import sys

import tensorflow as tf
import onnx
import keras2onnx

with open(os.path.join(sys.argv[1], 'models', 'squeeze_net.json')) as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights(os.path.join(sys.argv[1], 'models', 'squeeze_net.h5'))

onnx_model = keras2onnx.convert_keras(model, model.name)

onnx.save(onnx_model, 'squeezenet_cifar10.onnx')
