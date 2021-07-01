import os.path
import sys

import tensorflow as tf
import onnx
import tf2onnx.convert

with open(os.path.join(sys.argv[1], 'models', 'squeeze_net.json')) as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights(os.path.join(sys.argv[1], 'models', 'squeeze_net.h5'))

onnx_model, _ = tf2onnx.convert.from_keras(model)

onnx.save(onnx_model, 'squeezenet_cifar10.onnx')
