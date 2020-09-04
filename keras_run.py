import os.path
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_data_cifar10

with open(os.path.join(sys.argv[2], 'models', 'squeeze_net.json')) as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights(os.path.join(sys.argv[2], 'models', 'squeeze_net.h5'))

# Modified from https://stackoverflow.com/a/41712013/3786245
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

limit = 1
# Testing
labels, images = load_data_cifar10(sys.argv[1], start=0, limit=limit)
if limit == 1:
    layer_outs = [func([images]) for func in functors]
    for idx in range(len(layer_outs)):
        layer_out = layer_outs[idx][0]
        shape = tf.shape(layer_out)
        print(model.layers[idx].__class__.__name__.split('.')[-1], end='')
        print(f' layer: {model.layers[idx].name}')
        print(f'Shape: {shape}')
        if tf.shape(shape)[0] == 4:
            N, H, W, C = shape
            assert N == 1
            for c in range(C):
                print(f'Channel {c}')
                for h in range(H):
                    for w in range(W):
                        print('%13.6f' % layer_out[0, h, w, c], end='')
                    print()
                print()
        else:
            print(layer_out)
else:
    correct = 0
    for idx, image in enumerate(images):
        layer_outs = model(image)
        # Tensorflow 2.x uses .numpy instead of .eval for eager execution
        predicted = np.argmax(layer_outs.numpy()[0])
        if predicted == labels[idx]:
            print(f'Correct at idx={idx}')
            correct += 1
    total = len(labels)
    print(f'correct={correct} total={total} rate={correct/total}')
