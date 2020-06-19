import os.path
import sys

import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_data_cifar10

# argv[2] should be path to https://github.com/zshancock/SqueezeNet_vs_CIFAR10
sys.path.append(sys.argv[2])

from squeezenet_architecture import SqueezeNet

model = SqueezeNet()
model.load_weights(os.path.join(sys.argv[2], 'models', 'squeeze_net.h5'))

# Modified from https://stackoverflow.com/a/41712013/3786245
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

# Testing
labels, images = load_data_cifar10(sys.argv[1], limit=1)
layer_outs = [func([images]) for func in functors]
for idx in range(len(layer_outs)):
    layer_out = layer_outs[idx][0]
    shape = tf.shape(layer_out)
    print(f'Layer: {model.layers[idx].name}')
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
