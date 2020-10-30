import argparse
import functools
import os.path
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_data_cifar10, load_data_google_speech

limit = 1

# Modified from https://stackoverflow.com/a/41712013/3786245
def keras_get_intermediate_tensor(model, images):
    for layer in model.layers:
        output = layer.output
        yield output.name, K.function([model.input], [output])(images)

def tensorflow_get_intermediate_tensor(graph_def, wav_data):
    for idx, node in enumerate(graph_def.node):
        if node.op in ('Const', 'Identity', 'Placeholder'):
            continue
        tensor_name = tensor_values = None
        with tf.compat.v1.Session() as sess:
            op = sess.graph.get_operations()[idx]
            tensor_name = op.outputs[0].name
            tensor = sess.graph.get_tensor_by_name(tensor_name)
            tensor_values = sess.run(tensor, {'wav_data:0': wav_data[0][0]})
        yield tensor_name, tensor_values

def print_float(val):
    print('%13.6f' % val, end='')

def print_tensor(tensor):
    shape = tf.shape(tensor)
    print(f'Original shape: {shape}')
    dimensions = tf.shape(shape)[0]
    if dimensions and shape[0] == 1:
        tensor = tensor[0]
        dimensions -= 1
        shape = shape[1:]
    if dimensions and shape[-1] == 1:
        tensor = np.squeeze(tensor, axis=-1)
        dimensions -= 1
        shape = shape[:-1]
    print(f'New shape: {shape}')
    if dimensions == 4:
        N, H, W, C = shape
        assert N == 1
        for c in range(C):
            print(f'Channel {c}')
            for h in range(H):
                for w in range(W):
                    print_float(tensor[0, h, w, c])
                print()
            print()
    elif dimensions == 2:
        H, W = shape
        for h in range(H):
            for w in range(W):
                print_float(tensor[h, w])
            print()
    elif dimensions == 1:
        if shape[0] >= 1024:
            print(f'Skipping very long vector with length {shape[0]}')
            return
        for idx in range(shape[0]):
            print_float(tensor[idx])
            if idx % 16 == 15:
                print()
        print()
    else:
        print(f'Skip: unsupported {dimensions}-dimensional array')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=['cifar10', 'kws'])
    args = parser.parse_args()

    if args.config == 'cifar10':
        squeezenet_cifar10_path = './data/SqueezeNet_vs_CIFAR10/models'
        with open(os.path.join(squeezenet_cifar10_path, 'squeeze_net.json')) as f:
            model_json = f.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(os.path.join(squeezenet_cifar10_path, 'squeeze_net.h5'))

        get_intermediate_tensor = functools.partial(keras_get_intermediate_tensor, model)
        model_data = load_data_cifar10(start=0, limit=limit)
    elif args.config == 'kws':
        kws_root = pathlib.Path('./data/ML-KWS-for-MCU')
        with open(kws_root / 'Pretrained_models' / 'DNN' / 'DNN_S.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        get_intermediate_tensor = functools.partial(tensorflow_get_intermediate_tensor, graph_def)
        model_data = load_data_google_speech(start=0, limit=limit, for_onnx=False)

    # Testing
    if limit == 1:
        for layer_name, layer_out in get_intermediate_tensor([model_data.images]):
            print(f'Layer: {layer_name}')
            print_tensor(layer_out)
    else:
        correct = 0
        for idx, image in enumerate(model_data.images):
            layer_outs = model(image)
            # Tensorflow 2.x uses .numpy instead of .eval for eager execution
            predicted = np.argmax(layer_outs.numpy()[0])
            if predicted == model_data.labels[idx]:
                print(f'Correct at idx={idx}')
                correct += 1
        total = len(model_data.labels)
        print(f'correct={correct} total={total} rate={correct/total}')

if __name__ == '__main__':
    main()
