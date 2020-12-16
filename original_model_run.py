import argparse
import functools
import os.path
import pathlib

import numpy as np
import onnx
import onnxruntime.backend as backend
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import load_data_mnist, load_data_cifar10, load_data_google_speech, GOOGLE_SPEECH_SAMPLE_RATE

kws_root = pathlib.Path('./data/ML-KWS-for-MCU')

def onnxruntime_inference_one(model, images):
    rep = backend.prepare(model)
    return rep.run(images.astype(np.float32))

def onnxruntime_get_intermediate_tensor(model, images):
    # FIXME: only the last layer is returned for now.
    # Any way to extract intermediate layers?
    rep = backend.prepare(model)
    output_name = model.graph.output[0].name
    outputs = rep.run(images[0].astype(np.float32))
    yield output_name, outputs

# Modified from https://stackoverflow.com/a/41712013/3786245
def keras_get_intermediate_tensor(model, images):
    for layer in model.layers:
        output = layer.output
        yield output.name, K.function([model.input], [output])(images)

def keras_inference_one(model, images):
    layer_outs = model(images)
    # Tensorflow 2.x uses .numpy instead of .eval for eager execution
    return layer_outs.numpy()[0]

def tensorflow_inference_layer(decoded_wavs, idx):
    with tf.compat.v1.Session() as sess:
        op = sess.graph.get_operations()[idx]
        tensor = sess.graph.get_tensor_by_name(op.outputs[0].name)
        return sess.run(tensor, {
            'decoded_sample_data:0': decoded_wavs[0],
            'decoded_sample_data:1': GOOGLE_SPEECH_SAMPLE_RATE,
        })

def tensorflow_get_intermediate_tensor(graph_def, decoded_wavs):
    for idx, node in enumerate(graph_def.node):
        if node.op in ('Const', 'Identity', 'Placeholder'):
            continue
        tensor_name = node.name
        tensor_values = tensorflow_inference_layer(decoded_wavs, idx)
        yield tensor_name, tensor_values

def tensorflow_inference_one(decoded_wav):
    return tensorflow_inference_layer([decoded_wav], -1)[0]

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
    parser.add_argument('config', choices=['mnist', 'cifar10', 'kws'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    if args.limit == 0:
        args.limit = None

    if args.config == 'mnist':
        # model is from https://github.com/onnx/models/tree/master/mnist
        # https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
        model = onnx.load_model('./data/mnist-8.onnx')
        onnx.checker.check_model(model)

        get_intermediate_tensor = functools.partial(onnxruntime_get_intermediate_tensor, model)
        inference_one = functools.partial(onnxruntime_inference_one, model)
        model_data = load_data_mnist(start=0, limit=args.limit)
    elif args.config == 'cifar10':
        squeezenet_cifar10_path = './data/SqueezeNet_vs_CIFAR10/models'
        with open(os.path.join(squeezenet_cifar10_path, 'squeeze_net.json')) as f:
            model_json = f.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(os.path.join(squeezenet_cifar10_path, 'squeeze_net.h5'))

        get_intermediate_tensor = functools.partial(keras_get_intermediate_tensor, model)
        inference_one = functools.partial(keras_inference_one, model)
        model_data = load_data_cifar10(start=0, limit=args.limit)
    elif args.config == 'kws':
        with open(kws_root / 'Pretrained_models' / 'DNN' / 'DNN_S.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        get_intermediate_tensor = functools.partial(tensorflow_get_intermediate_tensor, graph_def)
        inference_one = tensorflow_inference_one
        model_data = load_data_google_speech(start=0, limit=args.limit, for_onnx=False)

    # Testing
    if args.limit == 1:
        for layer_name, layer_out in get_intermediate_tensor(model_data.images):
            print(f'Layer: {layer_name}')
            print_tensor(layer_out)
    else:
        correct = 0
        for idx, image in enumerate(model_data.images):
            layer_outs = inference_one(image)
            predicted = np.argmax(layer_outs)
            if predicted == model_data.labels[idx]:
                print(f'Correct at idx={idx}')
                correct += 1
        total = len(model_data.labels)
        print(f'correct={correct} total={total} rate={correct/total}')

if __name__ == '__main__':
    main()
