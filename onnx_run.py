import argparse

import numpy as np
import onnx
import onnxruntime.backend as backend

from utils import load_data, load_data_cifar10

np.set_printoptions(linewidth=1000)

parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='Input file. Can be example.png or mnist.txt.')
parser.add_argument('onnx_model')
args = parser.parse_args()

# model is from https://github.com/onnx/models/tree/master/mnist
# https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
model = onnx.load_model(args.onnx_model)

# model.graph.initializer[1].float_data[1] = 3

onnx.checker.check_model(model)

print(onnx.helper.printable_graph(model.graph))

rep = backend.prepare(model)
correct = 0

if 'mnist' in args.onnx_model:
    labels, images = load_data(args.input_file)
elif 'cifar10' in args.onnx_model:
    labels, images = load_data_cifar10(args.input_file)
else:
    raise NotImplementedError

for idx, im in enumerate(images):
    outputs = rep.run(im.astype(np.float32))
    predicted = np.argmax(outputs[0])
    if not labels:
        print(outputs[0])
        print(predicted)
    else:
        expected = labels[idx]
        if isinstance(expected, list):
            expected = np.argmax(expected)
        correct += (1 if predicted == expected else 0)

if labels:
    print('Correct rate: {}'.format(correct / len(labels)))
