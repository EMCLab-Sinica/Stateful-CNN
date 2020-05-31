import argparse

import numpy as np
import onnx
import onnxruntime.backend as backend

from utils import load_data

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
labels, images = load_data(args.input_file)
for idx, im in enumerate(images):
    outputs = rep.run(im.astype(np.float32))
    predicted = np.argmax(outputs[0])
    if not labels:
        print(outputs[0])
        print(predicted)
    else:
        correct += (1 if predicted == np.argmax(labels[idx]) else 0)

if labels:
    print('Correct rate: {}'.format(correct / len(labels)))
