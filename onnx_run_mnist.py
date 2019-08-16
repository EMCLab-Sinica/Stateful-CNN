import re
import sys

import cv2
import numpy as np
import onnx
import onnxruntime.backend as backend

np.set_printoptions(linewidth=1000)

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print('usage: {} [example.png|mnist.txt]'.format(sys.argv[0]), file=sys.stderr)
    sys.exit(1)

# model is from https://github.com/onnx/models/tree/master/mnist
model = onnx.load_model("../models/mnist/model_optimized.onnx")

onnx.checker.check_model(model)

print(onnx.helper.printable_graph(model.graph))

rep = backend.prepare(model)
images = []
labels = []
correct = 0
if filename.endswith('.png'):
    # images from https://github.com/tensorflow/models/tree/master/official/mnist
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # Check CNTK_103*.ipynb in https://github.com/microsoft/CNTK/tree/master/Tutorials
    # for data formats
    im = 255 - im
    print(im)
    im = np.expand_dims(im, axis=0)
    im = np.expand_dims(im, axis=0)
    images.append(im)
else:
    def parse_line(line):
        mobj = re.match(r'\|labels ([\d ]+) \|features ([\d ]+)', line)
        labels.append(np.array(list(map(int, mobj.group(1).split(' ')))))
        im = np.reshape(np.array(list(map(int, mobj.group(2).split(' ')))), (28, 28))
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)
        images.append(im)

    with open(filename) as f:
        for line in f:
            parse_line(line)

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
