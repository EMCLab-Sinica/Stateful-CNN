import sys

import cv2
import numpy as np
import onnx
import onnxruntime.backend as backend

np.set_printoptions(linewidth=1000)

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print('usage: {} example.png'.format(sys.argv[0]), file=sys.stderr)
    sys.exit(1)

# model is from https://github.com/onnx/models/tree/master/mnist
model = onnx.load_model("../models/mnist/model_optimized.onnx")

onnx.checker.check_model(model)

print(onnx.helper.printable_graph(model.graph))

rep = backend.prepare(model)
# images from https://github.com/tensorflow/models/tree/master/official/mnist
im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# Check CNTK_103*.ipynb in https://github.com/microsoft/CNTK/tree/master/Tutorials
# for data formats
im = 255 - im
print(im)
# TI's DSPLib requires values to be in [-1, 1)
im = im / 256
im = np.expand_dims(im, axis=0)
im = np.expand_dims(im, axis=0)
outputs = rep.run(im.astype(np.float32))
print(outputs[0])
print(np.argmax(outputs[0]))
