import pickle
import re

import numpy as np

def load_data(filename, start: int, limit: int):
    # XXX: implement start for MNIST
    images = []
    labels = []

    def append_img(im):
        # Check CNTK_103*.ipynb in https://github.com/microsoft/CNTK/tree/master/Tutorials
        # for data formats
        im = im / 256
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)
        images.append(im)

    if filename.endswith('.png'):
        # images from https://github.com/tensorflow/models/tree/master/official/mnist
        import cv2
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im = 255 - im
        print(im)
        append_img(im)
    else:
        def parse_line(line):
            mobj = re.match(r'\|labels ([\d ]+) \|features ([\d ]+)', line)
            labels.append(np.argmax(list(map(int, mobj.group(1).split(' ')))))
            im = np.reshape(np.array(list(map(int, mobj.group(2).split(' ')))), (28, 28))
            append_img(im)

        with open(filename) as f:
            counter = 0
            for line in f:
                parse_line(line)
                counter += 1
                if limit is not None and counter >= limit:
                    break

    return labels, images

def load_data_cifar10(filename: str, start: int, limit: int):
    if filename.endswith('.png'):
        import cv2
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        im = np.expand_dims(im, 0)
        im = im / 256
        return [None], [im]
    else:
        with open(filename, 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
        if limit is None:
            limit = len(test_data[b'labels'])
        labels = test_data[b'labels'][start:start+limit]
        images = []
        H = 32
        W = 32
        for im_data in test_data[b'data'][start:start+limit]:
            # ONNX models transformed from Keras ones uses NHWC as input
            im = np.array(im_data)
            im = np.reshape(im, (3, H, W))
            im = im / 256
            im = np.moveaxis(im, 0, -1)
            im = np.expand_dims(im, 0)
            images.append(im)
        return labels, images
