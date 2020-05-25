import re

import cv2
import numpy as np

def load_data(filename, limit=None):
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
