import os.path
import pathlib
import pickle
import re
from typing import Dict, List, NamedTuple
from urllib.request import urlretrieve

import numpy as np

class ModelData(NamedTuple):
    labels: List[int]
    images: List[np.array]
    input_mapping: Dict[str, str] = {}

def load_data_mnist(start: int, limit: int) -> ModelData:
    # XXX: implement start for MNIST
    images = []
    labels = []

    filename = 'data/MNIST/Test-28x28_cntk_text.txt'

    with open(filename) as f:
        counter = 0
        for line in f:
            mobj = re.match(r'\|labels ([\d ]+) \|features ([\d ]+)', line)
            if mobj is None:
                raise ValueError
            labels.append(np.argmax(list(map(int, mobj.group(1).split(' ')))))
            im = np.reshape(np.array(list(map(int, mobj.group(2).split(' ')))), (28, 28))

            # Check CNTK_103*.ipynb in https://github.com/microsoft/CNTK/tree/master/Tutorials
            # for data formats
            im = im / 256
            im = np.expand_dims(im, axis=0)
            im = np.expand_dims(im, axis=0)
            images.append(im)

            counter += 1
            if limit is not None and counter >= limit:
                break

    return ModelData(labels=labels, images=images)

def load_data_cifar10(start: int, limit: int) -> ModelData:
    filename = 'data/cifar-10-batches-py/test_batch'

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
    return ModelData(labels=labels, images=images)

GOOGLE_SPEECH_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
GOOGLE_SPEECH_SAMPLE_RATE = 16000

def load_data_google_speech(start: int, limit: int, for_onnx=True) -> ModelData:
    import tensorflow as tf
    import torchaudio

    cache_dir = pathlib.Path('~/.cache/torchaudio/speech_commands_v2').expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = torchaudio.datasets.SPEECHCOMMANDS(root=cache_dir, url=GOOGLE_SPEECH_URL, download=True)

    # From https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Pretrained_models/labels.txt
    new_labels = '_silence_ _unknown_ yes no up down left right on off stop go'.split(' ')

    decoded_wavs = []
    labels = []
    # The first few _unknown_ samples are not recognized by Hello Edge's DNN model - use good ones instead
    for idx, data in enumerate(reversed(dataset)):
        if idx < start:
            continue
        waveform, sample_rate, label, _, _ = data
        assert sample_rate == GOOGLE_SPEECH_SAMPLE_RATE
        decoded_wavs.append(np.expand_dims(np.squeeze(waveform), axis=-1))
        labels.append(new_labels.index(label))
        if limit and idx == limit - 1:
            break
    if for_onnx:
        with open(kws_dnn_model(), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        mfccs = []
        with tf.compat.v1.Session() as sess:
            mfcc_tensor = sess.graph.get_tensor_by_name('Mfcc:0')
            for decoded_wav in decoded_wavs:
                mfcc = sess.run(mfcc_tensor, {
                    'decoded_sample_data:0': decoded_wav,
                    'decoded_sample_data:1': GOOGLE_SPEECH_SAMPLE_RATE,
                })
                mfccs.append(np.expand_dims(mfcc, 0))

        input_mapping = {'wav_data:0': 'Mfcc:0'}

        return ModelData(labels=labels, images=mfccs, input_mapping=input_mapping)
    else:
        return ModelData(labels=labels, images=decoded_wavs)

def kws_dnn_model():
    kws_dnn_url = 'https://github.com/ARM-software/ML-KWS-for-MCU/raw/master/Pretrained_models/DNN/DNN_S.pb'
    kws_dnn_local_path = pathlib.Path(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))) / 'KWS-DNN_S.pb'
    if not kws_dnn_local_path.exists():
        urlretrieve(kws_dnn_url, kws_dnn_local_path)
    return kws_dnn_local_path
