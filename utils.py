import enum
import functools
import itertools
import logging
import os.path
import pathlib
import pickle
import re
import struct
import sys
import tarfile
import zipfile
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional
from urllib.request import urlretrieve

import filelock
import numpy as np
import onnx
import onnxoptimizer
import onnxruntime
import onnxruntime.backend as backend
import platformdirs

logger = logging.getLogger(__name__)

OPS_WITH_MERGE = ['Conv', 'Gemm']

TOPDIR = pathlib.Path(__file__).absolute().parent

audio_ops = ['DecodeWav', 'AudioSpectrogram', 'Mfcc']

class DataLayout(enum.Enum):
    NEUTRAL = 0
    NCW = 1
    NWC = 2
    NCHW = 3
    NHWC = 4

class ModelData(NamedTuple):
    labels: List[int]
    images: np.array
    data_layout: DataLayout

def extract_archive(archive_path: pathlib.Path, subdir: str):
    archive_dir = archive_path.with_name(subdir)
    if not archive_dir.exists():
        if '.tar' in str(archive_path):
            with tarfile.open(archive_path) as tar:
                members = [member for member in tar.getmembers() if member.name.startswith(subdir)]
                tar.extractall(archive_path.parent, members=members)
        elif str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(archive_path) as zip_f:
                members = [member for member in zip_f.namelist() if member.startswith(subdir)]
                zip_f.extractall(archive_path.parent, members=members)
    return archive_dir

def load_data_mnist(start: int, limit: int) -> ModelData:
    images = []
    labels = []

    filename = download_file('https://github.com/microsoft/NativeKeras/raw/master/Datasets/cntk_mnist/Test-28x28_cntk_text.txt',
                             'MNIST-Test-28x28_cntk_text.txt')

    with open(filename) as f:
        counter = 0
        for line in f:
            if start > 0:
                start -= 1
                continue
            mobj = re.match(r'\|labels ([\d ]+) \|features ([\d ]+)', line)
            if mobj is None:
                raise ValueError
            labels.append(np.argmax(list(map(int, mobj.group(1).split(' ')))))
            im = np.reshape(np.array(list(map(int, mobj.group(2).split(' ')))), (28, 28))

            # Check CNTK_103*.ipynb in https://github.com/microsoft/CNTK/tree/master/Tutorials
            # for data formats
            im = im / 256
            im = np.expand_dims(im, axis=0)
            images.append(im)

            counter += 1
            if limit is not None and counter >= limit:
                break

    return ModelData(labels=labels, images=np.array(images, dtype=np.float32), data_layout=DataLayout.NCHW)

def load_data_cifar10(start: int, limit: int) -> ModelData:
    archive_dir = download_file('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                                'cifar-10-python.tar.gz', functools.partial(extract_archive, subdir='cifar-10-batches-py'))

    with open(archive_dir / 'test_batch', 'rb') as f:
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
        images.append(im)
    # XXX: the actual data layout is NCHW, while the first node is Transpose - take the resultant
    return ModelData(labels=labels, images=np.array(images, dtype=np.float32), data_layout=DataLayout.NHWC)

GOOGLE_SPEECH_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
GOOGLE_SPEECH_SAMPLE_RATE = 16000

def load_data_google_speech(start: int, limit: int) -> ModelData:
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
            mfccs.append(mfcc[0])


    return ModelData(labels=labels, images=np.array(mfccs, dtype=np.float32), data_layout=DataLayout.NEUTRAL)

def kws_dnn_model():
    return download_file('https://github.com/ARM-software/ML-KWS-for-MCU/raw/master/Pretrained_models/DNN/DNN_S.pb', 'KWS-DNN_S.pb')

def load_har(start: int, limit: int):
    try:
        orig_sys_path = sys.path.copy()
        sys.path.append(str(TOPDIR / 'models' / 'deep-learning-HAR' / 'utils'))
        from utilities import read_data, standardize

        archive_dir = download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
                                    filename='UCI HAR Dataset.zip', post_processor=functools.partial(extract_archive, subdir='UCI HAR Dataset'))
        X_test, labels_test, _ = read_data(archive_dir, split='test')
        _, X_test = standardize(np.random.rand(*X_test.shape), X_test)
        return ModelData(labels=labels_test[:limit]-1, images=X_test[:limit, :, :].astype(np.float32), data_layout=DataLayout.NCW)
    finally:
        sys.path = orig_sys_path

def download_file(url: str, filename: str, post_processor: Optional[Callable] = None) -> os.PathLike:
    xdg_cache_home = platformdirs.user_cache_path()

    lock_path = xdg_cache_home / f'{filename}.lock'

    # Inspired by https://stackoverflow.com/a/53643011
    class ProgressHandler:
        def __init__(self):
            self.last_reported = 0

        def __call__(self, block_num, block_size, total_size):
            progress = int(block_num * block_size / total_size * 100)
            if progress > self.last_reported + 5:
                logger.info('Downloaded: %d%%', progress)
                self.last_reported = progress

    with filelock.FileLock(lock_path):
        local_path = xdg_cache_home / filename
        if not local_path.exists():
            urlretrieve(url, local_path, ProgressHandler())

        ret = local_path
        if post_processor:
            ret = post_processor(local_path)

    return ret

def extract_data(params):
    if params.data_type == onnx.TensorProto.FLOAT and params.float_data:
        ret = params.float_data
    elif params.data_type == onnx.TensorProto.INT64 and params.int64_data:
        ret = params.int64_data

    else:
        format_char = {
            onnx.TensorProto.FLOAT: 'f',
            onnx.TensorProto.INT64: 'q',
        }[params.data_type]
        ret = list(map(lambda t: t[0], struct.iter_unpack(format_char, params.raw_data)))

    # Undocumented (?) - empty dims means scalar
    # https://github.com/onnx/onnx/issues/1131
    if not len(params.dims):
        assert len(ret) == 1
        return ret[0]

    return np.reshape(ret, params.dims)

def find_initializer(onnx_model: onnx.ModelProto, name: str) -> Optional[onnx.TensorProto]:
    for initializer in onnx_model.graph.initializer:
        if initializer.name == name:
            return initializer

def find_tensor_value_info(onnx_model: onnx.ModelProto, name: str) -> onnx.ValueInfoProto:
    if name.endswith('_before_merge'):
        name = name[:-len('_before_merge')]
    g = onnx_model.graph
    for value_info in itertools.chain(g.value_info, g.input, g.output):
        if value_info.name == name:
            return value_info
    raise ValueError(f'No value_info found for {name}')

def find_node_by_output(nodes: List[onnx.NodeProto], output_name: str) -> onnx.NodeProto:
    for node in nodes:
        for output in node.output:
            if output == output_name:
                return node

def numpy_type_to_onnx_elem_type(numpy_type):
    if numpy_type == np.float32:
        return onnx.TensorProto.FLOAT
    if numpy_type == np.int64:
        return onnx.TensorProto.INT64
    if numpy_type == np.bool_:
        return onnx.TensorProto.BOOL
    raise Exception(f'Unsupported type {numpy_type}')

def get_model_ops(onnx_model):
    # Retrieving information for operators. Inspired by the script for generating
    # https://github.com/onnx/onnx/blob/v1.10.2/docs/Operators.md [1,2]
    # [1] https://github.com/onnx/onnx/blob/v1.10.2/onnx/defs/gen_doc.py
    # [2] https://github.com/onnx/onnx/blob/v1.10.2/onnx/onnx_cpp2py_export/defs.pyi
    ops = set()
    for schema in onnx.defs.get_all_schemas():
        ops.add(schema.name)

    ops = ops.intersection(node.op_type for node in onnx_model.graph.node)
    for op in OPS_WITH_MERGE:
        if op in ops:
            ops.add(op + 'Merge')
    ops = sorted(ops)

    return ops

def load_model(config, for_deployment):
    # https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
    onnx_model = onnx.load_model(TOPDIR / config['onnx_model'])

    # onnxoptimizer requires known dimensions, so set the batch size=1.
    # The batch size will be changed to a variable after dynamic_shape_inference, anyway.
    # https://github.com/onnx/optimizer/blob/v0.2.6/onnxoptimizer/passes/fuse_matmul_add_bias_into_gemm.h#L60
    change_batch_size(onnx_model)

    # https://zhuanlan.zhihu.com/p/41255090
    onnx_model = onnxoptimizer.optimize(onnx_model, [
        'eliminate_nop_dropout',
        'extract_constant_to_initializer',
        'fuse_add_bias_into_conv',
        'fuse_matmul_add_bias_into_gemm',
    ])

    dynamic_shape_inference(onnx_model, config['sample_size'])

    if for_deployment:
        add_merge_nodes(onnx_model)

    return onnx_model

def add_merge_nodes(model):
    # Split Conv/Gemm into Conv/Gemm and ConvMerge/GemmMerge (for merging OFMs from channel tiling)
    new_nodes = []
    for idx, n in enumerate(model.graph.node):
        if n.op_type in audio_ops:
            logger.warning('skipping audio operator %s', n.op_type)
            continue
        new_nodes.append(n)
        if n.op_type in OPS_WITH_MERGE:
            output_name = n.output[0]
            new_node = onnx.NodeProto()
            new_node.name = (n.name or n.op_type) + ':merge'
            new_node.op_type = n.op_type + 'Merge'
            new_node.input[:] = n.output[:] = [output_name + '_before_merge']
            new_node.output[:] = [output_name]
            new_nodes.append(new_node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

def onnxruntime_prepare_model(model):
    return backend.prepare(onnxruntime.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ))

def onnxruntime_get_intermediate_tensor(model, image):
    # Creating a new model with all nodes as outputs
    # https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-979901463
    tmp_model = onnx.ModelProto()
    tmp_model.CopyFrom(model)

    orig_outputs = list(tmp_model.graph.output)
    orig_output_names = [node.name for node in orig_outputs]
    del tmp_model.graph.output[:]
    for node in tmp_model.graph.node:
        for output in node.output:
            if output not in orig_output_names:
                tmp_model.graph.output.append(onnx.ValueInfoProto(name=output))
    tmp_model.graph.output.extend(orig_outputs)

    rep = onnxruntime_prepare_model(tmp_model)
    outputs = rep.run(image)
    for idx, output in enumerate(outputs):
        output_name = tmp_model.graph.output[idx].name
        node = find_node_by_output(tmp_model.graph.node, output_name)
        yield output_name, node.op_type, output

def change_batch_size(onnx_model: onnx.ModelProto):
    g = onnx_model.graph
    initializer_names = set([initializer.name for initializer in g.initializer])
    constant_names = set([node.output[0] for node in g.node if node.op_type == 'Constant'])
    for value_info in itertools.chain(g.value_info, g.input, g.output):
        if value_info.name in initializer_names or value_info.name in constant_names:
            continue
        shape = value_info.type.tensor_type.shape
        if shape.dim and shape.dim[0].dim_param:
            shape.dim[0].dim_value = 1

    # make sure above steps did not break the model
    onnx.shape_inference.infer_shapes(onnx_model)

def dynamic_shape_inference(onnx_model: onnx.ModelProto, sample_size: Iterable[int]) -> None:
    for node in itertools.chain(onnx_model.graph.input, onnx_model.graph.output):
        if not node.type.tensor_type.shape.dim:
            continue
        node.type.tensor_type.shape.dim[0].dim_param = 'N'

    del onnx_model.graph.value_info[:]

    BATCH_SIZE = 2  # Any number larger than 1 is OK. Here I pick the smallest one for performance considerations

    dummy_images = np.expand_dims(np.zeros(sample_size, dtype=np.float32), axis=0)
    shapes = {
        layer_name: np.shape(layer_out)
        for layer_name, _, layer_out in onnxruntime_get_intermediate_tensor(onnx_model, dummy_images)
    }
    dummy_images = np.concatenate([
        np.expand_dims(np.random.rand(*sample_size).astype(np.float32), axis=0) for _ in range(BATCH_SIZE)
    ], axis=0)

    value_infos = []
    for layer_name, layer_type, layer_out in onnxruntime_get_intermediate_tensor(onnx_model, dummy_images):
        larger_shape = np.shape(layer_out)
        smaller_shape = shapes[layer_name]
        if larger_shape[1:] != smaller_shape[1:]:
            logger.info('Skipping OFM %s for %s node with mismatched shapes: %r, %r', layer_name, layer_type, larger_shape, smaller_shape)
            continue

        new_shape = list(larger_shape)
        if larger_shape:
            if larger_shape[0] == smaller_shape[0] * BATCH_SIZE:
                new_shape[0] = 'N'
            elif larger_shape[0] == smaller_shape[0]:
                pass
            else:
                logger.info('Skipping OFM %s for %s node with mismatched batch sizes: %d, %d', layer_name, layer_type, larger_shape[0], smaller_shape[0])
                continue

        elem_type = numpy_type_to_onnx_elem_type(layer_out.dtype)
        value_info = onnx.helper.make_tensor_value_info(layer_name, elem_type, new_shape)
        value_infos.append(value_info)

    onnx_model.graph.value_info.extend(value_infos)

def remap_inputs(model: onnx.ModelProto, input_mapping: Dict[str, str]):
    new_inputs = list(input_mapping.values())
    for new_input in new_inputs:
        model.graph.input.append(onnx.ValueInfoProto(name=new_input))
    for node in model.graph.node:
        node.input[:] = [input_mapping.get(inp, inp) for inp in node.input]
        node.output[:] = [
            output + '_unused' if output in new_inputs else output
            for output in node.output
        ]
    for idx, inp in enumerate(model.graph.input):
        if inp.name in input_mapping.keys():
            del model.graph.input[idx]

    return onnxoptimizer.optimize(model, ['eliminate_deadend'])

def import_model_output_pb2():
    try:
        orig_sys_path = sys.path.copy()
        sys.path.append(str(pathlib.Path(__file__).resolve().parent / 'build'))
        import model_output_pb2
        return model_output_pb2
    finally:
        sys.path = orig_sys_path
