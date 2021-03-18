#!/usr/bin/python
import argparse
import ctypes
import dataclasses
import io
import itertools
import logging
import math
import os.path
import pprint
import struct
import textwrap
import warnings
from typing import List

import onnx
import onnx.helper
import numpy as np

from utils import (
    load_data_mnist,
    load_data_cifar10,
    load_data_google_speech,
)

logging.basicConfig()
logger = logging.getLogger(__name__)

"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(g.input)-1: input nodes
    len(g.input)~ : other (hidden) nodes
"""

class Constants:
    SLOT_PARAMETERS = 0xf0
    SLOT_TEST_SET = 0xff
    SLOT_CONSTANTS_MIN = SLOT_PARAMETERS
    SLOT_INTERMEDIATE_VALUES = 0b01
    NODE_NAME_LEN = 24
    EXTRA_INFO_LEN = 3  # for memory alignment
    TURNING_POINTS_LEN = 8
    MODEL_NODES_LEN = 0
    INPUTS_DATA_LEN = 0
    NUM_INPUTS = 3
    N_INPUT = 0
    # Match the size of external FRAM
    NVM_SIZE = 512 * 1024
    N_SAMPLES = 20
    # to make the code clearer; used in Conv
    TEMP_FILTER_WIDTH = 1
    LEA_BUFFER_SIZE = 0
    ARM_PSTATE_LEN = 8704
    USE_ARM_CMSIS = 0
    CONFIG = None

    DEFAULT_TILE_C = 4
    DEFAULT_TILE_H = 8
    CUR_BATCH_SIZE = 1
    STATEFUL = 0
    HAWAII = 0
    JAPARI = 0
    INTERMITTENT = 0
    INDIRECT_RECOVERY = 0
    METHOD = "Baseline"
    FIRST_SAMPLE_OUTPUTS = []

# https://github.com/onnx/onnx/blob/master/docs/Operators.md
# [expected_inputs_len, inplace_update]
ops = {
    'Add': [2, 0],
    # Concat actually accepts 1~infinity inputs. Use 2 to fit SqueezeNet
    'Concat': [2, 0],
    'Conv': [3, 0],
    'ConvMerge': [1, 0],
    'Dropout': [1, 1],
    'Gemm': [3, 0],
    # two inputs as GemmMerge also adds up the bias
    'GemmMerge': [2, 0],
    'GlobalAveragePool': [1, 0],
    'MaxPool': [1, 0],
    'Relu': [1, 0],
    'Reshape': [2, 1],
    'Softmax': [1, 1],
    'Squeeze': [1, 1],
    # XXX: Transpose does nothing as we happens to need NHWC
    'Transpose': [1, 1],
}

audio_ops = ['DecodeWav', 'AudioSpectrogram', 'Mfcc']

other_flags = [
    'AUTO_PAD_VALID',
    'NHWC2NCHW',
    'TRANSPOSED',
    # Tiles in different channels are actually in different slots
    'SEPARATE_TILING',
]

def op_flag(flag):
    return 2 ** other_flags.index(flag)

def _Q15(num):
    """Transform a floating point number to TI's fixed point _q15 format"""

    # See DSPLib_1_30_00_02/include/DSPLib_support.h

    lower = -1
    upper = 32767.0 / 32768.0

    if num < lower or num >= upper:
        if num != 1.0:
            warnings.warn(
                'Number {} goes beyond the range of _q15 ({}, {})'.format(
                    num, lower, upper))
        num = max(min(num, upper), lower)

    return int(num * 2 ** 15)

# https://stackoverflow.com/a/11481471/3786245
class ConvNodeFlags(ctypes.Structure):
    _fields_ = [
        ("input_tile_c", ctypes.c_uint8, 8),
        ("output_tile_c", ctypes.c_uint8, 8),
    ]

class GemmNodeFlags(ctypes.Structure):
    _fields_ = [
        ("tile_channel", ctypes.c_uint16, 16),
        ("tile_width", ctypes.c_uint16, 16),
    ]

class ExtraNodeFlags(ctypes.Union):
    _fields_ = [
        ("conv", ConvNodeFlags),
        ("gemm", GemmNodeFlags),
    ]

class NodeFlags_bits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("generic", ctypes.c_uint8, 8),
        ("kernel_size", ctypes.c_uint8, 4),
        ("stride", ctypes.c_uint8, 4),
        ("extra", ExtraNodeFlags),
    ]

class NodeFlags(ctypes.Union):
    _fields_ = [
        ("b", NodeFlags_bits),
        ("as_bytes", ctypes.c_uint64),
    ]

    def __repr__(self):
        ret = '<NodeFlags'
        for field in NodeFlags_bits._fields_:
            key = field[0]
            ret += f' {key}={getattr(self.b, key)}'
        ret += '>'
        return ret

class ONNXNodeWrapper:
    def __init__(self, orig_node: onnx.NodeProto):
        self.orig_node = orig_node
        self.flags = NodeFlags()

    def __getattr__(self, name):
        return getattr(self.orig_node, name)


def get_prev_node(n):
    return nodes[names[n.input[0]] - Constants.N_INPUT]


# intermediate_values_size should < 65536, or TI's compiler gets confused
configs = {
    'mnist': {
        # https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx
        'onnx_model': 'data/mnist-8.onnx',
        'scale': 8,
        'input_scale': 8,
        'num_slots': 2,
        'intermediate_values_size': 26000,
        'data_loader': load_data_mnist,
        'n_all_samples': 10000,
        # multiply by 2 for Q15
        'sample_size': 2 * 28 * 28,
        'op_filters': 4,
        'first_sample_outputs': [ -1.247997, 0.624493, 8.609308, 9.392411, -13.685033, -6.018567, -23.386677, 28.214134, -6.762523, 3.924627 ],
        'fp32_accuracy': 0.9889,
    },
    'cifar10': {
        'onnx_model': 'data/squeezenet_cifar10.onnx',
        'scale': 8,
        'input_scale': 8,
        'num_slots': 3,
        'intermediate_values_size': 65000,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'sample_size': 2 * 32 * 32 * 3,
        'op_filters': 4,
        'first_sample_outputs': [ 4.895500, 4.331344, 4.631835, 11.602396, 4.454658, 10.819544, 5.423588, 6.451203, 5.806091, 5.272837 ],
        'fp32_accuracy': 0.7704,
    },
    'kws': {
        'onnx_model': 'data/KWS-DNN_S.onnx',
        'scale': 8,
        'input_scale': 120,
        'num_slots': 2,
        'intermediate_values_size': 20000,
        'data_loader': load_data_google_speech,
        'n_all_samples': 4890,
        'sample_size': 2 * 25 * 10,  # MFCC gives 25x10 tensors
        'op_filters': 4,
        'first_sample_outputs': [ -29.228327, 5.429047, 22.146973, 3.142066, -10.448060, -9.513299, 15.832925, -4.655487, -14.588447, -1.577156, -5.864228, -6.609077 ],
        # Much lower than reported on the paper due to mismatched window_size_ms/window_stride_ms (?)
        # See: https://github.com/ARM-software/ML-KWS-for-MCU/issues/44
        'fp32_accuracy': 0.6323,
    },
}

lea_buffer_size = {
    # (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)
    'msp430': 1884,
    # determined by trial and error
    'msp432': 18000,
}

parser = argparse.ArgumentParser()
parser.add_argument('config', choices=configs.keys())
parser.add_argument('--all-samples', action='store_true')
parser.add_argument('--write-images', action='store_true')
parser.add_argument('--batch-size', type=int, default=Constants.DEFAULT_TILE_C)
parser.add_argument('--target', choices=('msp430', 'msp432'), required=True)
parser.add_argument('--debug', action='store_true')
intermittent_methodology = parser.add_mutually_exclusive_group(required=True)
intermittent_methodology.add_argument('--baseline', action='store_true')
intermittent_methodology.add_argument('--hawaii', action='store_true')
intermittent_methodology.add_argument('--japari', action='store_true')
intermittent_methodology.add_argument('--stateful', action='store_true')
args = parser.parse_args()
if args.debug:
    logger.setLevel(logging.DEBUG)
config = configs[args.config]
Constants.CONFIG = args.config
Constants.FIRST_SAMPLE_OUTPUTS = config['first_sample_outputs']
if args.all_samples:
    Constants.N_SAMPLES = config['n_all_samples']
    Constants.NVM_SIZE += config['n_all_samples'] * config['sample_size']
model_data = config['data_loader'](start=0, limit=Constants.N_SAMPLES)

Constants.CUR_BATCH_SIZE = args.batch_size
if args.stateful:
    Constants.STATEFUL = 1
    Constants.METHOD = "STATEFUL"
if args.hawaii:
    Constants.HAWAII = 1
    Constants.METHOD = "HAWAII"
if args.japari:
    Constants.JAPARI = 1
    Constants.METHOD = "JAPARI"
    config['intermediate_values_size'] *= 2
Constants.INTERMITTENT = Constants.STATEFUL | Constants.HAWAII | Constants.JAPARI
Constants.INDIRECT_RECOVERY = Constants.STATEFUL | Constants.JAPARI
if args.target == 'msp432':
    Constants.USE_ARM_CMSIS = 1
Constants.LEA_BUFFER_SIZE = lea_buffer_size[args.target]

onnx_opt_model_name = config['onnx_model'].replace('.onnx', '-opt.onnx')
if os.path.exists(onnx_opt_model_name):
    onnx_model = onnx.load(onnx_opt_model_name)
else:
    onnx_model = onnx.load(config['onnx_model'])
    try:
        import onnx.optimizer
        # https://zhuanlan.zhihu.com/p/41255090
        onnx_model = onnx.optimizer.optimize(onnx_model, [
            'fuse_add_bias_into_conv',
            'fuse_matmul_add_bias_into_gemm',
        ])
    except IndexError:
        # Somehow the optimizer cannot handle models transformed from keras2onnx
        pass
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save_model(onnx_model, onnx_opt_model_name)
g = onnx_model.graph
names = {}

def get_attr(node, attr_name):
    for attr in node.attribute:
        if attr.name != attr_name:
            continue
        return onnx.helper.get_attribute_value(attr)

    # Not found
    return None

# Remove Squeeze and Reshape nodes with constants as the input
replaced_nodes_map = {}

def find_initializer(name):
    for initializer in g.initializer:
        if initializer.name == name:
            return initializer

def replace_squeeze(node, inp):
    axes_name = node.input[1]
    axes = find_initializer(axes_name).int64_data
    new_dims = [dim for dim_idx, dim in enumerate(inp.dims) if dim_idx not in axes]
    # Repeated fields cannot be assigned directly
    # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-fields
    inp.dims[:] = new_dims

def replace_reshape(node, inp):
    dims_name = node.input[1]
    new_dims = find_initializer(dims_name).int64_data
    assert new_dims
    inp.dims[:] = new_dims

replace_handlers = {
    'Squeeze': replace_squeeze,
    'Reshape': replace_reshape,
}

def replace_nodes():
    for n in g.node:
        if n.op_type not in ('Squeeze', 'Reshape'):
            continue
        inp = find_initializer(n.input[0])
        if inp:
            replace_handlers[n.op_type](n, inp)
            replaced_nodes_map[n.output[0]] = n.input[0]

replace_nodes()

# Split Conv/Gemm into Conv/Gemm and ConvMerge/GemmMerge (for OFM scaling up and merge of OFMs from channel tiling)
new_nodes = []
for idx, n in enumerate(g.node):
    if n.op_type in audio_ops:
        logger.warning('skipping audio operator %s', n.op_type)
        continue
    new_nodes.append(n)
    if n.op_type in ('Conv', 'Gemm'):
        output_name = n.output[0]
        new_node = onnx.NodeProto()
        new_node.name = (n.name or n.op_type) + ':merge'
        new_node.op_type = n.op_type + 'Merge'
        new_node.input[:] = n.output[:] = [output_name + '_before_merge']
        if n.op_type == 'Gemm':
            new_node.input[:] = new_node.input[:] + [n.input[2]]
        new_node.output[:] = [output_name]
        new_nodes.append(new_node)

new_nodes = [n for n in new_nodes if n.output[0] not in replaced_nodes_map.keys()]
for n in new_nodes:
    for idx, inp in enumerate(n.input):
        n.input[idx] = replaced_nodes_map.get(inp, inp)

nodes = [ONNXNodeWrapper(n) for n in new_nodes]

conv_param_names = set()

input_mapping = model_data.input_mapping
for idx, inp in enumerate(g.input):
    inp_name = input_mapping.get(inp.name, inp.name)
    names[inp_name] = idx

# For some ONNX models (e.g., squeezenet-cifar10 converted from Keras), inputs
# do not include initializers. Merge them here.
inputs_len = len(names.keys())
for idx, initializer in enumerate(g.initializer):
    if initializer.name not in names:
        names[initializer.name] = idx + inputs_len

Constants.N_INPUT = len(names.keys())
print("Constants.N_INPUT = {}".format(Constants.N_INPUT))

prev_node = None
for idx, n in enumerate(nodes):
    if n.op_type == 'Dropout':
        output = n.output[:1]  # we don't care the second output `mask`
    else:
        output = n.output
    assert len(output) == 1
    if n.op_type == 'Conv':
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv
        conv_param_names.add(n.input[1])
        auto_pad = get_attr(n, 'auto_pad')
        if auto_pad == b'VALID':
            n.flags.b.generic += op_flag('AUTO_PAD_VALID')
    if n.op_type == 'MaxPool':
        kernel_shape = get_attr(n, 'kernel_shape')
        if kernel_shape is not None:
            n.flags.b.kernel_size = kernel_shape[0]
    if n.op_type in ('MaxPool', 'Conv'):
        stride = get_attr(n, 'strides')[0]
        n.flags.b.stride = stride
    if n.op_type == 'Reshape' and prev_node and prev_node.op_type == 'MaxPool':
        prev_node.flags.b.generic += op_flag('NHWC2NCHW')
    names[output[0]] = idx + Constants.N_INPUT
    prev_node = n

pprint.pprint(names)

@dataclasses.dataclass
class Node:
    name: str
    inputs: List[int]
    op_type: str
    flags: NodeFlags
    max_output_id: int

def find_tensor_value_info(name: str):
    if name.endswith('_before_merge'):
        name = name[:-len('_before_merge')]
    for value_info in g.value_info:
        if value_info.name == name:
            return value_info
    raise ValueError(f'No value_info found for {name}')

def find_node_by_output(output_name):
    for node in g.node:
        if node.output[0] == output_name:
            return node

def determine_conv_tile_c(n):
    logger.debug('Determine tile size for Conv node %s', n.name)

    output_value_info = find_tensor_value_info(n.output[0])
    filter_info = find_initializer(n.input[1])
    node_flags = n.flags.b.extra.conv

    is_separate_tiling = False
    if not find_initializer(n.input[0]):
        input_node = find_node_by_output(n.input[0])
        if input_node and input_node.op_type == 'Concat':
            is_separate_tiling = True

    shape = output_value_info.type.tensor_type.shape
    OUTPUT_CHANNEL = shape.dim[1].dim_value
    OUTPUT_H = shape.dim[2].dim_value
    OUTPUT_W = shape.dim[3].dim_value
    CHANNEL = filter_info.dims[1]
    kH = filter_info.dims[2]
    kW = filter_info.dims[3]

    max_continuous_channels = CHANNEL
    if is_separate_tiling:
        max_continuous_channels /= 2
    if max_continuous_channels % 2:
        node_flags.input_tile_c = max_continuous_channels
    else:
        node_flags.input_tile_c = 1
        while max_continuous_channels % (node_flags.input_tile_c * 2) == 0 and node_flags.input_tile_c < 128:
            node_flags.input_tile_c *= 2

    logger.debug('Initial input_tile_c=%d', node_flags.input_tile_c)

    def get_memory_usage(output_tile_c, filter_len):
        # *2 as in JAPARI, the number of footprint weights is up to the number of
        # filters (e.g., batch size=1)
        ret = ((output_tile_c * 2 + 1) + Constants.TEMP_FILTER_WIDTH) * filter_len
        logger.debug('Checking output_tile_c=%d, filter_len=%d, memory usage=%d', output_tile_c, filter_len, ret)
        return ret

    while True:
        input_tile_too_large = False
        # inner +1 for biases
        filter_len = ((node_flags.input_tile_c * kW + 1) + 1) // 2 * 2 * 2 * kH
        output_tile_c = OUTPUT_CHANNEL
        while get_memory_usage(output_tile_c, filter_len) > Constants.LEA_BUFFER_SIZE:
            logger.debug('output_tile_c=%d', output_tile_c)
            output_tile_c //= 2
            if output_tile_c % 2 or output_tile_c < config['op_filters']:
                # current input_tile_c is too large such that no even output_tile_c fits
                input_tile_too_large = True
                logger.debug("Input too large!")
                break

        if not input_tile_too_large:
            params_len = CHANNEL / node_flags.input_tile_c * OUTPUT_CHANNEL * OUTPUT_H * OUTPUT_W * 2
            if params_len < config['intermediate_values_size']:
                break
        node_flags.input_tile_c //= 2
        assert node_flags.input_tile_c
        logger.debug('input_tile_c=%d', node_flags.input_tile_c)
    node_flags.output_tile_c = output_tile_c

def determine_gemm_tile_sizes(n):
    logger.debug('Determine tile size for Gemm node %s', n.name)

    A = find_tensor_value_info(n.input[0])
    B = find_initializer(n.input[1])
    A_shape = A.type.tensor_type.shape
    A_rows = A_shape.dim[0].dim_value
    A_cols = A_shape.dim[1].dim_value
    B_rows = B.dims[0]
    B_cols = B.dims[1]
    node_flags = n.flags.b.extra.gemm

    # writing a batch at a time is simpler and faster
    tile_size_unit = config['op_filters']

    node_flags.tile_width = tile_size_unit
    # +2 for bias multiplier and padding
    total_buffer_size = Constants.LEA_BUFFER_SIZE - (A_rows * A_cols + 2)
    output_len = A_rows * B_cols

    # Extend to the max possible output_len (JAPARI, B=1)
    output_len += output_len // Constants.CUR_BATCH_SIZE

    while True:
        logger.debug("tile_width=%d", node_flags.tile_width)
        # LEA wants addresses to be 4 byte-aligned, or 2 Q15-aligned
        node_flags.tile_channel = min([(Constants.ARM_PSTATE_LEN / node_flags.tile_width) / 2 * 2 - 2, B_rows]) // tile_size_unit * tile_size_unit
        while node_flags.tile_channel > 0:
            tmp = int(math.ceil(B_rows / node_flags.tile_channel))
            logger.debug("tile_channel=%d, tmp=%d", node_flags.tile_channel, tmp)
            # * 2 to fit JAPARI footprint kernels
            if total_buffer_size - (node_flags.tile_channel + 2) * node_flags.tile_width * 2 >= output_len * tmp:
                break
            node_flags.tile_channel -= tile_size_unit
        logger.debug("tile_channel = %d", node_flags.tile_channel)
        if node_flags.tile_channel > 0:
            break
        assert node_flags.tile_width % tile_size_unit == 0
        node_flags.tile_width += tile_size_unit

    while node_flags.tile_width * (node_flags.tile_channel + 2) > Constants.ARM_PSTATE_LEN:
        assert node_flags.tile_width > tile_size_unit
        node_flags.tile_width -= tile_size_unit

graph = []
for n in nodes:
    if n.op_type == 'Conv':
        determine_conv_tile_c(n)
    if n.op_type == 'Gemm':
        determine_gemm_tile_sizes(n)
    graph.append(Node(name=n.name or n.op_type,
                      inputs=[names[i] for i in n.input],
                      op_type=n.op_type,
                      flags=n.flags,
                      max_output_id=0))

for idx, node in enumerate(graph):
    for inp in node.inputs:
        if inp < Constants.N_INPUT:
            continue
        used_node = graph[inp - Constants.N_INPUT]
        used_node.max_output_id = max([idx, used_node.max_output_id])

# Inputs of Concat should be kept until Concat is processed
for idx, node in enumerate(graph):
    if node.op_type != 'Concat':
        continue
    for inp in node.inputs:
        if inp < Constants.N_INPUT:
            continue
        used_node = graph[inp - Constants.N_INPUT]
        used_node.max_output_id = max([used_node.max_output_id, node.max_output_id])

parameters = [None for _ in range(Constants.N_INPUT)]

for params in g.initializer:
    if params.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
        raise Exception('unsupported data type {}'.format(params.data_type))

    assert parameters[names[params.name]] is None
    parameters[names[params.name]] = params

pprint.pprint(graph)

def to_bytes(i, size=16):
    if size == 8:
        return struct.pack('B', i)  # unsigned char
    elif size == 16:
        return struct.pack('h', i)
    elif size == 32:
        return struct.pack('i', i)
    elif size == 64:
        return struct.pack('q', i)
    else:
        raise ValueError(f'Unsupported size {size}')

def nchw2nhwc(arr, dims):
    N, C, H, W = dims
    ret = [0] * (N * C * H * W)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    old_idx = n * C * H * W + c * H * W + h * W + w
                    new_idx = n * H * W * C + h * W * C + w * C + c
                    ret[new_idx] = arr[old_idx]
    return ret, (N, H, W, C)

outputs = {
    'parameters': io.BytesIO(),
    'samples': io.BytesIO(),
    'model': io.BytesIO(),
    'nodes': io.BytesIO(),
    'model_parameters_info': io.BytesIO(),
    'intermediate_parameters_info': io.BytesIO(),
    'labels': io.BytesIO(),
}

Constants.MODEL_NODES_LEN = len(graph)

model = outputs['model']
model.write(to_bytes(0))  # Model.running
model.write(to_bytes(0))  # Model.run_counter
model.write(to_bytes(0))  # Model.layer_idx
for _ in range(config['num_slots']): # Model.slots_info
    if Constants.INDIRECT_RECOVERY:
        model.write(to_bytes(0, size=8)) # SlotInfo.state_bit
        model.write(to_bytes(0, size=8)) # SlotInfo.n_turning_points
        for __ in range(Constants.TURNING_POINTS_LEN):
            model.write(to_bytes(-1))   # SlotInfo.turning_points
    model.write(to_bytes(-1))       # SlotInfo.user
model.write(to_bytes(0, size=8))  # Model.dummy
model.write(to_bytes(0, size=8))  # Model.version

@dataclasses.dataclass
class ParametersSlot:
    offset: int
    target: io.BytesIO
    slot_id: int

parameters_slot = ParametersSlot(offset=0, target=outputs['parameters'], slot_id=Constants.SLOT_PARAMETERS)

output_nodes = outputs['nodes']
for node in graph:
    node_name = node.name[:Constants.NODE_NAME_LEN]
    output_nodes.write(node_name.encode('ascii') + b'\0' * (Constants.NODE_NAME_LEN - len(node_name)))
    output_nodes.write(to_bytes(len(node.inputs)))
    assert len(node.inputs) <= Constants.NUM_INPUTS
    for inp in node.inputs:
        output_nodes.write(to_bytes(inp))
    for _ in range(Constants.NUM_INPUTS - len(node.inputs)):
        output_nodes.write(to_bytes(0))
    output_nodes.write(to_bytes(node.max_output_id))
    output_nodes.write(to_bytes(list(ops.keys()).index(node.op_type)))
    output_nodes.write(to_bytes(0))                     # max_multiplier
    output_nodes.write(to_bytes(node.flags.as_bytes, size=64))
    if Constants.HAWAII:
        for _ in range(2):
            output_nodes.write(to_bytes(0, size=32))  # Node::Footprint

parameter_info_idx = 0

def decode_raw_data(params):
    format_char = {
        onnx.TensorProto.FLOAT: 'f',
        onnx.TensorProto.INT64: 'q',
    }[params.data_type]
    return list(map(lambda t: t[0], struct.iter_unpack(format_char, params.raw_data)))

model_parameters_info = outputs['model_parameters_info']
for params in parameters:
    if params is None:  # input
        # Actual data for test samples are added last
        _, input_channel, dimX, dimY = model_data.images[0].shape
        model_parameters_info.write(to_bytes(parameters_slot.offset, size=32))  # params_offset
        model_parameters_info.write(to_bytes(input_channel* dimX * dimY * 2, size=32))  # A _q15 is 16-bit
        model_parameters_info.write(to_bytes(16, size=8))                # bitwidth
        model_parameters_info.write(to_bytes(Constants.SLOT_TEST_SET, size=8))     # slot
        model_parameters_info.write(to_bytes(0))                     # dummy
        # extend_dims
        model_parameters_info.write(to_bytes(1))
        model_parameters_info.write(to_bytes(input_channel))
        model_parameters_info.write(to_bytes(dimX))
        model_parameters_info.write(to_bytes(dimY))
        model_parameters_info.write(to_bytes(config['input_scale']))     # scale
    else:
        assert len(params.dims) <= 4
        if params.data_type == onnx.TensorProto.FLOAT:
            if params.float_data:
                float_data = params.float_data
            else:
                float_data = decode_raw_data(params)
            data_len = len(float_data)
            assert data_len > 0
            slot = parameters_slot
            model_parameters_info.write(to_bytes(slot.offset, size=32))  # params_offset
            model_parameters_info.write(to_bytes(data_len * 2, size=32))  # A _q15 is 16-bit
            if params.name in conv_param_names:
                print(f'Reorder conv param {params.name}')
                float_data, _ = nchw2nhwc(float_data, params.dims)
            for param in float_data:
                slot.target.write(to_bytes(_Q15(param / config['scale'])))
                slot.offset += 2
            model_parameters_info.write(to_bytes(16, size=8)) # bitwidth
        elif params.data_type == onnx.TensorProto.INT64:
            if params.int64_data:
                int64_data = params.int64_data
            else:
                int64_data = decode_raw_data(params)
            data_len = len(int64_data)
            assert data_len > 0
            slot = parameters_slot
            model_parameters_info.write(to_bytes(slot.offset, size=32))  # params_offset
            model_parameters_info.write(to_bytes(data_len * 8, size=32))
            for param in int64_data:
                slot.target.write(to_bytes(param, size=64))
                slot.offset += 8
            model_parameters_info.write(to_bytes(64, size=8)) # bitwidth
        else:
            assert False
        model_parameters_info.write(to_bytes(slot.slot_id, size=8))  # slot
        if len(params.dims) == 4:
            channels = params.dims[1]
        else:
            channels = 0
        model_parameters_info.write(to_bytes(0, size=16))        # dummy
        print('dims = {}, length = {}'.format(params.dims, data_len))
        for dim in params.dims:
            model_parameters_info.write(to_bytes(dim))
        # dims are always 4 uint16_t's in C++
        for _ in range(4 - len(params.dims)):
            model_parameters_info.write(to_bytes(0))
        model_parameters_info.write(to_bytes(config['scale']))       # scale

    # common to input and non-inputs
    model_parameters_info.write(to_bytes(0, size=8))                 # flags
    for _ in range(Constants.EXTRA_INFO_LEN):
        model_parameters_info.write(to_bytes(0, size=8))             # extra_info
    model_parameters_info.write(to_bytes(parameter_info_idx))        # parameter_info_idx
    parameter_info_idx += 1

# Placeholder for ParameterInfo of intermediate values
intermediate_parameters_info = outputs['intermediate_parameters_info']
for idx, n in enumerate(nodes):
    intermediate_parameters_info.write(to_bytes(0, size=32))  # params_offset
    intermediate_parameters_info.write(to_bytes(0, size=32))  # params_len
    intermediate_parameters_info.write(to_bytes(0, size=8))  # bitwidth
    intermediate_parameters_info.write(to_bytes(0, size=8))  # slot
    intermediate_parameters_info.write(to_bytes(0))         # dummy
    for _ in range(4):  # dims[4]
        intermediate_parameters_info.write(to_bytes(0))
    intermediate_parameters_info.write(to_bytes(config['scale']))   # scale
    intermediate_parameters_info.write(to_bytes(0, size=8))     # flags
    for _ in range(Constants.EXTRA_INFO_LEN):
        intermediate_parameters_info.write(to_bytes(0, size=8)) # extra_info
    intermediate_parameters_info.write(to_bytes(parameter_info_idx))             # parameter_info_idx
    parameter_info_idx += 1

for idx, im in enumerate(model_data.images):
    # load_data returns NCHW
    for idx_c in range(im.shape[1]):
        for idx_h in range(im.shape[2]):
            for idx_w in range(im.shape[3]):
                outputs['samples'].write(to_bytes(_Q15(im[0, idx_c, idx_h, idx_w] / config['input_scale'])))
    if args.write_images:
        import cv2
        # Restore conanical image format (H, W, C)
        im = np.squeeze(im * 256)
        if args.config == 'mnist':
            im = np.expand_dims(im, axis=-1)
            im = 255 - im
        cv2.imwrite(f'images/test{idx:02d}.png', im)

for label in model_data.labels:
    outputs['labels'].write(to_bytes(label, size=8))

if args.write_images:
    with open('images/ans.txt', 'w') as f:
        f.write(' '.join(map(str, model_data.labels)))

with open('common/data.cpp', 'w') as output_c, open('common/data.h', 'w') as output_h:
    output_h.write('''
#pragma once

#include <stdint.h>

struct ParameterInfo;
struct Model;
struct NodeFlags;

''')
    for item in itertools.chain(dir(Constants), config.keys()):
        if hasattr(Constants, item):
            if item.startswith('__'):
                continue
            val = getattr(Constants, item)
        else:
            val = config[item]
            if not isinstance(val, (int, float)):
                continue
        # Making it long to avoid overflow for expressions like
        # INTERMEDIATE_VALUES_SIZE * NUM_SLOTS on 16-bit systems
        suffix = 'l' if item == 'intermediate_values_size' else ''
        output_h.write(f'#define {item.upper()} ')
        if isinstance(val, str):
            output_h.write(f'"{val}"')
        elif isinstance(val, list):
            output_h.write('{' + ', '.join(map(str, val)) + '}')
        else:
            output_h.write(f'{val}')
        output_h.write(f'{suffix}\n')

    output_c.write('''
#include "data.h"
#include "cnn_common.h"
#include "platform.h"
''')

    # ops
    keys = list(ops.keys())
    output_h.write('\n')
    for idx, op in enumerate(keys):
        output_h.write(f'#define {op} {idx}\n')

    output_c.write('const uint8_t expected_inputs_len[] = {')
    for op in keys:
        output_c.write(f'{ops[op][0]}, ')
    output_c.write('};\n\n')

    for op in keys:
        output_h.write('void alloc_{}(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct NodeFlags* flags);\n'.format(op.lower()))
        output_h.write('void handle_{}(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct NodeFlags* flags);\n'.format(op.lower()))
    output_c.write('const handler handlers[] = {\n')
    for op in keys:
        output_c.write(f'    handle_{op},\n'.lower())
    output_c.write('};\n')
    output_c.write('const allocator allocators[] = {\n')
    for op in keys:
        output_c.write(f'    alloc_{op},\n'.lower())
    output_c.write('};\n')
    for op in keys:
        if ops[op][1]:
            output_c.write(textwrap.dedent(f'''
                void alloc_{op.lower()}(struct Model *model, const struct ParameterInfo *[], struct ParameterInfo *output, const struct NodeFlags*) {{
                    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
                    if (cur_slot_info) {{
                        cur_slot_info->user = model->layer_idx;
                    }}
                }}
            '''))

    # data
    for idx, name in enumerate(other_flags):
        output_h.write(f'#define {name} {2**idx}\n')

    def hex_str(arr):
        return '  ' + ', '.join([f'0x{num:02x}' for num in arr]) + ',\n'

    def define_var(var_name, data):
        output_h.write(f'''
extern const uint8_t * const {var_name};
#define {var_name.upper()}_LEN {len(data)}
''')
        # #define with _Pragma seems to be broken :/
        output_c.write(f'''
const uint8_t _{var_name}[{len(data)}] = {{
''')
        n_pieces, remaining = divmod(len(data), 16)
        for idx in range(n_pieces):
            output_c.write(hex_str(data[idx*16:(idx+1)*16]))
        if remaining:
            output_c.write(hex_str(data[len(data) - remaining:len(data)]))
        output_c.write(f'''}};
const uint8_t * const {var_name} = _{var_name};
''')

    for var_name, data_obj in outputs.items():
        full_var_name = var_name + '_data'
        data_obj.seek(0)
        if full_var_name == 'samples_data':
            data = data_obj.read(config['sample_size'])
        else:
            data = data_obj.read()
        define_var(full_var_name, data)

with open('samples.bin', 'wb') as f:
    samples = outputs['samples']
    samples.seek(0)
    f.write(samples.read())
