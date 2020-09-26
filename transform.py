#!/usr/bin/python
import argparse
import ctypes
import dataclasses
import io
import itertools
import pprint
import struct
import textwrap
import warnings
from typing import List

import onnx
import onnx.helper
import onnx.optimizer
import numpy as np

from utils import load_data, load_data_cifar10

"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(g.input)-1: input nodes
    len(g.input)~ : other (hidden) nodes
"""

class Constants:
    SLOT_PARAMETERS2 = 0xf1
    SLOT_PARAMETERS = 0xf0
    SLOT_TEST_SET = 0xff
    SLOT_CONSTANTS_MIN = SLOT_PARAMETERS
    SLOT_INTERMEDIATE_VALUES = 0b01
    COUNTERS_LEN = 64
    # To make the Node struct exactly 32 bytes
    NODE_NAME_LEN = 18
    EXTRA_INFO_LEN = 3  # for memory alignment
    TURNING_POINTS_LEN = 8
    STATEFUL_CNN = 1
    MODEL_NODES_LEN = 0
    INPUTS_DATA_LEN = 0
    NUM_INPUTS = 3

# https://github.com/onnx/onnx/blob/master/docs/Operators.md
# [expected_inputs_len, inplace_update]
ops = {
    'Add': [2, 0],
    # Concat actually accepts 1~infinity inputs. Use 2 to fit SqueezeNet
    'Concat': [2, 0],
    'Conv': [3, 0],
    'ConvMerge': [1, 0],
    'Dropout': [1, 1],
    'GlobalAveragePool': [1, 0],
    'MatMul': [2, 0],
    'MaxPool': [1, 0],
    'Relu': [1, 0],
    'Reshape': [2, 1],
    'Softmax': [1, 1],
    'Squeeze': [1, 1],
    # XXX: Transpose does nothing as we happens to need NHWC
    'Transpose': [1, 1],
}

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
class NodeFlags_bits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("generic", ctypes.c_uint8, 8),
        ("kernel_size", ctypes.c_uint8, 4),
        ("stride", ctypes.c_uint8, 4),
    ]

class NodeFlags(ctypes.Union):
    _fields_ = [
        ("b", NodeFlags_bits),
        ("as_bytes", ctypes.c_uint16),
    ]

    def __repr__(self):
        return f'<NodeFlags generic={self.b.generic} kernel_size={self.b.kernel_size} stride={self.b.stride}>'

class ONNXNodeWrapper:
    def __init__(self, orig_node: onnx.NodeProto):
        self.orig_node = orig_node
        self.flags = NodeFlags()

    def __getattr__(self, name):
        return getattr(self.orig_node, name)


def get_prev_node(n):
    return nodes[names[n.input[0]] - n_input]

# intermediate_values_size should < 65536, or TI's compiler gets confused
configs = {
    'mnist': {
        # https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx
        'onnx_model': 'data/mnist-8.onnx',
        'input_file': 'data/MNIST/Test-28x28_cntk_text.txt',
        'scale': 8,
        'num_slots': 2,
        'intermediate_values_size': 20000,
        'nvm_size': 256 * 1024,
        'data_loader': load_data,
        'n_samples': 20,
        'n_all_samples': 10000,
        'fp32_accuracy': 0.9889,
    },
    'cifar10': {
        'onnx_model': 'data/squeezenet_cifar10.onnx',
        'input_file': 'data/cifar-10-batches-py/test_batch',
        'scale': 8,
        'num_slots': 3,
        'intermediate_values_size': 30000,
        'nvm_size': 1024 * 1024,
        'data_loader': load_data_cifar10,
        'n_samples': 20,
        'n_all_samples': 10000,
        'fp32_accuracy': 0.7704,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('config', choices=configs.keys())
parser.add_argument('--without-stateful-cnn', action='store_true')
parser.add_argument('--all-samples', action='store_true')
parser.add_argument('--write-images', action='store_true')
args = parser.parse_args()
config = configs[args.config]
if args.all_samples:
    config['nvm_size'] *= 64
    config['n_samples'] = config['n_all_samples']

if args.without_stateful_cnn:
    Constants.STATEFUL_CNN = 0

original_model = onnx.load(config['onnx_model'])
try:
    # https://zhuanlan.zhihu.com/p/41255090
    onnx_model = onnx.optimizer.optimize(original_model, ['fuse_add_bias_into_conv'])
except IndexError:
    # Somehow the optimizer cannot handle models transformed from keras2onnx
    onnx_model = original_model
g = onnx_model.graph
names = {}

# Remoe Squeeze nodes with constants as the input
replaced_squeeze_map = {}
for n in g.node:
    if n.op_type != 'Squeeze':
        continue
    input_name = n.input[0]
    for inp in g.initializer:
        if input_name != inp.name:
            continue
        axes = next(attr.ints for attr in n.attribute if attr.name == 'axes')
        new_dims = [dim for dim_idx, dim in enumerate(inp.dims) if dim_idx not in axes]
        # Repeated fields cannot be assigned directly
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-fields
        inp.dims[:] = new_dims
        replaced_squeeze_map[n.output[0]] = input_name
        break

# Split Conv into Conv and ConvMerge (for OFM scaling up and merge of OFMs from  channel tiling)
new_nodes = []
for idx, n in enumerate(g.node):
    new_nodes.append(n)
    if n.op_type != 'Conv':
        continue
    output_name = n.output[0]
    new_node = onnx.NodeProto()
    new_node.name = n.name + ':merge'
    new_node.op_type = 'ConvMerge'
    new_node.input[:] = n.output[:] = [output_name + '_before_merge']
    new_node.output[:] = [output_name]
    new_nodes.append(new_node)

new_nodes = [n for n in new_nodes if n.output[0] not in replaced_squeeze_map.keys()]
for n in new_nodes:
    for idx, inp in enumerate(n.input):
        n.input[idx] = replaced_squeeze_map.get(inp, inp)

nodes = [ONNXNodeWrapper(n) for n in new_nodes]

conv_param_names = set()

for idx, inp in enumerate(g.input):
    names[inp.name] = idx

# For some ONNX models (e.g., squeezenet-cifar10 converted from Keras), inputs
# do not include initializers. Merge them here.
inputs_len = len(names.keys())
for idx, initializer in enumerate(g.initializer):
    if initializer.name not in names:
        names[initializer.name] = idx + inputs_len

n_input = len(names.keys())
print("n_input = {}".format(n_input))

def get_attr(node, attr_name):
    for attr in node.attribute:
        if attr.name != attr_name:
            continue
        return onnx.helper.get_attribute_value(attr)

    # Not found
    return None

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
    names[output[0]] = idx + n_input
    prev_node = n

pprint.pprint(names)

@dataclasses.dataclass
class Node:
    name: str
    inputs: List[int]
    op_type: str
    flags: NodeFlags
    max_output_id: int

model = []
for n in nodes:
    model.append(Node(n.name, [names[i] for i in n.input], n.op_type, n.flags, 0))

for idx, node in enumerate(model):
    for inp in node.inputs:
        if inp < n_input:
            continue
        used_node = model[inp - n_input]
        used_node.max_output_id = max([idx, used_node.max_output_id])

# Inputs of Concat should be kept until Concat is processed
for idx, node in enumerate(model):
    if node.op_type != 'Concat':
        continue
    for inp in node.inputs:
        if inp < n_input:
            continue
        used_node = model[inp - n_input]
        used_node.max_output_id = max([used_node.max_output_id, node.max_output_id])

parameters = [None for _ in range(n_input)]

for params in g.initializer:
    if params.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
        raise Exception('unsupported data type {}'.format(params.data_type))

    assert parameters[names[params.name]] is None
    parameters[names[params.name]] = params

pprint.pprint(model)

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
    'parameters2': io.BytesIO(),
    'samples': io.BytesIO(),
    'model': io.BytesIO(),
    'nodes': io.BytesIO(),
    'parameters_info': io.BytesIO(),
    'labels': io.BytesIO(),
    'counters': io.BytesIO(),
}

Constants.MODEL_NODES_LEN = len(model)

outputs['model'].write(to_bytes(n_input))
outputs['model'].write(to_bytes(0))  # Model.running
outputs['model'].write(to_bytes(1))  # Model.first_time
outputs['model'].write(to_bytes(0))  # Model.run_counter
outputs['model'].write(to_bytes(0))  # Model.layer_idx
outputs['model'].write(to_bytes(0))  # Model.sample_idx
for _ in range(config['num_slots']): # Model.slots_info
    if Constants.STATEFUL_CNN:
        outputs['model'].write(to_bytes(0, size=8)) # SlotInfo.state_bit
        outputs['model'].write(to_bytes(0, size=8)) # SlotInfo.n_turning_points
        for __ in range(Constants.TURNING_POINTS_LEN):
            outputs['model'].write(to_bytes(-1))   # SlotInfo.turning_points
    outputs['model'].write(to_bytes(-1))       # SlotInfo.user

@dataclasses.dataclass
class ParametersSlot:
    offset: int
    target: io.BytesIO
    slot_id: int

parameters_slot = ParametersSlot(offset=0, target=outputs['parameters'], slot_id=Constants.SLOT_PARAMETERS)
parameters2_slot = ParametersSlot(offset=0, target=outputs['parameters2'], slot_id=Constants.SLOT_PARAMETERS2)

for node in model:
    node_name = node.name[:Constants.NODE_NAME_LEN]
    outputs['nodes'].write(node_name.encode('ascii') + b'\0' * (Constants.NODE_NAME_LEN - len(node_name)))
    outputs['nodes'].write(to_bytes(len(node.inputs)))
    assert len(node.inputs) <= Constants.NUM_INPUTS
    for inp in node.inputs:
        outputs['nodes'].write(to_bytes(inp))
    for _ in range(Constants.NUM_INPUTS - len(node.inputs)):
        outputs['nodes'].write(to_bytes(0))
    outputs['nodes'].write(to_bytes(node.max_output_id))
    outputs['nodes'].write(to_bytes(list(ops.keys()).index(node.op_type)))
    outputs['nodes'].write(to_bytes(node.flags.as_bytes))

labels, images = config['data_loader'](config['input_file'], start=0, limit=config['n_samples'])

def select_parameters_slot(data_len):
    if data_len <= 1024:  # XXX: random heuristic
        return parameters_slot
    else:
        return parameters2_slot

parameter_info_idx = 0

for params in parameters:
    if params is None:  # input
        # Actual data for test samples are added last
        _, input_channel, dimX, dimY = images[0].shape
        outputs['parameters_info'].write(to_bytes(parameters_slot.offset, size=32))  # params_offset
        outputs['parameters_info'].write(to_bytes(input_channel* dimX * dimY * 2, size=32))  # A _q15 is 16-bit
        outputs['parameters_info'].write(to_bytes(16, size=8))                # bitwidth
        outputs['parameters_info'].write(to_bytes(Constants.SLOT_TEST_SET, size=8))     # slot
        outputs['parameters_info'].write(to_bytes(input_channel, size=16))    # tile_c
        # extend_dims
        outputs['parameters_info'].write(to_bytes(1))
        outputs['parameters_info'].write(to_bytes(input_channel))
        outputs['parameters_info'].write(to_bytes(dimX))
        outputs['parameters_info'].write(to_bytes(dimY))
    else:
        assert len(params.dims) <= 4
        if params.data_type == onnx.TensorProto.FLOAT:
            if params.float_data:
                float_data = params.float_data
            else:
                float_data = list(map(lambda t: t[0], struct.iter_unpack('f', params.raw_data)))
            data_len = len(float_data)
            assert data_len > 0
            slot = select_parameters_slot(data_len * 2)
            outputs['parameters_info'].write(to_bytes(slot.offset, size=32))  # params_offset
            outputs['parameters_info'].write(to_bytes(data_len * 2, size=32))  # A _q15 is 16-bit
            if params.name in conv_param_names:
                print(f'Reorder conv param {params.name}')
                float_data, _ = nchw2nhwc(float_data, params.dims)
            for param in float_data:
                slot.target.write(to_bytes(_Q15(param / config['scale'])))
                slot.offset += 2
            outputs['parameters_info'].write(to_bytes(16, size=8)) # bitwidth
        elif params.data_type == onnx.TensorProto.INT64:
            data_len = len(params.int64_data)
            slot = select_parameters_slot(data_len * 8)
            outputs['parameters_info'].write(to_bytes(slot.offset, size=32))  # params_offset
            outputs['parameters_info'].write(to_bytes(data_len * 8, size=32))
            for param in params.int64_data:
                slot.target.write(to_bytes(param, size=64))
                slot.offset += 8
            outputs['parameters_info'].write(to_bytes(64, size=8)) # bitwidth
        else:
            assert False
        outputs['parameters_info'].write(to_bytes(slot.slot_id, size=8))  # slot
        if len(params.dims) == 4:
            tile_c = params.dims[1]
        else:
            tile_c = 0
        outputs['parameters_info'].write(to_bytes(tile_c, size=16))       # tile_c
        print('dims = {}, length = {}'.format(params.dims, data_len))
        for dim in params.dims:
            outputs['parameters_info'].write(to_bytes(dim))
        # dims are always 4 uint16_t's in C++
        for _ in range(4 - len(params.dims)):
            outputs['parameters_info'].write(to_bytes(0))

    # common to input and non-inputs
    outputs['parameters_info'].write(to_bytes(0, size=8))                 # flags
    for _ in range(Constants.EXTRA_INFO_LEN):
        outputs['parameters_info'].write(to_bytes(0, size=8))             # extra_info
    outputs['parameters_info'].write(to_bytes(config['scale']))           # scale
    outputs['parameters_info'].write(to_bytes(parameter_info_idx))        # parameter_info_idx
    parameter_info_idx += 1

# Placeholder for ParameterInfo of intermediate values
for idx, n in enumerate(nodes):
    outputs['parameters_info'].write(to_bytes(0, size=32))  # params_offset
    outputs['parameters_info'].write(to_bytes(0, size=32))  # params_len
    outputs['parameters_info'].write(to_bytes(0, size=8))  # bitwidth
    outputs['parameters_info'].write(to_bytes(0, size=8))  # slot
    outputs['parameters_info'].write(to_bytes(0, size=16))  # tile_c
    for _ in range(4):  # dims[4]
        outputs['parameters_info'].write(to_bytes(0))
    outputs['parameters_info'].write(to_bytes(0, size=8))     # flags
    for _ in range(Constants.EXTRA_INFO_LEN):
        outputs['parameters_info'].write(to_bytes(0, size=8)) # extra_info
    outputs['parameters_info'].write(to_bytes(config['scale']))   # scale
    outputs['parameters_info'].write(to_bytes(parameter_info_idx))             # parameter_info_idx
    parameter_info_idx += 1

for idx, im in enumerate(images):
    # load_data returns NCHW
    for idx_c in range(im.shape[1]):
        for idx_h in range(im.shape[2]):
            for idx_w in range(im.shape[3]):
                outputs['samples'].write(to_bytes(_Q15(im[0, idx_c, idx_h, idx_w] / config['scale'])))
    if args.write_images:
        import cv2
        # Restore conanical image format (H, W, C)
        im = np.squeeze(im * 256)
        if args.config == 'mnist':
            im = np.expand_dims(im, axis=-1)
            im = 255 - im
        cv2.imwrite(f'images/test{idx:02d}.png', im)

for label in labels:
    outputs['labels'].write(to_bytes(label, size=8))

if args.write_images:
    with open('images/ans.txt', 'w') as f:
        f.write(' '.join(map(str, labels)))

outputs['counters'].write(b'\0' * (4 * Constants.COUNTERS_LEN + 2))

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
        output_h.write(f'#define {item.upper()} {val}{suffix}\n')

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

    output_c.write('uint8_t expected_inputs_len[] = {')
    for op in keys:
        output_c.write(f'{ops[op][0]}, ')
    output_c.write('};\n\n')

    for op in keys:
        output_h.write('void alloc_{}(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct NodeFlags* flags);\n'.format(op.lower()))
        output_h.write('void handle_{}(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct NodeFlags* flags);\n'.format(op.lower()))
    output_c.write('handler handlers[] = {\n')
    for op in keys:
        output_c.write(f'    handle_{op},\n'.lower())
    output_c.write('};\n')
    output_c.write('allocator allocators[] = {\n')
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

    def define_var(var_name, data, will_modify):
        const_qualifier = 'const ' if not will_modify else ''

        output_h.write(f'''
extern {const_qualifier}uint8_t *{var_name};
#define {var_name.upper()}_LEN {len(data)}
''')
        # #define with _Pragma seems to be broken :/
        if var_name == 'parameters_data':
            section = 'nvm'
        else:
            section = 'nvm2'
        output_c.write(f'''
#ifdef NEED_DATA_VARS
#pragma DATA_SECTION(".{section}")
{const_qualifier}uint8_t _{var_name}[{len(data)}] = {{
''')
        n_pieces, remaining = divmod(len(data), 16)
        for idx in range(n_pieces):
            output_c.write(hex_str(data[idx*16:(idx+1)*16]))
        if remaining:
            output_c.write(hex_str(data[len(data) - remaining:len(data)]))
        output_c.write(f'''}};
{const_qualifier}uint8_t *{var_name} = _{var_name};
#endif
''')

    for var_name, data_obj in outputs.items():
        if var_name in ('samples', 'labels'):
            continue
        full_var_name = var_name + '_data'
        data_obj.seek(0)
        define_var(full_var_name, data_obj.read(),
                   will_modify=var_name in ('model', 'parameters_info', 'counters'))

    outputs['samples'].seek(0)
    samples_data = outputs['samples'].read()
    outputs['labels'].seek(0)
    labels_data = outputs['labels'].read()

    define_var('samples_data', samples_data[:len(samples_data)//len(labels)], False)
    define_var('labels_data', labels_data[:len(labels_data)//len(labels)], False)

with open('nvm.bin', 'wb') as f:
    f.write(config['nvm_size'] * b'\0')
    f.seek(config['num_slots'] * config['intermediate_values_size'])
    for data_obj in outputs.values():
        data_obj.seek(0)
        f.write(data_obj.read())
        needed_nvm_size = f.tell()
        assert needed_nvm_size < config['nvm_size'], f'Need NVM size {needed_nvm_size}'
