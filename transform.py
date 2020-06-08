import argparse
import io
import pprint
import struct
import warnings

import cv2
import onnx
import numpy as np

import ops
from utils import load_data

"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(g.input)-1: input nodes
    len(g.input)~ : other (hidden) nodes
"""

# XXX: Heuristics for scaling: only scale biases and the input
SCALE = 100
NUM_SLOTS = 2
INTERMEDIATE_VALUES_SIZE = 26000
CACHED_FILTERS_LEN = 8000
N_SAMPLES = 20
COUNTERS_LEN = 64


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


class Node:
    def __init__(self, orig_node: onnx.NodeProto, flags: int = 0):
        self.orig_node = orig_node
        self.flags = flags

    def __getattr__(self, name):
        return getattr(self.orig_node, name)


def get_prev_node(n):
    return nodes[names[n.input[0]] - n_input]

parser = argparse.ArgumentParser()
parser.add_argument('onnx_model')
parser.add_argument('input_file')
args = parser.parse_args()
onnx_model = onnx.load(args.onnx_model)
g = onnx_model.graph
names = {}
n_input = len(g.input)

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

new_nodes = [n for n in g.node if n.output[0] not in replaced_squeeze_map.keys()]
for n in new_nodes:
    for idx, inp in enumerate(n.input):
        n.input[idx] = replaced_squeeze_map.get(inp, inp)

nodes = [Node(n) for n in new_nodes]
print("n_input = {}".format(n_input))

conv_param_names = set()

for idx, inp in enumerate(g.input):
    names[inp.name] = idx

for idx, n in enumerate(nodes):
    if n.op_type == 'Dropout':
        output = n.output[:1]  # we don't care the second output `mask`
    else:
        output = n.output
    assert len(output) == 1
    if n.op_type == 'Conv':
        # https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv
        conv_param_names.add(n.input[1])
    if n.op_type == 'MaxPool':
        stride = next(attr.ints[0] for attr in n.attribute if attr.name == 'strides')
        n.flags = stride
    names[output[0]] = idx + n_input

pprint.pprint(names)

model = []
for n in nodes:
    model.append(([names[i] for i in n.input], n.op_type, n.flags))
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

inputs_data = io.BytesIO()
outputs = {
    'parameters': io.BytesIO(),
    'samples': io.BytesIO(),
    'model': io.BytesIO(),
    'labels': io.BytesIO(),
    'counters': io.BytesIO(),
}

outputs['model'].write(to_bytes(len(model)))
outputs['model'].write(to_bytes(n_input))
outputs['model'].write(to_bytes(0))  # Model.running
outputs['model'].write(to_bytes(0))  # Model.recovery
outputs['model'].write(to_bytes(0))  # Model.run_counter
outputs['model'].write(to_bytes(0))  # Model.state_bit
outputs['model'].write(to_bytes(0))  # Model.sample_idx
parameters_bin_offset = 0
for inputs, op_type, flags in model:
    outputs['model'].write(to_bytes(len(inputs)))
    outputs['model'].write(to_bytes(inputs_data.tell()))  # Node.inputs_offset
    for inp in inputs:
        # the lowest bit is used as a flag in topological sort
        inputs_data.write(to_bytes(inp * 2))
    outputs['model'].write(to_bytes(ops.ops[op_type]))
    print(f'flags: {flags}')
    outputs['model'].write(to_bytes(flags))
    outputs['model'].write(to_bytes(0))  # Node.scheduled

# Keep these in sync with cnn_common.h
FLAG_SLOTS = 0b11
FLAG_TEST_SET = 0b10

labels, images = load_data(args.input_file, limit=N_SAMPLES)

for params in parameters:
    outputs['model'].write(to_bytes(parameters_bin_offset, size=32))  # params_offset
    if params is None:  # input
        # Actual data for test samples are added last
        _, _, dimX, dimY = images[0].shape
        outputs['model'].write(to_bytes(dimX * dimY * 2, size=32))  # A _q15 is 16-bit
        outputs['model'].write(to_bytes(16, size=8))                # bitwidth
        outputs['model'].write(to_bytes(FLAG_TEST_SET, size=8))     # slot
        outputs['model'].write(to_bytes(0, size=8))                 # flag
        outputs['model'].write(to_bytes(0, size=8))                 # dummy
        # extend_dims
        outputs['model'].write(to_bytes(1))
        outputs['model'].write(to_bytes(1))
        outputs['model'].write(to_bytes(dimX))
        outputs['model'].write(to_bytes(dimY))
    else:
        assert len(params.dims) <= 4
        if params.data_type == onnx.TensorProto.FLOAT:
            if params.float_data:
                float_data = params.float_data
            else:
                float_data = [None] * (len(params.raw_data) // 4)
                for i in range(len(params.raw_data) // 4):
                    float_data[i] = struct.unpack_from(
                        'f', params.raw_data, offset=4 * i)[0]
            data_len = len(float_data)
            assert data_len > 0
            outputs['model'].write(to_bytes(data_len * 2, size=32))  # A _q15 is 16-bit
            if params.name in conv_param_names:
                print(f'Reorder conv param {params.name}')
                float_data, _ = nchw2nhwc(float_data, params.dims)
            for param in float_data:
                if len(params.dims) != 4:  # most likely biases
                    outputs['parameters'].write(to_bytes(_Q15(param / SCALE)))
                else:
                    outputs['parameters'].write(to_bytes(_Q15(param)))
                parameters_bin_offset += 2
            outputs['model'].write(to_bytes(16, size=8)) # bitwidth
        elif params.data_type == onnx.TensorProto.INT64:
            data_len = len(params.int64_data)
            outputs['model'].write(to_bytes(data_len * 8, size=32))
            for param in params.int64_data:
                outputs['parameters'].write(to_bytes(param, size=64))
                parameters_bin_offset += 8
            outputs['model'].write(to_bytes(64, size=8)) # bitwidth
        else:
            assert False
        outputs['model'].write(to_bytes(FLAG_SLOTS, size=8))    # slot
        outputs['model'].write(to_bytes(0, size=8))             # flag
        outputs['model'].write(to_bytes(0, size=8))             # dummy
        print('dims = {}, length = {}'.format(params.dims, data_len))
        for dim in params.dims:
            outputs['model'].write(to_bytes(dim))
        # dims are always 4 uint16_t's in C
        for _ in range(4 - len(params.dims)):
            outputs['model'].write(to_bytes(0))

# Placeholder for ParameterInfo of intermediate values
for idx, n in enumerate(nodes):
    outputs['model'].write(to_bytes(0, size=32))  # params_offset
    outputs['model'].write(to_bytes(0, size=32))  # params_len
    outputs['model'].write(to_bytes(0, size=8))  # bitwidth
    outputs['model'].write(to_bytes(0, size=8))  # slot
    outputs['model'].write(to_bytes(0, size=8))  # flags
    outputs['model'].write(to_bytes(0, size=8))  # dummy
    for _ in range(4):  # dims[4]
        outputs['model'].write(to_bytes(0))

inputs_data.seek(0)
outputs['model'].write(inputs_data.read())

for idx, im in enumerate(images):
    # load_data returns NCHW
    for i in range(im.shape[2]):
        for j in range(im.shape[3]):
            outputs['samples'].write(to_bytes(_Q15(im[0, 0, i, j] / SCALE)))
    # Restore conanical image format (H, W, C)
    im = np.expand_dims(np.squeeze(im * 256), axis=-1)
    im = 255 - im
    cv2.imwrite(f'images/test{idx:02d}.png', im)

for label in labels:
    outputs['labels'].write(to_bytes(label, size=8))

with open('images/ans.txt', 'w') as f:
    f.write(' '.join(map(str, labels)))

outputs['counters'].write(b'\0' * (4 * COUNTERS_LEN + 2))

with open('data.c', 'w') as output_c, open('data.h', 'w') as output_h:
    output_c.write('''
#include "data.h"
''')
    output_h.write(f'''
#pragma once
#include <stdint.h>
#include "platform.h"

#define SCALE {SCALE}
#define NUM_SLOTS {NUM_SLOTS}
#define INTERMEDIATE_VALUES_SIZE {INTERMEDIATE_VALUES_SIZE}u
#define CACHED_FILTERS_LEN {CACHED_FILTERS_LEN}
#define COUNTERS_LEN {COUNTERS_LEN}
''')

    def hex_str(arr):
        return '  ' + ', '.join([f'0x{num:02x}' for num in arr]) + ',\n'

    def define_var(var_name, data):
        output_h.write(f'''
extern uint8_t *{var_name};
#define {var_name.upper()}_LEN {len(data)}
''')
        # #define with _Pragma seems to be broken :/
        output_c.write(f'''
#ifdef __MSP430__
#pragma DATA_SECTION(_{var_name}, ".nvm")
#endif
uint8_t _{var_name}[{len(data)}] = {{
''')
        n_pieces, remaining = divmod(len(data), 16)
        for idx in range(n_pieces):
            output_c.write(hex_str(data[idx*16:(idx+1)*16]))
        if remaining:
            output_c.write(hex_str(data[len(data) - remaining:len(data)]))
        output_c.write(f'''}};
uint8_t *{var_name} = _{var_name};
''')

    for var_name, data_obj in outputs.items():
        if var_name in ('samples', 'labels'):
            continue
        var_name += '_data'
        data_obj.seek(0)
        define_var(var_name, data_obj.read())

    outputs['samples'].seek(0)
    samples_data = outputs['samples'].read()
    outputs['labels'].seek(0)
    labels_data = outputs['labels'].read()

    output_h.write('\n#if USE_ALL_SAMPLES\n')
    output_c.write('\n#if USE_ALL_SAMPLES\n')
    define_var('samples_data', samples_data)
    define_var('labels_data', labels_data)
    output_h.write('\n#else\n')
    output_c.write('\n#else\n')
    define_var('samples_data', samples_data[:len(samples_data)//len(labels)])
    define_var('labels_data', labels_data[:len(labels_data)//len(labels)])
    output_h.write('\n#endif\n')
    output_c.write('\n#endif\n')


with open('nvm.bin', 'wb') as f:
    f.write(256 * 1024 * b'\0')
    f.seek(CACHED_FILTERS_LEN + NUM_SLOTS * INTERMEDIATE_VALUES_SIZE)
    for data_obj in outputs.values():
        data_obj.seek(0)
        f.write(data_obj.read())
        assert f.tell() < 256 * 1024
