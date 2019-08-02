import pprint
import struct
import sys
import warnings

import cv2
import onnx

import ops

"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(g.input)-1: input nodes
    len(g.input)~ : other (hidden) nodes
"""

POINTER_SIZE = 64  # 64-bit systems


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


model = onnx.load(sys.argv[1])
g = model.graph
names = {}
n_input = len(g.input)
print("n_input = {}".format(n_input))

for idx, inp in enumerate(g.input):
    names[inp.name] = idx

for idx, n in enumerate(g.node):
    if n.op_type == 'Dropout':
        output = n.output[:1]  # we don't care the second output `mask`
    else:
        output = n.output
    assert len(output) == 1
    names[output[0]] = idx + n_input

pprint.pprint(names)

model = [
    (sorted([names[i] for i in n.input]), n.op_type)
    for n in g.node]
parameters = [None for _ in range(n_input)]

for params in g.initializer:
    if params.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
        raise Exception('unsupported data type {}'.format(params.data_type))

    assert parameters[names[params.name]] is None
    parameters[names[params.name]] = params


def to_bytes(i, size=16):
    if size == 16:
        return struct.pack('h', i)
    elif size == 32:
        return struct.pack('i', i)
    elif size == 64:
        return struct.pack('q', i)
    else:
        raise ValueError(f'Unsupported size {size}')


outputs = {}
model_bin = to_bytes(len(model))
model_bin += to_bytes(n_input)
model_bin += to_bytes(0, size=POINTER_SIZE)  # placeholder for nodes pointer
model_bin += to_bytes(0, size=POINTER_SIZE)  # placeholder for parameters pointer
inputs_bin = b''
parameters_bin = open('parameters.bin', 'wb')
parameters_bin_offset = 0
for inputs, op_type in model:
    model_bin += to_bytes(len(inputs))
    model_bin += to_bytes(len(inputs_bin))  # Node.inputs_offset
    for inp in inputs:
        # the lowest bit is used as a flag in topological sort
        inputs_bin += to_bytes(inp * 2)
    model_bin += to_bytes(ops.ops[op_type])
    model_bin += to_bytes(0)  # Node.scheduled

for params in parameters:
    model_bin += to_bytes(parameters_bin_offset, size=32)  # params_offset
    if params is None:  # input
        im = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
        # See https://github.com/microsoft/CNTK/blob/master/Tutorials/CNTK_103*
        # for data format
        im = 255 - im
        im = im / 256  # to fit into range of _q15
        dimX, dimY = im.shape
        model_bin += to_bytes(dimX * dimY * 2, size=32)  # A _q15 is 16-bit
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                parameters_bin.write(to_bytes(_Q15(im[i, j])))
                parameters_bin_offset += 2
        model_bin += to_bytes(16)
        # extend_dims
        model_bin += to_bytes(1)
        model_bin += to_bytes(1)
        model_bin += to_bytes(dimX)
        model_bin += to_bytes(dimY)
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
            model_bin += to_bytes(data_len * 2, size=32)  # A _q15 is 16-bit
            for param in float_data:
                parameters_bin.write(to_bytes(_Q15(param)))
                parameters_bin_offset += 2
            model_bin += to_bytes(16)
        elif params.data_type == onnx.TensorProto.INT64:
            data_len = len(params.int64_data)
            model_bin += to_bytes(data_len * 8, size=32)
            for param in params.int64_data:
                parameters_bin.write(to_bytes(param, size=64))
                parameters_bin_offset += 8
            model_bin += to_bytes(64)
        else:
            assert False
        print('dims = {}, length = {}'.format(params.dims, data_len))
        for dim in params.dims:
            model_bin += to_bytes(dim)
        # dims are always 4 uint16_t's in C
        for _ in range(4 - len(params.dims)):
            model_bin += to_bytes(0)

outputs['model.bin'] = model_bin
outputs['inputs.bin'] = inputs_bin
parameters_bin.close()

for filename, data in outputs.items():
    with open(filename, 'wb') as f:
        f.write(data)
