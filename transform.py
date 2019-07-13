import sys
import warnings

import onnx

"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(g.input)-1: input nodes
    len(g.input)~ : other (hidden) nodes
"""


def _Q15(num):
    """Transform a floating point number to TI's fixed point _q15 format"""

    # See DSPLib_1_30_00_02/include/DSPLib_support.h

    lower = -1
    upper = 32767.0 / 32768.0

    if num < lower or num >= upper:
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
    assert len(n.output) == 1
    names[n.output[0]] = idx + n_input

model = [
    sorted([names[i] for i in n.input])
    for n in g.node]
parameters = [None for _ in range(n_input)]

for params in g.initializer:
    if params.data_type == onnx.TensorProto.FLOAT:
        print('dims = {}, length = {}'.format(
            params.dims, len(params.float_data)))
        data = [_Q15(num) for num in params.float_data]
    elif params.data_type == onnx.TensorProto.INT64:
        data = params.int64_data
    else:
        raise Exception('unsupported data type {}'.format(params.data_type))

    assert parameters[names[params.name]] is None
    parameters[names[params.name]] = data


def to_bytes(i):
    return i.to_bytes(2, byteorder=sys.byteorder)


output = to_bytes(len(model))
output += to_bytes(n_input)
for inputs in model:
    output += to_bytes(len(inputs))
    for inp in inputs:
        output += to_bytes(inp)

with open('model.bin', 'wb') as f:
    f.write(output)
