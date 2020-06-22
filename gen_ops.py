# https://github.com/onnx/onnx/blob/master/docs/Operators.md
# [expected_inputs_len, inplace_update]
ops = {
    'Add': [2, 0],
    # Concat actually accepts 1~infinity inputs. Use 2 to fit SqueezeNet
    'Concat': [2, 1],
    'Conv': [3, 0],
    'Dropout': [1, 1],
    'GlobalAveragePool': [1, 0],
    'MatMul': [2, 0],
    'MaxPool': [1, 0],
    'Relu': [1, 1],
    'Reshape': [2, 1],
    'Softmax': [1, 1],
    'Squeeze': [1, 1],
    # XXX: Transpose does nothing as we happens to need NHWC
    'Transpose': [1, 1],
}

other_flags = [
    'AUTO_PAD_VALID',
    'NHWC2NCHW',
]

with open('ops.py', 'w') as f_py, open('ops.h', 'w') as f_h, open('ops.c', 'w') as f_c:
    f_h.write('#pragma once\n\n')
    f_h.write('struct ParameterInfo;\n\n');
    f_c.write('#include "cnn_common.h"\n\n')
    f_c.write('#include "ops.h"\n\n')
    f_py.write('ops = {}\n')
    keys = list(ops.keys())
    for idx, op in enumerate(keys):
        f_h.write(f'#define {op} {idx}\n')
        f_py.write(f'ops["{op}"] = {idx}\n')

    f_c.write('uint8_t expected_inputs_len[] = {')
    for op in keys:
        f_c.write(f'{ops[op][0]}, ')
    f_c.write('};\n\n')
    f_c.write('uint8_t inplace_update[] = {')
    for op in keys:
        f_c.write(f'{ops[op][1]}, ')
    f_c.write('};\n\n')

    for op in keys:
        f_h.write('void handle_{}(struct Model *model, struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);\n'.format(op.lower()))
        f_h.write('uint32_t alloc_{}(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);\n'.format(op.lower()))
    f_c.write('handler handlers[] = {\n')
    for op in keys:
        f_c.write(f'\thandle_{op},\n'.lower())
    f_c.write('};\n')
    f_c.write('allocator allocators[] = {\n')
    for op in keys:
        f_c.write(f'\talloc_{op},\n'.lower())
    f_c.write('};\n')
    for op in keys:
        if ops[op][1]:
            f_c.write(f'uint32_t alloc_{op.lower()}(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags)\n')
            f_c.write('{\n')
            f_c.write('\tUNUSED(flags);\n')
            f_c.write('\tmy_memcpy(output, input[0], sizeof(struct ParameterInfo));\n')
            f_c.write('\treturn 0;\n')
            f_c.write('}\n')

    for idx, name in enumerate(other_flags):
        f_h.write(f'#define {name} {2**idx}\n')
        f_py.write(f'{name} = {2**idx}\n')
