ops = {
    'Add': 2,
    'Conv': 3,
    'MatMul': 2,
    'MaxPool': 1,
    'Relu': 1,
    'Reshape': 2,
    'Squeeze': 1,
}

other_flags = [
    'CONV_BIAS_MERGED',
    'TRANSPOSED',
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
        f_c.write(f'{ops[op]}, ')
    f_c.write('};\n\n')

    for op in keys:
        f_h.write('void handle_{}(struct ParameterInfo *input[], struct ParameterInfo *output, uint16_t flags);\n'.format(op.lower()))
    f_c.write('handler handlers[] = {\n')
    for op in keys:
        f_c.write(f'\thandle_{op},\n'.lower())
    f_c.write('};\n')

    for idx, name in enumerate(other_flags):
        f_h.write(f'#define {name} {2**idx}\n')
        f_py.write(f'{name} = {2**idx}\n')
