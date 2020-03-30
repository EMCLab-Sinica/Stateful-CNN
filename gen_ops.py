ops = {
    'Add': 2,
    'Conv': 3,
    'MatMul': 2,
    'MaxPool': 1,
    'Relu': 1,
    'Reshape': 2,
    'Squeeze': 1,
}

other_flags = {
    'CONV': ['BIAS_MERGED'],
}

with open('ops.py', 'w') as f_py, open('ops.h', 'w') as f_h, open('ops.c', 'w') as f_c:
    f_h.write('#pragma once\n\n')
    f_h.write('#include "common.h"\n\n')
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
        f_h.write('uint8_t handle_{}(ParameterInfo *input[], ParameterInfo *output, OpExtraData *extra_data, uint16_t flags);\n'.format(op.lower()))
    f_c.write('handler handlers[] = {\n')
    for op in keys:
        f_c.write(f'\thandle_{op},\n'.lower())
    f_c.write('};\n')

    for key, values in other_flags.items():
        for idx, value in enumerate(values):
            f_h.write(f'#define {key}_{value} {2**idx}\n')
            f_py.write(f'{key}_{value} = {2**idx}\n')
