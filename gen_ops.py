ops = (
    'Add',
    'Conv',
    'MatMul',
    'MaxPool',
    'Relu',
    'Reshape',
)

with open('ops.py', 'w') as f_py, open('ops.h', 'w') as f_c:
    f_c.write('#pragma once\n\n')
    f_py.write('ops = {}\n')
    for idx, op in enumerate(ops):
        f_c.write(f'const uint16_t {op} = {idx};\n')
        f_py.write(f'ops["{op}"] = {idx}\n')
