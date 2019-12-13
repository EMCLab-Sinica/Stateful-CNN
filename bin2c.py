import os

output_c = '''
#include "data.h"
'''
output_h = '''
#include <stdint.h>
'''
for filename in os.listdir('.'):
    if not filename.endswith('.bin'):
        continue
    var_name = filename[:-len('.bin')] + '_data'
    with open(filename, 'rb') as f:
        data = f.read()
    output_h += f'''
#ifdef __MSP430__
#define GLOBAL_CONST const
#else
#define GLOBAL_CONST
#endif
extern GLOBAL_CONST uint8_t {var_name}[{len(data)}];
'''
    output_c += f'''
#ifdef __MSP430__
#pragma NOINIT({var_name})
#endif
GLOBAL_CONST uint8_t {var_name}[{len(data)}] = {{'''
    output_c += ', '.join([hex(b) for b in data])
    output_c += '};\n'

with open('data.c', 'w') as f, open('data.h', 'w') as g:
    f.write(output_c)
    g.write(output_h)
