import sys

import onnx

model = onnx.load(sys.argv[1])
g = model.graph
names = {}
n_input = len(g.input)

for idx, inp in enumerate(g.input):
    names[inp.name] = idx

for idx, n in enumerate(g.node):
    assert len(n.output) == 1
    names[n.output[0]] = idx + n_input

inputs_table = [
    [names[i] for i in n.input]
    for n in g.node]

print(inputs_table)

def to_bytes(i):
    return i.to_bytes(2, byteorder=sys.byteorder)

output = to_bytes(len(inputs_table))
for inputs in inputs_table:
    output += to_bytes(len(inputs))
    for inp in inputs:
        output += to_bytes(inp)

with open('inputs_table.bin', 'wb') as f:
    f.write(output)
