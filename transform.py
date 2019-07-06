import sys

import onnx

"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(g.input)-1: input nodes
    len(g.input)~ : other (hidden) nodes
"""

model = onnx.load(sys.argv[1])
g = model.graph
names = {}
n_input = len(g.input)
print(n_input)

for idx, inp in enumerate(g.input):
    names[inp.name] = idx

for idx, n in enumerate(g.node):
    assert len(n.output) == 1
    names[n.output[0]] = idx + n_input

model = [
    sorted([names[i] for i in n.input])
    for n in g.node]

print(model)

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
