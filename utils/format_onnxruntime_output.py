import re
import sys

# Requires onnxruntime to be built with -Donnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=2

def main():
    dims = None
    for line in sys.stdin:
        line = line.rstrip('\r\n')
        mobj = re.match(r'^Output .*Shape: \{([\d,]+)\}', line)
        if mobj:
            dims = list(map(int, mobj.group(1).split(',')))
            print(line)
            continue
        if not dims:
            print(line)
            continue
        numbers = list(map(lambda s: float(s.strip()), line.split(',')))
        offset = 0
        C, H, W = 1, dims[-2], dims[-1]
        if len(dims) >= 3:
            C = dims[-3]
        c = C
        while c:
            h = H
            while h:
                print(' '.join(map(lambda f: '{: .6f}'.format(f), numbers[offset:offset+W])))
                offset += W
                h -= 1
            print()
            c -= 1
        dims = None

if __name__ == '__main__':
    main()
