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
        C, H, W = 1, dims[-2], dims[-1]
        if len(dims) >= 3:
            C = dims[-3]
        # Somehow dumped values use format NHWC
        for c in range(C):
            print(f'Channel {c}')
            for h in range(H):
                for w in range(W):
                    print('{: 13.6f}'.format(numbers[h * W * C + w * C + c]), end='')
                print()
            print()
        dims = None

if __name__ == '__main__':
    main()
