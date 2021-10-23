import argparse

import numpy as np
import onnx
import onnxruntime
import onnxruntime.backend as backend
import onnxoptimizer

from configs import configs
from utils import change_batch_size, find_tensor_value_info

def onnxruntime_prepare_model(model):
    return backend.prepare(
        onnxruntime.InferenceSession(model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ))

def onnxruntime_inference(model, images):
    rep = onnxruntime_prepare_model(model)
    return rep.run(np.array(images).astype(np.float32))

def onnxruntime_get_intermediate_tensor(model, images):
    for idx, node in enumerate(model.graph.node):
        # Creating a new model with a given node as the output
        # XXX: Is there a faster way?
        tmp_model = onnx.ModelProto()
        tmp_model.CopyFrom(model)
        new_output = find_tensor_value_info(model, node.output[0])
        tmp_model.graph.output[0].CopyFrom(new_output)
        onnx.checker.check_model(tmp_model)

        rep = onnxruntime_prepare_model(tmp_model)
        outputs = rep.run(np.expand_dims(images[0], 0).astype(np.float32))
        yield new_output.name, outputs

def print_float(val):
    print('%13.6f' % val, end='')

def print_tensor(tensor):
    shape = np.shape(tensor)
    print(f'Original shape: {shape}')
    dimensions = np.shape(shape)[0]
    if dimensions and shape[0] == 1:
        tensor = tensor[0]
        dimensions -= 1
        shape = shape[1:]
    print(f'New shape: {shape}')
    if dimensions == 4:
        N, C, H, W = shape
        assert N == 1
        for c in range(C):
            print(f'Channel {c}')
            for h in range(H):
                for w in range(W):
                    print_float(tensor[0, c, h, w])
                print()
            print()
    elif dimensions == 2:
        H, W = shape
        for h in range(H):
            for w in range(W):
                print_float(tensor[h, w])
            print()
    elif dimensions == 1:
        if shape[0] >= 1024:
            print(f'Skipping very long vector with length {shape[0]}')
            return
        for idx in range(shape[0]):
            print_float(tensor[idx])
            if idx % 16 == 15:
                print()
        print()
    else:
        print(f'Skip: unsupported {dimensions}-dimensional array')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=configs.keys())
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    if args.limit == 0:
        args.limit = None

    config = configs[args.config]

    # https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
    model = onnx.load_model(config['onnx_model'].replace('.onnx', '-opt.onnx'))
    model_data = config['data_loader'](start=0, limit=args.limit)

    new_inputs = list(model_data.input_mapping.values())
    for new_input in new_inputs:
        model.graph.input.append(find_tensor_value_info(model, new_input))
    for node in model.graph.node:
        node.input[:] = [model_data.input_mapping.get(inp, inp) for inp in node.input]
        node.output[:] = [
            output + '_unused' if output in new_inputs else output
            for output in node.output
        ]
    for idx, inp in enumerate(model.graph.input):
        if inp.name in model_data.input_mapping.keys():
            del model.graph.input[idx]

    model = onnxoptimizer.optimize(model, ['eliminate_deadend'])

    change_batch_size(model, 'N')
    onnx.checker.check_model(model)

    # Testing
    if args.limit == 1:
        for layer_name, layer_out in onnxruntime_get_intermediate_tensor(model, model_data.images):
            print(f'Layer: {layer_name}')
            print_tensor(layer_out)
    else:
        correct = 0
        layer_outs = onnxruntime_inference(model, model_data.images)[0]
        for idx, layer_out in enumerate(layer_outs):
            predicted = np.argmax(layer_out)
            if predicted == model_data.labels[idx]:
                print(f'Correct at idx={idx}')
                correct += 1
        total = len(model_data.labels)
        print(f'correct={correct} total={total} rate={correct/total}')

if __name__ == '__main__':
    main()
