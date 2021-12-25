import argparse

import numpy as np
import onnx

from configs import configs
from utils import dynamic_shape_inference, onnxruntime_prepare_model, onnxruntime_get_intermediate_tensor, load_model, import_model_output_pb2

def print_float(val):
    print('%13.6f' % val, end='')

def print_tensor(tensor):
    shape = np.shape(tensor)
    print(f'Shape: {shape}')
    dimensions = np.shape(shape)[0]
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
    if dimensions >= 1 and np.prod(shape) != 0:
        print(f'Max={np.max(tensor)}, min={np.min(tensor)}')

def prepare_model_and_data(config, limit):
    model = load_model(config)
    model_data = config['data_loader'](start=0, limit=limit)

    dynamic_shape_inference(model, config['sample_size'])
    onnx.checker.check_model(model)

    return model, model_data

def run_model(model, model_data, limit, verbose=True, save_file=None):
    # Testing
    if limit == 1:
        last_layer_out = None
        if verbose:
            print('Input')
            print_tensor(model_data.images)
        if save_file:
            model_output_pb2 = import_model_output_pb2()
            model_output = model_output_pb2.ModelOutput()
        for layer_name, op_type, layer_out in onnxruntime_get_intermediate_tensor(model, model_data.images[0:1]):
            if verbose:
                print(f'{op_type} layer: {layer_name}')
                print_tensor(layer_out)
            if save_file:
                layer_out_obj = model_output_pb2.LayerOutput()
                layer_out_obj.name = layer_name
                layer_out_obj.dims.extend(layer_out.shape)
                if layer_out.shape:
                    linear_shape = [np.prod(layer_out.shape)]
                    layer_out_obj.value.extend(np.reshape(layer_out, linear_shape))
                else:
                    # zero-dimension tensor -> scalar
                    layer_out_obj.value.append(layer_out)
                model_output.layer_out.append(layer_out_obj)
            # Softmax is not implemented yet - return the layer before Softmax
            if op_type != 'Softmax':
                last_layer_out = layer_out
        if save_file:
            with open(save_file, 'wb') as f:
                f.write(model_output.SerializeToString())
        return last_layer_out
    else:
        correct = 0
        layer_outs = onnxruntime_prepare_model(model).run(model_data.images)[0]
        for idx, layer_out in enumerate(layer_outs):
            predicted = np.argmax(layer_out)
            if predicted == model_data.labels[idx]:
                if verbose:
                    print(f'Correct at idx={idx}')
                correct += 1
        total = len(model_data.labels)
        accuracy = correct/total
        if verbose:
            print(f'correct={correct} total={total} rate={accuracy}')
        return accuracy

def compare_configs(config, model, model_data):
    last_layer_out = run_model(model, model_data, limit=1, verbose=False)
    recorded_last_layer_out = config['first_sample_outputs']
    if not np.allclose(last_layer_out, recorded_last_layer_out):
        raise Exception(f'Computed outputs are different! {last_layer_out} != {recorded_last_layer_out}')

    accuracy = run_model(model, model_data, limit=None, verbose=False)
    recorded_accuracy = config['fp32_accuracy']
    if not np.isclose(accuracy, recorded_accuracy):
        raise Exception(f'Computed accuracies are different! {accuracy} != {recorded_accuracy}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', choices=configs.keys())
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--compare-configs', action='store_true')
    parser.add_argument('--save-file')
    args = parser.parse_args()

    if args.limit == 0:
        args.limit = None

    config = configs[args.config]
    model, model_data = prepare_model_and_data(config, args.limit)
    if args.compare_configs:
        compare_configs(config, model, model_data)
    else:
        run_model(model, model_data, args.limit,
                  verbose=not args.save_file, save_file=args.save_file)

if __name__ == '__main__':
    main()
