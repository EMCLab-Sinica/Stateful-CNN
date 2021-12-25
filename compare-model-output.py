import argparse

import numpy as np

from utils import import_model_output_pb2

def get_tensor(layer_out):
    arr = np.array(layer_out.value)
    if not len(arr):
        return []
    dims = np.array(layer_out.dims)
    return np.reshape(arr, dims[dims!=0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--topk', type=int, required=True)
    args = parser.parse_args()

    baseline_data = {}
    model_output = import_model_output_pb2().ModelOutput()
    with open(args.baseline, 'rb') as f:
        model_output.ParseFromString(f.read())
    for layer_out in model_output.layer_out:
        baseline_data[layer_out.name] = get_tensor(layer_out)

    with open(args.target, 'rb') as f:
        model_output.ParseFromString(f.read())
    for layer_out in model_output.layer_out:
        name = layer_out.name
        if name.endswith('_before_merge'):
            continue

        print(f'Layer output {name}')
        max_num = np.max(np.abs(baseline_data[name]))
        cur_baseline_data = baseline_data[name]
        cur_target_data = get_tensor(layer_out)
        errors = np.abs(cur_baseline_data - cur_target_data) / max_num

        # Sort on negative values to get indices for decreasing values
        # https://www.kite.com/python/answers/how-to-use-numpy-argsort-in-descending-order-in-python
        error_indices = np.unravel_index(np.argsort(-errors, axis=None), errors.shape)
        top_error_indices = np.array(error_indices)[:, :args.topk]
        for index_idx in range(top_error_indices.shape[1]):
            # indices should be a tuple
            value_idx = tuple(top_error_indices[:, index_idx])
            value_idx_str = '(' + ', '.join(f'{idx:3d}' for idx in value_idx) + ')'
            # :e => scientific notation
            print(', '.join([f'index={value_idx_str}',
                             f'baseline={cur_baseline_data[value_idx]:+e}',
                             f'target={cur_target_data[value_idx]:+e}',
                             f'error={errors[value_idx]:e}']))

if __name__ == '__main__':
    main()
