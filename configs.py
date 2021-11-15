from utils import (
    load_data_mnist,
    load_data_cifar10,
    load_data_google_speech,
)

# intermediate_values_size should < 65536, or TI's compiler gets confused
configs = {
    'mnist': {
        # https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx
        'onnx_model': 'data/mnist-8.onnx',
        'scale': 2,
        'input_scale': 4,
        'num_slots': 2,
        'intermediate_values_size': 26000,
        'data_loader': load_data_mnist,
        'n_all_samples': 10000,
        # multiply by 2 for Q15
        'sample_size': 2 * 28 * 28,
        'op_filters': 4,
        'first_sample_outputs': [ -1.247997, 0.624493, 8.609308, 9.392411, -13.685033, -6.018567, -23.386677, 28.214134, -6.762523, 3.924627 ],
        'fp32_accuracy': 0.9890,
    },
    'cifar10': {
        'onnx_model': 'data/squeezenet_cifar10.onnx',
        'scale': 2,
        'input_scale': 4,
        'num_slots': 3,
        'intermediate_values_size': 65000,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'sample_size': 2 * 32 * 32 * 3,
        'op_filters': 4,
        'first_sample_outputs': [ 4.895500, 4.331344, 4.631835, 11.602396, 4.454658, 10.819544, 5.423588, 6.451203, 5.806091, 5.272837 ],
        'fp32_accuracy': 0.7704,
    },
    'kws': {
        'onnx_model': 'data/KWS-DNN_S.onnx',
        'scale': 1,
        'input_scale': 120,
        'num_slots': 2,
        'intermediate_values_size': 20000,
        'data_loader': load_data_google_speech,
        'n_all_samples': 4890,
        'sample_size': 2 * 25 * 10,  # MFCC gives 25x10 tensors
        'op_filters': 4,
        'first_sample_outputs': [ -29.228327, 5.429047, 22.146973, 3.142066, -10.448060, -9.513299, 15.832925, -4.655487, -14.588447, -1.577156, -5.864228, -6.609077 ],
        'fp32_accuracy': 0.7983,
    },
}

