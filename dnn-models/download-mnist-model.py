import urllib.request

import onnx
import onnx.helper

def main():
    URL = 'https://github.com/onnx/models/blob/main/vision/classification/mnist/model/mnist-8.onnx?raw=true'
    with urllib.request.urlopen(URL) as req:
        model = onnx.load_model_from_string(req.read())
    for initializer in model.graph.initializer:
        if initializer.name == 'Pooling160_Output_0_reshape0_shape':
            initializer.int64_data[:] = [-1, 256]
    onnx.save_model(model, 'dnn-models/mnist-8.onnx')

if __name__ == '__main__':
    main()
