import pathlib
import sys

import onnx
import onnx.helper
import tensorflow as tf
import tf2onnx

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from utils import kws_dnn_model, find_tensor_value_info, remap_inputs
from configs import configs

# Simplied from tf2onnx/convert.py and added code for shape information
def main():
    graph_def = tf.compat.v1.GraphDef()
    with open(kws_dnn_model(), 'rb') as f:
        content = f.read()
        graph_def.ParseFromString(content)

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.compat.v1.Session(graph=tf_graph):
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            tf_graph, input_names=['wav_data:0'], output_names=['labels_softmax:0'])

    # Eliminate the Cast node
    onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)

    model_proto = onnx_graph.make_model('KWS-DNN_S')

    model_proto = remap_inputs(model_proto, {'wav_data:0': 'Mfcc:0'})

    input_value_info = find_tensor_value_info(model_proto, 'Mfcc:0')
    input_value_info.CopyFrom(onnx.helper.make_tensor_value_info('Mfcc:0', onnx.TensorProto.FLOAT, [1] + configs['kws']['sample_size']))

    onnx.save_model(model_proto, 'models/KWS-DNN_S.onnx')

if __name__ == "__main__":
    main()
