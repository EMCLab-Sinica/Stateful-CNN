import onnx
import tensorflow as tf
import tf2onnx

# Simplied from tf2onnx/convert.py and added code for shape information
def main():
    graph_def = tf.compat.v1.GraphDef()
    with open('data/ML-KWS-for-MCU/Pretrained_models/DNN/DNN_S.pb', 'rb') as f:
        content = f.read()
        graph_def.ParseFromString(content)

    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')
    with tf.compat.v1.Session(graph=tf_graph):
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            tf_graph, input_names=['wav_data:0'], output_names=['labels_softmax:0'])

    # Eliminate the Cast node
    onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)

    # Borrow shape information from Tensorflow as some operators (e.g., Mfcc) are
    # not available in ONNX and thus shape inference does not work.
    # XXX: The following lines are also pushed to my fork. However, I didn't
    # submit a pull request as it breaks existing tests.
    # https://github.com/yan12125/tensorflow-onnx/commit/6263c68b94d8c9e5573583d94b56587b3aac8fd4
    all_outputs = set()
    for op in onnx_graph.get_nodes():
        all_outputs.update(op.output)
    value_infos = onnx_graph.make_onnx_graph_io(all_outputs)

    model_proto = onnx_graph.make_model('KWS-DNN_S')

    model_proto.graph.value_info.extend(value_infos)

    onnx.save_model(model_proto, 'data/KWS-DNN_S.onnx')

if __name__ == "__main__":
    main()
