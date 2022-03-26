import numpy as np
import onnx
import onnx.helper
import tensorflow as tf
import tf2onnx

from tensorflow.core.framework.attr_value_pb2 import AttrValue
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

def replace_placeholder_with_constant(graph_def, placeholder_name, my_value):
    # Modified from https://stackoverflow.com/a/56296195
    for node in graph_def.node:
        if node.name != placeholder_name:
            continue

        # Make graph node
        tensor_content = my_value.tobytes()
        dt = tf.as_dtype(my_value.dtype).as_datatype_enum
        tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=s) for s in my_value.shape])
        tensor_proto = TensorProto(tensor_content=tensor_content,
                                   tensor_shape=tensor_shape,
                                   dtype=dt)
        node.CopyFrom(NodeDef(name=node.name, op='Const',
                              attr={'value': AttrValue(tensor=tensor_proto),
                                    'dtype': AttrValue(type=dt)}))

def main():
    graph_def = tf.compat.v1.GraphDef()
    with open('dnn-models/deep-learning-HAR/HAR-CNN.pb', 'rb') as f:
        content = f.read()
        graph_def.ParseFromString(content)

    replace_placeholder_with_constant(graph_def, 'keep', np.array(1.0, dtype=np.float32))

    # Not using newer tf2onnx.convert.from_graph_def() API as that seems not including steps
    # on recognizing the dropout structure
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(graph_def, name='')

        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            tf_graph, input_names=['inputs:0'], output_names=['dense/BiasAdd:0'])

    # Eliminate the Cast node
    onnx_graph = tf2onnx.optimizer.optimize_graph(onnx_graph)

    model_proto = onnx_graph.make_model('HAR-CNN')

    model_proto = onnx.shape_inference.infer_shapes(model_proto)

    onnx.save_model(model_proto, 'dnn-models/HAR-CNN.onnx')

if __name__ == "__main__":
    main()
