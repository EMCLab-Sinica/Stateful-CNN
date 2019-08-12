# https://zhuanlan.zhihu.com/p/41255090
import sys

import onnx
from onnx import optimizer

model_path = sys.argv[1]
original_model = onnx.load(model_path)

passes = ['fuse_add_bias_into_conv']

optimized_model = optimizer.optimize(original_model, passes)

onnx.save(optimized_model, model_path.replace('.onnx', '_optimized.onnx'))
