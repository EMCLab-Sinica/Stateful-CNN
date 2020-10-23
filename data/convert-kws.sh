#!/bin/sh
python -m tf2onnx.convert --input data/ML-KWS-for-MCU/Pretrained_models/DNN/DNN_S.pb --inputs wav_data:0 --outputs labels_softmax:0 --output data/KWS-DNN_S.onnx
