#!/bin/bash
[[ -d data/MNIST ]] && exit 0
mkdir -p data/MNIST
cd data/MNIST
curl -LO https://github.com/microsoft/NativeKeras/raw/master/Datasets/cntk_mnist/Test-28x28_cntk_text.txt
curl -LO https://github.com/microsoft/NativeKeras/raw/master/Datasets/cntk_mnist/Train-28x28_cntk_text.txt
