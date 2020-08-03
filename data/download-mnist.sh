#!/bin/sh
mkdir -p data/MNIST
cd data/MNIST
wget https://github.com/microsoft/NativeKeras/raw/master/Datasets/cntk_mnist/Test-28x28_cntk_text.txt
wget https://github.com/microsoft/NativeKeras/raw/master/Datasets/cntk_mnist/Train-28x28_cntk_text.txt
