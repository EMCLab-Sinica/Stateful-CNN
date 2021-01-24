#!/bin/bash
cd data
[[ -d cifar-10-batches-py ]] && exit 0
curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
bsdtar xf cifar-10-python.tar.gz
rm -v cifar-10-python.tar.gz
