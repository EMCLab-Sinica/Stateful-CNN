## Building on Linux

# Requirements

* CMake >= 2.8
* Python >= 3.6

# Install needed Python packages

* `pip3 install numpy onnx`

If you are using Python 3.6, install one more Python package:

* `pip3 install dataclasses`

# Steps

* `git submodule update --init --recursive`
* `git clone -b patched https://github.com/EMCLab-Sinica/ARM-CMSIS_5 ../ARM-CMSIS_5 && ln -s ../ARM-CMSIS_5/CMSIS ARM-CMSIS` if you want to use ARM CMSIS DSP library
* `git clone ssh://git@github.com/EMCLab-Sinica/DSPLib TI-DSPLib` if you want to use TI DSPLib
* `./data/download-mnist.sh` and `./data/download-cifar10.sh` to download MNIST and CIFAR-10 datasets
* `./transform.py (mnist|cifar10)`
* `cmake . -D USE_ARM_CMSIS=(ON|OFF)`
* `make`
* `./intermittent-cnn`

See README in https://github.com/EMCLab-Sinica/stateful-cnn-msp430 for how to use this on MSP430.
