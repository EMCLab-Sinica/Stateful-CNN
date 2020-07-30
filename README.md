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
* `git clone -b patched https://github.com/EMCLab-Sinica/ARM-CMSIS_5 ARM-CMSIS`
* `./transform.py mnist`
* `cmake .`
* `make`
* `./intermittent-cnn`

See README in https://github.com/EMCLab-Sinica/stateful-cnn-msp430 for how to use this on MSP430.
