## Building on Linux

# Requirements

* CMake >= 2.8
* Python >= 3.6

# Install needed Python packages

* `pip3 install numpy onnx`

If you are using Python 3.6, install one more Python package:

* `pip3 install dataclasses`

# Preparation steps for all platforms

* `git submodule update --init --recursive`
* `git clone -b patched https://github.com/EMCLab-Sinica/ARM-CMSIS_5 ../ARM-CMSIS_5 && cd ARM-CMSIS && ./create_symlinks.sh "$PWD"/../ARM-CMSIS_5/CMSIS` if you want to use ARM CMSIS DSP library
* `git clone ssh://git@github.com/EMCLab-Sinica/DSPLib TI-DSPLib` if you want to use TI DSPLib
* `./data/download-mnist.sh` and `./data/download-cifar10.sh` to download MNIST and CIFAR-10 datasets
* `./transform.py (mnist|cifar10)`

# Building for POSIX-compliant systems

* `cmake . -D USE_ARM_CMSIS=(ON|OFF)`
* `make`
* `./intermittent-cnn`

# Building for MSP430FR5994

In the `msp430` folder, run:

* `git clone ssh://git@github.com/EMCLab-Sinica/Tools.git Tools`
* `git clone ssh://git@github.com/EMCLab-Sinica/driverlib-msp430.git driverlib`

And then import this project into CCStudio.

# Building for MSP432P401R

In the `msp432` folder, run:

* `git clone ssh://git@github.com/EMCLab-Sinica/Tools.git Tools`
* `git clone ssh://git@github.com/EMCLab-Sinica/driverlib-msp432.git driverlib`

And then import this project into CCStudio.
