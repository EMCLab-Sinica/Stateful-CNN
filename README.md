## Building on Linux

# Requirements

* CMake >= 2.8
* Python >= 3.6

# Install needed Python packages

* `pip3 install numpy onnx==1.8.0`

If you are using Python 3.6, install one more Python package:

* `pip3 install dataclasses`

# Preparation steps for all platforms

* `git submodule update --init --recursive`
* `git submodule update --remote --merge`
* `pushd ARM-CMSIS && ./download-extract-cmsis.sh && popd` if you want to use ARM CMSIS DSP library
* `pushd TI-DSPLib && ./download-extract-dsplib.sh && popd` if you want to use TI DSPLib
* `./data/download-mnist.sh` and `./data/download-cifar10.sh` to download MNIST and CIFAR-10 datasets
* `./transform.py --target (msp430|msp432) (--baseline|--hawaii|--japari|--stateful) --batch-size 1 (mnist|cifar10|kws)`

# Building for POSIX-compliant systems

* `cmake .`
* `make`
* `./intermittent-cnn`

# Building for MSP430FR5994

In the `msp430` folder, run:

* `git clone ssh://git@github.com/EMCLab-Sinica/driverlib-msp430.git driverlib`

And then import this project into CCStudio.

# Building for MSP432P401R

In the `msp432` folder, run:

* `git clone ssh://git@github.com/EMCLab-Sinica/driverlib-msp432.git driverlib`

And then import this project into CCStudio.
