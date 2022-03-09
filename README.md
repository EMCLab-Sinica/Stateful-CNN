# Stateful Neural Networks for Intermittent Systems

<!-- ABOUT THE PROJECT -->
## Overview

This project develops an inference engine that turns a common neural network into a Stateful neural network (Stateful NN), which inherently contains the inference progress as *states* and is suitable to be deployed on intermittently-powered systems.
The inference engine includes four key components besides common techniques used in existing inference engines:

* State embedder: transforms values of the selected network components to *represent inference progress* as states.
* State assigner: assigns specific states to embed to ensure the job output currently on NVM and the one overwritting the same location have different states and can *be distinctively identified*.
* State clearer: ensures tile computation results are *not corrupted* by removing inference states possibly contained in input features.
* Progress seeker: searches for the last preserved job output in NVM using states upon power resumption for *correctly resuming the inference process*.

The Stateful NN inference engine is deployed to two platforms: TI MSP-EXP430FR5994 Launchpad with 16-bit MCU, 8KB SRAM and the low-energy accelerator (LEA) for accelerated computation, as well as MSP-EXP432P401R Launchpad with ARM-based 32-bit MCU, 64KB SRAM and single instruction multiple data (SIMD) instructions for accelerated computation. The inference engine can also be deployed as a simulator for testing inference correctness under power failures with PC, where the NVM is simulated by a file and all computation operations are carried out by CPU without acceleration.
For more technical details, please refer to our paper **TODO**.

Demo video: **TODO**

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup and Build](#setup-and-build)

## Directory/File Structure

Below is an explanation of the directories/files found in this repo.

```
├── ARM-CMSIS
│   └── ...
├── configs.py
├── common
│   ├── cnn_common.*
│   ├── conv.cpp
│   ├── counters.*
│   ├── gemm.cpp
│   ├── intermittent-cnn.*
│   ├── my_debug.*
│   ├── my_dsplib.*
│   ├── op_handlers.cpp
│   ├── op_utils.*
│   ├── plat-pc.*
│   ├── plat-mcu.*
│   ├── platform.*
│   └── pooling.cpp
├── exp
│   └── ...
├── models
│   └── ...
├── msp430
├── msp432
├── patches
├── TI-DSPLib
│   └── ...
├── Tools
│   └── ...
├── transform.py
└── utils.py
```

* `ARM-CMSIS/` and `TI-DSPLib/` are vendor-supplied libraries for hardware-accelerated computation on ARM-based and MSP430-based platforms, respectively. `patches` includes changes to those vendor supplied libraries.
* `msp430/` and `msp432/` include platform-dependent hardware initialization routines.
* `Tools/` includes helper functions for various hardware components, including UART, system clocks and external FRAM.
* `common/conv.cpp`, `common/gemm.cpp`, `common/pooling.cpp`, `common/op_handlers.cpp` include functions for various neural network layers.
* `common/cnn_common.*` include a typical embedded inference engine, which handles layer-by-layer processing and memory management of input/output feature maps.
* `common/intermittent-cnn.*` include functions for enabling Stateful neural networks, including the progress seeker and auxiliary routines for managing states.
* `common/op_utils.*` include auxiliary routines shared among functions for processing layers
* `common/plat-pc.*`, `common/plat-mcu.*` and `platform.*` abstract platform differences for building a portable inference engine. Notably, `common/plat-pc.*` implements the simulator running on PC.
* `common/my_dsplib.*` abstract differences between hardware for accelerated computation.
* `common/my_debug.*` and `common/counters.*` include helper functions for developing inference engines and measuring the performance.
* `exp` includes scripts for measuring model accuracy and inference latency in experiments.
* `models` includes training scripts, pre-trained models used in experiments and scripts for converting different model formats to ONNX.
* `transform.py` is responsible for transforming an off-the-shelf deep learning model into a custom format recognized by the Stateful NN inference engine, and `configs.py` includes model parameters used during transformation.
* `utils.py` includes Python functions shared by `transform.py` and scripts under `exp/`.

## Getting Started

### Prerequisites

Here are basic software and hardware requirements to build the Stateful NN inference engine:

* A Unix-like operating system. Currently Windows is not supported.
* Python >= 3.7
* Several deep learning Python libraries defined in `requirements.txt`. Those libraries can be installed with `pip3 install -r requirements.txt`.
* [Code composer studio](https://www.ti.com/tool/CCSTUDIO) >= 11.0
* [MSP-EXP430FR5994 LaunchPad](https://www.ti.com/tool/MSP-EXP430FR5994)
* [MSP-EXP432P401R LaunchPad](https://www.ti.com/tool/MSP-EXP432P401R)
* [ARM CMSIS Library](https://github.com/ARM-software/CMSIS_5/) 5.7.0
* [MSP DSP Library](https://www.ti.com/tool/MSP-DSPLIB)

To build the simulator, basic toolchain for C/C++ development is also necessary:

* CMake >= 2.8.12
* A modern compiler (gcc or clang) supporting C++ 14

### Setup and Build

#### Common steps for all platforms

1. Clone `Tools/` from https://github.com/EMCLab-Sinica/Tools
1. Prepare libraries for hardware-accelerated computation. We modified ARM CMSIS Library and MSP DSP Library to support pipelined computation and data preservation. After downloading and extracting both libraries to `ARM-CMSIS/` and `TI-DSPLib/`, respecively, applying patches with the following commands:
    ```
    cd ./ARM-CMSIS && patch -Np1 -i ../patches/ARM-CMSIS.diff
    cd ./TI-DSPLib && patch -Np1 -i ../patches/TI-DSPLib.diff
    ```
1. Convert pre-trained models with the command `./transform.py --target (msp430|msp432) (--ideal|--hawaii|--japari|--stateful) (cifar10|har|kws)` to specify the target platform, the intermittency management approach and the model to deploy.

#### Building the simulator running on PC

After transforming the model with `transform.py`, the simulator can be built with `cmake -B build -S .` and `make -C build`. The built program can be run with `./build/intermittent-cnn`.

#### Building for MSP430FR5994

1. Download and extract [MSP430 driverlib 2.91.13.01](https://www.ti.com/tool/MSPDRIVERLIB), and copy `driverlib/MSP430FR5xx_6xx` folder into the `msp430/` folder.
1. Import the folder `msp430/` as a project in CCStudio.

#### Building for MSP432P401R

1. Download and extract [MSP432 driverlib 3.21.00.05](https://www.ti.com/tool/MSPDRIVERLIB), and copy `driverlib/MSP432P4xx` folder into the `msp432/` folder.
1. Import the folder `msp432/` as a project in CCStudio.
