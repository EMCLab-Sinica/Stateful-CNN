# Stateful Neural Networks for Intermittent Systems

<!-- ABOUT THE PROJECT -->
## Overview

This project develops a middleware stack for converting a deployed Neural Network into a Stateful Neural Network (referred to as *Stateful*), which enables the network to indicate the inference progress itself. Our runtime middleware embeds state information into the network such that the computed and preserved output features intrinsically contain progress indicators, avoiding the need to preserve them separately.

We implemented our Stateful design on two Texas Instruments devices, namely the MSP-EXP430FR5994 and MSP-EXP432P401R. The former is a 16-bit MCU with 8KB SRAM and the low-energy vector math accelerator (LEA), and the latter is an ARM-based 32-bit MCU with 64KB SRAM and single instruction multiple data (SIMD) instructions for accelerated computation. An external NVM module (Cypress CY15B104Q serial FRAM) was integrated to both platforms.

Our middleware is built on top of a lightweight inference engine, and contains four key design components which interacts with the inference engine at runtime.


* State embedder: transforms the values of the selected network components to represent inference progress as states, without corrupting the output feature computations.
* State assigner: maintains a state table to determine the specific states to embed, to ensure the last preserved job output in NVM can be correctly identified.
* State clearer: removes the states from the input features to avoid computation corruption.
* Progress seeker: searches for the last preserved job output in NVM using states upon power resumption for correctly resuming the inference process.


<!-- For more technical details, please refer to our paper **TODO**. -->

Demo video: https://www.youtube.com/watch?v=nANmUAO-1Ag

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Directory/File Structure](#directory/file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Setup and Build](#setup-and-build)

## Directory/File Structure

Below is an explanation of the directories/files found in this repo.

* `common/conv.cpp`, `common/fc.cpp`, `common/pooling.cpp`, `common/op_handlers.cpp`: functions implementing various neural network layers.
* `common/cnn_common.*`: a lightweight inference engine, which handles layer-by-layer processing and memory management of input/output feature maps.
* `common/intermittent-cnn.*`: functions implementing the aforementioned key design components of Stateful.
* `common/op_utils.*` : auxiliary functions shared among different layers.
* `common/platform.*`, `common/plat-mcu.*` and `common/plat-pc.*`: high-level wrappers for handling platform-specific peripherals. Notably, `common/plat-pc.*` implements a testbench to evaluate the accuracy of Stateful on a PC.
* `common/my_dsplib.*`: high-level wrappers for accessing different vendor-specific library calls performing accelerated computations.
* `common/counters.*` : helper functions for measuring runtime overhead.
* `dnn-models/`: pre-trained models and python scripts for model training, converting different model formats to ONNX and converting a model into a custom format recognized by the lightweight inference engine.
* `msp430/` and `msp432/`: platform-speicific hardware initialization functions.
* `tools/`: helper functions for various system peripherals (e.g., UART, system clocks and external FRAM).
* `vendor-patches/`: necessary adaptations to vendor-supplied MSP430 and MSP432 libraries, to correctly interface with our Stateful middleware (e.g. pipelined computation and output feature preservation).

## Getting Started

### Prerequisites

Here are basic software and hardware requirements to build Stateful and the lightweight inference engine:

* Python >= 3.7
* Several deep learning Python libraries defined in `requirements.txt`. Those libraries can be installed with `pip3 install -r requirements.txt`.
* [Code composer studio](https://www.ti.com/tool/CCSTUDIO) >= 11.0
* [MSP-EXP430FR5994 LaunchPad](https://www.ti.com/tool/MSP-EXP430FR5994) or [MSP-EXP432P401R LaunchPad](https://www.ti.com/tool/MSP-EXP432P401R)
* For MSP432, [ARM CMSIS Library](https://github.com/ARM-software/CMSIS_5/) 5.7.0 and [MSP432 driverlib 3.21.00.05](https://www.ti.com/tool/MSPDRIVERLIB)
* For MSP430, [MSP DSP Library](https://www.ti.com/tool/MSP-DSPLIB) 1.30.00.02 and [MSP430 driverlib 2.91.13.01](https://www.ti.com/tool/MSPDRIVERLIB)

### Setup and Build

#### Common steps for all platforms

1. Prepare vendor-supplied libraries for hardware-accelerated computation. After downloading and extracting both libraries to `ARM-CMSIS/` and `TI-DSPLib/`, respectively, apply the patches with the following commands:
    ```
    cd ./ARM-CMSIS && patch -Np1 -i ../vendor-patches/ARM-CMSIS.diff
    cd ./TI-DSPLib && patch -Np1 -i ../vendor-patches/TI-DSPLib.diff
    ```
1. Convert the provided pre-trained models with the command `./dnn-models/transform.py --target (msp430|msp432) (--ideal|--hawaii|--japari|--stateful) (cifar10|har|kws)` to specify the target platform, the intermittent inference approach and the model to deploy.

#### Building for MSP430FR5994

1. Download and extract MSP430 driverlib, and copy `driverlib/MSP430FR5xx_6xx` folder into the `msp430/` folder.
1. Import the folder `msp430/` as a project in CCStudio.

#### Building for MSP432P401R

1. Download and extract MSP432 driverlib, and copy `driverlib/MSP432P4xx` folder into the `msp432/` folder.
1. Import the folder `msp432/` as a project in CCStudio.
