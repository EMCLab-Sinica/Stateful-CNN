# Intermittent CNN

## Prerequisites

* DSPLib for MSP. Available at http://www.ti.com/tool/MSP-DSPLIB (a TI account needed)

## Build the project

```
$ make
```

## Parse an ONNX model using C
```sh
$ ./parse_model < model.onnx
```

## Transform the model

Change the nodes in a model from string-based to integer-based.

```sh
$ python transform.py model.onnx
```

This command will create `input_tables.bin`.

## Load transformed model

```
$ ./main
```
