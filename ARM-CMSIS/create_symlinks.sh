#!/bin/bash

set -x
set -e

CMSIS_ROOT="$1"

[[ $CMSIS_ROOT != "" ]] || exit 1

mkdir -p Core DSP/Source
ln -sf "$CMSIS_ROOT"/Core/Include Core/
ln -sf "$CMSIS_ROOT"/DSP/Include DSP/
ln -sf "$CMSIS_ROOT"/DSP/PrivateInclude DSP/
for component in BasicMathFunctions MatrixFunctions StatisticsFunctions SupportFunctions; do
    ln -sf "$CMSIS_ROOT"/DSP/Source/$component DSP/Source/
done
