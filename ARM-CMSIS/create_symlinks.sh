#!/bin/bash

set -x
set -e

CMSIS_ROOT="$1"

[[ $CMSIS_ROOT != "" ]] || exit 1

# Keep this list in sync with CMakeLists.txt
sources=(
    DSP/Source/BasicMathFunctions/arm_add_q15.c
    DSP/Source/BasicMathFunctions/arm_offset_q15.c
    DSP/Source/BasicMathFunctions/arm_scale_q15.c
    DSP/Source/MatrixFunctions/arm_mat_init_q15.c
    DSP/Source/MatrixFunctions/arm_mat_mult_fast_q15.c
    DSP/Source/StatisticsFunctions/arm_max_q15.c
    DSP/Source/StatisticsFunctions/arm_min_q15.c
    DSP/Source/SupportFunctions/arm_fill_q15.c
)

mkdir -p Core DSP/Source
ln -sf "$CMSIS_ROOT"/Core/Include Core/
ln -sf "$CMSIS_ROOT"/DSP/Include DSP/
ln -sf "$CMSIS_ROOT"/DSP/PrivateInclude DSP/
for f in ${sources[@]}; do
    dir="$(dirname $f)"
    mkdir -p "$dir"
    ln -sf "$CMSIS_ROOT/$f" "$dir"
done
