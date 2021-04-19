#!/bin/bash

set -x
set -e

CMSIS_VERSION=5.7.0
CMSIS_ARCHIVE="ARM.CMSIS.$CMSIS_VERSION.pack"

cleanup() {
    rm -rvf CMSIS
    rm -vf $CMSIS_ARCHIVE
}

trap cleanup EXIT SIGINT

cleanup

[[ -d DSP ]] && exit 0

curl -LO "https://github.com/ARM-software/CMSIS_5/releases/download/$CMSIS_VERSION/$CMSIS_ARCHIVE"
bsdtar xf $CMSIS_ARCHIVE 'CMSIS/Core/*' 'CMSIS/DSP/*'

CMSIS_ROOT="$PWD/CMSIS"

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
cp -vrf "$CMSIS_ROOT"/Core/Include Core/
cp -vrf "$CMSIS_ROOT"/DSP/Include DSP/
cp -vrf "$CMSIS_ROOT"/DSP/PrivateInclude DSP/
for f in ${sources[@]}; do
    dir="$(dirname $f)"
    mkdir -p "$dir"
    cp -vf "$CMSIS_ROOT/$f" "$dir"
done

# CMSIS sources use CR/LF, and CR/LF in the patch is converted to CR by git
dos2unix DSP/Include/arm_math.h DSP/Source/MatrixFunctions/arm_mat_mult_fast_q15.c

patch -Np2 -i pipeline-dma.diff
