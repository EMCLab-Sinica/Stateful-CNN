#!/bin/bash
set -ex
# Download https://github.com/ARM-software/CMSIS_5/releases/download/5.6.0/ARM.CMSIS.5.6.0.pack,
# extract it and use the path here
ARM_CMSIS_PATH="$1"

[ -n "$ARM_CMSIS_PATH" ] || exit 1

rm -rv DSP Include
mkdir -p Include DSP/Source/MatrixFunctions
cp -v "$ARM_CMSIS_PATH"/CMSIS/Include/cmsis_{compiler,gcc}.h Include/
cp -rv "$ARM_CMSIS_PATH"/CMSIS/DSP/Include DSP/
cp -v "$ARM_CMSIS_PATH"/CMSIS/DSP/Source/MatrixFunctions/{arm_mat_init_q15.c,arm_mat_mult_fast_q15.c} DSP/Source/MatrixFunctions
patch -Np0 -i ./arm_mat_mult_fast_q15_skip_transpose.diff
