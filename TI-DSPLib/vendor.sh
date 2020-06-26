#!/bin/sh
set -ex
TI_DSPLIB_PATH="$1"

[ -n "$TI_DSPLIB_PATH" ] || exit 1

rm -rv include source
mkdir -p source/{lea,matrix,vector,utility}
cp -rv "$TI_DSPLIB_PATH"/include .
cp -rv "$TI_DSPLIB_PATH"/source/lea ./source/
cp -v "$TI_DSPLIB_PATH"/source/matrix/msp_matrix_mpy_q15.c ./source/matrix
cp -v "$TI_DSPLIB_PATH"/source/vector/msp_add_q15.c ./source/vector
cp -v "$TI_DSPLIB_PATH"/source/utility/{msp_interleave_q15.c,msp_fill_q15.c} ./source/utility
patch -Np1 -F0 -i ./eliminate-warnings.diff
