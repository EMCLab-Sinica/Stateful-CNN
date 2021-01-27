#!/bin/bash

set -x
set -e

DSPLIB_VERSION=1_30_00_02
DSPLIB_DIR=DSPLib_$DSPLIB_VERSION
DSPLIB_ARCHIVE=${DSPLIB_DIR}_linux.zip

cleanup() {
    rm -rvf $DSPLIB_DIR
    rm -vf $DSPLIB_ARCHIVE
}

trap cleanup EXIT SIGINT

cleanup

[[ -d source ]] && exit 0

curl -LO "https://software-dl.ti.com/msp430/msp430_public_sw/mcu/msp430/DSPLib/latest/exports/$DSPLIB_ARCHIVE"
bsdtar xf $DSPLIB_ARCHIVE

find $DSPLIB_DIR -type f -print0 | xargs -0 chmod -x
mv $DSPLIB_DIR/{include,source} .
