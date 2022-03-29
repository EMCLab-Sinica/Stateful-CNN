#!/bin/sh
git -C ARM-CMSIS diff origin/main..HEAD > vendor-patches/ARM-CMSIS.diff
git -C TI-DSPLib diff origin/unmodified..HEAD > vendor-patches/TI-DSPLib.diff
