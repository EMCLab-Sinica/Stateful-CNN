#!/bin/sh

# Somehow lcov cannot find gcno files if they are not in the same directory as gcda files
cp build/*.gcno "$1"
lcov --capture --output-file "$1.info" --directory "$1"
genhtml "$1.info" --output-directory gcov/result
