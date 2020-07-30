# dependencies

set -e

# System already up-to-date after install-git.sh
pacman -S --noconfirm --needed base-devel cmake python-pip
pip install --user numpy onnx

# preparation
transform_args="--all-samples"
cmake_args=""
run_args=""
if [[ $WITH_PROGRESS_EMBEDDING = 0 ]]; then
    transform_args="$transform_args --without-progress-embedding"
fi

if [[ $USE_ARM_CMSIS = 1 ]]; then
    cmake_args="$cmake_args -D USE_ARM_CMSIS=ON"
fi

if [[ $DEBUG_BUILD = 1 ]]; then
    sed -i 's/#define MY_NDEBUG//' debug.h
    run_args="$run_args 1"
fi

python transform.py $transform_args "$CONFIG"
cmake -B build $cmake_args
make -C build
./build/intermittent-cnn $run_args
