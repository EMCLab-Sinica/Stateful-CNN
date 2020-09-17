# dependencies

set -e

pushd ARM-CMSIS
./create_symlinks.sh "$PWD/../ARM-CMSIS_5/CMSIS"
popd
pacman -Syu --noconfirm
pacman -S --noconfirm --needed base-devel cmake python-pip wget
pip install --user numpy onnx

# preparation
transform_args="--all-samples"
cmake_args=""
MY_DEBUG="1"
run_args=""
if [[ $WITH_PROGRESS_EMBEDDING = 0 ]]; then
    transform_args="$transform_args --without-progress-embedding"
fi

if [[ $USE_ARM_CMSIS = 1 ]]; then
    cmake_args="$cmake_args -D USE_ARM_CMSIS=ON"
fi

if [[ $DEBUG_BUILD = 1 ]]; then
    MY_DEBUG="2"
    run_args="$run_args 1"
fi
cmake_args="$cmake_args -D MY_DEBUG=$MY_DEBUG"

if [[ $CONFIG = mnist ]]; then
    ./data/download-mnist.sh
fi
if [[ $CONFIG = cifar10 ]]; then
    ./data/download-cifar10.sh
fi

python transform.py $transform_args "$CONFIG"
cmake -B build $cmake_args
make -C build
./build/intermittent-cnn $run_args
