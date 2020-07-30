# dependencies

set -e

pacman -Syu --noconfirm
pacman -S --noconfirm --needed base-devel cmake python-pip
pip install --user numpy onnx

# preparation
transform_args="--all-samples"
run_args=""
if [[ $WITH_PROGRESS_EMBEDDING = 0 ]]; then
    transform_args="$transform_args --without-progress-embedding"
fi

if [[ $DEBUG_BUILD = 1 ]]; then
    sed -i 's/#define MY_NDEBUG//' debug.h
    run_args="$run_args 1"
fi

python transform.py $transform_args "$CONFIG"
cmake -B build
make -C build
./build/intermittent-cnn $run_args
