# dependencies
pacman -Syu --noconfirm
pacman -S --noconfirm --needed base-devel cmake python-pip
pip install --user numpy onnx

# preparation
args="--all-samples"
if [[ $WITH_PROGRESS_EMBEDDING = 0 ]]; then
    args="$args --without-progress-embedding"
fi

if [[ $DEBUG_BUILD = 1 ]]; then
    sed -i 's/#define MY_NDEBUG//' debug.h
fi

python transform.py $args "$CONFIG"
cmake -B build
make -C build
./build/intermittent-cnn
