# dependencies
pacman -Syu --noconfirm
pacman -S --noconfirm --needed base-devel cmake python-pip
pip install --user numpy onnx

# preparation
if [[ $WITH_PROGRESS_EMBEDDING = 0 ]]; then
    sed -i 's/#define WITH_PROGRESS_EMBEDDING//' cnn_common.h
fi

python transform.py "$CONFIG"
cmake -B build
make -C build
./build/intermittent-cnn
