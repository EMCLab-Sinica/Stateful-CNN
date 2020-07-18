# dependencies
pacman -Syu --noconfirm
pacman -S --noconfirm --needed base-devel cmake python-pip
pip install --user numpy onnx

# preparation
args=''
if [[ $WITH_PROGRESS_EMBEDDING = 0 ]]; then
    args='--without-progress-embedding'
fi

python transform.py $args "$CONFIG"
cmake -B build
make -C build
./build/intermittent-cnn
