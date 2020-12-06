set -e
set -x

pushd ARM-CMSIS
./create_symlinks.sh "$PWD/../ARM-CMSIS_5/CMSIS"
popd

cat >> /etc/pacman.conf <<EOF
[archlinuxcn]
Server = https://repo.archlinuxcn.org/$arch
SigLevel = Optional
EOF

pacman -Syu --noconfirm
pacman -S --noconfirm --needed base-devel cmake python-numpy python-onnx wget

# preparation
cmake_args=""
MY_DEBUG="1"
run_args=""

if [[ $USE_ARM_CMSIS = 1 ]]; then
    cmake_args="$cmake_args -D USE_ARM_CMSIS=ON"
fi

if [[ $DEBUG_BUILD = 1 ]]; then
    MY_DEBUG="2"
    run_args="$run_args 1"
fi
cmake_args="$cmake_args -D MY_DEBUG=$MY_DEBUG"

if [[ $CONFIG = *mnist* ]]; then
    ./data/download-mnist.sh
fi
if [[ $CONFIG = *cifar10* ]]; then
    ./data/download-cifar10.sh
fi

python transform.py $CONFIG
cmake -B build $cmake_args
make -C build
./build/intermittent-cnn $run_args
