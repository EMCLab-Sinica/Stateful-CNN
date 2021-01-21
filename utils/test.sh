set -e
set -x

cat >> /etc/pacman.conf <<EOF
[archlinuxcn]
Server = https://repo.archlinuxcn.org/\$arch
SigLevel = Never
EOF

pacman -Syu --noconfirm
pacman -S --noconfirm --needed base-devel cmake python-numpy python-onnx python-tensorflow wget unzip
pacman -U --noconfirm https://build.archlinuxcn.org/~yan12125/python-torchaudio-git-r628.9c484027-1-x86_64.pkg.tar.zst

# preparation
cmake_args=""
run_args=""

if [[ $USE_ARM_CMSIS = 1 ]]; then
    pushd ARM-CMSIS && ./download-extract-cmsis.sh && popd
    cmake_args="$cmake_args -D USE_ARM_CMSIS=ON"
fi

cmake_args="$cmake_args -D MY_DEBUG=1"

rounds=100
power_cycle=0.01
if [[ $CONFIG = *mnist* ]]; then
    ./data/download-mnist.sh
fi
if [[ $CONFIG = *cifar10* ]]; then
    ./data/download-cifar10.sh
    rounds=50
    power_cycle=0.02
fi
if [[ $CONFIG = *kws* ]]; then
    git submodule init
    git submodule update data/ML-KWS-for-MCU
fi

python transform.py $CONFIG
cmake -B build $cmake_args
make -C build
./build/intermittent-cnn $run_args

# Test intermittent running
if [[ ! $CONFIG = *baseline* ]]; then
    rm -vf nvm.bin
    cmake_args=${cmake_args/MY_DEBUG=1/MY_DEBUG=2}
    cmake -B build $cmake_args
    make -C build
    python ./run-intermittently.py --rounds $rounds --interval $power_cycle --compress ./build/intermittent-cnn
fi
