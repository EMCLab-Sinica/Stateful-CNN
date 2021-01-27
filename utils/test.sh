set -e
set -x

# preparation
cmake_args=""
run_args=""

if [[ $USE_ARM_CMSIS = 1 ]]; then
    pushd ARM-CMSIS && ./download-extract-cmsis.sh && popd
    cmake_args="$cmake_args -D USE_ARM_CMSIS=ON"
else
    # TODO
    pushd TI-DSPLib && ./download-extract-dsplib.sh && popd
    cmake_args="$cmake_args -D USE_ARM_CMSIS=OFF"
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
if [[ $CONFIG = *japari* ]]; then
    power_cycle=$(awk "BEGIN {print $power_cycle+0.01}")
fi

rm -vf nvm.bin
python transform.py $CONFIG
cmake -B build $cmake_args
make -C build
./build/intermittent-cnn $run_args

# Test intermittent running
if [[ ! $CONFIG = *baseline* ]]; then
    rm -vf nvm.bin

    # somehow a large samples.bin breaks intermittent
    # execution - regenerate samples when needed
    if [[ $CONFIG = *all-samples* ]]; then
        python transform.py ${CONFIG/--all-samples/}
    fi

    cmake_args=${cmake_args/MY_DEBUG=1/MY_DEBUG=2}
    cmake -B build $cmake_args
    make -C build
    python ./run-intermittently.py --rounds $rounds --interval $power_cycle --suffix $LOG_SUFFIX --compress ./build/intermittent-cnn
fi
