set -e
set -x

# preparation
cmake_args=""

if [[ $LOG_SUFFIX = *baseline_b1_cmsis ]]; then
    model=${LOG_SUFFIX/_*/}
    python original_model_run.py $model --compare-configs
fi

cmake_args="$cmake_args -D MY_DEBUG=1"

rounds=100
power_cycle=0.01
if [[ $CONFIG = *cifar10* ]]; then
    rounds=50
    power_cycle=0.02
fi
if [[ $CONFIG = *japari* ]]; then
    power_cycle=$(awk "BEGIN {print $power_cycle+0.01}")
fi

rm -vf nvm.bin
python transform.py $CONFIG
cmake -B build $cmake_args
make -C build
./build/intermittent-cnn

# Test intermittent running
if [[ ! $CONFIG = *baseline* ]]; then
    rm -vf nvm.bin

    # somehow a large samples.bin breaks intermittent
    # execution - regenerate samples when needed
    if [[ $CONFIG = *all-samples* ]]; then
        python transform.py ${CONFIG/--all-samples/}
    fi

    cmake_args=${cmake_args/MY_DEBUG=1/MY_DEBUG=3}
    cmake -B build $cmake_args
    make -C build
    TMPDIR=/var/tmp python ./run-intermittently.py --rounds $rounds --interval $power_cycle --suffix $LOG_SUFFIX --compress ./build/intermittent-cnn
fi
