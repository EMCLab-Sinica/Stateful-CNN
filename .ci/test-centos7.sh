# dependencies
yum -y update
yum -y install gcc-c++ make cmake python36
python3.6 -m ensurepip --user
pip3 install --user dataclasses numpy onnx

python3.6 transform.py mnist
cmake -B build
make -C build
./build/intermittent-cnn
