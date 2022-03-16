python3 -m pip install --user --upgrade pip setuptools
python3 -m pip install -r requirements.txt

python3 transform.py --target msp430 --stateful har
cmake -B build
make -C build
./build/intermittent-cnn
