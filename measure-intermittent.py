import argparse
import sys
import time
import warnings

import numpy as np
import serial

from utils import TOPDIR

sys.path.append(str(TOPDIR / 'Tools'))

from emclab import uart_utils

class DataHandler:
    N = 100

    def __init__(self, verbose):
        self.recharging = 0
        self.power_failures = 0
        self.last_n = []
        self.verbose = verbose

    @staticmethod
    def report(data):
        with_recharging, without_recharging, power_failures = data
        print(f'With recharging: {with_recharging/1000:.4f}, without recharging: {without_recharging/1000:.4f}, power failures: {power_failures:.1f}')

    def feed(self, data):
        cmd, arg = data[0], data[1:]
        if cmd == 'I':
            if self.verbose:
                print(data)
            with_recharging = int(arg)
            if with_recharging < self.recharging:
                # FIXME: why this occurs!?
                warnings.warn(f'Incorrect recharging time! {with_recharging} < {self.recharging}')
                self.recharging = 0
                self.power_failures = 0
                return
            without_recharging = with_recharging - self.recharging
            last_one = (with_recharging, without_recharging, self.power_failures)
            self.report(last_one)

            self.last_n.append(last_one)
            if len(self.last_n) > self.N:
                self.last_n.pop(0)
            print(f'Average of last {len(self.last_n)}: ', end='')
            self.report(np.mean(self.last_n, axis=0))

            self.recharging = 0
            self.power_failures = 0
        elif cmd == 'R':
            if self.verbose:
                print(data, end=' ', flush=True)
            self.recharging += int(data[1:])
            self.power_failures += 1

def read_from_port(ser, verbose):
    ser.reset_input_buffer()
    ser.flushInput()

    handler = DataHandler(verbose)

    reading = ''
    while True:
        reading += ser.readline().decode()
        if '\n' not in reading:
            continue
        line, reading = reading.split('\n', maxsplit=1)
        handler.feed(line.rstrip('\r'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    ports = [port for port in uart_utils.find_uart() if 'XDS110' in port]
    assert len(ports) == 1, 'No or mutiple XDS110 UART ports detected!'
    baud = 9600
    serial_port = serial.Serial(ports[0], baud, timeout=0)
    serial_port.reset_input_buffer()
    serial_port.flushInput()
    time.sleep(1)
    serial_port.reset_input_buffer()
    serial_port.reset_output_buffer()
    serial_port.flushInput()
    # Avoid busy-waiting
    serial_port.timeout = 1
    try:
        read_from_port(serial_port, verbose=args.verbose)
    finally:
        serial_port.close()

if __name__ == '__main__':
    main()
