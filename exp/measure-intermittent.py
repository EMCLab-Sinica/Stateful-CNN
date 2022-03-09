import argparse
import datetime
import pathlib
import sys
import time
import warnings

import numpy as np
import serial

TOPDIR = pathlib.Path(__file__).absolute().parents[1]
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
    def report(data, prefix=''):
        with_recharging, without_recharging, power_failures = data
        # Recording timestamps for easier debugging of fluctuating experimental results (ex: comparing results with minicom)
        print(f'[{datetime.datetime.now()}] {prefix}With recharging: {with_recharging:.4f}, without recharging: {without_recharging:.4f}, power failures: {power_failures:.1f}')

    def feed(self, data):
        cmd, arg = data[0], data[1:]
        if cmd == 'I':
            if self.verbose:
                print(data)
            with_recharging = int(arg)/1000
            if with_recharging < self.recharging:
                # FIXME: why this occurs!?
                warnings.warn(f'Incorrect recharging time! {with_recharging} < {self.recharging}')
                self.recharging = 0
                self.power_failures = 0
                return
            without_recharging = with_recharging - self.recharging

            last_one = (with_recharging, without_recharging, self.power_failures)
            self.report(last_one)

            if len(self.last_n) >= 2:
                _, without_recharging_mean, _ = np.mean(self.last_n, axis=0)
                # Sometimes "inference complete" signals are missed (after state table for the last
                # layer is updated and before the signal is sent), resulting in double or higher
                # inference latency - ignore it
                if without_recharging >= 1.5 * without_recharging_mean:
                    warnings.warn('Detected an abnormal value (>= 1.5 avg)')
                    self.recharging = 0
                    self.power_failures = 0
                    return

            self.last_n.append(last_one)
            if len(self.last_n) > self.N:
                self.last_n.pop(0)
            self.report(np.mean(self.last_n, axis=0), prefix=f'Average of last {len(self.last_n)}: ')

            self.recharging = 0
            self.power_failures = 0
        elif cmd == 'R':
            if self.verbose:
                print(data, end=' ', flush=True)
            self.recharging += int(data[1:])/1000
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
