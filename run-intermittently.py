from subprocess import Popen, PIPE, TimeoutExpired
import signal
import sys

def main():
    while True:
        with Popen(['./out/intermittent-cnn'], stdout=PIPE) as proc:
            try:
                proc.wait(timeout=0.5)
            except TimeoutExpired:
                proc.send_signal(signal.SIGINT)
            current = proc.stdout.read()
            sys.stdout.buffer.write(current)
            if b'Copied size' in current:
                break


if __name__ == '__main__':
    main()
