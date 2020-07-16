from subprocess import Popen, TimeoutExpired
import signal
import sys

def main():
    while True:
        with Popen([sys.argv[1], '1']) as proc:
            try:
                proc.wait(timeout=0.001)
            except TimeoutExpired:
                proc.send_signal(signal.SIGINT)
            proc.wait()
            if proc.returncode in (-signal.SIGABRT, -signal.SIGFPE):
                raise Exception('Crashed')
            if proc.returncode == 0:
                break


if __name__ == '__main__':
    main()
