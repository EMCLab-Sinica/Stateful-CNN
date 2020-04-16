from subprocess import Popen, TimeoutExpired
import signal

def main():
    while True:
        with Popen(['./out/intermittent-cnn', '1']) as proc:
            try:
                proc.wait(timeout=0.001)
            except TimeoutExpired:
                proc.send_signal(signal.SIGINT)
            if proc.returncode == 0:
                break


if __name__ == '__main__':
    main()
