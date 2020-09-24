import os.path
from subprocess import Popen, TimeoutExpired
import signal
import sys
import tempfile

def run_one_inference(logfile):
    while True:
        with Popen([sys.argv[1], '1'], stdout=logfile, stderr=logfile) as proc:
            try:
                outs, errs = proc.communicate(timeout=0.01)
            except TimeoutExpired:
                proc.send_signal(signal.SIGINT)
            proc.wait()
            if proc.returncode in (1, -signal.SIGFPE):
                raise Exception('Crashed')
            if proc.returncode == 0:
                logfile.seek(-2000, 2)
                last_1000 = logfile.read()
                print('\n'.join(last_1000.decode('ascii').split('\n')[-20:]))
                return

def main():
    logfile_path = os.path.join(tempfile.gettempdir(), 'intermittent-cnn')
    while True:
        with open(logfile_path, mode='w+b') as logfile:
            run_one_inference(logfile)


if __name__ == '__main__':
    main()
