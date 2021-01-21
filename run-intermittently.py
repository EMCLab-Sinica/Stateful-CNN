import argparse
import logging
import pathlib
import signal
import tempfile
import sys
from subprocess import Popen, TimeoutExpired, check_call

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('run-intermittently')

CHUNK_SIZE = 2000
CHUNK_LINES = 20

def run_one_inference(program, interval, logfile, shutdown_after_writes, power_cycles_limit):
    first_run = True
    timeout_counter = 0
    while True:
        cmd = [program, '1']
        if first_run and shutdown_after_writes:
            cmd.extend(['-c', str(shutdown_after_writes)])
        with Popen(cmd, stdout=logfile, stderr=logfile) as proc:
            try:
                kwargs = {}
                if not shutdown_after_writes:
                    kwargs['timeout'] = interval
                outs, errs = proc.communicate(**kwargs)
            except TimeoutExpired:
                timeout_counter += 1
                if timeout_counter >= power_cycles_limit:
                    logger.error('The program does not run intermittently')
                    return False
                proc.send_signal(signal.SIGINT)
            proc.wait()
            first_run = False
            if proc.returncode == 2:
                # simulated power failure
                continue
            if proc.returncode in (1, -signal.SIGFPE, -signal.SIGSEGV):
                logger.error('Program crashed!')
                return False
            if proc.returncode == 0:
                logfile.seek(0, 2)
                file_size = logfile.tell()
                logfile.seek(-(CHUNK_SIZE if file_size > CHUNK_SIZE else file_size), 2)
                last_chunk = logfile.read()
                print('\n'.join(last_chunk.decode('ascii').split('\n')[-CHUNK_LINES:]))
                return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=0)
    parser.add_argument('--interval', type=float, default=0.01)
    parser.add_argument('--shutdown-after-writes', type=int, default=0)
    parser.add_argument('--power-cycles-limit', type=int, default=200)
    parser.add_argument('--compress', default=False, action='store_true')
    parser.add_argument('program')
    args = parser.parse_args()

    rounds = 0
    ret = 0
    logdir = pathlib.Path(tempfile.gettempdir())
    log_archive = logdir / 'intermittent.tar'
    while True:
        logfile_path = logdir / f'intermittent-cnn-{rounds}'
        compressed_logfile_path = logfile_path.with_suffix('.zst')
        with open(logfile_path, mode='w+b') as logfile:
            success = run_one_inference(args.program, args.interval, logfile, args.shutdown_after_writes, args.power_cycles_limit)

        if args.compress:
            check_call(['touch', log_archive])
            check_call(['zstd', logfile_path, '-o', compressed_logfile_path, '-f'])
            check_call(['tar', '-C', logdir, '-uvf', log_archive, compressed_logfile_path.name])
            check_call(['rm', '-v', logfile_path])
            check_call(['rm', '-v', compressed_logfile_path])

        if not success:
            ret = 1
            break
        rounds += 1
        if args.rounds and rounds >= args.rounds:
            break
    return ret

if __name__ == '__main__':
    sys.exit(main())
