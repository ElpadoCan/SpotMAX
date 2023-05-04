import subprocess
import multiprocessing
import argparse

ap = argparse.ArgumentParser(
    prog='spotMAX process', description='Used to spawn a separate process', 
    formatter_class=argparse.RawTextHelpFormatter
)
ap.add_argument(
    '-c', '--command', required=True, type=str, metavar='COMMAND',
    help='String of commands separated by comma.'
)

def worker(*commands):
    subprocess.run(list(commands)) # [sys.executable, r'spotmax\test.py'])
    # sys.stdout.flush()

if __name__ == '__main__':
    args = vars(ap.parse_args())
    command = args['command']
    commands = command.split(',')
    commands = [command.lstrip() for command in commands]
    process = multiprocessing.Process(target=worker, args=commands)
    process.start()
    process.join()
