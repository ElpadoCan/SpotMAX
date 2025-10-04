# Monitor memory usage and write to a file in case the main process uses too
# much memory. Useful in UNIX systems to warn the user that the process might 
# have been killed by the OS.

import os

import argparse

import psutil
import time

from spotmax import get_watchdog_filepaths

description = (
    'Watchdog for SpotMAX process. Monitors memory usage and writes to a file '
    'if the memory usage is too high.'
)

ap = argparse.ArgumentParser(
    prog='Watchdog for SpotMAX process', 
    description=description, 
    formatter_class=argparse.RawTextHelpFormatter
)
ap.add_argument(
    '-id', '--identifier', 
    required=True, 
    type=str, 
    metavar='IDENTIFIER',
    help='Text identifier to distinguish multiple SpotMAX instances.'
)

def run():
    args = vars(ap.parse_args())
    identifier = args['identifier']
    watchdog_filepaths = get_watchdog_filepaths(identifier)
    (stop_watchdog_flag_filepath, watchdog_log_filepath, 
    watchdog_stopped_flag) = watchdog_filepaths
    
    while True:
        if os.path.exists(stop_watchdog_flag_filepath):
            break
        
        time.sleep(1)
        
        if psutil.virtual_memory().percent < 85:
            continue
        
        with open(watchdog_log_filepath, 'a') as f:
            f.write(
                'WARNING: SpotMAX memory usage passed 95%.'
            )
    
    open(watchdog_stopped_flag, 'w').close()

if __name__ == '__main__':
    run()