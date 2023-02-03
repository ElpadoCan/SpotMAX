import sys

try:
    import acdctools
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-U',
        'git+https://github.com/SchmollerLab/acdc_tools']
    )

import os
import inspect
from datetime import datetime
from pprint import pprint
import pathlib
import numpy as np

rng = np.random.default_rng(seed=6490)

spotmax_path = os.path.dirname(os.path.abspath(__file__))
home_path = pathlib.Path.home()
spotmax_appdata_path = os.path.join(home_path, 'spotmax_appdata')

logs_path = os.path.join(spotmax_appdata_path, 'logs')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

settings_path = os.path.join(spotmax_appdata_path, 'settings')
if not os.path.exists(settings_path):
    os.makedirs(settings_path)

default_ini_path = os.path.join(spotmax_appdata_path, 'config.ini')
colorItems_path = os.path.join(settings_path, 'colorItems.json')

def printl(*objects, pretty=False, is_decorator=False, **kwargs):
    # Copy current stdout, reset to default __stdout__ and then restore current
    current_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    timestap = datetime.now().strftime('%H:%M:%S')
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe)
    idx = 2 if is_decorator else 1
    callingframe = outerframes[idx].frame
    callingframe_info = inspect.getframeinfo(callingframe)
    filpath = callingframe_info.filename
    filename = os.path.basename(filpath)
    print_func = pprint if pretty else print
    print('*'*30)
    print(f'{timestap} - File "{filename}", line {callingframe_info.lineno}:')
    print_func(*objects, **kwargs)
    print('='*30)
    sys.stdout = current_stdout

is_linux = sys.platform.startswith('linux')
is_mac = sys.platform == 'darwin'
is_win = sys.platform.startswith("win")
is_win64 = (is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"))

issues_url = 'https://github.com/SchmollerLab/spotMAX/issues'
