import sys
import os
import inspect

def printl(*objects, **kwargs):
    cf = inspect.currentframe()
    filpath = inspect.getframeinfo(cf).filename
    filename = os.path.basename(filpath)
    print('*'*30)
    print(f'File "{filename}", line {cf.f_back.f_lineno}:')
    print(*objects, **kwargs)
    print('='*30)

is_linux = sys.platform.startswith('linux')
is_mac = sys.platform == 'darwin'
is_win = sys.platform.startswith("win")
is_win64 = (is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"))

issues_url = 'https://github.com/SchmollerLab/spotMAX/issues'
