# print('Setting up required libraries...')
from cellacdc._run import _install_tables
_install_tables(parent_software='SpotMAX')

import os
import sys
import traceback
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from functools import wraps

is_cli = True

try:
    from cellacdc import gui as acdc_gui
    from qtpy.QtGui import QFont
    font = QFont()
    font.setPixelSize(11)
    font_small = QFont()
    font_small.setPixelSize(9)
    GUI_INSTALLED = True
except ModuleNotFoundError:
    GUI_INSTALLED = False
    
spotmax_path = os.path.dirname(os.path.abspath(__file__))
qrc_resources_path = os.path.join(spotmax_path, 'qrc_resources_spotmax.py')
resources_folderpath = os.path.join(spotmax_path, 'resources')

# Replace 'from PyQt5' with 'from qtpy' in qrc_resources.py file
try:
    save_qrc = False
    with open(qrc_resources_path, 'r') as qrc_py:
        text = qrc_py.read()
        if text.find('from PyQt5') != -1:
            text = text.replace('from PyQt5', 'from qtpy')
            save_qrc = True
    if save_qrc:
        with open(qrc_resources_path, 'w') as qrc_py:
            qrc_py.write(text)
except Exception as err:
    raise err


import inspect
from datetime import datetime
from pprint import pprint
import pathlib
import numpy as np

rng = np.random.default_rng(seed=6490)

spotMAX_path = os.path.dirname(spotmax_path)
html_path = os.path.join(spotmax_path, 'html_files')

home_path = pathlib.Path.home()
spotmax_appdata_path = os.path.join(home_path, 'spotmax_appdata')
last_used_ini_text_filepath = os.path.join(
    spotmax_appdata_path, 'last_used_ini_filepath.txt'
)
last_cli_log_file_path = os.path.join(
    spotmax_appdata_path, 'last_cli_log_file_path.txt'
)
data_path = os.path.join(spotMAX_path, 'data')

logs_path = os.path.join(spotmax_appdata_path, 'logs')
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

settings_path = os.path.join(spotmax_appdata_path, 'settings')
if not os.path.exists(settings_path):
    os.makedirs(settings_path)

default_ini_path = os.path.join(spotmax_appdata_path, 'config.ini')
colorItems_path = os.path.join(settings_path, 'colorItems.json')
gui_settings_csv_path = os.path.join(settings_path, 'gui_settings.csv')

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

help_text = (
    'Welcome to spotMAX!\n\n'
    'You can run spotmax both as a GUI or in the command line.\n'
    'To run the GUI type `spotmax`. To run the command line type `spotmax -p <path_to_params_file>`.\n'
    'The `<path_to_params_file>` can either be a CSV or INI file.\n'
    'If you do not have one, use the GUI to set up the parameters.\n\n'
    'See below other arguments you can pass to the command line. Enjoy!'
)

base_lineage_table_values = {
    'cell_cycle_stage': 'G1',
    'generation_num': 2,
    'relative_ID': -1,
    'relationship': 'mother',
    'emerg_frame_i': -1,
    'division_frame_i': -1
}

error_up_str = '^'*50
error_up_str = f'\n{error_up_str}'
error_down_str = '^'*50
error_down_str = f'\n{error_down_str}'

ZYX_GLOBAL_COLS = ['z', 'y', 'x']
ZYX_LOCAL_COLS = ['z_local', 'y_local', 'x_local']
ZYX_AGGR_COLS = ['z_aggr', 'y_aggr', 'x_aggr']
ZYX_LOCAL_EXPANDED_COLS = [
    'z_local_expanded', 'y_local_expanded', 'x_local_expanded'
]
ZYX_FIT_COLS = ['z_fit', 'y_fit', 'x_fit']
ZYX_RESOL_COLS = ['z_resolution_pxl', 'y_resolution_pxl', 'x_resolution_pxl']

DFs_FILENAMES = {
    'spots_detection': '*rn*_0_detected_spots*desc*',
    'spots_gop': '*rn*_1_valid_spots*desc*',
    'spots_spotfit': '*rn*_2_spotfit*desc*'
}

valid_true_bool_str = {
    'true', 'yes', 'on'
}
valid_false_bool_str = {
    'false', 'no', 'off'
}

def exception_handler_cli(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                result = func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                result = func(self, *args)
            else:
                result = func(self, *args, **kwargs)
        except Exception as e:
            result = None
            if self.is_cli:
                self.logger.exception(e)
            if not self.is_batch_mode:
                self.quit(error=e)
            else:
                raise e
        return result
    return inner_function

def handle_log_exception_cli(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                result = func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                result = func(self, *args)
            else:
                result = func(self, *args, **kwargs)
        except Exception as error:
            result = None
            self.log_exception_report(error, traceback.format_exc())
        return result
    return inner_function

def read_version():
    try:
        from setuptools_scm import get_version
        version = get_version(root='..', relative_to=__file__)
        return version
    except Exception as e:
        try:
            from . import _version
            return _version.version
        except Exception as e:
            return 'ND'