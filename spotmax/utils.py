import os
import sys
import pandas as pd
import datetime
import time
import difflib
import cv2
import logging
import traceback
from importlib import import_module
from typing import List, Union, Optional, Tuple, Dict
from tqdm import tqdm
import numpy as np
import pathlib
import re
import h5py

try:
    import yaml
except ModuleNotFoundError:
    # Needed only with Unet and prev users do not need it
    pass
    
from uuid import uuid4
import configparser
from urllib.parse import urlparse
from natsort import natsort_keygen, natsorted

from tifffile.tifffile import TiffWriter, TiffFile

import skimage.color
import colorsys

from math import log, pow, floor, sqrt

from . import last_cli_log_file_path
from . import GUI_INSTALLED
from . import DFs_FILENAMES
from . import valid_true_bool_str, valid_false_bool_str
from . import rng

if GUI_INSTALLED:
    import matplotlib.colors
    import matplotlib.pyplot as plt

    from qtpy.QtCore import QTimer

    from cellacdc import apps as acdc_apps
    from cellacdc import widgets as acdc_widgets
    from cellacdc.plot import imshow

    from . import widgets

    GUI_INSTALLED = True

from cellacdc import myutils as acdc_myutils
from cellacdc import load as acdc_load

from . import is_mac, is_linux, printl, settings_path, io
from . import core, config
from . import transformations
from .nnet import config_yaml_path

class _Dummy:
    def __init__(self, *args, **kwargs):
        _name = kwargs.get('name')
        if _name is not None:
            self.__name__ = _name

def _check_cli_params_extension(params_path):
    _, ext = os.path.splitext(params_path)
    if ext == '.csv' or ext == '.ini':
        return
    else:
        raise FileNotFoundError(
            'The extension of the parameters file must be either `.ini` or `.csv`.'
            f'File path provided: "{params_path}"'
        )

def check_cli_file_path(file_path, desc='parameters'):
    if os.path.exists(file_path) and os.path.isabs(file_path):
        _check_cli_params_extension(file_path)
        return file_path
    
    # Try to check if user provided a relative path for the params file
    abs_file_path = io.get_abspath(file_path)
    if os.path.exists(abs_file_path):
        _check_cli_params_extension(abs_file_path)
        return abs_file_path

    raise FileNotFoundError(
        f'The following {desc} file provided does not exist: "{abs_file_path}"'
    )

def setup_cli_logger(name='spotmax_cli', custom_logs_folderpath=None): 
    from . import logs_path 
    acdc_myutils.delete_older_log_files(logs_path)
    
    if custom_logs_folderpath is None:
        custom_logs_folderpath = logs_path
    
    logger = acdc_myutils.Logger(name='spotmax-logger', module=name)
    sys.stdout = logger

    if not os.path.exists(custom_logs_folderpath):
        os.mkdir(custom_logs_folderpath)
                
    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    id = uuid4()
    log_filename = f'{date_time}_{name}_{id}_stdout.log'
    log_path = os.path.join(custom_logs_folderpath, log_filename)

    output_file_handler = logger_file_handler(log_path)
    logger._file_handler = output_file_handler
    logger.addHandler(output_file_handler)
    
    acdc_myutils._log_system_info(
        logger, log_path, is_cli=True, also_spotmax=True
    )
    
    # stdout_handler = logging.StreamHandler(sys.stdout)    
    # logger.addHandler(stdout_handler)
    
    store_current_log_file_path(log_path)

    return logger, log_path, custom_logs_folderpath

def is_valid_url(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except Exception as e:
        return False

def lighten_color(color, amount=0.3, hex=True):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = matplotlib.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
    lightened_c = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    if hex:
        lightened_c = tuple([int(round(v*255)) for v in lightened_c])
        lightened_c = '#%02x%02x%02x' % lightened_c
    return lightened_c

def rgba_str_to_values(rgbaString, errorRgb=(255,255,255,255)):
    try:
        r, g, b, a = re.findall(r'(\d+), (\d+), (\d+), (\d+)', rgbaString)[0]
        r, g, b, a = int(r), int(g), int(b), int(a)
    except TypeError:
        try:
            r, g, b, a = rgbaString
        except Exception as e:
            r, g, b, a = errorRgb
    return r, g, b, a

def getMostRecentPath():
    recentPaths_path = os.path.join(
        settings_path, 'recentPaths.csv'
    )
    if os.path.exists(recentPaths_path):
        df = pd.read_csv(recentPaths_path, index_col='index')
        if 'opened_last_on' in df.columns:
            df = df.sort_values('opened_last_on', ascending=False)
        mostRecentPath = df.iloc[0]['path']
        if not isinstance(mostRecentPath, str):
            mostRecentPath = ''
    else:
        mostRecentPath = ''
    return mostRecentPath

def showInExplorer(path):
    if is_mac:
        os.system(f'open "{path}"')
    elif is_linux:
        os.system(f'xdg-open "{path}"')
    else:
        os.startfile(path)

def is_iterable(item):
     try:
         iter(item)
         return True
     except TypeError as e:
         return False

def listdir(path):
    return natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f == 'desktop.ini'
        and not f == 'recovery'
        and not f == 'cached'
    ])

def get_salute_string():
    time_now = datetime.datetime.now().time()
    time_end_morning = datetime.time(12,00,00)
    time_end_afternoon = datetime.time(15,00,00)
    time_end_evening = datetime.time(20,00,00)
    time_end_night = datetime.time(4,00,00)
    if time_now >= time_end_night and time_now < time_end_morning:
        return 'Have a good day!'
    elif time_now >= time_end_morning and time_now < time_end_afternoon:
        return 'Have a good afternoon!'
    elif time_now >= time_end_afternoon and time_now < time_end_evening:
        return 'Have a good evening!'
    else:
        return 'Have a good night!'

def logger_file_handler(log_filepath, mode='w'):
    output_file_handler = logging.FileHandler(log_filepath, mode=mode)
    # Format your logs (optional)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s:\n'
        '------------------------\n'
        '%(message)s\n'
        '------------------------\n',
        datefmt='%d-%m-%Y, %H:%M:%S'
    )
    output_file_handler.setFormatter(formatter)
    return output_file_handler

def _bytes_to_MB(size_bytes):
    i = int(floor(log(size_bytes, 1024)))
    p = pow(1024, i)
    s = round(size_bytes / p, 2)
    return s

def getMemoryFootprint(files_list):
    required_memory = sum([
        48 if str(file).endswith('.h5') else os.path.getsize(file)
        for file in files_list
    ])
    return required_memory

def imagej_tiffwriter(new_path, data, metadata, SizeT, SizeZ, imagej=True):
    if data.dtype != np.uint8 or data.dtype != np.uint16:
        data = skimage.img_as_uint(data)
    with TiffWriter(new_path, imagej=imagej) as new_tif:
        if not imagej:
            new_tif.save(data)
            return

        if SizeZ > 1 and SizeT > 1:
            # 3D data over time
            T, Z, Y, X = data.shape
        elif SizeZ == 1 and SizeT > 1:
            # 2D data over time
            T, Y, X = data.shape
            Z = 1
        elif SizeZ > 1 and SizeT == 1:
            # Single 3D data
            Z, Y, X = data.shape
            T = 1
        elif SizeZ == 1 and SizeT == 1:
            # Single 2D data
            Y, X = data.shape
            T, Z = 1, 1
        data.shape = T, Z, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
        new_tif.save(data, metadata=metadata)

def index_4D_dset_for(dset, axis0_interval, axis1_interval, worker=None):
    is_compressed = dset.compression is not None
    Y, X = dset.shape[-2:]
    axis0_range = range(*axis0_interval)
    axis1_range = range(*axis1_interval)
    arr = np.empty((len(axis0_range), len(axis1_range), Y, X), dtype=dset.dtype)
    for t0, t in enumerate(axis0_range):
        for z0, z in enumerate(axis1_range):
            if worker is not None and worker.H5readWait and is_compressed:
                # Paused by main GUI to allow GUI completion of GUI tasks
                worker.pauseH5read()
                worker.H5readWait = False
            else:
                arr[t0, z0] = dset[t, z]
    return arr

def index_3D_dset_for(dset, axis0_interval, worker=None):
    is_compressed = dset.compression is not None
    Y, X = dset.shape[-2:]
    axis0_range = range(*axis0_interval)
    arr = np.empty((len(axis0_range), Y, X), dtype=dset.dtype)
    for z0, z in enumerate(axis0_range):
        if worker is not None and worker.H5readWait and is_compressed:
            # Paused by main GUI to allow GUI completion of GUI tasks
            worker.pauseH5read()
            worker.H5readWait = False
        else:
            arr[z0] = dset[z]
    return arr

def emit(txt, signals, level='INFO'):
    if signals is not None:
        signals.progress.emit(txt, level)

def shiftWindow_axis0(
        dset, window_arr, windowSize, coord0_window, current_idx,
        axis1_interval=None, worker=None
    ):
    """Get a window array centered at current_idx from a bigger dataset
    by minimizing the number of indexed elements from the bigger dataset.

    The window array can be a simple shift by one or a completely new array.

    If this is controlled by a slider there are 4 possible scenarios:
        1. The slider cursor is moved in the left boundary region
           --> return original window_arr without indexing
        2. The slider cursor is moved in the right boundary region
           --> return original window_arr without indexing
        3. The slider cursor is moved overlapping the current window_arr
           --> roll the original array and replace the chunk with newly
               indexed data from the bigger dataset
        4. The slider cursor is moved completely outside of the current window
           --> fully index a new window_arr from the bigger dataset


    Parameters
    ----------
    dset : h5py dataset or numpy array
        The bigger dataset.
    window_arr : numpy array
        The current window array (subset of dset).
    windowSize : int
        The size of window array along the required axis.
    coord0_window : int
        The global start index of the window_arr.
    current_idx : int
        Description of parameter `current_idx`.
    axis1_interval : tuple of (start, end) range or None
        This controls which elements need to be indexed on axis 1.
    signals : Signals or None
        Signals to emit if this function is called in a QThread.

    Returns
    -------
    tuple
        The new window array, the new start coordinate
        and the start coordinate of the axis 1.

    """
    signals = worker.signals

    if axis1_interval is None:
        axis1_c0 = 0

    coord1_window = coord0_window + windowSize - 1
    halfWindowSize = int(windowSize/2)

    coord0_chunk = coord1_window + 1
    chunkSize = current_idx + halfWindowSize - coord0_chunk + 1

    rightBoundary = dset.shape[0]-halfWindowSize
    leftBoundary = halfWindowSize
    if current_idx <= halfWindowSize:
        emit(f'Slider cursor moved to {current_idx} --> left boundary', signals)
        if leftBoundary < coord0_window:
            direction = 'new'
            current_idx = leftBoundary + 1
        else:
            emit('No need to load new chunk', signals)
            return window_arr, coord0_window, axis1_c0
    elif current_idx >= rightBoundary:
        emit(f'Slider cursor moved to {current_idx} --> right boundary', signals)
        if rightBoundary > coord1_window:
            direction = 'new'
            current_idx = rightBoundary
        else:
            return window_arr, coord0_window, axis1_c0

    if abs(chunkSize) >= windowSize:
        direction = 'new'
    elif chunkSize <= 0:
        direction = 'backward'
    else:
        direction = 'forward'

    if direction == 'new':
        coord0_chunk = current_idx - halfWindowSize - 1
        coord1_chunk = coord0_chunk + windowSize

        if signals is not None:
            signals.sigLoadingNewChunk.emit((coord0_chunk, coord1_chunk))
            # Give time to the GUI thread to finish updating
            time.sleep(0.05)
        emit(
            'Loading entire new window, '
            f'new time range = ({coord0_chunk}, {coord1_chunk})',
            signals
        )

        if axis1_interval is None:
            window_arr = dset[coord0_chunk:coord1_chunk]
            axis1_c0 = 0
        else:
            axis1_c0, axis1_c1 = axis1_interval
            window_arr = dset[coord0_chunk:coord1_chunk, axis1_c0:axis1_c1]
        coord0_window = coord0_chunk

        return window_arr, coord0_window, axis1_c0

    emit(
        f'Rolling current window with shift = {-chunkSize}',
        signals
    )
    window_arr = np.roll(window_arr, -chunkSize, axis=0)

    if direction == 'forward':
        emit(
            f'Loading chunk, range = ({coord0_chunk}, {coord0_chunk+chunkSize})',
            signals
        )
        axis0_interval = (coord0_chunk, coord0_chunk+chunkSize)
        if axis1_interval is None:
            axis1_c0 = 0
            chunk = index_3D_dset_for(dset, axis0_interval, worker=worker)
        else:
            axis1_c0, axis1_c1 = axis1_interval
            chunk = index_4D_dset_for(
                dset, axis0_interval, axis1_interval, worker=worker
            )

        window_arr[-chunkSize:] = chunk
        coord0_window += chunkSize
    elif direction == 'backward':
        coord0_chunk = coord0_window + chunkSize
        chunkSize = abs(chunkSize)

        emit(
            f'Loading chunk, range = ({coord0_chunk}, {coord0_chunk+chunkSize})',
            signals
        )

        axis0_interval = (coord0_chunk, coord0_chunk+chunkSize)
        if axis1_interval is None:
            # axis1_interval = (0, Z)
            chunk = index_3D_dset_for(dset, axis0_interval, worker=worker)
        else:
            axis1_c0, axis1_c1 = axis1_interval
            chunk = index_4D_dset_for(
                dset, axis0_interval, axis1_interval, worker=worker
            )

        window_arr[:chunkSize] = chunk
        coord0_window = coord0_chunk

    emit(
        f'New window range = ({coord0_window}, {window_arr.shape[0]})',
        signals
    )

    return window_arr, coord0_window, axis1_c0

def shiftWindow_axis1(
        dset, window_arr, windowSize, coord0_window, current_idx,
        axis0_interval=None, worker=None
    ):
    """
    See shiftWindow_axis0 for details
    """

    signals = worker.signals

    if axis0_interval is None:
        axis0_c0 = 0

    coord1_window = coord0_window + windowSize - 1
    halfWindowSize = int(windowSize/2)

    coord0_chunk = coord1_window + 1
    chunkSize = current_idx + halfWindowSize - coord0_chunk + 1

    rightBoundary = dset.shape[1]-halfWindowSize
    leftBoundary = halfWindowSize
    if current_idx <= halfWindowSize:
        emit(f'Slider cursor moved to {current_idx} --> left boundary', signals)
        if leftBoundary < coord0_window:
            direction = 'new'
            current_idx = leftBoundary + 1
        else:
            return window_arr, axis0_c0, coord0_window
    elif current_idx >= rightBoundary:
        emit(f'Slider cursor moved to {current_idx} --> right boundary', signals)
        if rightBoundary > coord1_window:
            direction = 'new'
            current_idx = rightBoundary
        else:
            return window_arr, axis0_c0, coord0_window

    if abs(chunkSize) >= windowSize:
        direction = 'new'
    elif chunkSize <= 0:
        direction = 'backward'
    else:
        direction = 'forward'

    if direction == 'new':
        coord0_chunk = current_idx - halfWindowSize - 1
        coord1_chunk = coord0_chunk + windowSize

        if signals is not None:
            signals.sigLoadingNewChunk.emit((coord0_chunk, coord1_chunk))
            time.sleep(0.05)
        emit(
            'Loading entire new window, '
            f'new time range = ({coord0_chunk}, {coord1_chunk})',
            signals
        )

        if axis0_interval is None:
            window_arr = dset[:, coord0_chunk:coord1_chunk]
        else:
            axis0_c0, axis0_c1 = axis0_interval
            window_arr = dset[axis0_c0:axis0_c1, coord0_chunk:coord1_chunk]
        coord0_window = coord0_chunk
        return window_arr, axis0_c0, coord0_window

    emit(
        f'Rolling current window with shift = {-chunkSize}',
        signals
    )
    window_arr = np.roll(window_arr, -chunkSize, axis=1)

    T, Z, Y, X = dset.shape

    if direction == 'forward':
        emit(
            f'Loading chunk, range = ({coord0_chunk}, {coord0_chunk+chunkSize})',
            signals
        )
        axis1_interval = (coord0_chunk, coord0_chunk+chunkSize)
        if axis0_interval is None:
            axis0_c0 = 0
            axis0_interval = (0, T)
        else:
            axis0_c0, axis0_c1 = axis0_interval

        chunk = index_4D_dset_for(
            dset, axis0_interval, axis1_interval, worker=worker
        )
        window_arr[:, -chunkSize:] = chunk
        coord0_window += chunkSize
    elif direction == 'backward':
        coord0_chunk = coord0_window + chunkSize
        chunkSize = abs(chunkSize)

        emit(
            f'Loading chunk, range = ({coord0_chunk}, {coord0_chunk+chunkSize})',
            signals
        )

        axis1_interval = (coord0_chunk, coord0_chunk+chunkSize)
        if axis0_interval is None:
            axis0_c0 = 0
            axis0_interval = 0, T
        else:
            axis0_c0, axis0_c1 = axis0_interval

        chunk = index_4D_dset_for(
            dset, axis0_interval, axis1_interval, worker=worker
        )
        window_arr[:, :chunkSize] = chunk
        coord0_window = coord0_chunk

    emit(
        f'New window range = ({coord0_window}, {window_arr.shape[1]})',
        signals
    )

    return window_arr, axis0_c0, coord0_window

def singleSpotGOFmeasurementsName():
    names = {
        'QC_passed': 'Quality control passed?',
        'solution_found': 'Solution found?',
        'reduced_chisq': 'Reduced Chi-square',
        'p_chisq': 'p-value of Chi-squared test',
        'null_chisq_test': 'Failed to reject Chi-squared test null?',
        'KS_stat': 'Kolmogorov–Smirnov test statistic',
        'p_KS': 'Kolmogorov–Smirnov test p-value',
        'null_ks_test': 'Failed to reject Kolmogorov–Smirnov null?',
        'RMSE': 'Root mean squared error',
        'NRMSE': 'Normalized mean squared error',
        'F_NRMSE': 'Rescaled normalized mean squared error'
    }
    return names

def singleSpotFitMeasurentsName():
    names = {
        'spot_B_min': 'Background lower bound',
        'obj_id': 'spot ID',
        'num_intersect': 'Number of touching spots',
        'num_neigh': 'Number of spots per object',
        'z_fit': 'Spot center Z-coordinate',
        'y_fit': 'Spot center Y-coordinate',
        'x_fit': 'Spot center X-coordinate',
        'sigma_z_fit': 'Spot Z-sigma',
        'sigma_y_fit': 'Spot Y-sigma',
        'sigma_x_fit': 'Spot Z-sigma',
        'sigma_yx_mean': 'Spot mean of Y- and X- sigmas',
        'spotfit_vol_vox': 'Spot volume (voxel)',
        'A_fit': 'Fit parameter A',
        'B_fit': 'Fit parameter B (local background)',
        'I_tot': 'Total integral of fitted curve',
        'I_foregr': 'Foreground integral of fitted curve'
    }
    return names

def singleSpotSizeMeasurentsName():
    names = {
        'spotsize_yx_radius_um': 'yx- radius (μm)',
        'spotsize_z_radius_um': 'z- radius (μm)',
        'spotsize_yx_radius_pxl': 'yx- radius (pixel)',
        'spotsize_z_radius_pxl': 'z- radius (pixel)',
        'spotsize_limit': 'Stop limit',
        'spot_surf_50p': 'Median of outer surface intensities',
        'spot_surf_5p': '5%ile of outer surface intensities',
        'spot_surf_mean': 'Mean of outer surface intensities',
        'spot_surf_std': 'Std. of outer surface intensities'
    }
    return names

def singleSpotEffectsizeMeasurementsName():
    names = {
        'effsize_cohen_s': 'Cohen\'s effect size (sample)',
        'effsize_hedge_s': 'Hedges\' effect size (sample)',
        'effsize_glass_s': 'Glass\' effect size (sample)',
        'effsize_cliffs_s': 'Cliff\'s Delta (sample)',
        'effsize_cohen_pop': 'Cohen\'s effect size (population)',
        'effsize_hedge_pop': 'Hedges\' effect size (population)',
        'effsize_glass_pop': 'Glass\' effect size (population)',
        'effsize_cohen_s_95p': '95%ile Cohen\'s effect size (sample)',
        'effsize_hedge_s_95p': '95%ile Hedges\' effect size (sample)',
        'effsize_glass_s_95p': '95%ile Glass\' effect size (sample)',
        'effsize_cliffs_s_95p': '95%ile Cliff\'s effect size (sample)',
        'effsize_cohen_pop_95p': '95%ile Cohen\'s effect size (population)',
        'effsize_hedge_pop_95p': '95%ile Hedges\' effect size (population)',
        'effsize_glass_pop_95p': '95%ile Glass\' effect size (population)'
    }
    return names

def singleSpotCountMeasurementsName():
    names = {
        'Timestamp': 'Timestamps',
        'Time (min)': 'Time (minutes)',
        'vox_spot': 'Spot center pixel intensity',
        'vox_ref': 'Reference ch. center pixel intensity',
        '|abs|_spot': 'Spot mean intensity',
        '|abs|_ref': 'Reference ch. mean intensity at spot',
        '|norm|_spot': 'Spot normalised mean Intensity',
        '|norm|_ref': 'Ref. ch. normalised mean intensity at spot',
        '|spot|:|ref| t-value': 't-statistic of t-test',
        '|spot|:|ref| p-value (t)': 'p-value of t-test',
        'z': 'Spot Z coordinate',
        'y': 'Spot Y coordinate',
        'x': 'Spot X coordinate',
        'peak_to_background ratio': 'Spot/background center pixel intensity ratio',
        'backgr_INcell_OUTspot_mean': 'IN-cell background mean intensity',
        'backgr_INcell_OUTspot_median': 'IN-cell background median intensity',
        'backgr_INcell_OUTspot_75p': 'IN-cell background 75%ile intensity',
        'backgr_INcell_OUTspot_25p': 'IN-cell background 25%ile intensity',
        'backgr_INcell_OUTspot_std': 'IN-cell background std. intensity',
        'is_spot_inside_ref_ch': 'Is spot inside reference channel?',
        'Creation DateTime': 'File creation Timestamp'
    }
    return names

def singleCellMeasurementsName():
    names = {
        'frame_i': 'Frame index',
        'Cell_ID': 'Cell ID',
        'timestamp': 'Timestamps',
        'time_min': 'Time (minutes)',
        'cell_area_pxl': 'Cell area (pixel)',
        'cell_area_um2': 'Cell area (μm<sup>2</sup>)',
        'ratio_areas_bud_moth': 'Ratio areas bud/mother',
        'ratio_volumes_bud_moth': 'Ratio volumes bud/mother',
        'cell_vol_vox': 'Cell volume (voxel)',
        'cell_vol_fl': 'Cell volume (fl)',
        'predicted_cell_cycle_stage': 'Predicted cell cycle stage',
        'generation_num': 'Generation number',
        'num_spots': 'Number of spts',
        'ref_ch_vol_vox': 'Reference channel volume (voxel)',
        'ref_ch_vol_um3': 'Reference channel volume (μm<sup>3</sup>)',
        'ref_ch_vol_len_um': 'Reference channel length (μm)',
        'ref_ch_num_fragments': 'Reference channel number of fragments',
        'cell_cycle_stage': 'Cell cycle stage',
        'relationship': 'Mother or bud',
        'relative_ID': 'ID of relative cell',
    }
    return names

def splitPathlibParts(path):
    return pd.Series(path.parts)

def natSortExpPaths(expPaths):
    df = (
        pd.DataFrame(expPaths)
        .transpose()
        .reset_index()
        .rename(columns={'index': 'key'})
    )
    df['key'] = df['key'].apply(pathlib.Path)
    df_split = df['key'].apply(splitPathlibParts).add_prefix('part')
    df = df.join(df_split, rsuffix='split')
    df = df.sort_values(
        by=list(df_split.columns),
        key=natsort_keygen()
    )
    expPaths = {}
    for series in df.itertuples():
        expPaths[str(series.key)] = {
            'channelDataPaths': series.channelDataPaths,
            'path': series.path
        }
    return expPaths

def orderedUnique(iterable):
    # See https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in iterable if not (x in seen or seen_add(x))]

def RGBtoGray(img):
    img = skimage.img_as_ubyte(skimage.color.rgb2gray(img))
    return img

def isRGB(img):
    if img.ndim == 3 and img.shape[-1] == 3:
        return True
    elif 2 < img.ndim > 2:
        raise IndexError(
            f'Image is not 2D (shape = {img.shape}) '
            'and last dimension is not == 3'
        )
    else:
        return False

def img_to_imageJ(img, folderPath, filenameNOext):
    if isRGB(img):
        img = RGBtoGray(img)
    tif_path = os.path.join(folderPath, f'{filenameNOext}.tif')
    if img.ndim == 3:
        SizeT = img.shape[0]
        SizeZ = 1
    elif img.ndim == 4:
        SizeT = img.shape[0]
        SizeZ = img.shape[1]
    else:
        SizeT = 1
        SizeZ = 1
    is_imageJ_dtype = (
        img.dtype == np.uint8
        or img.dtype == np.uint16
        or img.dtype == np.float32
    )
    if not is_imageJ_dtype:
        img = skimage.img_as_ubyte(img)

    imagej_tiffwriter(
        tif_path, img, {}, SizeT, SizeZ
    )
    return tif_path

def mergeChannels(channel1_img, channel2_img, color, alpha):
    if not isRGB(channel1_img):
        channel1_img = skimage.color.gray2rgb(channel1_img/channel1_img.max())
    if not isRGB(channel2_img):
        if channel2_img.max() > 0:
            channel2_img = skimage.color.gray2rgb(channel2_img/channel2_img.max())
        else:
            channel2_img = skimage.color.gray2rgb(channel2_img)

    colorRGB = [v/255 for v in color][:3]
    merge = (channel1_img*(1.0-alpha)) + (channel2_img*alpha*colorRGB)
    merge = merge/merge.max()
    merge = (np.clip(merge, 0, 1)*255).astype(np.uint8)
    return merge

def objContours(obj):
    contours, _ = cv2.findContours(
        obj.image.astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    min_y, min_x, _, _ = obj.bbox
    cont = np.squeeze(contours[0], axis=1)
    cont = np.vstack((cont, cont[0]))
    cont += [min_x, min_y]
    return cont

def pdDataFrame_boolTo0s1s(df, labelsToCast, axis=0):
    df = df.copy()

    if isinstance(labelsToCast, str) and labelsToCast == 'allRows':
        labelsToCast = df.index
        axis=0

    for label in labelsToCast:
        if axis==0:
            series = df.loc[label]
        else:
            series = df[label]

        isObject = pd.api.types.is_object_dtype(series)
        isString = pd.api.types.is_string_dtype(series)
        isBool = pd.api.types.is_bool_dtype(series)

        if isBool:
            series = series.replace({True: 'yes', False: 'no'})
            df[label] = series
        elif (isObject or isString):
            series = (series.str.lower()=='true') | (series.str.lower()=='yes')
            series = series.replace({True: 'yes', False: 'no'})
            if axis==0:
                if ((df.loc[label]=='True') | (df.loc[label]=='False')).any():
                    df.loc[label] = series
            else:
                if ((df[label]=='True') | (df[label]=='False')).any():
                    df[label] = series
    return df

def seconds_to_ETA(seconds):
    seconds = round(seconds)
    ETA = datetime.timedelta(seconds=seconds)
    ETA_split = str(ETA).split(':')
    if seconds >= 86400:
        days, hhmmss = str(ETA).split(',')
        h, m, s = hhmmss.split(':')
        ETA = f'{days}, {int(h):02}h:{int(m):02}m:{int(s):02}s'
    else:
        h, m, s = str(ETA).split(':')
        ETA = f'{int(h):02}h:{int(m):02}m:{int(s):02}s'
    return ETA

class widgetBlinker:
    def __init__(
            self, widget,
            styleSheetOptions=('background-color',),
            color='limegreen',
            duration=2000
        ):
        self._widget = widget
        self._color = color

        self._on_style = ''
        for option in styleSheetOptions:
            if option.find('color') != -1:
                self._on_style = f'{self._on_style};{option}: {color}'
            elif option.find('font-weight')!= -1:
                self._on_style = f'{self._on_style};{option}: bold'
        self._on_style = self._on_style[1:]

        self._off_style = ''
        for option in styleSheetOptions:
            if option.find('color')!= -1:
                self._off_style = f'{self._off_style};{option}: none'
            elif option.find('font-weight')!= -1:
                self._off_style = f'{self._off_style};{option}: normal'
        self._off_style = self._off_style[1:]

        self._flag = True
        self._blinkTimer = QTimer()
        self._blinkTimer.timeout.connect(self.blinker)

        self._stopBlinkTimer = QTimer()
        self._stopBlinkTimer.timeout.connect(self.stopBlinker)
        self._duration = duration

    def start(self):
        self._blinkTimer.start(100)
        self._stopBlinkTimer.start(self._duration)

    def blinker(self):
        if self._flag:
            self._widget.setStyleSheet(f'{self._on_style}')
        else:
            self._widget.setStyleSheet(f'{self._off_style}')
        self._flag = not self._flag

    def stopBlinker(self):
        self._blinkTimer.stop()
        self._widget.setStyleSheet(f'{self._off_style}')

def _get_all_filepaths(start_path):
    filepaths = []
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if os.path.islink(fp):
                continue
            filepaths.append(fp)
    return filepaths

def get_sizes_path(start_path, return_df=False):
    filepaths = _get_all_filepaths(start_path) 
    sizes = {'rel_path': [], 'size_bytes': []}
    for filepath in tqdm(filepaths, ncols=100):
        try:
            sizes['size_bytes'].append(os.path.getsize(filepath))
        except Exception as e:
            continue
        sizes['rel_path'].append(os.path.relpath(filepath, start_path))
    if not return_df:
        return sizes
    else:
        df = pd.DataFrame(sizes).sort_values('size_bytes')
        df['size_MB'] = df['size_bytes']*1e-6
        df['size_GB'] = df['size_bytes']*1e-9
        return df

def is_perfect_square(start_num):
    if (sqrt(start_num) - floor(sqrt(start_num)) != 0):
        return False
    return True

def get_closest_square(start_num, direction='ceil'):
    if is_perfect_square(start_num):
        return start_num
    
    if direction == 'ceil':
        num = start_num + 1
        while True:
            if is_perfect_square(num):
                return num
            num += 1
    elif direction == 'floor':
        num = start_num - 1
        while True:
            if is_perfect_square(num):
                return num
            num -= 1

def store_current_log_file_path(log_path):
    with open(last_cli_log_file_path, 'w') as file:
        file.write(log_path)

def get_current_log_file_path():
    if not os.path.exists(last_cli_log_file_path):
        return
    
    with open(last_cli_log_file_path, 'r') as file:
        log_path = file.read()
    return log_path

def parse_log_file(log_path=None):
    if log_path is None:
        log_path = get_current_log_file_path()
        if log_path is None:
            return '', []
    
    with open(log_path, 'r') as file:
        log_text = file.read()
    
    errors = re.findall(r'(^\[ERROR\]: [\w\W]*?^)\^.*', log_text, re.M)
    tracebacks = re.findall(
        r'^Traceback[\w\W]+?(?=^\[|\Z)', log_text, re.M | re.X
    )
    
    errors.extend(tracebacks)
    
    return log_path, errors

def resolve_path(rel_or_abs_path, abs_path=''):
    if os.path.isabs(rel_or_abs_path):
        return rel_or_abs_path
    parts = rel_or_abs_path.replace('\\', '/').split('/')
    
def get_runs_num_and_desc(exp_path, pos_foldernames=None):
    if pos_foldernames is None:
        pos_foldernames = acdc_myutils.get_pos_foldernames(exp_path)
    pattern = DFs_FILENAMES['spots_detection']
    pattern = pattern.replace('*rn*', r'(\d+)')
    pattern = pattern.replace('*desc*', r'(_?.*)_aggregated\.')
    
    all_runs = set()
    for pos in pos_foldernames:
        pos_path = os.path.join(exp_path, pos)
        spotmax_output_path = os.path.join(pos_path, 'spotMAX_output')
        if not os.path.exists(spotmax_output_path):
            continue
        
        for file in acdc_myutils.listdir(spotmax_output_path):
            m = re.findall(pattern, file)
            all_runs.update(m)
    
    return all_runs

def to_dtype(value, dtype):
    if dtype == bool:
        if isinstance(value, str):
            if value.lower() in valid_true_bool_str:
                return True
            elif value.lower() in valid_false_bool_str:
                return False
            else:
                error = f'"{value}" is not a valid boolean expression.'
        elif isinstance(value, bool):
            return bool
        else:
            error = f'"{value}" is not a valid boolean expression.'
    else:
        return dtype(value)
    
    raise TypeError(error)    

def get_local_backgr_ring_width_pixel(
        local_background_ring_width: str, pixel_size: float
    ):
    try:
        value, unit = local_background_ring_width.split()
    except ValueError as err:
        raise ValueError(
            'Local background ring width unit is not specified. '
            'It must be either `pixel` or `micrometre`.'
        )
    
    if unit not in ('pixel', 'micrometre'):
        raise ValueError(
            f'`{unit}` is not a valid unit for the local background ring width. '
            'It must be either `pixel` or `micrometre`.'
        )
    
    if unit == 'pixel':
        return round(float(value))
    
    value_pixel = float(value) / pixel_size
    return round(value_pixel)

def get_spotfit_image(df_spotfit: pd.DataFrame, shape: Tuple[int, int, int]):
    if len(shape) == 2:
        shape = (1, *shape)
    
    model = core.GaussianModel()
    
    img = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.uint32)
    
    SizeZ, SizeY, SizeX = shape
    
    spheroid = core.Spheroid(img, show_progress=False)
    
    pixel_size_cols = ['voxel_size_z', 'pixel_size_y', 'pixel_size_x']
    zyx_pixel_size = df_spotfit.iloc[0][pixel_size_cols].to_numpy()
    
    for row in df_spotfit.itertuples():
        ID, spot_id = row.Index
        
        zyx_center = np.array(
            (row.z_fit, row.y_fit, row.x_fit)
        ).round().astype(int)
        zyx_sigmas = np.array(
            (row.sigma_z_fit, row.sigma_y_fit, row.sigma_x_fit)
        )
        spot_zyx_radii = zyx_sigmas*2
        
        semiax_len = spheroid.calc_semiax_len(0, zyx_pixel_size, spot_zyx_radii)
        local_spot_mask = spheroid.get_local_spot_mask(semiax_len)
        mask, _, slice_to_local, slice_crop = (
            spheroid.index_local_into_global_mask(
                mask, local_spot_mask, zyx_center, semiax_len, 
                SizeZ, SizeY, SizeX, return_slice=True
            )
        )
        local_spot_mask = local_spot_mask[slice_crop]
        labels[slice_to_local][local_spot_mask] = spot_id
        
        local_zz, local_yy, local_xx = np.nonzero(local_spot_mask)
        local_zyx_center = zyx_center - [s.start for s in slice_to_local]
        
        coeffs = (*local_zyx_center, *zyx_sigmas, row.A_fit)
        B = row.spot_B_fit
        spot_vals = model.func(local_zz, local_yy, local_xx, coeffs, B=B)
        
        img[slice_to_local][local_zz, local_yy, local_xx] += spot_vals
        
    return img, mask, labels

def sort_strings_by_template(iterable, template: str):
    sorted_list = sorted(
        iterable, 
        key=lambda x: difflib.SequenceMatcher(a=x, b=template).ratio(), 
        reverse=True
    )
    return sorted_list

def _load_spots_masks_training_workflow(pos_path, masks_endname):
    masks_filepath = acdc_load.search_filepath_in_pos_path_from_endname(
            pos_path, masks_endname
    )
    if masks_filepath is None:
        print(
            '[WARNING]: The following position does not have the '
            f'{masks_endname} spots masks file. Skipping it. '
            f'"{pos_path}"'
        )
        return 
    
    spots_masks = acdc_load.load_image_file(masks_filepath)
    return spots_masks

def _generate_spots_masks_training_workflow(
        pos_path, img_data, spheroid_radii, spots_coords_endname
    ):
    spots_coords_filepath = (
        acdc_load.search_filepath_in_pos_path_from_endname(
            pos_path, spots_coords_endname
    ))
    df_spots = io.load_table_to_df(spots_coords_filepath)
    
    if img_data.ndim == 2:
        img_data = img_data[np.newaxis]
        df_spots['z'] = 1

    try:
        xy_radius = int(spheroid_radii)
        spheroid_radii = (1, xy_radius)
    except Exception as err:
        pass
            
    if spots_coords_filepath is None:
        print(
            '\n[WARNING]: The following position does not have the '
            f'{spots_coords_endname} spots coords file. Skipping it. '
            f'"{pos_path}"'
        )
        return 
    
    if img_data.ndim == 4:
        spots_masks = np.zeros(img_data.shape, dtype=bool)
        if 't' in df_spots.columns:
            df_spots['frame_i'] = df_spots['t']
    else:
        df_spots['frame_i'] = 0
        spots_masks = np.zeros((1, *img_data.shape), dtype=bool)
    
    for t, df_t in df_spots.groupby('frame_i'):
        zyx_spot_centers = (
            df_t[['z', 'y', 'x']].round().to_numpy().astype(int)
        )
        spheroid = core.Spheroid(img_data, show_progress=False)
        spots_masks_t = spheroid.get_spots_mask(
            0, None, None, zyx_spot_centers, 
            semiax_len=spheroid_radii
        )
        spots_masks[t] = spots_masks_t
    
    return spots_masks

def _crop_background_training_workflow(
        img_data, spots_masks, crop_background_pad
    ):
    img_data = transformations.add_missing_axes_4D(img_data)         
    crop_info = transformations.crop_from_segm_data_info(
        spots_masks, crop_background_pad
    )
    segm_slice, pad_widths, crop_to_global_coords = crop_info
    img_data = img_data[segm_slice]
    img_data = np.pad(
        img_data, pad_widths, constant_values=img_data.min()
    )
    return img_data

def generate_dataset_training_workflow(
        exp_path, 
        channel_names_exp, 
        training_positions, 
        val_positions, 
        pixel_size,
        datasets_folderpath,
        masks_endnames=None,
        spot_masks_size=None,
        spots_coords_endnames=None,
        data_augment_params=None, 
        crop_background=True,
        crop_background_pad=5, 
        visualize=False
    ):
    os.makedirs(datasets_folderpath, exist_ok=True)
    search_file_func = acdc_load.search_filepath_in_pos_path_from_endname
    positions_mapper = {
        'TRAIN': training_positions, 
        'VAL': val_positions
    }
    pbar_total = len(training_positions) + len(val_positions)
    pbar = tqdm(total=pbar_total, ncols=100, leave=False, position=1, unit='pos')
    for category, positions in positions_mapper.items():
        X_list = []
        y_list = []
        for pos_foldername in positions:
            pos_path = os.path.join(exp_path, pos_foldername)
            for ch, channel_name in enumerate(channel_names_exp):
                channel_filepath = search_file_func(
                    pos_path, channel_name
                )
                img_data = acdc_load.load_image_file(channel_filepath)
                
                if masks_endnames is not None:
                    masks_endname = masks_endnames[exp_path][ch]
                    spots_masks = _load_spots_masks_training_workflow(
                        pos_path, masks_endname
                    )
                else:
                    spheroid_radii = spot_masks_size[exp_path]
                    spots_coords_endname = spots_coords_endnames[exp_path][ch]
                    spots_masks = _generate_spots_masks_training_workflow(
                        pos_path, img_data, spheroid_radii, 
                        spots_coords_endname
                    )
                
                if spots_masks is None:
                    continue
                
                Y, X = img_data.shape[-2:]
                flat_2d_spots_masks = spots_masks.reshape(-1, Y, X)
                y_list.extend(flat_2d_spots_masks)
                
                # Data augmentation
                for da_section, filter_kwargs in data_augment_params.items():
                    filter_module = da_section.split(';')[-1]
                    module_parts = filter_module.split('.')
                    module_name = '.'.join(module_parts[:-1])
                    filter_name = module_parts[-1]
                    module = import_module(module_name)
                    filter_func = getattr(module, filter_name)
                    filtered_img = filter_func(img_data, **filter_kwargs)
                    if crop_background:
                        filtered_img = _crop_background_training_workflow(
                            filtered_img, spots_masks, crop_background_pad
                        )
                    flat_2d_filtered_img = filtered_img.reshape(-1, Y, X)
                    X_list.extend(flat_2d_filtered_img)
                    y_list.extend(flat_2d_spots_masks)
                
                if crop_background:
                    img_data = _crop_background_training_workflow(
                        img_data, spots_masks, crop_background_pad
                    )
                    
                flat_2d_img_data = img_data.reshape(-1, Y, X)
                X_list.extend(flat_2d_img_data)
                
                if visualize:
                    imshow(flat_2d_img_data, flat_2d_spots_masks)
                    import pdb; pdb.set_trace()
                
                pbar.update()
        pbar.close()
        
        exp_path_parts = exp_path.replace('\\', '/').split('/')
        ep = exp_path_parts
        h5_filename = f'{ep[-3]}_{ep[-2]}_{ep[-1]}_{category}.h5'
        h5_filepath = os.path.join(datasets_folderpath, h5_filename)
        i = 4
        while os.path.exists(h5_filepath):
            try:
                h5_filename = f'{ep[-i]}_{h5_filename}'
            except IndexError:
                i = 1
                h5_filename = f'{i:02d}_{h5_filename}'
            h5_filepath = os.path.join(datasets_folderpath, h5_filename)
            i += 1
            
        dset = h5py.File(h5_filepath, 'w')
        dset['pixel_size'] = pixel_size
        dset['X'] = np.array(X_list)
        dset['y'] = np.array(y_list)

def generate_unet_training_workflow_files(
        src_train_pos_paths: Dict[str, List[str]], 
        src_val_pos_paths: Dict[str, List[str]], 
        channel_names: Dict[str, Union[str, List[str]]],
        workflow_filepath: os.PathLike,
        pixel_sizes: Dict[str, float],
        rescale_to_pixel_size: float=-1.0,
        model_size: str='Large',
        spots_coords_endnames: Optional[Dict[str, str]]=None, 
        masks_endnames: Optional[Dict[str, str]]=None, 
        spot_masks_size: Optional[Union[Dict[str, float], Dict[str, Tuple[float]]]]=None,
        crops_shapes: Optional[Tuple[float]]=(256, 256), 
        max_number_of_crops: int=-1,
        data_augment_params: Optional[Dict[str, dict]]=None, 
        crop_background: bool=True,
        crop_background_pad: int=5,
        visualize: bool=False, 
        do_not_generate_datasets=False,
    ):
    """Generate training workflow files to train SpotMAX AI model. 

    Parameters
    ----------
    src_train_pos_paths : Dict[str, List[str]]
        Dictionary with experiment folder paths as keys and list of Position 
        folders to use as training positions as values.
    src_val_pos_paths : Dict[str, List[str]]
        Dictionary with experiment folder paths as keys and list of Position 
        folders to use as validation positions as values.
    channel_names : Dict[str, str | List[str]]
        Dictionary with experiment folder paths as keys and channel names 
        for the spots channel images as values. A list of multiple channels 
        can also be passed for each experiment folder. 
    workflow_filepath : os.PathLike
        Filepath of the generated INI workflow file. The extension is '.ini'.
    pixel_sizes : Dict[str, float]
        Dictionary with experiment folder paths as keys and single number for 
        the pixel size in x- and y-direction as value.
    rescale_to_pixel_size : float, optional
        Single number representing the pixel size target of image rescaling. 
        For example, if the pixel size is 0.1 and the rescale pixel size is 
        0.2, the images will be upscaled by factor 2. This is useful if you 
        want to predict on images with variable pixel size. Note that rescaling 
        if performed when you run the training routine, not in this function. 
        If -1.0 do not rescale images. Default is -1.0
    model_size : {'Large', 'Medium', 'Small'}, optional
        Model size. The larger the model, the more parameters it has. 
    spots_coords_endnames : Optional[Dict[str, str]], optional
        Dictionary with experiment folder paths as keys and the endname of 
        the table file with coordinates of the spots as values. This table must 
        have the 'x', and 'y' column with additional 'z' column for 3D 
        z-stacks images, and 't' or 'frame_i' column for timelapse data. 
        Default is None
    masks_endnames : Optional[Dict[str, str]], optional
        Dictionary with experiment folder paths as keys and the endname of 
        the file with the spots masks (same shape of the image data) as values. 
        Default is None
    spot_masks_size : Optional[Union[Dict[str, float], Dict[str, Tuple[float]]]], optional
        Dictionary with experiment folder paths as keys and (y, x) or 
        (z, y, x) size of the spots (in pixels) as values. These values will 
        be used to generate the spheroid spots masks together with the 
        relative `spots_coords_endnames`. Default is None
    crops_shapes : Optional[Tuple[float]], optional
        (Y, X) values for the target shape of the single images input of 
        the neural network. The larger the shape, the more memory required 
        on the GPU. Default is (256, 256)
    max_number_of_crops : int, optional
        Maximum number of crops per image. A value of -1 means no upper 
        limit to the number of crops. Default is -1
    data_augment_params : Optional[Dict[str, dict]], optional
        Dictionary of with 'N;importable_function' as keys 
        (e.g. '1:spotmax.filters.gaussian') and keyword arguments of the 
        relative function as values. 
        The keys can be any importable function, e.g. 'spotmax.filters.gaussian' 
        for a Gaussian filter. Default is None
    crop_background : bool, optional
        If True, crop as much background as possible around the spots 
        masks. This is useful if you have multiple cells in the image 
        but you did not annotate all of them. Default is True
    crop_background_pad : float, optional
        Number of pixels for padding the spots masks when cropping the 
        background (when `crop_background` is True). Default is 5.
    visualize : bool, optional
        If True, generation will pause and you will be able to visualize the
        loaded images and spots masks. Default is False
    """    
    cp = config.ConfigParser()
    workflow_folderpath = os.path.dirname(workflow_filepath).replace('\\', '/')
    datasets_folderpath = f'{workflow_folderpath}/datasets'
    
    if data_augment_params is None:
        data_augment_params = {}
        
    for filter_module, filter_kwargs in data_augment_params.items():
        cp[f'data_augmentation_{filter_module}'] = {
            kwarg:str(value) for kwarg, value in filter_kwargs.items()
        }
            
    training_params = {}
    exp_paths = src_train_pos_paths.keys()
    
    for exp_path in tqdm(exp_paths, ncols=100, unit='exp'):
        exp_path = exp_path.replace('\\', '/')
        exp_path_params = {}
        channel_names_exp = channel_names.get(exp_path)
        if channel_names_exp is not None:
            if isinstance(channel_names_exp, str):
                channel_names_exp = [channel_names_exp]
            channels = '\n'.join(channel_names_exp)
            exp_path_params['channels'] = channels
        
        pixel_size = pixel_sizes.get(exp_path)
        if pixel_size is not None:
            exp_path_params['pixel_size'] = pixel_size

        if masks_endnames is not None:
            masks_endnames_exp = masks_endnames.get(exp_path)
            if masks_endnames_exp is not None:
                if isinstance(masks_endnames_exp, str):
                    masks_endnames_exp = [masks_endnames_exp]
                masks_endnames_exp_str = '\n'.join(channel_names_exp)
                exp_path_params['masks_endnames'] = masks_endnames_exp_str
        
        if spots_coords_endnames is not None:
            spots_coords_endnames_exp = spots_coords_endnames.get(exp_path)
            if spots_coords_endnames_exp is not None:
                if isinstance(spots_coords_endnames_exp, str):
                    spots_coords_endnames_exp = [spots_coords_endnames_exp]
                spots_coords_endnames_exp_str = '\n'.join(
                    spots_coords_endnames_exp
                )
                exp_path_params['spots_coords_endnames'] = (
                    spots_coords_endnames_exp_str
                )
        
        if spot_masks_size is not None:
            spot_masks_size_exp = spot_masks_size.get(exp_path)
            if spot_masks_size_exp is not None:
                exp_path_params['spot_masks_size'] = str(spot_masks_size_exp)

        training_positions = src_train_pos_paths[exp_path]
        training_positions_str = '\n'.join(training_positions)
        exp_path_params['training_positions'] = training_positions_str
        
        val_positions = []
        try:
            val_positions = src_val_pos_paths[exp_path]
            val_positions_str = '\n'.join(val_positions)
        except KeyError:
            val_positions_str = ''
        
        exp_path_params['validation_positions'] = val_positions_str
        
        cp[exp_path] = exp_path_params
        
        if do_not_generate_datasets:
            continue
        
        generate_dataset_training_workflow(
            exp_path, 
            channel_names_exp, 
            training_positions, 
            val_positions, 
            pixel_size,
            datasets_folderpath,
            masks_endnames=masks_endnames,
            spot_masks_size=spot_masks_size,
            spots_coords_endnames=spots_coords_endnames,
            data_augment_params=data_augment_params, 
            crop_background=crop_background,
            crop_background_pad=crop_background_pad, 
            visualize=visualize
        )

    if model_size:
        training_params['model_size'] = model_size
    
    if crops_shapes:
        training_params['crops_shapes'] = str(crops_shapes)
        
    if max_number_of_crops:
        training_params['max_number_of_crops'] = str(max_number_of_crops)
    
    training_params['rescale_to_pixel_size'] = str(rescale_to_pixel_size)

    training_params['crop_background'] = str(crop_background)
    training_params['crop_background_pad'] = str(crop_background_pad)
    
    cp['training_params'] = training_params
    
    with open(workflow_filepath, 'w', encoding="utf-8") as ini:
        cp.write(ini)
    
    if do_not_generate_datasets:
        return
    
    with open(config_yaml_path, 'r') as yaml_file:
        config_yaml = yaml.safe_load(yaml_file)
        
    workflow_filename = os.path.basename(workflow_filepath)[:-4]
    checkpoint_3d_folderpath = (
        f'{workflow_folderpath}/unet_checkpoints/unet3D'
    )
    config_yaml['unet3D']['train']['trainer']['checkpoint_dir'] = (
        f'{checkpoint_3d_folderpath}/training'
    )
    config_yaml['unet3D']['predict']['model_path'] = (
        f'{checkpoint_3d_folderpath}/{workflow_filename}.pytorch'
    )

    checkpoint_2d_folderpath = (
        f'{workflow_folderpath}/unet_checkpoints/unet2D'
    )
    config_yaml['unet2D']['model']['model_dir'] = (
        f'{checkpoint_2d_folderpath}'
    )
    config_yaml['unet2D']['model']['best_model_path'] = (
        f'{checkpoint_2d_folderpath}/{workflow_filename}.pth'
    )
    config_yaml['unet2D']['model']['training_path'] = (
        f'{checkpoint_2d_folderpath}/training'
    )
    config_yaml['base_pixel_size_nm'] = rescale_to_pixel_size
    
    config_yaml_name = f'{workflow_filename}.yaml'
    new_config_yaml_path = os.path.join(workflow_folderpath, config_yaml_name)
    with open(new_config_yaml_path, 'w') as yaml_file:
        yaml.dump(config_yaml, yaml_file)

def random_choice_pos_foldernames(pos_foldernames, train_perc=80, val_perc=20):
    num_pos = len(pos_foldernames)
    num_train_pos = int(np.ceil(num_pos*train_perc/100))
    num_val_pos = num_pos - num_train_pos
    train_positions = rng.choice(
        pos_foldernames, num_train_pos, replace=False
    )
    val_positions = [
        pos for pos in pos_foldernames if pos not in train_positions
    ]
    return train_positions, val_positions

def get_info_version_text(is_cli=False, include_platform=True):
    from spotmax import read_version, spotmax_path
    from cellacdc.myutils import get_date_from_version
    version = read_version()
    release_date = get_date_from_version(version, package='spotmax')
    py_ver = sys.version_info
    python_version = f'{py_ver.major}.{py_ver.minor}.{py_ver.micro}'
    info_txts = [
        f'Version {version}',
        f'Released on: {release_date}',
        f'Installed in "{spotmax_path}"',
    ]
    
    if include_platform:
        import platform
        info_txts.extend([
            f'Python {python_version}',
            f'Platform: {platform.platform()}',
            f'System: {platform.system()}',
        ])
        if GUI_INSTALLED and not is_cli:
            try:
                from qtpy import QtCore
                info_txts.append(f'Qt {QtCore.__version__}')
            except Exception as err:
                info_txts.append('Qt: Not installed')
        
        info_txts.append(f'Working directory: {os.getcwd()}')
        
    info_txts = [f'  - {txt}' for txt in info_txts]
    
    max_len = max([len(txt) for txt in info_txts]) + 2
    
    formatted_info_txts = []
    for txt in info_txts:
        horiz_spacing = ' '*(max_len - len(txt))
        txt = f'{txt}{horiz_spacing}|'
        formatted_info_txts.append(txt)
    
    formatted_info_txts.insert(0, 'SpotMAX info:\n')
    formatted_info_txts.insert(0, '='*max_len)
    formatted_info_txts.append('='*max_len)
    info_txt = '\n'.join(formatted_info_txts)
    
    return info_txt
