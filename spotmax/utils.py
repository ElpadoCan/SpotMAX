import os
import sys
import pandas as pd
import datetime
import time
import math
import cv2
import logging
import traceback
import numpy as np
import tkinter as tk
import pathlib
import configparser
from functools import wraps, partial
from urllib.parse import urlparse
from natsort import natsort_keygen

from tifffile.tifffile import TiffWriter, TiffFile

import skimage.color
import colorsys
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

from PyQt5.QtCore import QTimer

from . import config, widgets

def exception_handler(func):
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
            self.logger.exception(e)
            msg = widgets.myMessageBox()
            err_msg = (f"""
            <p style="font-size:13px">
                Error in function <b>{func.__name__}</b> when trying to
                {self.funcDescription}.<br><br>
                More details below or in the terminal/console.<br><br>
                Note that the error details from this session are also saved
                in the file<br>
                {self.log_path}<br><br>
                Please <b>send the log file</b> when reporting a bug, thanks!
            </p>
            """)
            msg = widgets.myMessageBox()
            msg.setDetailedText(traceback.format_exc())
            msg.addShowInFileManagerButton(self.logs_path, 'Show log file...')
            msg.critical(
                self, 'Critical error', err_msg
            )
            self.loadingDataAborted()
        return result
    return inner_function

def is_valid_url(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except Exception as e:
        return False

def _resizeWarningHandler(msg_type, msg_log_context, msg_string):
    if msg_string.find('Unable to set geometry') != -1:
        self.timer.stop()
    elif msg_string:
        print(msg_string)

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
        config.settings_path, 'recentPaths.csv'
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
    return [f for f in os.listdir(path) if not f.startswith('.')]

def setupLogger():
    logger = logging.getLogger('spotMAX')
    logger.setLevel(logging.INFO)

    src_path = os.path.dirname(os.path.abspath(__file__))
    logs_path = os.path.join(src_path, 'logs')
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    else:
        # Keep 20 most recent logs
        ls = listdir(logs_path)
        if len(ls)>20:
            ls = [os.path.join(logs_path, f) for f in ls]
            ls.sort(key=lambda x: os.path.getmtime(x))
            for file in ls[:-20]:
                os.remove(file)

    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f'{date_time}_stdout.log'
    log_path = os.path.join(logs_path, log_filename)

    output_file_handler = logging.FileHandler(log_path, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)

    # Format your logs (optional)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s:\n'
        '------------------------\n'
        '%(message)s\n'
        '------------------------\n',
        datefmt='%d-%m-%Y, %H:%M:%S')
    output_file_handler.setFormatter(formatter)

    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)

    return logger, log_path, logs_path

def _bytes_to_MB(size_bytes):
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
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
    signals : pyqtSignals or None
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
            axis1_interval = (0, Z)
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
        '|norm|_spot': 'Spot normalised mean intesity',
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

class imshow_tk:
    def __init__(self, img, dots_coords=None, x_idx=1, axis=None,
                       additional_imgs=[], titles=[], fixed_vrange=False,
                       run=True):
        if img.ndim == 3:
            if img.shape[-1] > 4:
                img = img.max(axis=0)
                h, w = img.shape
            else:
                h, w, _ = img.shape
        elif img.ndim == 2:
            h, w = img.shape
        elif img.ndim != 2 and img.ndim != 3:
            raise TypeError(f'Invalid shape {img.shape} for image data. '
            'Only 2D or 3D images.')
        for i, im in enumerate(additional_imgs):
            if im.ndim == 3 and im.shape[-1] > 4:
                additional_imgs[i] = im.max(axis=0)
            elif im.ndim != 2 and im.ndim != 3:
                raise TypeError(f'Invalid shape {im.shape} for image data. '
                'Only 2D or 3D images.')
        n_imgs = len(additional_imgs)+1
        if w/h > 1:
            fig, ax = plt.subplots(n_imgs, 1, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots(1, n_imgs, sharex=True, sharey=True)
        if n_imgs == 1:
            ax = [ax]
        self.ax0img = ax[0].imshow(img)
        if dots_coords is not None:
            ax[0].plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
        if axis:
            ax[0].axis('off')
        if fixed_vrange:
            vmin, vmax = img.min(), img.max()
        else:
            vmin, vmax = None, None
        self.additional_aximgs = []
        for i, img_i in enumerate(additional_imgs):
            axi_img = ax[i+1].imshow(img_i, vmin=vmin, vmax=vmax)
            self.additional_aximgs.append(axi_img)
            if dots_coords is not None:
                ax[i+1].plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
            if axis:
                ax[i+1].axis('off')
        for title, a in zip(titles, ax):
            a.set_title(title)
        sub_win = embed_tk('Imshow embedded in tk', [800,600,400,150], fig)
        sub_win.root.protocol("WM_DELETE_WINDOW", self._close)
        self.sub_win = sub_win
        self.fig = fig
        self.ax = ax
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        if run:
            sub_win.root.mainloop()

    def _close(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

class embed_tk:
    """Example:
    -----------
    img = np.ones((600,600))
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    ax.imshow(img)

    sub_win = embed_tk('Embeddding in tk', [1024,768,300,100], fig)

    def on_key_event(event):
        print('you pressed %s' % event.key)

    sub_win.canvas.mpl_connect('key_press_event', on_key_event)

    sub_win.root.mainloop()
    """
    def __init__(self, win_title, geom, fig):
        root = tk.Tk()
        root.wm_title(win_title)
        root.geometry("{}x{}+{}+{}".format(*geom)) # WidthxHeight+Left+Top
        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas = canvas
        self.toolbar = toolbar
        self.root = root

if __name__ == '__main__':
    dset = np.random.randint(1,255, size=(16,50,50))

    loadSizeZ = 10
    SizeZ = dset.shape[-3]

    midZ = int(SizeZ/2)
    halfZ = int(loadSizeZ/2)
    z0 = midZ-halfZ
    z1 = midZ+halfZ
    z0_window = z0

    a = dset[z0:z1]

    print(a.shape)

    direction = 'backward'
    chunkSizeZ = 1
