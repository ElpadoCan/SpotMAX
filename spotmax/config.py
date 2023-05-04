print('Configuring files...')
import os
import re
import json
import pathlib
from pprint import pprint
import pandas as pd
import configparser
import skimage.filters

from . import GUI_INSTALLED

if GUI_INSTALLED:
    from PyQt5.QtGui import QFont
    from PyQt5.QtCore import QObject, pyqtSignal, qInstallMessageHandler

    from cellacdc import widgets as acdc_widgets

    from . import widgets
else:
    from . import utils

from . import io, colorItems_path, default_ini_path

class ConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, allow_no_value=True, **kwargs)
        self.optionxform = str
    
    def read(self, filepath, encoding='utf-8'):
        super().read(filepath, encoding=encoding)
        self._filename = os.path.basename(filepath)
        self._filepath = filepath

    def filepath(self):
        return self._filepath

    def filename(self):
        return self._filename

def initColorItems():
    if os.path.exists(colorItems_path):
        return

    colors = {
      "left": {
        "Image": None,
        "Overlay image": [0, 255, 255, 255],
        "Text on segmented objects": [255, 255, 255, 255],
        "Contours of segmented objects": [255, 0, 0, 255],
        "Contour color...": [255, 0, 0, 255],
        "Clicked spot": [255, 0, 0, 255],
        "Spots inside ref. channel": [255, 0, 0, 1],
        "Spots outside ref. channel": [255, 0, 0, 1],
        "Skeleton color...": [0, 255, 255, 255]
      },
      "right": {
        "Image": None,
        "Overlay image": [255, 0, 255, 255],
        "Text on segmented objects": [255, 255, 255, 255],
        "Contours of segmented objects": [255, 0, 0, 255],
        "Contour color...": [255, 0, 0, 255],
        "Clicked spot": [255, 0, 0, 255],
        "Spots inside ref. channel": [255, 0, 0, 255],
        "Spots outside ref. channel": [255, 0, 0, 255],
        "Skeleton color...": [255, 0, 0, 255]
      }
    }

    with open(colorItems_path, mode='w') as file:
        json.dump(colors, file, indent=2)

def font(pixelSizeDelta=0):
    normalPixelSize = 13
    font = QFont()
    font.setPixelSize(normalPixelSize+pixelSizeDelta)
    return font

def get_bool(text):
    if isinstance(text, bool):
        return text
    if text.lower() == 'yes':
        return True
    if text.lower() == 'no':
        return False
    if text.lower() == 'true':
        return True
    if text.lower() == 'false':
        return False
    raise TypeError(f'The object "{text}" cannot be converted to a valid boolean object')

def get_threshold_func(func_name):
    return getattr(skimage.filters, func_name)

def get_valid_text(text):
    return re.sub('[^\w\-.]', '_', text)

def parse_threshold_func(threshold_func):
    if isinstance(threshold_func, str):
        return threshold_func
    else:
        return threshold_func.__name__

def get_exp_paths(exp_paths):
    # Remove white spaces at the start
    exp_paths = exp_paths.lstrip()

    # Remove brackets at the start if user provided a list
    exp_paths = exp_paths.lstrip('[')
    exp_paths = exp_paths.lstrip('(')

    # Remove white spaces at the ena
    exp_paths = exp_paths.rstrip()

    # Remove brackets at the end if user provided a list
    exp_paths = exp_paths.rstrip(']')
    exp_paths = exp_paths.rstrip(')')

    # Replace commas with end of line
    exp_paths = exp_paths.replace('\n',',')

    # Replace eventual double commas with comma
    exp_paths = exp_paths.replace(',,',',')

    # Split paths and remove possible end charachters 
    exp_paths = exp_paths.split(',')
    exp_paths = [path.strip() for path in exp_paths if path]
    exp_paths = [path.rstrip('\\') for path in exp_paths if path]
    exp_paths = [path.rstrip('/') for path in exp_paths if path]

    exp_paths = [io.get_abspath(path) for path in exp_paths]
    return exp_paths

def get_log_folderpath(folder_path):
    # User can provide the home path as '~'
    log_path = folder_path.replace(r'%userprofile%', '~')
    log_path = io.to_system_path(log_path)
    log_path = os.path.expanduser(log_path)
    log_path = io.get_abspath(log_path)
    return log_path

def parse_log_folderpath(log_path):
    log_path = io.get_abspath(log_path)
    try:
        log_path = pathlib.Path(log_path).relative_to(pathlib.Path.home())
        log_path = os.path.normpath(f'~{os.sep}{log_path}')
    except ValueError as e:
        log_path = log_path
    return log_path

def parse_list_to_configpars(iterable: list):
    if isinstance(iterable, str):
        iterable = [iterable]
    
    li_str = [f'\n{p}' for p in iterable]
    li_str = ''.join(li_str)
    return li_str

def gop_thresholds_comment():
    s = (
        '# Save the features to use for filtering truw spots as `feature_name,max,min`.\n'
        '# You can write as many features as you want. Write each feature on its own indented line.\n'
        '# Example: `spot_vs_ref_ch_ttest_pvalue,None,0.025` means `keep only spots whose p-value\n'
        '# is smaller than 0.025` where `None` indicates that there is no minimum.'
    )
    return s

def get_gop_thresholds(gop_thresholds_to_parse):
    """_summary_

    Parameters
    ----------
    gop_thresholds_to_parse : str
        String formatted to contain feature names and min,max values to use 
        when filtering spots in goodness-of-peak test.

        Multiple features are separated by the "/" charachter. Feature name and 
        thresholds values are separated by comma.

        Examples:
            `spot_vs_bkgr_glass_effect_size,0.8,None`: Filter all the spots 
            that have the Glass' effect size greater than 0.8. There is no max 
            set.
    """    
    features_thresholds = gop_thresholds_to_parse.split('\n')
    gop_thresholds = {}
    for feature_thresholds in features_thresholds:
        feature_name, *thresholds_str = feature_thresholds.split(',')
        if not feature_name:
            continue
        thresholds = [None, None]
        for t, thresh in enumerate(thresholds_str):
            try:
                thresholds[t] = float(thresh)
            except Exception as e:
                pass
        gop_thresholds[feature_name] = tuple(thresholds)
    return gop_thresholds

def _filepaths_params():
    filepaths_params = {
        'filePathsToAnalyse': {
            'desc': 'Experiment folder path(s) to analyse',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': True,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': get_exp_paths,
            'parser': parse_list_to_configpars,
            'editButtonCallback': 'dock_params_callbacks.editFilePathsToAnalyse'
        },
        'spotsEndName': {
            'desc': 'Spots channel end name or path',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str
        },
        'segmEndName': {
            'desc': 'Cells segmentation end name or path',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str
        },
        'refChEndName': {
            'desc': 'Reference channel end name or path',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str
        },
        'refChSegmEndName': {
            'desc': 'Ref. channel segmentation end name or path',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str
        },
        'lineageTableEndName': {
            'desc': 'Table with lineage info end name or path',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str
        },
        'runNumber': {
            'desc': 'Run number',
            'initialVal': 1,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addEditButton': False,
            'formWidgetFunc': 'acdc_widgets.SpinBox',
            'actions': None,
            'dtype': int
        },
        'textToAppend': {
            'desc': 'Text to append at the end of the output files',
            'initialVal': '',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': get_valid_text
        },
        'dfSpotsFileExtension': {
            'desc': 'File extension of the output tables',
            'initialVal': '.h5',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._dfSpotsFileExtensionsWidget',
            'actions': None,
            'dtype': str, 
            'parser_arg': 'output_tables_file_ext'
        },
    }
    return filepaths_params

def _configuration_params():
    config_params = {
        'pathToLog': {
            'desc': 'Folder path of the log file',
            'initialVal': f'~{os.sep}{os.path.join("spotmax_appdata", "logs")}',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': get_log_folderpath, 
            'parser': parse_log_folderpath,
            'parser_arg': 'log_folderpath'
        },
        'pathToReport': {
            'desc': 'Folder path of the final report',
            'initialVal': '',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': io.get_abspath,
            'parser_arg': 'report_folderpath'
        },
        'reportFilename': {
            'desc': 'Filename of final report',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str, 
            'parser_arg': 'report_filename'
        },
        'disableFinalReport': {
            'desc': 'Disable saving of the final report',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addEditButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool, 
            'parser_arg': 'disable_final_report'
        },
        'forceDefaultValues': {
            'desc': 'Use default values for missing parameters',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool, 
            'parser_arg': 'force_default_values'
        },
        'raiseOnCritical': {
            'desc': 'Stop analysis on critical error',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool, 
            'parser_arg': 'raise_on_critical'
        },
        'useGpu': {
            'desc': 'Use CUDA-compatible GPU',
            'initialVal': True,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool, 
            'parser_arg': 'gpu'
        },
        'numbaNumThreads': {
            'desc': 'Number of threads used by numba',
            'initialVal': -1,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'acdc_widgets.SpinBox',
            'actions': None,
            'dtype': int, 
            'parser_arg': 'num_threads'
        },
        'reduceVerbosity': {
            'desc': 'Reduce logging verbosity',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool, 
            'parser_arg': 'reduce_verbosity'
        },
    }
    return config_params

def _metadata_params():
    metadata_params = {
        'SizeT': {
            'desc': 'Number of frames (SizeT)',
            'initialVal': 1,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'widgets.intLineEdit',
            'actions': None,
            'dtype': int
        },
        'stopFrameNum': {
            'desc': 'Analyse until frame number',
            'initialVal': 1,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'widgets.intLineEdit',
            'actions': None,
            'dtype': int
        },
        'SizeZ': {
            'desc': 'Number of z-slices (SizeZ)',
            'initialVal': 1,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'widgets.intLineEdit',
            'actions': None,
            'dtype': int
        },
        'pixelWidth': {
            'desc': 'Pixel width (μm)',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.floatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float
        },
        'pixelHeight': {
            'desc': 'Pixel height (μm)',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.floatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float
        },
        'voxelDepth': {
            'desc': 'Voxel depth (μm)',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.floatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float
        },
        'numAperture': {
            'desc': 'Numerical aperture',
            'initialVal': 1.4,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.floatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float
        },
        'emWavelen': {
            'desc': 'Spots reporter emission wavelength (nm)',
            'initialVal': 500.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.floatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float
        },
        'zResolutionLimit': {
            'desc': 'Spot minimum z-size (μm)',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.floatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float,
            'autoTuneWidget': 'widgets.ReadOnlyLineEdit'
        },
        'yxResolLimitMultiplier': {
            'desc': 'Resolution multiplier in y- and x- direction',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.floatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float,
            'autoTuneWidget': 'widgets.ReadOnlyLineEdit'
        },
        'spotMinSizeLabels': {
            'desc': 'Spot (z,y,x) minimum dimensions',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'formWidgetFunc': 'widgets._spotMinSizeLabels',
            'actions': None,
            'isParam': False
        }
    }
    return metadata_params

def _pre_processing_params():
    pre_processing_params = {
        'aggregate': {
            'desc': 'Aggregate cells prior analysis',
            'initialVal': True,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'gaussSigma': {
            'desc': 'Initial gaussian filter sigma',
            'initialVal': 0.75,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.floatLineEdit',
            'actions': None,
            'dtype': float
        },
        'sharpenSpots': {
            'desc': 'Sharpen spots signal prior detection',
            'initialVal': True,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
    }
    return pre_processing_params

def _ref_ch_params():
    ref_ch_params = {
        'segmRefCh': {
            'desc': 'Segment reference channel',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'keepPeaksInsideRef': {
            'desc': 'Keep only spots that are inside ref. channel mask',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': True,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'filterPeaksInsideRef': {
            'desc': 'Filter spots by comparing to reference channel',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'refChSingleObj': {
            'desc': 'Ref. channel is single object (e.g., nucleus)',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': True,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'refChThresholdFunc': {
            'desc': 'Ref. channel threshold function',
            'initialVal': 'threshold_li',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets._refChThresholdFuncWidget',
            'actions': None,
            'dtype': get_threshold_func,
            'parser': parse_threshold_func
        },
        'calcRefChNetLen': {
            'desc': 'Calculate reference channel network length',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'saveRefChMask': {
            'desc': 'Save reference channel segmentation masks',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        }
    }
    return ref_ch_params

def _spots_ch_params():
    spots_ch_params = {
        'spotDetectionMethod': {
            'desc': 'Spots detection method',
            'initialVal': 'peak_local_max', # or 'label_prediction_mask'
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets._spotDetectionMethod',
            'actions': None
        },
        'spotPredictionMethod': {
            'desc': 'Spots segmentation method',
            'initialVal': 'Thresholding',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': True,
            'formWidgetFunc': 'widgets._spotPredictionMethod',
            'actions': None
        },
        'spotThresholdFunc': {
            'desc': 'Spot detection threshold function',
            'initialVal': 'threshold_li',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets._spotThresholdFunc',
            'actions': None,
            'dtype': get_threshold_func,
            'parser': parse_threshold_func,
            'autoTuneWidget': 'widgets.ReadOnlyLineEdit'
        },
        'gopThresholds': {
            'desc': 'Features and thresholds for filtering true spots',
            'initialVal': None,
            'stretchWidget': True,
            'addLabel': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._GopFeaturesAndThresholdsButton',
            'actions': None,
            'dtype': get_gop_thresholds,
            'parser': parse_list_to_configpars,
            'comment': gop_thresholds_comment,
            'autoTuneWidget': 'widgets.SelectFeaturesAutoTune'
        },
        'optimiseWithEdt': {
            'desc': 'Optimise detection for high spot density',
            'initialVal': True,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'doSpotFit': {
            'desc': 'Compute spots size',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        # 'highSpotDensityFit': {
        #     'desc': 'Optimise spots size estimation for high spot density',
        #     'initialVal': True,
        #     'stretchWidget': False,
        #     'addInfoButton': True,
        #     'addComputeButton': True,
        #     'addApplyButton': False,
        #     'formWidgetFunc': 'acdc_widgets.Toggle',
        #     'actions': None,
        #     'dtype': get_bool
        # },
        # 'minSpotSize': {
        #     'desc': 'Discard spots with radius less than (pixels)',
        #     'initialVal': 0.0,
        #     'stretchWidget': True,
        #     'addInfoButton': True,
        #     'addComputeButton': False,
        #     'addApplyButton': False,
        #     'formWidgetFunc': 'widgets.floatLineEdit',
        #     'actions': None,
        #     'dtype': float
        # },
        # 'maxSpotSize': {
        #     'desc': 'Discard spots with radius greater than (pixels)',
        #     'initialVal': 0.0,
        #     'stretchWidget': True,
        #     'addInfoButton': True,
        #     'addComputeButton': False,
        #     'addApplyButton': False,
        #     'formWidgetFunc': 'widgets.floatLineEdit',
        #     'actions': None,
        #     'dtype': float
        # }
    }
    return spots_ch_params


def analysisInputsParams(params_path=default_ini_path):
    # NOTE: if you change the anchors (i.e., the key of each second level
    # dictionary, e.g., 'spotsEndName') remember to change them also in
    # docs.paramsInfoText dictionary keys
    params = {
        'File paths and channels': _filepaths_params(),
        'METADATA': _metadata_params(),
        'Pre-processing': _pre_processing_params(),
        'Reference channel': _ref_ch_params(),
        'Spots channel': _spots_ch_params(),
        'Configuration': _configuration_params()
    
    }
    if params_path.endswith('.ini'):
        params = io.readStoredParamsINI(params_path, params)
    else:
        params = io.readStoredParamsCSV(params_path, params)
    return params

def skimageAutoThresholdMethods():
    methodsName = [
        'threshold_li',
        'threshold_isodata',
        'threshold_otsu',
        'threshold_minimum',
        'threshold_triangle',
        'threshold_mean',
        'threshold_yen'
    ]
    return methodsName

if GUI_INSTALLED:
    class QtWarningHandler(QObject):
        sigGeometryWarning = pyqtSignal(str)

        def _resizeWarningHandler(self, msg_type, msg_log_context, msg_string):
            if msg_string.find('Unable to set geometry') != -1:
                try:
                    self.sigGeometryWarning.emit(msg_type)
                except Exception as e:
                    pass
            elif msg_string:
                print(msg_string)

    # Install Qt Warnings handler
    warningHandler = QtWarningHandler()
    qInstallMessageHandler(warningHandler._resizeWarningHandler)

    # Initialize color items
    initColorItems()
