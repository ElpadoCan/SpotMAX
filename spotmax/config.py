# print('Configuring files...')
import os
import re
import json
import pathlib
from pprint import pprint
from typing import Any
import pandas as pd
import configparser
import skimage.filters

from . import GUI_INSTALLED

if GUI_INSTALLED:
    from qtpy.QtGui import QFont
    from qtpy.QtCore import QObject, Signal, qInstallMessageHandler

    from cellacdc import apps as acdc_apps
    from cellacdc import widgets as acdc_widgets

    from . import widgets
else:
    from . import utils

from . import io, colorItems_path

class ConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, allow_no_value=True, **kwargs)
        self.optionxform = str
    
    def read(self, filepath, encoding='utf-8'):
        super().read(filepath, encoding=encoding)        
        self._filename = os.path.basename(filepath)
        self._filepath = filepath
        filepaths_section = 'File paths and channels'
        if not self.has_section(filepaths_section):
            return
        
        for option in self.options(filepaths_section):
            if not option.endswith('end name or path'):
                continue
            
            value = self.get(filepaths_section, option)
            new_option = option.replace('end name or path', 'end name')
            self[filepaths_section][new_option] = value
            self.remove_option(filepaths_section, option)

    def filepath(self):
        return self._filepath

    def filename(self):
        return self._filename

    def get(self, section, option, **kwargs):
        value = super().get(section, option, **kwargs)
        try:
            comment_idx = value.find('#')
            if comment_idx > 0:
                value = value[:comment_idx]
        except Exception as e:
            pass
        return value

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
    raise TypeError(f'The text "{text}" cannot be converted to a valid boolean object')

def get_gauss_sigma(text):
    if not text:
        return 0.0
    
    try:
        sigma = float(text)
        return sigma
    except Exception as e:
        pass
    
    try:
        if text.startswith('[') or text.startswith('('):
            text = text[1:]
        if text.endswith(']') or text.startswith(')'):
            text = text[:-1]
        sigma = [float(val) for val in text.split(',')]
    except Exception as e:
        raise TypeError(
            f'{text} is not a valid value for the gaussian sigma. '
            'Pass either a single number or one number per dimension '
            'of the image data.'
        )
    return sigma

def get_stack_3d_segm_range(text):
    if not text:
        return (0, 0)
    
    low, high = re.findall(r'(\d+), ?(\d+)', text)[0]
    return int(low), int(high)

def get_sigma_xy_bounds(text):
    if text == 'Default' or not text:
        return ('0.5', 'spotsize_yx_radius_pxl')
    
    text = text.replace(' ', '')
    return text.split(',')

def get_sigma_z_bounds(text):
    if text == 'Default' or not text:
        return ('0.5', 'spotsize_z_radius_pxl')

    text = text.replace(' ', '')
    return text.split(',')

def get_A_fit_bounds(text):
    if text == 'Default' or not text:
        return ('0.0', 'spotsize_A_max')

    text = text.replace(' ', '')
    return text.split(',')

def get_B_fit_bounds(text):
    if text == 'Default' or not text:
        return ('spot_B_min', 'inf')

    text = text.replace(' ', '')
    return text.split(',')

def get_ridge_sigmas(text):
    if not text:
        return 0.0
    
    try:
        sigmas = [float(text)]
        return sigmas
    except Exception as e:
        pass
    
    try:
        if text.startswith('[') or text.startswith('('):
            text = text[1:]
        if text.endswith(']') or text.startswith(')'):
            text = text[:-1]
        sigmas = [float(val) for val in text.split(',')]
    except Exception as e:
        raise TypeError(
            f'{text} is not a valid value for the ridge filter sigmas. '
            'Pass either a single number or a list of numbers'
        )
    return sigmas

def get_custom_combined_measurements(text):
    ...

class InvalidThresholdFunc:
    def __init__(self, func_name):
        self._name = func_name
    
    def __call__(self, *args, **kwds):
        raise AttributeError(
            f'`{self._name}` is not a valid thresholding function. '
            'Carefully check for typos. Available automatic thresholding '
            'methods are the following:\n\n'
            '  * threshold_li\n'
            '  * threshold_otsu\n' 
            '  * threshold_isodata\n'
            '  * threshold_triangle\n'
            '  * threshold_minimum\n'
            '  * threshold_mean\n'
            '  * threshold_yen\n\n'
            'You can find more details here '
            'https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html'
        )

def get_threshold_func(func_name):
    func_name = func_name.strip()
    try:
        func = getattr(skimage.filters, func_name)
    except Exception as e:
        func = InvalidThresholdFunc(func_name)
    return func

def get_valid_text(text):
    return re.sub(r'[^\w\-.]', '_', text)

def parse_threshold_func(threshold_func):
    if isinstance(threshold_func, str):
        return threshold_func
    else:
        return threshold_func.__name__

def exp_paths_to_str(params, configparser):
    SECTION = 'File paths and channels'
    ANCHOR = 'folderPathsToAnalyse'
    option = params[SECTION][ANCHOR]['desc']
    exp_paths = params[SECTION][ANCHOR]['loadedVal']
    exp_paths_str = '\n'.join(exp_paths)
    configparser[SECTION][option] = exp_paths_str
    return configparser

def parse_exp_paths(ini_filepath):
    """Check if experiment folder to analyse is a text file

    Parameters
    ----------
    ini_filepath : os.PathLike
        Path to the ini configuration file
    """    
    ini_folderpath = os.path.dirname(ini_filepath)
    cp = io.read_ini(ini_filepath)
    SECTION = 'File paths and channels'
    input_exp_paths = cp[SECTION]['Experiment folder path(s) to analyse']
    exp_path = input_exp_paths.replace('\n', '')
    
    paths_to_analyse = None
    try:
        # Check if text file is in same folder of ini file (relative)
        filepath = os.path.join(ini_folderpath, exp_path)
        if filepath.endswith('.txt'):
            with open(filepath, 'r') as file:
                paths_to_analyse = file.read()
    except Exception as err:
        pass
    
    try:
        # Check if text file path is absolute
        filepath = exp_path
        if filepath.endswith('.txt'):
            with open(filepath, 'r') as file:
                paths_to_analyse = file.read()
    except Exception as err:
        pass
    
    if paths_to_analyse is None:
        paths_to_analyse = input_exp_paths
    
    paths_to_analyse = get_exp_paths(
        paths_to_analyse, ini_folderpath=ini_folderpath
    )
    
    return paths_to_analyse

def get_exp_paths(exp_paths, ini_folderpath=''):    
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

    exp_paths = [
        io.get_abspath(path, src_folderpath=ini_folderpath) 
        for path in exp_paths
    ]
    
    return exp_paths

def parse_log_folderpath(log_path):
    log_path = io.get_abspath(log_path)
    try:
        log_path = pathlib.Path(log_path).relative_to(pathlib.Path.home())
        log_path = os.path.normpath(f'~{os.sep}{log_path}')
    except ValueError as e:
        log_path = log_path
    return log_path

def parse_list_to_configpars(iterable: list):
    if iterable is None:
        return ''
        
    if isinstance(iterable, str):
        iterable = [iterable]
    
    li_str = [f'\n{p}' for p in iterable]
    li_str = ''.join(li_str)
    return li_str

def features_thresholds_comment():
    s = (
        '# Save the features to use for filtering truw spots as `feature_name,max,min`.\n'
        '# You can write as many features as you want. Write each feature on its own indented line.\n'
        '# Example: `spot_vs_ref_ch_ttest_pvalue,None,0.025` means `keep only spots whose p-value\n'
        '# is smaller than 0.025` where `None` indicates that there is no minimum.'
    )
    return s

def get_features_thresholds_filter(features_thresholds_to_parse):
    """_summary_

    Parameters
    ----------
    features_thresholds_to_parse : str
        String formatted to contain feature names and min,max values to use 
        when filtering spots in goodness-of-peak test.

        Multiple features are separated by the "/" charachter. Feature name and 
        thresholds values are separated by comma.

        Examples:
            `spot_vs_bkgr_glass_effect_size,0.8,None`: Filter all the spots 
            that have the Glass' effect size greater than 0.8. There is no max 
            set.
    """    
    in_features_thresholds = features_thresholds_to_parse.split('\n')
    out_features_thresholds = {}
    for feature_thresholds in in_features_thresholds:
        feature_name, *thresholds_str = feature_thresholds.split(',')
        feature_name = feature_name.strip()
        if not feature_name:
            continue
        if feature_name == 'None':
            continue
        thresholds = [None, None]
        for t, thresh in enumerate(thresholds_str):
            if thresh.endswith(')'):
                thresh = thresh[:-1]
                feature_name = f'{feature_name})'
            try:
                thresholds[t] = float(thresh)
            except Exception as e:
                pass
        
        feature_name = feature_name.replace('OR ', '| ')
        feature_name = feature_name.replace('or ', '| ')
        feature_name = feature_name.replace('AND ', '& ')
        feature_name = feature_name.replace('and ', '& ')
        
        out_features_thresholds[feature_name] = tuple(thresholds)
    return out_features_thresholds

def _filepaths_params():
    filepaths_params = {
        'folderPathsToAnalyse': {
            'desc': 'Experiment folder path(s) to analyse',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addWarningButton': True,
            'addEditButton': True,
            'editSlot': 'addFoldersToAnalyse',
            'formWidgetFunc': 'widgets.ReadOnlyElidingLineEdit',
            'actions': None,
            'dtype': get_exp_paths,
            'parser': parse_list_to_configpars,
            'valueSetter': 'setText'
        },
        'spotsEndName': {
            'desc': 'Spots channel end name',
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
            'valueSetter': 'setText'
        },
        'segmEndName': {
            'desc': 'Cells segmentation end name',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'browseExtensions': {'Segm. masks': ['.npz', '.npy']},
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str,
            'valueSetter': 'setText'
        },
        'refChEndName': {
            'desc': 'Reference channel end name',
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
        'spotChSegmEndName': {
            'desc': 'Spots channel segmentation end name',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'browseExtensions': {'Segm. masks': ['.npz', '.npy']},
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str
        },
        'refChSegmEndName': {
            'desc': 'Ref. channel segmentation end name',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'browseExtensions': {'Segm. masks': ['.npz', '.npy']},
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str
        },
        'inputDfSpotsEndname': {
            'desc': 'Spots coordinates table end name',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'browseExtensions': {'Table': ['.csv', '.h5']},
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str
        },
        'lineageTableEndName': {
            'desc': 'Table with lineage info end name',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': True,
            'browseExtensions': {'CSV': ['.csv']},
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
            'formWidgetFunc': 'widgets.RunNumberSpinbox',
            'actions': None,
            'dtype': int,
            'valueSetter': 'setValue'
        },
        'textToAppend': {
            'desc': 'Text to append at the end of the output files',
            'initialVal': '',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addEditButton': False,
            'formWidgetFunc': 'widgets.CenteredAlphaNumericLineEdit',
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
            'addBrowseButton': False,
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
            'isFolderBrowse': True,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str, 
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
            'isFolderBrowse': True,
            'addEditButton': False,
            'formWidgetFunc': 'widgets._CenteredLineEdit',
            'actions': None,
            'dtype': str,
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
            'formWidgetFunc': 'widgets.CenteredAlphaNumericLineEdit',
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
            'parser_arg': 'raise_on_critical'
        },
        'useGpu': {
            'desc': 'Use CUDA-compatible GPU',
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
            'formWidgetFunc': 'widgets.SpinBox',
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
        }
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
            'formWidgetFunc': 'acdc_widgets.IntLineEdit',
            'actions': None,
            'dtype': int,
            'valueSetter': 'setValue'
        },
        'stopFrameNum': {
            'desc': 'Analyse until frame number',
            'initialVal': -1,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'acdc_widgets.IntLineEdit',
            'actions': None,
            'dtype': int,
            'valueSetter': 'setValue'
        },
        'SizeZ': {
            'desc': 'Number of z-slices (SizeZ)',
            'initialVal': 1,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'acdc_widgets.IntLineEdit',
            'actions': (
                ('valueChanged', 'SizeZchanged'),
            ),
            'dtype': int,
            'valueSetter': 'setValue'
        },
        'pixelWidth': {
            'desc': 'Pixel width (μm)',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.FloatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
                ('valueChanged', 'updateLocalBackgroundValue'),
            ),
            'dtype': float,
            'valueSetter': 'setValue'
        },
        'pixelHeight': {
            'desc': 'Pixel height (μm)',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.FloatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float,
            'valueSetter': 'setValue'
        },
        'voxelDepth': {
            'desc': 'Voxel depth (μm)',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.FloatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float,
            'valueSetter': 'setValue'
        },
        'numAperture': {
            'desc': 'Numerical aperture',
            'initialVal': 1.4,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.FloatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float,
            'valueSetter': 'setValue'
        },
        'emWavelen': {
            'desc': 'Spots reporter emission wavelength (nm)',
            'initialVal': 500.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.FloatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float,
            'valueSetter': 'setValue'
        },
        'zResolutionLimit': {
            'desc': 'Spot minimum z-size (μm)',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.FloatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float,
            'autoTuneWidget': 'widgets.ResolutMultiplierAutoTuneWidget'
            # 'autoTuneWidget': 'widgets.ReadOnlyLineEdit'
        },
        'yxResolLimitMultiplier': {
            'desc': 'Resolution multiplier in y- and x- direction',
            'initialVal': 1.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.FloatLineEdit',
            'actions': (
                ('valueChanged', 'updateMinSpotSize'),
            ),
            'dtype': float,
            'autoTuneWidget': 'widgets.ResolutMultiplierAutoTuneWidget'
        },
        'spotMinSizeLabels': {
            'desc': 'Spot (z, y, x) minimum dimensions (radius)',
            'initialVal': """""",
            'stretchWidget': True,
            'addInfoButton': True,
            'addWarningButton': True,
            'addComputeButton': True,
            'formWidgetFunc': 'widgets.SpotMinSizeLabels',
            'actions': None,
            'isParam': False
        }
    }
    return metadata_params

def ini_metadata_anchor_to_acdc_metadata_mapper(acdc_metadata_df, channel_name):
    mapper = {
        'SizeT': ('SizeT', int),
        'SizeZ': ('SizeZ', int),
        'pixelWidth': ('PhysicalSizeX', float),
        'pixelHeight': ('PhysicalSizeY', float),
        'voxelDepth': ('PhysicalSizeZ', float),
        'numAperture': ('LensNA', float),
        
    }
    try:
        channel_idx = (
            acdc_metadata_df[acdc_metadata_df['values'] == channel_name]
            .index.to_list()[0]
            .split('_')[1]
        )
        mapper['emWavelen'] = (f'channel_{channel_idx}_emWavelen', float)
    except Exception as err:
        pass
    
    return mapper

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
        'thresholdWithObjsMask': {
            'desc': 'Threshold only inside segmented objects',
            'initialVal': True,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'removeHotPixels': {
            'desc': 'Remove hot pixels',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': True,
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
            'formWidgetFunc': 'widgets.Gaussian3SigmasLineEdit',
            'actions': None,
            'dtype': get_gauss_sigma
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
        'extend3DsegmRange': {
            'desc': 'Extend 3D input segm. objects in Z',
            'initialVal': '(0, 0)',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.Extend3DsegmRangeWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': get_stack_3d_segm_range
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
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'bkgrMaskOutsideRef': {
            'desc': 'Use the ref. channel mask to determine background',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
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
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'keepTouchObjectsIntact': {
            'desc': 'Keep external touching objects intact',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'refChGaussSigma': {
            'desc': 'Ref. channel gaussian filter sigma',
            'initialVal': 0.75,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.Gaussian3SigmasLineEdit',
            'actions': None,
            'dtype': get_gauss_sigma
        },
        'refChRidgeFilterSigmas': {
            'desc': 'Sigmas used to enhance network-like structures',
            'initialVal': 0.0,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.VectorLineEdit',
            'actions': None,
            'dtype': get_ridge_sigmas
        },
        'refChSegmentationMethod': {
            'desc': 'Ref. channel segmentation method',
            'initialVal': 'Thresholding',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.RefChPredictionMethodWidget',
            'actions': None,
            'dtype': str
        },
        'refChThresholdFunc': {
            'desc': 'Ref. channel threshold function',
            'initialVal': 'threshold_otsu',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets._refChThresholdFuncWidget',
            'actions': None,
            'dtype': get_threshold_func,
            'parser': parse_threshold_func
        },
        'refChFilteringFeatures': {
            'desc': 'Features for filtering ref. channel objects',
            'initialVal': None,
            'stretchWidget': True,
            'addLabel': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addEditButton': False,
            'formWidgetFunc': 'widgets.RefChannelFeaturesThresholdsButton',
            'actions': None,
            'dtype': get_features_thresholds_filter,
            'parser': parse_list_to_configpars
        },
        'saveRefChFeatures': {
            'desc': 'Save reference channel features',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool, 
            'ignoreIfMissing': True
        },
        'saveRefChMask': {
            'desc': 'Save reference channel segmentation masks',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'saveRefChPreprocImage': {
            'desc': 'Save pre-processed reference channel image',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        }
    }
    return ref_ch_params

def _spots_ch_params():
    spots_ch_params = {
        'spotPredictionMethod': {
            'desc': 'Spots segmentation method',
            'initialVal': 'Thresholding',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': True,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.SpotPredictionMethodWidget',
            'actions': None, 
            # 'autoTuneWidget': 'widgets.TuneSpotPredictionMethodWidget'
        },
        'minSizeSpotMask': {
            'desc': 'Minimum size of spot segmentation mask',
            'initialVal': 5,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'widgets.SpinBox',
            'actions': None,
            'dtype': int,
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
            'dtype': get_features_thresholds_filter,
            'parser': parse_list_to_configpars,
            'comment': features_thresholds_comment,
            'autoTuneWidget': 'widgets.SelectFeaturesAutoTune'
        },
        'localBkgrRingWidth': {
            'desc': 'Local background ring width',
            'initialVal': '5 pixel',
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'widgets.LocalBackgroundRingWidthWidget',
            'valueSetter': 'setText',
            'actions': None,
            'dtype': str,
        },
        'optimiseWithEdt': {
            'desc': 'Optimise detection for high spot density',
            'initialVal': True,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'doSpotFit': {
            'desc': 'Compute spots size (fit gaussian peak(s))',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'dtype': get_bool, 
            'actions': (
                ('toggled', 'doSpotFitToggled'),
            ),
        },
        'dropSpotsMinDistAfterSpotfit': {
            'desc': 'After spotFIT, drop spots that are too close',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool, 
            'parentActivator': ('Spots channel', 'doSpotFit')
        },
        'checkMergeSpotfit': {
            'desc': 'Merge spots pairs where single peak fits better',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool, 
            'parentActivator': ('Spots channel', 'doSpotFit')
        },
        'maxNumPairs': {
            'desc': 'Maximum number of spot pairs to check',
            'initialVal': 11,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'addBrowseButton': False,
            'addAutoButton': False,
            'formWidgetFunc': 'widgets.SpinBox',
            'actions': None,
            'dtype': int, 
            'parentActivator': ('Spots channel', 'doSpotFit')
        },
        'saveSpotsMask': {
            'desc': 'Save spots segmentation masks',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'saveSpotsPreprocImage': {
            'desc': 'Save pre-processed spots image',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
        'skipInvalidSpotsLabels': {
            'desc': 'Skip objects where segmentation failed',
            'initialVal': False,
            'stretchWidget': False,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'acdc_widgets.Toggle',
            'actions': None,
            'dtype': get_bool
        },
    }
    return spots_ch_params

def _spotfit_params():
    spotfit_params = {
        'XYcenterBounds': {
            'desc': 'Bounds interval for the x and y peak center coord.',
            'initialVal': 0.1, 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.PlusMinusFloatLineEdit',
            'actions': None,
            'dtype': float
        },
        'ZcenterBounds': {
            'desc': 'Bounds interval for the z peak center coord.',
            'initialVal': 0.2, 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.PlusMinusFloatLineEdit',
            'actions': None,
            'dtype': float
        },
        'sigmaXBounds': {
            'desc': 'Bounds for sigma in x-direction',
            'initialVal': '0.5, spotsize_yx_radius_pxl', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.sigmaXBoundsWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': get_sigma_xy_bounds
        },
        'sigmaYBounds': {
            'desc': 'Bounds for sigma in y-direction',
            'initialVal': '0.5, spotsize_yx_radius_pxl', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.sigmaYBoundsWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': get_sigma_xy_bounds
        },
        'sigmaZBounds': {
            'desc': 'Bounds for sigma in z-direction',
            'initialVal': '0.5, spotsize_z_radius_pxl', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.sigmaZBoundsWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': get_sigma_z_bounds
        },
        'A_fit_bounds': {
            'desc': 'Bounds for the peak amplitude',
            'initialVal': '0.0, spotsize_A_max', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.AfitBoundsWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': get_A_fit_bounds
        },
        'B_fit_bounds': {
            'desc': 'Bounds for the peak background level',
            'initialVal': 'spot_B_min, inf', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.BfitBoundsWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': get_B_fit_bounds
        },
        'sigmaXinitGuess': {
            'desc': 'Initial guess for sigma in x-direction',
            'initialVal': 'x_resolution_pxl/2.35', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.SetValueFromFeaturesWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': str
        },
        'sigmaYinitGuess': {
            'desc': 'Initial guess for sigma in y-direction',
            'initialVal': 'y_resolution_pxl/2.35', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.SetValueFromFeaturesWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': str
        },
        'sigmaZinitGuess': {
            'desc': 'Initial guess for sigma in z-direction',
            'initialVal': 'z_resolution_pxl/2.35', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.SetValueFromFeaturesWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': str
        },
        'A_fit_initGuess': {
            'desc': 'Initial guess for the peak amplitude',
            'initialVal': 'spotsize_A_max', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.SetValueFromFeaturesWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': str
        },
        'B_fit_initGuess': {
            'desc': 'Initial guess for the peak background level',
            'initialVal': 'spotsize_surface_median', 
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'formWidgetFunc': 'widgets.SetValueFromFeaturesWidget',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': str
        },
    }
    return spotfit_params

def _custom_combined_measurements_params():
    custom_combined_meas_params = {
        'customCombinedMeas': {
            'desc': 'Column name',
            'confvalText': 'Custom combined measurement',
            'initialVal': '',
            'useEditableLabel': True,
            'stretchFactors': (2, 0, 3, 0),
            'addAddFieldButton': True,
            'stretchWidget': True,
            'addInfoButton': True,
            'addComputeButton': False,
            'addApplyButton': False,
            'labelTextMiddle': '\n = ',
            'formWidgetFunc': 'widgets.SetCustomCombinedMeasurement',
            'actions': None,
            'valueSetter': 'setValue',
            'dtype': str
        },
    }
    return custom_combined_meas_params

def get_section_from_anchor(anchor_to_search):
    params = analysisInputsParams()
    for section, section_params in params.items():
        for anchor in section_params.keys():
            if anchor == anchor_to_search:
                return section

def getDefaultParams():
    params = {
        'File paths and channels': _filepaths_params(),
        'METADATA': _metadata_params(),
        'Pre-processing': _pre_processing_params(),
        'Reference channel': _ref_ch_params(),
        'Spots channel': _spots_ch_params(),
        'SpotFIT': _spotfit_params(),
        'Custom combined measurements': _custom_combined_measurements_params(),
        'Configuration': _configuration_params()
    }
    return params

def analysisInputsParams(params_path=None, cast_dtypes=True):
    # NOTE: if you change the anchors (i.e., the key of each second level
    # dictionary, e.g., 'spotsEndName') remember to change them also in
    # _docs.paramsInfoText dictionary keys
    params = getDefaultParams()
    if params_path is None:
        return params
    
    if params_path.endswith('.ini'):
        params = io.readStoredParamsINI(
            params_path, params, cast_dtypes=cast_dtypes
        )
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
        sigGeometryWarning = Signal(str)

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
