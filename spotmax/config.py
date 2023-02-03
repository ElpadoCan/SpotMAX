print('Configuring files...')
import os
import json
import pathlib
from pprint import pprint
import pandas as pd
import configparser
import skimage.filters

try:
    from PyQt5.QtGui import QFont
    from PyQt5.QtCore import QObject, pyqtSignal, qInstallMessageHandler

    from cellacdc import widgets as acdc_widgets

    from . import widgets
    GUI_INSTALLED = True
except ModuleNotFoundError:
    GUI_INSTALLED = False
    from . import utils

from . import io, colorItems_path, default_ini_path

class ConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optionxform = str

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

def get_exp_paths(exp_paths):
    exp_paths = exp_paths.lstrip()
    exp_paths = exp_paths.lstrip('[')
    exp_paths = exp_paths.lstrip('(')

    exp_paths = exp_paths.rstrip()
    exp_paths = exp_paths.rstrip(']')
    exp_paths = exp_paths.rstrip(')')

    exp_paths = exp_paths.split(',')
    exp_paths = [path.strip() for path in exp_paths]
    return exp_paths

def analysisInputsParams(params_path=default_ini_path):
    # NOTE: if you change the anchors (i.e., the key of each second level
    # dictionary, e.g., 'spotsEndName') remember to change them also in
    # docs.paramsInfoText dictionary keys
    params = {
        # Section 0 (GroupBox)
        'File paths and channels': {
            'filePathsToAnalyse': {
                'desc': 'Experiment folder path(s) to analyse',
                'initialVal': """""",
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': False,
                'addEditButton': True,
                'formWidgetFunc': 'acdc_widgets.alphaNumericLineEdit',
                'actions': None,
                'dtype': get_exp_paths
            },
            'spotsEndName': {
                'desc': 'Spots channel end name or path',
                'initialVal': """""",
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': False,
                'addEditButton': True,
                'formWidgetFunc': 'acdc_widgets.alphaNumericLineEdit',
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
                'addBrowseButton': False,
                'addEditButton': True,
                'formWidgetFunc': 'acdc_widgets.alphaNumericLineEdit',
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
                'addBrowseButton': False,
                'addEditButton': True,
                'formWidgetFunc': 'acdc_widgets.alphaNumericLineEdit',
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
                'addBrowseButton': False,
                'addEditButton': True,
                'formWidgetFunc': 'acdc_widgets.alphaNumericLineEdit',
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
                'addBrowseButton': False,
                'addEditButton': True,
                'formWidgetFunc': 'acdc_widgets.alphaNumericLineEdit',
                'actions': None,
                'dtype': str
            },
        },
        # Section 1 (GroupBox)
        'METADATA': {
            'SizeT': {
                'desc': 'Number of frames (SizeT)',
                'initialVal': 1,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addAutoButton': True,
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
                'addAutoButton': True,
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
                'addAutoButton': True,
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
                'dtype': float
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
                'dtype': float
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
        },

        # Section 2 (GroupBox)
        'Pre-processing': {
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
        },

        # Section 3 (GroupBox)
        'Reference channel': {
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
            'filterPeaksInsideRefMethod': {
                'desc': 'Method to filter spots when comparing to ref. channel',
                'initialVal': 'Keep only spots that are brighter than ref. channel',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': 'widgets._filterSpotsVsRefChMethodWidget',
                'actions': None,
                'dtype': str
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
                'dtype': get_threshold_func
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
            }
        },

        # Section 4 (GroupBox)
        'Spots channel': {
            'spotDetectionMethod': {
                'desc': 'Spots detection method',
                'initialVal': 'peak_local_max',
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
                'dtype': get_threshold_func
            },
            'gopMethod': {
                'desc': 'Method for filtering true spots',
                'initialVal': 'Effect size',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': 'widgets._gopMethod',
                'actions': None
            },
            'gopLimit': {
                'desc': 'Threshold value for filtering valid spots',
                'initialVal': 0.8,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': 'widgets.floatLineEdit',
                'actions': None,
                'dtype': float
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
            'minSpotSize': {
                'desc': 'Discard spots with size less than',
                'initialVal': 0.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': 'widgets.floatLineEdit',
                'actions': None,
                'dtype': float
            },
            'maxSpotSize': {
                'desc': 'Discard spots with size greater than',
                'initialVal': 0.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': 'widgets.floatLineEdit',
                'actions': None,
                'dtype': float
            }
        }
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

def filterSpotsVsRefChMethods():
    methods = [
        'Keep only spots that are brighter than ref. channel',
        'Keep only spots that are as bright as ref. channel',
        'Keep only spots that are brighter or as bright as ref. channel',
        'Keep only spots that are darker than ref. channel'
    ]
    return methods

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
