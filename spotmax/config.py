print('Configuring files...')
import os
import json
from pprint import pprint
import pandas as pd

from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject, pyqtSignal, qInstallMessageHandler

from . import widgets, html_func, io

spotmax_path = os.path.dirname(os.path.abspath(__file__))
settings_path = os.path.join(spotmax_path, 'settings')
default_ini_path = os.path.join(spotmax_path, 'config.ini')
colorItems_path = os.path.join(settings_path, 'colorItems.json')

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

def analysisInputsParams(ini_path=default_ini_path):
    # NOTE: if you change the anchors (i.e., the key of each second level
    # dictionary, e.g., 'spotsFilePath') remember to change them also in
    # docs.paramsInfoText dictionary keys
    params = {
        # Section 0 (GroupBox)
        'File paths': {
            'spotsFilePath': {
                'desc': 'Spots channel file path',
                'initialVal': """""",
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': True,
                'formWidgetFunc': widgets.tooltipLineEdit,
                'actions': None
            },
            'segmFilePath': {
                'desc': 'Cells segmentation file path',
                'initialVal': """""",
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': True,
                'formWidgetFunc': widgets.tooltipLineEdit,
                'actions': None
            },
            'refChFilePath': {
                'desc': 'Reference channel file path',
                'initialVal': """""",
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': True,
                'formWidgetFunc': widgets.tooltipLineEdit,
                'actions': None
            },
            'refChSegmFilePath': {
                'desc': 'Ref. channel segmentation file path',
                'initialVal': """""",
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': True,
                'formWidgetFunc': widgets.tooltipLineEdit,
                'actions': None
            }
        },
        # Section 1 (GroupBox)
        'METADATA': {
            'pixelWidth': {
                'desc': 'Pixel width (μm)',
                'initialVal': 0.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': (
                    ('valueChanged', 'updateMinSpotSize'),
                ),
            },
            'pixelHeight': {
                'desc': 'Pixel height (μm)',
                'initialVal': 0.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': (
                    ('valueChanged', 'updateMinSpotSize'),
                ),
            },
            'voxelDepth': {
                'desc': 'Voxel depth (μm)',
                'initialVal': 0.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': (
                    ('valueChanged', 'updateMinSpotSize'),
                ),
            },
            'numAperture': {
                'desc': 'Numerical aperture',
                'initialVal': 1.4,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': (
                    ('valueChanged', 'updateMinSpotSize'),
                ),
            },
            'emWavelen': {
                'desc': 'Spots reporter emission wavelength (nm)',
                'initialVal': 500.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': (
                    ('valueChanged', 'updateMinSpotSize'),
                )
            },
            'zResolutionLimit': {
                'desc': 'Spot minimum z-size (μm)',
                'initialVal': 1.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': (
                    ('valueChanged', 'updateMinSpotSize'),
                )
            },
            'yxResolLimitMultiplier': {
                'desc': 'Resolution multiplier in y- and x- direction',
                'initialVal': 1.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': (
                    ('valueChanged', 'updateMinSpotSize'),
                )
            },
            'spotMinSizeLabels': {
                'desc': 'Spot (z,y,x) minimum dimensions',
                'initialVal': """""",
                'stretchWidget': True,
                'addInfoButton': False,
                'formWidgetFunc': widgets._spotMinSizeLabels,
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
                'formWidgetFunc': widgets.Toggle,
                'actions': None
            },
            'gaussSigma': {
                'desc': 'Initial gaussian filter sigma',
                'initialVal': 0.75,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': None
            },
            'sharpenSpots': {
                'desc': 'Sharpen spots signal prior detection',
                'initialVal': True,
                'stretchWidget': False,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': widgets.Toggle,
                'actions': None
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
                'formWidgetFunc': widgets.Toggle,
                'actions': None
            },
            'keepPeaksInsideRef': {
                'desc': 'Keep only spots that are inside ref. channel mask',
                'initialVal': False,
                'stretchWidget': False,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': True,
                'formWidgetFunc': widgets.Toggle,
                'actions': None
            },
            'filterPeaksInsideRef': {
                'desc': 'Filter spots by comparing to reference channel',
                'initialVal': False,
                'stretchWidget': False,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': widgets.Toggle,
                'actions': None
            },
            'refChSingleObj': {
                'desc': 'Ref. channel is single object (e.g., nucleus)',
                'initialVal': False,
                'stretchWidget': False,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': True,
                'formWidgetFunc': widgets.Toggle,
                'actions': None
            },
            'refChThresholdFunc': {
                'desc': 'Ref. channel threshold function',
                'initialVal': 'threshold_li',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': widgets._refChThresholdFuncWidget,
                'actions': None
            },
            'calcRefChNetLen': {
                'desc': 'Calculate reference channel network length',
                'initialVal': False,
                'stretchWidget': False,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': widgets.Toggle,
                'actions': None
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
                'formWidgetFunc': widgets._spotDetectionMethod,
                'actions': None
            },
            'spotPredictionMethod': {
                'desc': 'Spots segmentation method',
                'initialVal': 'Thresholding',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': True,
                'formWidgetFunc': widgets._spotPredictionMethod,
                'actions': None
            },
            'spotThresholdFunc': {
                'desc': 'Spot detection threshold function',
                'initialVal': 'threshold_li',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets._spotThresholdFunc,
                'actions': None
            },
            'gopMethod': {
                'desc': 'Method for filtering true spots',
                'initialVal': 'Effect size',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets._gopMethod,
                'actions': None
            },
            'gopLimit': {
                'desc': 'Threshold value for filtering valid spots',
                'initialVal': 0.8,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': None
            },
            'doSpotFit': {
                'desc': 'Compute spots size',
                'initialVal': False,
                'stretchWidget': False,
                'addInfoButton': True,
                'addComputeButton': True,
                'addApplyButton': False,
                'formWidgetFunc': widgets.Toggle,
                'actions': None
            },
            'minSpotSize': {
                'desc': 'Discard spots with size less than',
                'initialVal': 0.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': None
            },
            'maxSpotSize': {
                'desc': 'Discard spots with size greater than',
                'initialVal': 0.0,
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'formWidgetFunc': widgets.floatLineEdit,
                'actions': None
            }
        }
    }
    params = io.readStoredParamsINI(ini_path, params)
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

class QtWarningHandler(QObject):
    sigGeometryWarning = pyqtSignal(str)

    def _resizeWarningHandler(self, msg_type, msg_log_context, msg_string):
        if msg_string.find('Unable to set geometry') != -1:
            print('warning caught')
            self.sigGeometryWarning.emit(msg_type)
        elif msg_string:
            print(msg_string)

# Install Qt Warnings handler
warningHandler = QtWarningHandler()
qInstallMessageHandler(warningHandler._resizeWarningHandler)

# Initialize color items
initColorItems()
