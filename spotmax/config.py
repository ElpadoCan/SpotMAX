import os
import configparser

from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject, pyqtSignal, qInstallMessageHandler

from . import widgets, spotmax_path

class QtWarningHandler(QObject):
    sigGeometryWarning = pyqtSignal(str)

    def _resizeWarningHandler(self, msg_type, msg_log_context, msg_string):
        if msg_string.find('Unable to set geometry') != -1:
            print('warning caught')
            self.sigGeometryWarning.emit(msg_type)
        elif msg_string:
            print(msg_string)

warningHandler = QtWarningHandler()
qInstallMessageHandler(warningHandler._resizeWarningHandler)

default_ini_path = os.path.join(spotmax_path, 'config.ini')

paramsInfoText = {
    'spotsFilePath': (
        'Path of the image file with the <b>spots channel signal</b>.<br><br>'
        'Allowed <b>file formats</b>: .npy, .npz, .h5, .png, .tif, .tiff, '
        '.jpg, .jpeg, .mov, .avi, and .mp4'
    ),
    'segmFilePath': (
        'OPTIONAL: Path of the file with the <b>segmentation masks of '
        'the objects of interest</b>. Typically the objects are the <b>cells '
        'or the nuclei</b>, but it can be any object.<br><br>'
        'While this is optional, <b>it improves accuracy</b>, because spotMAX will '
        'detect only the spots that are inside the segmented objects.<br><br>'
        'It needs to be a 2D or 3D (z-stack) array, eventually with '
        'an additional dimension for frames over time.<br>'
        'The Y and X dimensions <b>MUST be the same</b> as the spots '
        'or reference channel images.<br><br>'
        'Each pixel beloging to the object must have a <b>unique</b> integer '
        'or RGB(A) value, while background pixels must have 0 or black RGB(A) '
        'value.<br><br>'
        'Allowed <b>file formats</b>: .npy, .npz, .h5, .png, .tif, .tiff, '
        '.jpg, .jpeg, .mov, .avi, and .mp4'
    ),
    'refChFilePath': (
        ''
    ),
    'refChSegmFilePath': (
        ''
    ),
    'pixelWidth': (
        ''
    ),
    'pixelHeight': (
        ''
    ),
    'voxelDepth': (
        ''
    ),
    'numAperture': (
        ''
    ),
    'emWavelen': (
        ''
    ),
    'zResolutionLimit': (
        ''
    ),
    'yxResolLimitMultiplier': (
        ''
    ),
    'spotMinSizeLabels': (
        ''
    ),
    'aggregate': (
        ''
    ),
    'gaussSigma': (
        ''
    ),
    'sharpenSpots': (
        ''
    ),
    'segmRefCh': (
        ''
    ),
    'keepPeaksInsideRef': (
        ''
    ),
    'filterPeaksInsideRef': (
        ''
    ),
    'refChSingleObj': (
        ''
    ),
    'refChThresholdFunc': (
        ''
    ),
    'calcRefChNetLen': (
        ''
    ),
    'spotDetectionMethod': (
        ''
    ),
    'spotPredictionMethod': (
        ''
    ),
    'spotThresholdFunc': (
        ''
    ),
    'gopMethod': (
        ''
    ),
    'gopLimit': (
        ''
    ),
    'doSpotFit': (
        ''
    ),
    'minSpotSize': (
        ''
    ),
    'maxSpotSize': (
        ''
    )
}

def font(pixelSizeDelta=0):
    normalPixelSize = 13
    font = QFont()
    font.setPixelSize(normalPixelSize+pixelSizeDelta)
    return font

def readStoredParamsCSV(csv_path, params):
    """Read old format of analysis_inputs.csv file from spotMAX v1"""
    old_csv_options_to_anchors = {
        'Calculate ref. channel network length?':
            ('Reference channel', 'calcRefChNetLen'),
        'Compute spots size?':
            ('Spots channel', 'Compute spots size'),
        'EGFP emission wavelength (nm):':
            ('METADATA', 'emWavelen'),
        'Effect size used:':
            ('Spots channel', 'gopLimit'),
        'Filter good peaks method:':
            ('Spots channel', 'gopMethod'),
        'Filter spots by reference channel?':
            ('Spots channel', 'filterPeaksInsideRef'),
        'Fit 3D Gaussians?':
            ('Spots channel', 'doSpotFit'),
        'Gaussian filter sigma:':
            ('Pre-processing', 'gaussSigma'),
        'Is ref. channel a single object per cell?':
            ('Reference channel', 'refChSingleObj'),
        'Load a reference channel?':
            ('Reference channel', 'segmRefCh'),
        'Local or global threshold for spot detection?':
            ('Reference channel', 'aggregate'),
        'Numerical aperture:':
            ('METADATA', 'numAperture'),
        'Peak finder threshold function:':
            ('Spots channel', 'spotThresholdFunc'),
        'Reference channel threshold function:':
            ('Reference channel', 'refChThresholdFunc'),
        'Sharpen image prior spot detection?':
            ('Pre-processing', 'sharpenSpots'),
        'Spotsize limits (pxl)':
            ('Spots channel', ('minSpotSize', 'maxSpotSize')),
        'YX resolution multiplier:':
            ('METADATA', 'yxResolLimitMultiplier'),
        'Z resolution limit (um):':
            ('METADATA', 'zResolutionLimit'),
        'ZYX voxel size (um):':
            ('METADATA', ('voxelDepth', 'pixelHeight', 'pixelWidth')),
        'p-value limit:':
            ('Spots channel', 'gopLimit'),
    }
    df = pd.read_csv(csv_path, index='Description')
    for idx, section_anchor in old_csv_options_to_anchors.items():
        section, anchor = section_anchor
        try:
            value = df.at[idx, 'Values']
        except Exception as e:
            value = None
        if isinstance(anchor, tuple):
            for val, sub_anchor in zip(value, anchor):
                params[section][sub_anchor]['loadedVal'] = val
        else:
            params[section][anchor]['loadedVal'] = value
    return params


def readStoredParamsINI(ini_path, params):
    sections = list(params.keys())
    section_params = list(params.values())
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read(ini_path, encoding="utf-8")
    configSections = config.sections()
    for section, section_params in zip(sections, section_params):
        anchors = list(section_params.keys())
        for anchor in anchors:
            option = section_params[anchor]['desc']
            defaultVal = section_params[anchor]['initialVal']
            config_value = None
            if section not in config:
                params[section][anchor]['isSectionInConfig'] = False
                params[section][anchor]['loadedVal'] = None
                continue

            if isinstance(defaultVal, bool):
                config_value = config.getboolean(section, option, fallback=None)
            elif isinstance(defaultVal, float):
                config_value = config.getfloat(section, option, fallback=None)
            elif isinstance(defaultVal, int):
                config_value = config.getint(section, option, fallback=None)
            elif isinstance(defaultVal, str):
                config_value = config.get(section, option, fallback=None)

            params[section][anchor]['isSectionInConfig'] = True
            params[section][anchor]['loadedVal'] = config_value
    return params

def analysisInputsParams(ini_path=default_ini_path):
    params = {
        # Section 0 (GroupBox)
        'File paths': {
            'spotsFilePath': {
                'desc': 'Spots channel file path',
                'initialVal': '',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': True,
                'formWidgetFunc': widgets.tooltipLineEdit,
                'actions': None,
            },
            'segmFilePath': {
                'desc': 'Cells segmentation file path',
                'initialVal': '',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': True,
                'formWidgetFunc': widgets.tooltipLineEdit,
                'actions': None,
            },
            'refChFilePath': {
                'desc': 'Reference channel file path',
                'initialVal': '',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': True,
                'formWidgetFunc': widgets.tooltipLineEdit,
                'actions': None,
            },
            'refChSegmFilePath': {
                'desc': 'Ref. channel segmentation file path',
                'initialVal': '',
                'stretchWidget': True,
                'addInfoButton': True,
                'addComputeButton': False,
                'addApplyButton': False,
                'addBrowseButton': True,
                'formWidgetFunc': widgets.tooltipLineEdit,
                'actions': None,
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
                'initialVal': '',
                'stretchWidget': True,
                'addInfoButton': False,
                'formWidgetFunc': widgets._spotMinSizeLabels,
                'actions': None
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
                'desc': 'How should I filter true spots?',
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
    params = readStoredParamsINI(ini_path, params)
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
