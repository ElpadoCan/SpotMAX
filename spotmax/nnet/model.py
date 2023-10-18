import os
import skimage
import yaml

import skimage.measure

import torch

from cellacdc import myutils as acdc_myutils
from .. import utils

from . import config_yaml_path
from . import transform
from .models.nd_model import  Data, Operation, NDModel, Models

try:
    with open(config_yaml_path, 'r') as f:
        default_config = yaml.safe_load(f)
except:
    default_config = None

class AvailableModels:
    values = ['2D', '3D']

class CustomSignals:
    def __init__(self):
        self.slots_info = [{
            'group': 'init',
            'widget_name': 'model_type',
            'signal': 'currentTextChanged',
            'slot': self.updateDefaultThrehsoldMethod    
        },
        ]
    
    def updateDefaultThrehsoldMethod(self, win, model_type):
        if default_config is None:
            return
        
        for argwidget in win.argsWidgets:
            if argwidget.name == 'threshold_value':
                if model_type == '2D':
                    default_params = default_config['unet2D']['default_params']
                    thresh_val = default_params['threshold_value']
                else:
                    default_params = default_config['unet3D']['default_params']
                    thresh_val = default_params['threshold_value']
                argwidget.widget.setValue(thresh_val)
                break

class Model:
    def __init__(
            self, 
            model_type: AvailableModels='2D', 
            preprocess_across_experiment=True,
            config_yaml_filepath: os.PathLike=config_yaml_path,
            PhysicalSizeX: float=0.073,
            use_gpu=False,
        ):
        self._config = self._load_config(config_yaml_filepath)
        self._scale_factor = self._get_scale_factor(PhysicalSizeX)
        self.x_transformer = self._init_data_transformer()
        self._config['device'] = self._get_device_str(use_gpu)
        self._batch_preprocess = preprocess_across_experiment
        self.model = self._init_model(model_type)
    
    def _load_config(self, config_yaml_filepath):
        with open(config_yaml_filepath, 'r') as f:
            _config = yaml.safe_load(f)
        return _config
    
    def _get_scale_factor(self, pixel_size):
        pixel_size_nm = pixel_size*1000
        return pixel_size_nm/self._config['base_pixel_size_nm']

    def _init_data_transformer(self):
        x_transformer = transform.ImageTransformer(logs=False)
        x_transformer.set_pipeline([
            transform._rescale,
            # transform._opening,
            transform._normalize,
        ])
        return x_transformer

    def _get_device_str(self, use_gpu: bool):
        if use_gpu and torch.backends.mps.is_available():
            device = 'mps'
        elif use_gpu and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        return device
    
    def _init_model(self, model_type):
        if model_type == '2D':
            MODEL = Models.UNET2D
        else:
            MODEL = Models.UNET3D
        model = NDModel(
            operation=Operation.PREDICT,
            model=MODEL,
            config=self._config
        )
        return model
    
    def preprocess(self, images):
        transformed_data = self.x_transformer.transform(
            images, scale=self._scale_factor
        )
        return transformed_data
    
    def segment(
            self, image,
            threshold_value=0.9,
            label_components=False
        ):
        if not self._batch_preprocess:
            image = self.preprocess(image)
        
        input_data = Data(
            images=image, masks=None, val_images=None, val_masks=None
        )
        prediction, _ = self.model(input_data)
        thresh = prediction > threshold_value
        if label_components:
            lab = skimage.measure.label(thresh)
        else:
            lab = thresh
        
        return lab

def get_nnet_params_from_ini_params(ini_params):
    sections = ['neural_network.init', 'neural_network.segment']
    if not any([section in ini_params for section in sections]):
        return 
    
    import spotmax.nnet.model as model_module
    init_params, segment_params = acdc_myutils.getModelArgSpec(model_module)
    params = {'init': {}, 'segment': {}}
    
    for section in sections:
        if section not in ini_params:
            continue
    
    section = sections[0]
    if section in ini_params:
        section_params = ini_params[section]
        for argWidget in init_params:
            value = section_params[argWidget.name]['loadedVal']
            if not isinstance(argWidget.default, str):
                try:
                    value = utils.to_dtype(value, type(argWidget.default))
                except Exception as err:
                    value = argWidget.default
            params['init'][argWidget.name] = value
    
    section = sections[1]
    if section in ini_params:
        section_params = ini_params[section]
        for argWidget in segment_params:
            value = section_params[argWidget.name]['loadedVal']
            if not isinstance(argWidget.default, str):
                try:
                    value = utils.to_dtype(value, type(argWidget.default))
                except Exception as err:
                    value = argWidget.default
            params['segment'][argWidget.name] = value
    
    return params

def url_help():
    return ''