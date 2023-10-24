import os
import numpy as np
import skimage
import yaml

import skimage.measure
import skimage.transform

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

class NotParam:
    not_a_param = True

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
            preprocess_across_experiment=False,
            preprocess_across_timepoints=True,
            remove_hot_pixels=False,
            config_yaml_filepath: os.PathLike=config_yaml_path,
            PhysicalSizeX: float=0.073,
            use_gpu=False,
        ):
        self._config = self._load_config(config_yaml_filepath)
        self._scale_factor = self._get_scale_factor(PhysicalSizeX)
        self.x_transformer = self._init_data_transformer(remove_hot_pixels)
        self._config['device'] = self._get_device_str(use_gpu)
        self._batch_preprocess = (
            preprocess_across_experiment or preprocess_across_timepoints
        )
        self.model_type = model_type
        self.model = self._init_model(model_type)
    
    def _load_config(self, config_yaml_filepath):
        with open(config_yaml_filepath, 'r') as f:
            _config = yaml.safe_load(f)
        return _config
    
    def _get_scale_factor(self, pixel_size):
        pixel_size_nm = pixel_size*1000
        return pixel_size_nm/self._config['base_pixel_size_nm']

    def _init_data_transformer(self, remove_hot_pixels):
        x_transformer = transform.ImageTransformer(logs=False)
        if remove_hot_pixels:
            x_transformer.set_pipeline([
                # transform._rescale,
                transform._opening,
                transform._normalize,
            ])
        else:
            x_transformer.set_pipeline([
                # transform._rescale,
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
    
    def pad_if_smaller_than_patch_shape(self, patch_shape, image):
        Y, X = image.shape[-2:]
        patch_y, patch_x = patch_shape[-2:]
        if Y >= patch_y and X > patch_x:
            return image, None
        
        pad_y = patch_y - Y
        pad_x = patch_x - X
        pad_y = pad_y if pad_y >= 0 else 0     
        pad_x = pad_x if pad_x >= 0 else 0  
        
        pad_width = ((0, 0), (0, pad_y), (0, pad_x))
        pad_value = image.min()
        image = np.pad(image, pad_width=pad_width, constant_values=pad_value)
        return image, pad_width
    
    def preprocess(self, images):
        transformed_data = self.x_transformer.transform(images)
        return transformed_data
    
    def rescale_to_base_pixel_width(self, image):
        if self._scale_factor == 1:
            return image
        
        if image.ndim == 2:
            scaled = skimage.transform.rescale(
                image, self._scale_factor, order=1
            )
        else:
            scaled = np.array([
                skimage.transform.rescale(img_z, self._scale_factor, order=1)
                for img_z in image
            ], dtype=image.dtype)
        return scaled
    
    def resize_to_orig_shape(self, thresh, orig_shape):
        if thresh.shape[-2:] == orig_shape:
            return thresh
        
        if thresh.ndim == 2:
            thresh_resized = skimage.transform.resize(thresh, orig_shape)
        else:
            thresh_resized = np.array([
                skimage.transform.resize(thresh_z, orig_shape) 
                for thresh_z in thresh
            ])
        return thresh_resized
    
    def _check_input_dtype_is_float(self, image):
        if isinstance(image[tuple([0]*image.ndim)], (np.floating, float)):
            return
        
        raise TypeError(
            f'Input image has data type {image.dtype}. The only supported types '
            'are float64, float32, and float16. Did you forget to pre-process '
            'your images? You can let spotMAX taking care of that by setting '
            'both `preprocess_across_experiment=False` and '
            '`preprocess_across_timepoints=False` when you initialize the model.'
        )
    
    def remove_padding(self, pad_width, image):
        y1, x1 = pad_width[1][1], pad_width[2][1]
        cropped = image[:, :-y1, :-x1]
        return cropped
    
    def segment(
            self, image,
            threshold_value=0.9,
            label_components=False,
            verbose: NotParam=True
        ):
        orig_yx_shape = image.shape[-2:]
        if not self._batch_preprocess:
            image = self.preprocess(image)
        
        self._check_input_dtype_is_float(image)
        
        rescaled = self.rescale_to_base_pixel_width(image)

        pad_width = None
        if self.model_type == '3D':
            loaders_config = self._config['unet3D']['predict']['loaders']
            slice_builder_config = loaders_config['test']['slice_builder']
            patch_shape = slice_builder_config['patch_shape']
            rescaled, pad_width = self.pad_if_smaller_than_patch_shape(
                patch_shape, rescaled
            )
                
        input_data = Data(
            images=rescaled, masks=None, val_images=None, val_masks=None
        )
        prediction, _ = self.model(input_data, verbose=verbose)
        
        if pad_width is not None:
            prediction = self.remove_padding(pad_width, prediction)
        
        thresh = prediction > threshold_value
        thresh = self.resize_to_orig_shape(thresh, orig_yx_shape)
        
        if label_components:
            lab = skimage.measure.label(thresh)
        else:
            lab = thresh
            
        return lab

def _raise_missing_param_ini(missing_option):
    raise KeyError(
        'The following parameter is missing from the INI configuration file: '
        f'`{missing_option}`. You can force using default value by setting '
        '`Use default values for missing parameters = True` in the '
        'INI file.'
    )

def get_nnet_params_from_ini_params(ini_params, use_default_for_missing=False):
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
            try:
                not_a_param = argWidget.type().not_a_param
                continue
            except Exception as err:
                pass
            option = section_params.get(argWidget.name)
            if option is None:
                if use_default_for_missing:
                    continue
                else:
                    _raise_missing_param_ini(argWidget.name)
            value = option['loadedVal']
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
            try:
                not_a_param = argWidget.type().not_a_param
                continue
            except Exception as err:
                pass
                
            option = section_params.get(argWidget.name)
            if option is None:
                if use_default_for_missing:
                    continue
                else:
                    _raise_missing_param_ini(argWidget.name)
            value = option['loadedVal']
            if not isinstance(argWidget.default, str):
                try:
                    value = utils.to_dtype(value, type(argWidget.default))
                except Exception as err:
                    value = argWidget.default
            params['segment'][argWidget.name] = value
    
    return params

def url_help():
    return ''