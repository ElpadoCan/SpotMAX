import os
import numpy as np
import skimage
import yaml

import skimage.measure
import skimage.transform

import torch

from cellacdc.types import Vector

from .. import io, printl
from . import install_and_download
from . import config_yaml_path

def install_and_import_modules():
    install_and_download()
    from . import transform
    from .models.nd_model import Data, Operation, NDModel, Models
    return transform, Data, Operation, NDModel, Models

def read_default_config():
    try:
        with open(config_yaml_path, 'r') as f:
            default_config = yaml.safe_load(f)
    except:
        default_config = None
    return default_config
        
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
        default_config = read_default_config()
        
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
            gaussian_filter_sigma: Vector=0.0,
            remove_hot_pixels=False,
            config_yaml_filepath: os.PathLike=config_yaml_path,
            PhysicalSizeX: float=0.073,
            resolution_multiplier_yx: float=1.0,
            use_gpu=False,
        ):
        modules = install_and_import_modules()
        transform, Data, Operation, NDModel, Models =  modules            
        self.transform = transform
        self.Data = Data
        self.Operation = Operation
        self.NDModel = NDModel
        self.Models = Models
        
        self._config = self._load_config(config_yaml_filepath)
        self._scale_factor = self._get_scale_factor(
            PhysicalSizeX, resolution_multiplier_yx
        )
        self.x_transformer = self._init_data_transformer(
            remove_hot_pixels, gaussian_filter_sigma, use_gpu
        )
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
    
    def _get_scale_factor(self, pixel_size_um, resolution_multiplier_yx):
        pixel_size_nm = pixel_size_um*1000
        sf = (
            pixel_size_nm
            /self._config['base_pixel_size_nm']
            /resolution_multiplier_yx
        )
        return sf

    def _init_data_transformer(
            self, remove_hot_pixels, gaussian_filter_sigma, use_gpu
        ):
        x_transformer = self.transform.ImageTransformer(logs=False)
        if remove_hot_pixels:
            x_transformer.add_step(self.transform._opening)
        x_transformer.add_step(
            self.transform._gaussian_filter, 
            sigma=gaussian_filter_sigma,
            use_gpu=use_gpu
        )
        x_transformer.add_step(self.transform._normalize)
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
            MODEL = self.Models.UNET2D
        else:
            MODEL = self.Models.UNET3D
        model = self.NDModel(
            operation=self.Operation.PREDICT,
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
                
        input_data = self.Data(
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

def get_model_params_from_ini_params(
        ini_params, use_default_for_missing=False, subsection='spots'
    ):
    sections = [
        f'neural_network.init.{subsection}', 
        f'neural_network.segment.{subsection}'
    ]
    if not any([section in ini_params for section in sections]):
        # Keep compatibility with previous versions that did not have subsection
        sections = [
            f'neural_network.init', 
            f'neural_network.segment'
        ]
        if not any([section in ini_params for section in sections]):
            return 
    
    import spotmax.nnet.model as model_module
    params = io.nnet_params_from_init_params(
        ini_params, sections, model_module, 
        use_default_for_missing=use_default_for_missing
    )
    
    return params

def url_help():
    return ''