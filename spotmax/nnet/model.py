import os
import skimage
import yaml

import skimage.measure

import torch

from . import config_yaml_path
from . import transform
from .models.nd_model import  Data, Operation, NDModel, Models

class AvailableModels:
    values = ['2D', '3D']

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

def url_help():
    return ''