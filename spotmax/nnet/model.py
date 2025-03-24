import os

from typing import Iterable, List

import numpy as np

import torch.nn as nn

import skimage
import skimage.measure
import skimage.transform

from cellacdc._types import Vector

from spotmax import io, printl
from spotmax.nnet import install_and_download, config_yaml_path

def install_and_import_modules():
    install_and_download()
    from spotmax.nnet import transform
    from spotmax.nnet.models.nd_model import (
        Data, Operation, NDModel, Models, models
    )
    return transform, Data, Operation, NDModel, Models, models

def read_default_config():
    import yaml
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

class Model(nn.Module):
    """SpotMAX neural network model for semantic segmentation. This is 
    also a PyTorch model with a forward method
    """
    def __init__(
            self, 
            model_type: AvailableModels='2D', 
            preprocess_across_experiment=False,
            preprocess_across_timepoints=True,
            gaussian_filter_sigma: Vector=0.0,
            remove_hot_pixels=False,
            config_yaml_filepath: os.PathLike='spotmax/nnet/config.yaml',
            threshold_value=0.0,
            PhysicalSizeX: float=0.073,
            resolution_multiplier_yx: float=1.0,
            use_gpu=False,
            save_prediction_map=False,
            verbose=True,
        ):
        """Initialization method of the model

        Parameters
        ----------
        model_type : {'2D', '3D'} str, optional
            Model type. If 2D model is applied to 3D data then it will run 
            on each z-slice. Default is '2D'
        preprocess_across_experiment : bool, optional
            If False and also `preprocess_across_timepoints` is False, the model 
            will assume that the image passed to the `segment` method is 
            pre-processed. If True, pre-processing will run on all input images. 
            Default is False
        preprocess_across_timepoints : bool, optional
            If False and also `preprocess_across_experiment` is False, the model 
            will assume that the image passed to the `segment` method is 
            pre-processed. If True, pre-processing will run on all input images. 
            Default is True
        gaussian_filter_sigma : float or 3 elements (z, y, x) sequence, optional
            Sigma value(s) of the gaussian filter. This can be a single 
            number or one value per dimension of the input image. 
            Default is 0.0
        remove_hot_pixels : bool, optional
            If True, uses morphological opening on the grayscale image that 
            will remove single bright pixels (hot pixels). Default is False
        config_yaml_filepath : os.PathLike, optional
            Path to the YAML configuration file of the model. 
            Pre-trained default is ``spotmax/nnet/config.yaml``
        threshold_value : float, optional
            Threshold value used to convert probability output to binary. 
            Increase or decrease this value to detect less or more spots, 
            respectively. Pass 0.0 to use default values of 0.9 for 2D 
            model and 0.7 for 3D model. Default is 0.0
        PhysicalSizeX : float, optional
            Pixel width in µm/pixel. This value is used to rescale the image 
            to the size of the training images. The pixel size of the training 
            images is defined in the YAML configuration file in nanometers (see 
            the 'base_pixel_size_nm' entry). Default is 0.073, 
            meaning no rescaling is applied when using the pretrained 
            models from ``spotmax/nnet/config.yaml``.
        resolution_multiplier_yx : float, optional
            Additional factor to reduce the scaling factor when resizing the 
            image. Pass a value > 1.0 when you need to detect spots that 
            are larger than the diffraction limit. Default is 1.0
        use_gpu : bool, optional
            If True, inference runs on the GPU. Make sure that the correct 
            version of the NVIDIA CUDA drivers and PyTorch with CUDA are 
            installed. Default is False
        save_prediction_map : bool, optional
            If True, the model will return the prediction map and if the model 
            is used as part of SpotMAX analysis, the map will be saved in 
            each Position folder loaded. Default is False
        verbose : bool, optional
            If True, print additional information text to the terminal. 
            Default is True
        """       
        nn.Module.__init__(self)
         
        modules = install_and_import_modules()
        transform, Data, Operation, NDModel, Models, models =  modules            
        self.transform = transform
        self.Data = Data
        self.Operation = Operation
        self.NDModel = NDModel
        self.Models = Models
        self.models = models
        
        config_yaml_filepath = config_yaml_filepath.replace('\\', '/')
        if config_yaml_filepath == 'spotmax/nnet/config.yaml':
            config_yaml_filepath = config_yaml_path
        else:
            from spotmax.io import get_abspath
            model_py_path = os.path.dirname(os.path.abspath(__file__))
            config_yaml_filepath = get_abspath(
                config_yaml_filepath, src_folderpath=model_py_path
            )
        
        self._config = self._load_config(config_yaml_filepath)
        self._config['verbose'] = verbose
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
        self._save_prediction_map = save_prediction_map
        self.threshold_value = threshold_value
        self.model_type = model_type
        
        self.init_inference_params()
        self.model = self._init_model(model_type)
    
    def _load_config(self, config_yaml_filepath):
        import yaml
        with open(config_yaml_filepath, 'r') as f:
            _config = yaml.safe_load(f)
        return _config
    
    def _get_scale_factor(self, pixel_size_um, resolution_multiplier_yx):
        pixel_size_nm = pixel_size_um*1000
        if self._config['base_pixel_size_nm'] == -1:
            base_pixel_size_nm = 1
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
        import torch
        if use_gpu and torch.backends.mps.is_available():
            device = 'mps'
        elif use_gpu and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        return device
    
    def _get_model_class(self, model_type=None):
        if model_type is None:
            model_type = self.model_type
        
        if model_type == '2D':
            model_class = self.Models.UNET2D
        else:
            model_class = self.Models.UNET3D
        return model_class
    
    def _init_model(self, model_type):
        model_class = self._get_model_class(model_type=model_type)
        model = self.NDModel(
            operation=self.Operation.PREDICT,
            model=model_class,
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
            'your images? You can let SpotMAX taking care of that by setting '
            'both `preprocess_across_experiment = True` or '
            '`preprocess_across_timepoints = True` when you initialize the model.'
        )
    
    def remove_padding(self, pad_width, image):
        y1, x1 = pad_width[1][1], pad_width[2][1]
        cropped = image
        if y1 > 0:
            cropped = cropped[:, :-y1]
        if x1 > 0:
            cropped = cropped[:, :, :-x1]
        return cropped
    
    def init_inference_params(
            self, 
            label_components=False,
        ):
        """Initialize additional parameters used by the `forward` method

        Parameters
        ----------
        label_components : bool, optional
            If True, the binary mask will be labelled with `skimage.measure.label`. 
            This will separate the connected components into objects with an 
            integer ID. Default is False
        """      
        self._label_components = label_components
        if self.threshold_value > 0:
            self._threshold_value = self.threshold_value
            return
        
        if self.model_type == '2D': 
            self._threshold_value = 0.9
        else:
            self._threshold_value = 0.7
        
    def load_state_dict(self, state):
        model_instance = self.model._init_model_instance(verbose=False)
        if self.model_type == '2D':
            model_instance.initialize_network()
            state_dict = model_instance.net.load_state_dict(
                state
            )
        else:
            from spotmax.nnet.models.unet3D.unet3d.model import get_model
            model = get_model(model_instance.config['model'])
            state_dict = model.load_state_dict(state['model_state_dict'])
        return state_dict
    
    def forward(self, x: np.ndarray) -> np.ndarray[bool]:
        if x.ndim != 4:
            raise TypeError(
                'Input images for the forward method must be an array of '
                '(N, Z, Y, X) shape where N is the number of individual images'
            )
        out = np.zeros(x.shape, dtype=bool)
        for n, img in enumerate(x):
            lab = self.segment(
                np.squeeze(img), 
                label_components=False
            )
            out[n] = lab
        return out
    
    def segment(
            self, image,
            label_components=False,
            return_pred: NotParam=False,
        ):
        """Run inference and return the segmentation result

        Parameters
        ----------
        image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray of floats in the range (-1, 1)
            Input 2D or 3D image.
        label_components : bool, optional
            If True, the binary mask will be labelled with `skimage.measure.label`. 
            This will separate the connected components into objects with an 
            integer ID. Default is False

        Returns
        -------
        (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray of ints or bools
            Segmentation mask with the same shape as the input image. If 
            `label_components` is `True`, the boolean masked is labelled 
            with `skimage.measure.label` before being returned.
        """        
        self.init_inference_params(
            label_components=label_components,
        )
        
        orig_yx_shape = image.shape[-2:]
        if not self._batch_preprocess:
            image = self.preprocess(image[np.newaxis])[0]
        
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
        prediction, _ = self.model(input_data)
        
        if pad_width is not None:
            prediction = self.remove_padding(pad_width, prediction)
        
        thresh = prediction > self._threshold_value
        thresh = self.resize_to_orig_shape(thresh, orig_yx_shape)
        
        if label_components:
            lab = skimage.measure.label(thresh)
        else:
            lab = thresh
        
        if return_pred or self._save_prediction_map:
            prediction = self.resize_to_orig_shape(prediction, orig_yx_shape)
            return lab, prediction
        else:
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
    params = io.nnet_params_from_ini_params(
        ini_params, sections, model_module, 
        use_default_for_missing=use_default_for_missing
    )
    
    return params

def url_help():
    return ''