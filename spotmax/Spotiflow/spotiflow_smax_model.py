import os

import numpy as np
import skimage.measure

from .. import io, printl
from . import install

class Devices:
    values = ['auto', 'cpu', 'cuda', 'mps', 'None']

class AvailableModels:
    values = [
        'general', 'synth_3d', 'smfish_3d', 'hybiss', 'synth_complex', 'custom'
    ]

class PeakModes:
    values = ['skimage', 'fast']

class SubPixel:
    values = ['True', 'False', 'None']

class OutputLabels:
    values = ['Boolean mask', 'Connected components']

class Model:
    """SpotMAX implementation of Spotiflow model
    """   
    def __init__(
            self, 
            device: Devices='auto',
            model_name: AvailableModels='general',
            custom_model_path: os.PathLike=''
        ):
        """Initialize Spotiflow model

        Parameters
        ----------
        device : {'auto', 'cpu', 'cuda', 'mps', 'None'}, optional
            Computing device to use. If 'None', will infer from model location. 
            If 'auto', will infer from available hardware. Default is 'auto'
        model_name : {'general', 'custom'}, optional
            Either the pre-trained model 'general', or your own 'custom' 
            trained model. If 'custom', please provide the model path in 
            the `custom_model_path` parameter. Default is 'general'
        custom_model_path : os.PathLike, optional
            Path to the custom trained model. Default is ''
        """        
        install()
        from spotiflow.model import Spotiflow
        if device == 'None':
            device = None
            map_location = 'auto'
        else:
            map_location = device
        
        self._device = device
        
        if model_name == 'general' and os.path.exists(custom_model_path):
            raise TypeError(
                'The "general" model cannot be loaded if the `custom_model_path` '
                'exists! Please, either pass `model_name="custom" with an '
                'existing custom model, or `model_name="custom" with empty '
                '`custom_model_path`, thanks.'
            )
        elif model_name == 'custom' and os.path.exists(custom_model_path):
            self.model = Spotiflow.from_folder(
                custom_model_path, 
                map_location=map_location
            )
        elif model_name == 'general':
            self.model = Spotiflow.from_pretrained(
                'general', 
                map_location=map_location
            )
        elif model_name == 'custom':
            raise FileNotFoundError(
                f'This model folder does not exist: "{custom_model_path}"'
            )
    
    def segment(
            self, image,
            run_on_3D_zstack_volume: bool=False,
            output_labels: OutputLabels='Boolean mask',
            prob_thresh: float=0.0,
            expected_spot_radius: int=1,
            exclude_border: bool = False, 
            scale: int = 1, 
            subpix:  SubPixel = 'None', 
            peak_mode: PeakModes = 'fast',
            verbose: bool = True
        ):
        """Run Spotiflow inference and build the segmentation masks from the 
        detecte coordinates. 

        Parameters
        ----------
        image : (Y, X) or (Z, Y, X) np.ndarray
            Input image
        output_labels : {'Boolean mask', 'Connected components'}, optional
            The type of the output labels. If 'Connected components' and the 
            image is 2D, each spot mask will receive a unique integer ID. If 
            the image is 3D then the boolean mask will be created first 
            and converted to connected components with the function 
            skimage.measure.label from the scikit-image package. 
            Default is 'Boolean mask'
        prob_thresh : float, optional
            Probability threshold for peak detection. If None, will load the 
            optimal one. Default is 0.0
        expected_spot_radius : int, optional
            Expected spot radius in pixels. This will be used as the minimum 
            distance allowed between spots' centers and as the radius of 
            the circular spot masks in the output labels. Default is 1
        exclude_border : bool, optional
            Whether to exclude spots at the border. Default is False
        scale : int, optional
            Scale factor to apply to the image. Default is 1
        subpix : {'True', 'False', 'None'}, optional
            Whether to use the stereographic flow to compute subpixel 
            localization. If 'None', will deduce from the model configuration. 
            Default is 'None'
        peak_mode : {'skimage', 'fast'}, optional
            Peak detection mode (can be either 'skimage' or 'fast', which is a 
            faster Spotiflow C++ implementation). Default is 'fast'
        verbose : bool, optional
            If `True`, additional text and progress will be displayed in 
            the terminal. Default is True
        """        
        if prob_thresh == 0:
            prob_thresh = None
        
        if scale == 1:
            scale = None
        
        if expected_spot_radius < 1:
            expected_spot_radius = 1
        
        subpix = eval(subpix)
        
        lab = np.zeros(image.shape, dtype=bool)
        is2D = False
        if image.ndim == 2:
            image = (image,)
            lab = lab[np.newaxis]
            is2D = True
            if output_labels != 'Boolean mask':
                lab = lab.astype(np.uint32)
        
        predict_kwargs = dict(
            prob_thresh=prob_thresh,
            min_distance=expected_spot_radius,
            exclude_border=exclude_border,
            scale=scale,
            subpix=subpix,
            peak_mode=peak_mode,
            device=self._device,
            verbose=verbose
        )
        
        if run_on_3D_zstack_volume:
            points, details = self.model.predict(image, **predict_kwargs)
        else:
            points = []
            for z, img in enumerate(image):
                yx_points, details = self.model.predict(img, **predict_kwargs)
                points.append(yx_points)
            points = np.array(points)
        
        for z, yx_coords in enumerate(points):
            z_slice = round(z)
            coords = np.round(yx_coords).astype(int)

            for yc, xc in coords:
                rr, cc = skimage.draw.disk(
                    (yc, xc), expected_spot_radius, shape=img.shape
                )
                if output_labels == 'Boolean mask' and is2D:
                    values = np.arange(1, len(rr)+1)
                else:
                    values = True
                
                lab[z_slice, rr, cc] = values
        
        if is2D:
            lab = lab[0]
        elif output_labels != 'Boolean mask':
            lab = skimage.measure.label(lab)
        
        return lab

def get_model_params_from_ini_params(
        ini_params, use_default_for_missing=False, subsection='spots'
    ):
    sections = [
        f'spotiflow.init.{subsection}', 
        f'spotiflow.segment.{subsection}'
    ]
    if not any([section in ini_params for section in sections]):
        # Keep compatibility with previous versions that did not have subsection
        sections = [
            f'spotiflow.init', 
            f'spotiflow.segment'
        ]
        if not any([section in ini_params for section in sections]):
            return 
    
    import spotmax.Spotiflow.spotiflow_smax_model as model_module
    params = io.nnet_params_from_ini_params(
        ini_params, sections, model_module, 
        use_default_for_missing=use_default_for_missing
    )
    
    return params

def url_help():
    return 'https://weigertlab.github.io/spotiflow/index.html'