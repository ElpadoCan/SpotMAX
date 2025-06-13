from typing import Tuple

import numpy as np

import skimage.measure

from .. import io

from . import install

class NotParam:
    not_a_param = True

class Model:
    """SpotMAX implementation of any BioImage.IO model
    """    
    def __init__(
            self, 
            model_doi_url_rdf_or_zip_path='',
            logger_func: NotParam=print
        ):
        """Initialize Bioimage.io Model class

        Parameters
        ----------
        model_doi_url_rdf_or_zip_path : str, optional
            Bioimage.io models can be lodaded using different representation.
            You can either provide the DOI of the model, the URL, or download it
            yourself and provide the path to the downloaded zip file.
            
            For more information and to visualize the available models 
            visit the BioImage.IO website at the followng link 
            `bioimage.io <https://bioimage.io/#/>`_.
        """       
        install()
        import bioimageio.core
        
        self.logger_func = logger_func
        self.model_description = bioimageio.core.load_model_description(
            model_doi_url_rdf_or_zip_path
        )
        self.kwargs = self.get_kwargs()
        self.prediction_pipeline = bioimageio.core.create_prediction_pipeline(
            self.model_description
        )
    
    def set_kwargs(self, kwargs):
        if not kwargs:
            return
        
        if kwargs is None:
            return
        
        architecture_yaml = (
            self.model_description.weights.pytorch_state_dict.architecture
        )
        model_kwargs = {**architecture_yaml.kwargs, **kwargs}
        architecture_yaml.kwargs = model_kwargs
    
    def get_kwargs(self):
        try:
            kwargs = (
                self.model_description.weights.pytorch_state_dict
                .architecture.kwargs
            )
        except Exception as err:
            kwargs = {} 
        
        return kwargs
    
    def _test_model(self):
        """
        The function 'test_model' from 'bioimageio.core' 
        can be used to fully test the model, including running prediction for 
        the test input(s) and checking that they agree with the test output(s).
        Before using a model, it is recommended to check that it properly works. 
        The 'test_model' function returns a dict with 'status'='passed'/'failed' 
        and more detailed information.
        """
        from bioimageio.core import test_model
        validation_summary = test_model(self.model_resource)
        self.logger_func(validation_summary.display())
        return validation_summary
    
    def _convert_dtype(self, image):
        try:
            input_dtype = self.model_description.inputs[0].data.type
            if input_dtype.startswith('float'):
                return image
            if input_dtype == 'uint8':
                from skimage.util import img_as_ubyte
                image = img_as_ubyte(image)
            elif input_dtype == 'uint16':
                from skimage.util import img_as_uint
                image = img_as_uint(image)
            return image
        except Exception as err:
            return image
    
    def create_input_sample(self, image: np.ndarray):
        import bioimageio.core
        from bioimageio.spec.model.v0_5 import TensorId
        
        image = np.squeeze(image)
        image = self._convert_dtype(image)
        
        axes = self.model_description.inputs[0].axes
        space_axis_ids = {'z', 'y', 'x'}
        output_index = []
        for axis in axes:
            if axis.id == 'z' and image.ndim == 2:
                image = image[np.newaxis]
                output_index.append(0)
        
        for axis in axes:
            if axis.id in space_axis_ids:
                continue
            
            image = image[np.newaxis]
            output_index.append(0)
        
        dims = [axis.id for axis in axes]
        input_tensor = bioimageio.core.Tensor(
            array=image, 
            dims=dims
        )
        
        input_ids = bioimageio.core.digest_spec.get_member_ids(
            self.model_description.inputs
        )
        
        sample = bioimageio.core.Sample(
            members={
                TensorId(input_ids[0]): input_tensor
            }, 
            stat={}, 
            id='inputs'
        )

        return sample, tuple(output_index)
    
    def get_prediction_mask(
            self,
            prediction_output, 
            reshape_output_index: Tuple[int],
            output_index: int=0
        ):
        members = prediction_output.members
        out_ids = list(members.keys())
        out_id = out_ids[output_index]
        prediction_arr = np.array(members[out_id])[reshape_output_index]
        prediction_mask = prediction_arr.astype(bool)
        return prediction_mask
    
    def segment(
            self, image, 
            output_index=0, 
            label_components=False
        ):
        """Run model prediction and return masks

        Parameters
        ----------
        image : 3D (Z, Y, X) or 2D (Y, X) np.ndarray
            3D z-stack or 2D input image as a numpy array
        output_index : int, optional
            Some BioImage.IO models returns multiple outputs. Check the documentation 
            of the specific model to understand which output could be more 
            useful for spot detection. By default 0
        label_components : bool, optional
            If True, the thresholded prediction array will be labelled using 
            the scikit-image function `skimage.measure.label`. 
            This will assign a unique integer ID to each separated object.
            By default False

        Returns
        -------
        np.ndarray
            Output of the model as a numpy array with same shape of as the input image. 
            If `label_components = True`, the output is the result of calling the 
            scikit-image function `skimage.measure.label` on the thresholded 
            array. If `label_components = False`, the returned array is simply 
            the thresholded binary output.
        
        See also
        --------
        `skimage.measure.label <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label>`__
        """
        sample, reshape_output_index = self.create_input_sample(image)
        
        prediction_output = (
            self.prediction_pipeline.predict_sample_without_blocking(
                sample
            )
        )
        
        prediction_mask = self.get_prediction_mask(
            prediction_output, reshape_output_index, output_index
        )
        
        if label_components:
            return skimage.measure.label(prediction_mask)
        else:
            return prediction_mask

def get_model_params_from_ini_params(
        ini_params, use_default_for_missing=False, subsection='spots'
    ):
    sections = [
        f'bioimageio_model.init.{subsection}', 
        f'bioimageio_model.segment.{subsection}'
    ]
    if not any([section in ini_params for section in sections]):
        return 
    
    sections.append(f'bioimageio_model.kwargs.{subsection}')
    
    import spotmax.BioImageIO.model as model_module
    params = io.nnet_params_from_ini_params(
        ini_params, sections, model_module, 
        use_default_for_missing=use_default_for_missing
    )
    
    return params

def url_help():
    return 'https://bioimage.io/#/'