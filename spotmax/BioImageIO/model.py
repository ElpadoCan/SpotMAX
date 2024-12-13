import numpy as np

import skimage.measure

from .. import io

from . import install

class Model:
    """SpotMAX implementation of any BioImage.IO model
    """    
    def __init__(self, model_doi_url_or_zip_path=''):
        """Initialize Bioimage.io Model class

        Parameters
        ----------
        model_doi_url_or_zip_path : str, optional
            Bioimage.io models can be lodaded using different representation.
            You can either provide the DOI of the model, the URL, or download it
            yourself (select "Ilastik" weight format) and provide the path to 
            the downloaded zip file.
            
            For more information and to visualize the available models 
            visit the BioImage.IO website at the followng link 
            `bioimage.io <https://bioimage.io/#/>`_.
        """       
        install()
        import bioimageio.core
        import xarray as xr
        
        self.bioimageio_core = bioimageio.core
        self.xr = xr
        
        self.model_resource = bioimageio.core.load_model_description(
            model_doi_url_or_zip_path
        )
        # self.prediction_pipeline = self.bioimageio_core.create_prediction_pipeline(
        #     self.model_resource, devices=None, weight_format=None
        # )
        self.dims = tuple(self.model_resource.inputs[0].axes)
    
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
    
    def reshape_to_required_shape(self, img):
        for axis in self.dims:
            if axis == 'y':
                continue
            if axis == 'z':
                continue
            if axis == 'x':
                continue
            img = img[np.newaxis]
        return img
    
    def segment(
            self, image, 
            threshold_value=0.5,
            output_index=0, 
            label_components=False
        ):
        """_summary_

        Parameters
        ----------
        image : 3D (Z, Y, X) or 2D (Y, X) np.ndarray
            3D z-stack or 2D input image as a numpy array
        threshold_value : float, optional
            Threshold value in the range 0-1 to convert the prediction output 
            of the model to a binary image. 
            Increasing this value might help removing artefacts. By default 0.5
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
        
        # Build slice object to get the correct output index
        output_index_loc = self.dims.index('c')
        output_index_slice = [slice(None) for _ in range(len(self.dims))]
        output_index_slice[output_index_loc] = output_index
        output_index_slice = tuple(output_index_slice)
        
        input_image = image
        if image.ndim == 2:
            # Add axis for z-slices
            input_image = image[np.newaxis]        
        
        if 'z' in self.dims:
            # Add fake axis because we want to predict on 3D since the model 
            # is 3D capable ('z' is in self.dims) --> input_image must be 4D
            input_image = input_image[np.newaxis]
        
        thresholded = np.zeros(input_image.shape, dtype=bool)
        
        for i, img in enumerate(input_image):
            img = self.reshape_to_required_shape(img)
            input_xarray = self.xr.DataArray(img, dims=self.dims)
            prediction_xarray = self.bioimageio_core.prediction.predict(
                self.prediction_pipeline, input_xarray
            )[0]
            
            prediction = prediction_xarray.to_numpy()[output_index_slice]
            
            prediction = np.squeeze(prediction)
            thresholded[i] = prediction > threshold_value
        
        thresholded = np.squeeze(thresholded)
        
        if label_components:
            return skimage.measure.label(thresholded)
        else:
            return thresholded

def get_model_params_from_ini_params(
        ini_params, use_default_for_missing=False, subsection='spots'
    ):
    sections = [
        f'bioimageio_model.init.{subsection}', 
        f'bioimageio_model.segment.{subsection}'
    ]
    if not any([section in ini_params for section in sections]):
        return 
    
    import spotmax.BioImageIO.model as model_module
    params = io.nnet_params_from_ini_params(
        ini_params, sections, model_module, 
        use_default_for_missing=use_default_for_missing
    )
    
    return params

def url_help():
    return 'https://bioimage.io/#/'