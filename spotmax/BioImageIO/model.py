import numpy as np

import skimage.measure

from .. import io

from . import install

class Model:
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
            visit the BioImage.IO website at the followng link `bioimage.io <https://bioimage.io/#/>`_.
        """       
        install()
        import bioimageio.core
        import xarray as xr
        
        self.bioimageio_core = bioimageio.core
        self.xr = xr
        
        self.model_resource = self.bioimageio_core.load_resource_description(
            model_doi_url_or_zip_path
        )
        self.prediction_pipeline = self.bioimageio_core.create_prediction_pipeline(
            self.model_resource, devices=None, weight_format=None
        )
        self.dims = tuple(self.model_resource.inputs[0].axes)
    
    def _test_model(self):
        """
        The function 'test_model' from 'bioimageio.core.resource_tests' 
        can be used to fully test the model, including running prediction for 
        the test input(s) and checking that they agree with the test output(s).
        Before using a model, it is recommended to check that it properly works. 
        The 'test_model' function returns a dict with 'status'='passed'/'failed' 
        and more detailed information.
        """
        from self.bioimageio_core.resource_tests import test_model
        test_result = test_model(self.model_resource)[0]
        if test_result["status"] == "failed":
            print("model test:", test_result["name"])
            print("The model test failed with:", test_result["error"])
            print("with the traceback:")
            print("".join(test_result["traceback"]))
        else:
            test_result["status"] == "passed"
            print("The model passed all tests")
        return test_result
    
    def _test_prediction(self):        
        # Load the example image for this model, which is stored in numpy file format.
        input_image = np.load(self.model_resource.test_inputs[0])
        
        # Create an xarray.DataArray from the input image.
        # DataArrays are like numpy arrays, but they have annotated axes.
        # The axes are used to validate that the axes of the input image match the axes expected by a model.
        input_array = self.xr.DataArray(
            input_image, dims=tuple(self.model_resource.inputs[0].axes)
        )
        
        # print the axis annotations ('dims') and the shape of the input array
        print(f'Input array annotations: {input_array.dims}')
        print(f'Input array shape: {input_array.shape}')

        # Next, create a 'prediction_pipeline'. The prediction_pipeline is used 
        # to run prediction with a given model.
        # This means it applies the preprocessing, runs inference with the 
        # model and applies the postprocessing.

        # The 'devices' argument can be used to specify which device(s) to 
        # use for inference with the model.
        # Hence it can be used to specify whether to use the cpu, a single gpu 
        # or multiple gpus (not implemented yet).
        # By default (devices=None) a gpu will be used if available and 
        # otherwise the cpu will be used.
        devices = None

        # The 'weight_format' argument can be used to specify which weight 
        # format available in the model to use.
        # By default (weight_format=None) the weight format with highest 
        # priority (as defined by bioimageio.core) will be used.
        weight_format = None

        prediction_pipeline = self.bioimageio_core.create_prediction_pipeline(
            self.model_resource, devices=devices, weight_format=weight_format
        )
        
        # Use the prediction pipeline to run prediction for the image we 
        # loaded before.
        # The prediction pipeline expects inputs to have a shape that fits 
        # the model exactly.
        # So if the input does not fit the expected input shape the prediction 
        # will fail.
        # Therefore, we can use the function `predict_with_padding`, 
        # which will pad the image to a shape that fits the model.
        prediction_xarray = self.bioimageio_core.predict_with_padding(
            prediction_pipeline, input_array
        )[0]
        prediction = prediction_xarray.to_numpy()
        return prediction, input_image
    
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
            # is 3D capable ('z' is in self.dimns)
            input_image = input_image[np.newaxis]
        
        thresholded = np.zeros(input_image.shape, dtype=bool)
        
        for i, img in enumerate(input_image):
            img = self.reshape_to_required_shape(img)
            input_xarray = self.xr.DataArray(img, dims=self.dims)
            prediction_xarray = self.bioimageio_core.predict_with_padding(
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
    params = io.nnet_params_from_init_params(
        ini_params, sections, model_module, 
        use_default_for_missing=use_default_for_missing
    )
    
    return params

def url_help():
    return 'https://bioimage.io/#/'