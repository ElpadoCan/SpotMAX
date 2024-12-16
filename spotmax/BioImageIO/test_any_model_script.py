try:
    import pytest
    pytest.skip('skipping this test', allow_module_level=True)
except Exception as e:
    pass

import os

import numpy as np

from typing import Tuple

import bioimageio.core
from bioimageio.spec.model import AnyModelDescr
from bioimageio.spec.model.v0_5 import TensorId

from cellacdc.plot import imshow
from spotmax import spotmax_path

MODEL_SOURCE = os.path.join(
    spotmax_path, 'BioImageIO', 'example_models', 'SpotMAX_AI_2D', 'rdf.yaml'
)
# INPUT_IMAGE_PATH = os.path.join(
#     spotmax_path, 'BioImageIO', 'SpotMAX_UNet_2D', 'input_sample.npy'
# )
INPUT_IMAGE_PATH = os.path.join(
    spotmax_path, 'BioImageIO', 'SpotMAX_UNet_2D', 'Input_sample_Positions',
    'Position_11', 'Images', 'ASY15-1_60nM-11_s11_mNeon.tif'
)

def load_and_squeeze_input_image():
    if INPUT_IMAGE_PATH.endswith('.npy'):
        input_img = np.load(INPUT_IMAGE_PATH)
    elif INPUT_IMAGE_PATH.endswith('.tif'):
        import skimage.io
        input_img = skimage.io.imread(INPUT_IMAGE_PATH)
    elif INPUT_IMAGE_PATH.endswith('.npz'):
        input_img = np.load(INPUT_IMAGE_PATH)['arr_0']
    
    return np.squeeze(input_img)

def create_input_sample(input_image: np.ndarray, model_descr: AnyModelDescr):
    axes = model_descr.inputs[0].axes
    space_axis_ids = {'z', 'y', 'x'}
    output_index = []
    for axis in axes:
        if axis.id == 'z' and input_image.ndim == 2:
            input_image = input_image[np.newaxis]
            output_index.append(0)
    
    for axis in axes:
        if axis.id in space_axis_ids:
            continue
        
        input_image = input_image[np.newaxis]
        output_index.append(0)
    
    dims = [axis.id for axis in axes]
    input_tensor = bioimageio.core.Tensor(
        array=input_image, 
        dims=dims
    )
    
    input_ids = bioimageio.core.digest_spec.get_member_ids(
        model_descr.inputs
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
        prediction_output: bioimageio.core.Sample, 
        model_descr: AnyModelDescr,
        output_index: Tuple[int]
    ):
    members = prediction_output.members
    out_ids = list(members.keys())
    out_id = out_ids[0]
    prediction_mask = np.array(members[out_id])[output_index].astype(bool)
    return prediction_mask

def load_model_descr():
    print(f'Loading model description from "{MODEL_SOURCE}"...')
    model_descr = bioimageio.core.load_model_description(MODEL_SOURCE)
    return model_descr

def test_model():
    model_descr = load_model_descr()
    validation_summary = bioimageio.core.test_model(model_descr)
    print(validation_summary.display())

def main():
    model_descr = load_model_descr()
    
    print('Loading input image...')
    input_img = load_and_squeeze_input_image()
    
    print('Reshaping input image...')
    sample, output_index = create_input_sample(input_img, model_descr)
    
    print('Creating prediction pipeline...')
    prediction_pipeline = bioimageio.core.create_prediction_pipeline(
        model_descr
    )
    
    print('Running prediction...')
    prediction_output = prediction_pipeline.predict_sample_without_blocking(
        sample
    )
    
    prediction_mask = get_prediction_mask(
        prediction_output, model_descr, output_index
    )
    
    imshow(input_img, prediction_mask)  
    
if __name__ == '__main__':
    main()
    # main()