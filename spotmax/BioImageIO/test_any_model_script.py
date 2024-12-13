import os

import numpy as np

import bioimageio.core
from bioimageio.spec.model import AnyModelDescr

from cellacdc.plot import imshow
from spotmax import spotmax_path

MODEL_SOURCE = os.path.join(
    spotmax_path, 'BioImageIO', 'SpotMAX_UNet_2D', 'SpotMAX_AI_2D.zip'
)
INPUT_IMAGE_PATH = os.path.join(
    spotmax_path, 'BioImageIO', 'SpotMAX_UNet_2D', 'input_sample.npy'
)

def load_and_squeeze_input_image():
    if INPUT_IMAGE_PATH.endswith('.npy'):
        input_img = np.load(INPUT_IMAGE_PATH)
    elif INPUT_IMAGE_PATH.endswith('.tif'):
        import skimage.io
        skimage.io.imread(INPUT_IMAGE_PATH)
    elif INPUT_IMAGE_PATH.endswith('.npz'):
        input_img = np.load(INPUT_IMAGE_PATH)['arr_0']
    
    return np.squeeze(input_img)

def reshape_input_image(input_image: np.ndarray, model_descr: AnyModelDescr):
    axes = model_descr.inputs[0].axes
    space_axis_ids = {'z', 'y', 'x'}
    for axis in axes:
        if axis.id == 'z' and input_image.ndim == 2:
            input_image = input_image[np.newaxis]
    
    for axis in axes:
        if axis.id in space_axis_ids:
            continue
        
        input_image = input_image[np.newaxis]
    
    return input_image

def main():
    print(f'Loading model description from "{MODEL_SOURCE}"...')
    model_descr = bioimageio.core.load_model_description(MODEL_SOURCE)
    
    print('Loading input image...')
    input_img = load_and_squeeze_input_image()
    
    print('Reshaping input image...')
    input_img = reshape_input_image(input_img, model_descr)
    
    print('Creating prediction pipeline...')
    prediction_pipeline = bioimageio.core.create_prediction_pipeline(
        model_descr
    )
    
    import pdb; pdb.set_trace()
    
    print('Running prediction...')
    predict_output = bioimageio.core.predict(
        model=model_descr, 
        inputs=input_img
    )
    import pdb; pdb.set_trace()
    
    
    

if __name__ == '__main__':
    main()