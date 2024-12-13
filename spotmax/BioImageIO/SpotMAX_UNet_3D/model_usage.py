import os

import numpy as np

from cellacdc.plot import imshow
from spotmax.nnet.model import Model

DO_BATCH = False

def run_spotmax_AI(input_sample):
    # Initialize model
    print('Initializing model...')
    model = Model(
        model_type='3D', 
        preprocess_across_experiment=False,
        preprocess_across_timepoints=False,
        gaussian_filter_sigma=0,
        remove_hot_pixels=False,
        config_yaml_filepath='spotmax/nnet/config.yaml', 
        PhysicalSizeX=0.06725,
        resolution_multiplier_yx=1, 
        use_gpu=True, 
        save_prediction_map=False, 
        verbose=True
    )
    model.init_inference_params(
        threshold_value=0.7, 
        label_components=False,
    )
    import pdb; pdb.set_trace()

    # Run inference on batch of images
    if DO_BATCH:
        print('Running inference (batch processing)...')
        output_sample_mask = model.forward(input_sample)
        output_mask = output_sample_mask[0]
        output_prediction = np.zeros_like(output_mask)
    else:
        print('Running inference (single image)...')
        input_img = input_sample[0]
        # Run inference on single image
        output_mask, output_prediction = model.segment(
            input_img, 
            threshold_value=0.9, 
            label_components=False,
            return_pred=True,
        )

    # Visualize result
    input_img = input_sample[0]

    imshow(
        input_img, output_mask, output_prediction, 
        axis_titles=(
            'Input image', 
            'Output mask', 
            'Output prediction'
        )
    )

if __name__ == '__main__':
    # Load test image
    print('Loading test image...')

    cwd_path = os.path.dirname(os.path.abspath(__file__))
    input_sample_filepath = os.path.join(cwd_path, 'input_sample.npy')
    input_sample = np.load(input_sample_filepath)
    
    run_spotmax_AI(input_sample)