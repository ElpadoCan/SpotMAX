import os

import numpy as np

from cellacdc.plot import imshow
from cellacdc.load import (
    load_image_data_from_channel, load_segm_file, load_metadata_df
)

from spotmax.nnet.model import Model
from spotmax.transformations import crop_from_segm_data_info

# Initialize paths and channel name
cwd_path = os.path.dirname(os.path.abspath(__file__))
sample_pos_path = os.path.join(cwd_path, 'Input_sample_Positions', 'Position_12')
sample_images_path = os.path.join(sample_pos_path, 'Images')
channel_name = 'mNeon'
end_name_segm_file = 'segm'

# Load image, crop at segm mask, and reshape
print('Loading image...')
df_metadata = load_metadata_df(sample_images_path)
SizeZ = int(df_metadata.at['SizeZ', 'values'])
SizeT = int(df_metadata.at['SizeT', 'values'])

if SizeT > 1:
    raise TypeError('Timelapse data not supported yet.')

image = load_image_data_from_channel(sample_images_path, channel_name)
if end_name_segm_file is not None:
    print('Loading segmentation file...')
    segm_data = load_segm_file(
        sample_images_path, 
        end_name_segm_file=end_name_segm_file
    )
    if SizeZ == 1:
        segm_data = segm_data[np.newaxis]
    elif segm_data.ndim == 2:
        segm_data = np.array([segm_data]*SizeZ)
        
    segm_data = segm_data[np.newaxis]
    
if SizeZ == 1:
    image = image[np.newaxis]
    
input_sample = image[np.newaxis]
if end_name_segm_file is not None:
    segm_slice = crop_from_segm_data_info(
        segm_data, delta_tolerance=(1, 5, 5)
    )[0]
    input_sample = input_sample[segm_slice]

print(input_sample.shape)

imshow(np.squeeze(input_sample))
import pdb; pdb.set_trace()

# Initialize model
print('Initializing model...')
model = Model(
    model_type='2D', 
    preprocess_across_experiment=False,
    preprocess_across_timepoints=False,
    gaussian_filter_sigma=0,
    remove_hot_pixels=True,
    config_yaml_filepath='spotmax/nnet/config.yaml', 
    PhysicalSizeX=0.06725,
    resolution_multiplier_yx=1, 
    use_gpu=True, 
    save_prediction_map=False, 
    verbose=True
)
model.init_inference_params(
    threshold_value=0.9, 
    label_components=False
)

# Run inference
print('Running inference...')
output_sample_mask = model.forward(input_sample)

# Visualize result
imshow(np.squeeze(input_sample), np.squeeze(output_sample_mask))
import pdb; pdb.set_trace()

# Save input and output samples
print('Saving input sample...')
input_sample_filepath = os.path.join(cwd_path, 'input_sample.npy')
np.save(input_sample_filepath, input_sample)

print('Saving output sample...')
output_sample_filepath = os.path.join(cwd_path, 'output_sample_mask.npy')
np.save(output_sample_filepath, output_sample_mask)

print('*'*100)
print(
    'Done. Input and output files saved at the following locations:\n\n'
    f'  * {input_sample_filepath}\n'
    f'  * {output_sample_filepath}\n'
)
print('*'*100)
