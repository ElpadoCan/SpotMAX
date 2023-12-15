import os
import tifffile
import numpy as np
import pandas as pd
import math

import spotmax.nnet.model
import spotmax.pipe

print('Loading images...')
"""Set paths"""
work_dir = os.path.dirname(os.path.abspath(__file__))
examples_dir = os.path.dirname(work_dir)
mitochondria_dir = os.path.join(examples_dir, 'images', 'mitochondria')
IMAGE_PATH = os.path.join(mitochondria_dir, 'ASY15-1_0nM-10_s10_mNeon.tif')
LABELS_PATH = os.path.join(mitochondria_dir, 'ASY15-1_0nM-10_s10_segm.npz')

"""Load images"""
image = tifffile.tifffile.imread(IMAGE_PATH)
lab = np.load(LABELS_PATH)['arr_0']

print('Segmenting spots...')
"""Segment the spots"""

# Initialize spotMAX neural network
nnet_model = spotmax.nnet.model.Model(
    model_type='2D',
    preprocess_across_experiment=False,
    preprocess_across_timepoints=False,
    gaussian_filter_sigma=0,
    remove_hot_pixels=False, 
    PhysicalSizeX=0.073,
    resolution_multiplier_yx=1.0,
    use_gpu=False,
)
nnet_params = {
    'segment': {
        'threshold_value': 0.9,
        'label_components': False
    }
}

# Run all predictions

# Rough estimate for the radii of the spots in pixels. This can be 
# estimated visually
spots_zyx_radii_pxl = (3.5, 5, 5)

spots_segm_result = spotmax.pipe.spots_semantic_segmentation(
    image, 
    lab=lab,
    gauss_sigma=0.75,
    spots_zyx_radii_pxl=spots_zyx_radii_pxl, 
    do_sharpen=True, 
    do_remove_hot_pixels=False,
    lineage_table=None,
    do_aggregate=True,
    use_gpu=False,
    logger_func=print,
    thresholding_method=None,
    keep_input_shape=True,
    nnet_model=nnet_model,
    nnet_params=nnet_params,
    nnet_input_data=None,
    bioimageio_model=None,
    bioimageio_params=None,
    do_preprocess=True,
    do_try_all_thresholds=True,
    return_only_segm=False,
    pre_aggregated=False,
    raw_image=None
)
print('Semantic segmentation done.')


print('Detecting spots...')
"""Detect spots"""
# Proceed to detect spots with 'neural_network' method
SELECTED_METHOD = 'neural_network'
spots_labels = spots_segm_result[SELECTED_METHOD]

df_spots_coords, spots_objs = spotmax.pipe.spot_detection(
    spots_segm_result['input_image'],
    spots_segmantic_segm=spots_labels,
    detection_method='peak_local_max',
    spot_footprint=None,
    lab=lab,
    spots_zyx_radii_pxl=spots_zyx_radii_pxl,
    return_spots_mask=False,
    return_df=True
)

print('Calculating spots features...')
"""Calculate spots features"""
keys, dfs_spots_det, _ = spotmax.pipe.spots_calc_features_and_filter(
    spots_segm_result['input_image'], 
    spots_zyx_radii_pxl,
    df_spots_coords,
    frame_i=0,
    sharp_spots_image=None,
    lab=lab,
    rp=None,
    gop_filtering_thresholds=None,
    delta_tol=None,   
    raw_image=image,
    ref_ch_mask_or_labels=None, 
    ref_ch_img=None,   
    keep_only_spots_in_ref_ch=False,
    min_size_spheroid_mask=None,
    optimise_for_high_spot_density=True,
    dist_transform_spheroid=None,
    get_backgr_from_inside_ref_ch_mask=False,
    show_progress=True,
    verbose=False,
    logger_func=print
)
print('Done')