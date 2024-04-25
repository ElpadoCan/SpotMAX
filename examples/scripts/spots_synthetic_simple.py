import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from cellacdc.plot import imshow
import spotmax as sm

# Generate synthetic data
spots_radii = np.array([4, 6, 6])
shape = (25, 256, 256)
spots_img, spots_gt_mask, spots_coords = sm.data.synthetic_spots(
    num_spots=20,
    shape=(25, 256, 256), 
    spots_radii=spots_radii, 
    noise_scale=0.05,
    noise_shape=0.03, 
    rng_seed=11
)

# Visualize the generated data
imshow(
    spots_img, spots_gt_mask, 
    points_coords=spots_coords,
    axis_titles=['Spots image', 'Spots mask']
)

# Segment the spots
print('Segmenting the spots...')
result = sm.pipe.spots_semantic_segmentation(
    spots_img, do_try_all_thresholds=True
)

# Visualize the result of segmentation
imshow(*result.values(), axis_titles=result.keys())

# Detect the spots
print('Detecting the spots...')
spots_pred_mask = result['threshold_otsu']
spots_detect_img = result['input_image']
spots_zyx_radii_pxl = spots_radii

df_spots_coords, _ = sm.pipe.spot_detection(
    spots_detect_img, 
    spots_segmantic_segm=spots_pred_mask, 
    spots_zyx_radii_pxl=spots_zyx_radii_pxl, 
    return_df=True
)
print('-'*100)
print(f'Number of detected spots = {len(df_spots_coords)}')
print('-'*100)

# Quantify the spots
print('Quantifying the spots...')
keys, dfs_spots_det, dfs_spots_gop = sm.pipe.spots_calc_features_and_filter(
    spots_img, spots_zyx_radii_pxl, df_spots_coords, 
    sharp_spots_image=spots_detect_img, 
    optimise_for_high_spot_density=True, 
    show_progress=False, 
    verbose=False
)
df_spots = pd.concat(
    dfs_spots_gop, keys=keys, names=['frame_i', 'Cell_ID']).loc[0]
print('')
columns = [
    'z', 'y', 'x', 'spot_center_raw_intensity', 
    'spot_vs_backgr_effect_size_glass'
]
print(df_spots[columns].head(10))

# Quantify spots with spotFIT
print('Runnning spotFIT...')
spotfit_kernel = sm.core.SpotFIT()
df_spotfit, _ = sm.pipe.spotfit(
    spotfit_kernel, spots_img, df_spots, 
    spots_zyx_radii_pxl=spots_zyx_radii_pxl, 
    return_df=True, 
    verbose=False,
    show_progress=False
)
print('')
columns = [
    'z', 'y', 'x', 'z_fit', 'y_fit', 'x_fit', 
    'sigma_z_fit', 'sigma_y_fit', 'sigma_x_fit', 
    'total_integral_fit', 'RMSE_fit'
]
print(df_spotfit[columns].head(10))

# Visualize spotfit results
spotfit_img, spotfit_mask, spotfit_labels = (
    sm.utils.get_spotfit_image(df_spotfit.loc[0], shape)
)
imshow(
    spots_img, spotfit_img, spotfit_labels,
    axis_titles=['Spots image', 'SpotFIT image', 'SpotFIT labels']
)