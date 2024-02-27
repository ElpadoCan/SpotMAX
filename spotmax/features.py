from typing import Union
from numbers import Number

import numpy as np
import pandas as pd

import scipy.stats

import skimage.feature

from . import docs
from . import printl
from . import transformations
from . import filters

def normalise_by_dist_transform_simple(
        spot_slice_z, dist_transf, backgr_vals_z_spot
    ):
    norm_spot_slice_z = spot_slice_z*dist_transf
    backgr_median = np.median(backgr_vals_z_spot)
    norm_spot_slice_z[norm_spot_slice_z<backgr_median] = backgr_median
    return norm_spot_slice_z

def normalise_by_dist_transform_range(
        spot_slice_z, dist_transf, backgr_vals_z_spot
    ):
    """Normalise the distance transform based on the distance from expected 
    value. 

    The idea is that intesities that are too high and far away from the center 
    should be corrected by the distance transform. On the other hand, if a 
    pixel is far but already at background level it doesn't need correction. 

    We do not allow corrected values below background median, so these values 
    are set to background median.

    Parameters
    ----------
    spot_slice_z : np.ndarray
        2D spot intensities image. This is the z-slice at spot's center
    dist_transf : np.ndarray, same shape as `spot_slice_z`
        2D distance transform image. Must be 1 in the center and <1 elsewhere.
    backgr_vals_z_spot : np.ndarray
        Bacgrkound values
    
    Returns
    -------
    norm_spot_slice_z : np.ndarray, same shape as `spot_slice_z`
        Normalised `spot_slice_z`.
    
    Examples
    --------
    >>> import numpy as np
    >>> from spotmax import features
    >>> backgr_vals_z_spot = np.array([0.4, 0.4, 0.4, 0.4])
    >>> dist_transf = np.array([0.25, 0.5, 0.75, 1.0])
    >>> spot_slice_z = np.array([2.5,0.5,3.4,0.7])
    >>> norm_spot_slice_z_range = features.normalise_by_dist_transform_range(
    ...    spot_slice_z, dist_transf, backgr_vals_z_spot)
    >>> norm_spot_slice_z_range
    [0.51568652 0.5        1.85727514 0.7       ]
    """    
    backgr_median = np.median(backgr_vals_z_spot)
    expected_values = (1 + (dist_transf-dist_transf.min()))*backgr_median
    spot_slice_z_nonzero = spot_slice_z.copy()
    # Ensure that we don't divide by zeros
    spot_slice_z_nonzero[spot_slice_z==0] = 1E-15
    dist_from_expected_perc = (spot_slice_z-expected_values)/spot_slice_z_nonzero
    dist_transf_range = 1 - dist_transf
    dist_transf_correction = np.abs(dist_from_expected_perc*dist_transf_range)
    dist_tranf_required = 1-np.sqrt(dist_transf_correction)
    dist_tranf_required[dist_tranf_required<0] = 0
    norm_spot_slice_z = spot_slice_z*dist_tranf_required
    norm_spot_slice_z[norm_spot_slice_z<backgr_median] = backgr_median
    return norm_spot_slice_z
    

def calc_pooled_std(s1, s2, axis=0):
    n1 = s1.shape[axis]
    n2 = s2.shape[axis]

    std1 = np.std(s1, axis=axis)
    std2 = np.std(s2)
    pooled_std = np.sqrt(
        ((n1-1)*(std1**2)+(n2-1)*(std2**2))/(n1+n2-2)
    )
    return pooled_std

def glass_effect_size(positive_sample, negative_sample, n_bootstraps=0):
    negative_std = np.std(negative_sample)

    positive_mean = np.mean(positive_sample)
    negative_mean = np.mean(negative_sample)

    eff_size = (positive_mean-negative_mean)/negative_std

    return eff_size, negative_mean, negative_std

def cohen_effect_size(positive_sample, negative_sample, n_bootstraps=0):
    pooled_std = np.std(np.concatenate((positive_sample, negative_sample)))

    positive_mean = np.mean(positive_sample)
    negative_mean = np.mean(negative_sample)

    eff_size = (positive_mean-negative_mean)/pooled_std
    return eff_size, negative_mean, pooled_std

def hedge_effect_size(positive_sample, negative_sample, n_bootstraps=0):
    n1 = len(positive_sample)
    n2 = len(negative_sample)
    correction_factor = 1 - (3/((4*(n1-n2))-9))
    eff_size_cohen, negative_mean, pooled_std = cohen_effect_size(
        positive_sample, negative_sample
    )
    eff_size = eff_size_cohen*correction_factor
    return eff_size, negative_mean, pooled_std

def _try_combine_pvalues(*args, **kwargs):
    try:
        result = scipy.stats.combine_pvalues(*args, **kwargs)
        try:
            stat, pvalue = result
        except Exception as e:
            pvalue = result.pvalue
        return pvalue
    except Exception as e:
        return 0.0

def get_aggregating_spots_feature_func():
    func = {
        'num_spots': ('x', 'count', 0),
        'num_spots_inside_ref_ch': ('is_spot_inside_ref_ch', 'sum', 0),
        'sum_foregr_integral_fit': ('foreground_integral_fit', 'sum', np.nan),
        'sum_tot_integral_fit': ('total_integral_fit', 'sum', np.nan),
        'mean_sigma_z_fit': ('sigma_z_fit', 'mean', np.nan),
        'mean_sigma_y_fit': ('sigma_y_fit', 'mean', np.nan),
        'mean_sigma_x_fit': ('sigma_x_fit', 'mean', np.nan),
        'std_sigma_z_fit': ('sigma_z_fit', 'std', np.nan),
        'std_sigma_y_fit': ('sigma_y_fit', 'std', np.nan),
        'std_sigma_x_fit': ('sigma_x_fit', 'std', np.nan),
        'sum_A_fit_fit': ('A_fit', 'sum', np.nan),
        'mean_B_fit_fit': ('B_fit', 'mean', np.nan),
        'solution_found_fit': ('solution_found_fit', 'mean', np.nan),
        'mean_reduced_chisq_fit': ('reduced_chisq_fit', 'mean', np.nan),
        'combined_p_chisq_fit': ('p_chisq_fit', _try_combine_pvalues, np.nan),
        'mean_RMSE_fit': ('RMSE_fit', 'mean', np.nan),
        'mean_NRMSE_fit': ('NRMSE_fit', 'mean', np.nan),
        'mean_F_NRMSE_fit': ('F_NRMSE_fit', 'mean', np.nan),
        'mean_ks_fit': ('KS_stat_fit', 'mean', np.nan),
        'combined_p_ks_fit': ('p_KS_fit', 'mean', np.nan),
        'mean_ks_null_fit': ('null_ks_test_fit', 'mean', np.nan),
        'mean_chisq_null_fit': ('null_chisq_test_fit', 'mean', np.nan),
        'mean_QC_passed_fit': ('QC_passed_fit', 'mean', np.nan)
    }
    return func
    

def _try_metric_func(func, *args):
    try:
        val = func(*args)
    except Exception as e:
        val = np.nan
    return val

def _try_quantile(arr, q):
    try:
        val = np.quantile(arr, q=q)
    except Exception as e:
        val = np.nan
    return val

def get_distribution_metrics_func():
    metrics_func = {
        'mean': lambda arr: _try_metric_func(np.mean, arr),
        'sum': lambda arr: _try_metric_func(np.sum, arr),
        'median': lambda arr: _try_metric_func(np.median, arr),
        'min': lambda arr: _try_metric_func(np.min, arr),
        'max': lambda arr: _try_metric_func(np.max, arr),
        'q25': lambda arr: _try_quantile(arr, 0.25),
        'q75': lambda arr: _try_quantile(arr, 0.75),
        'q05': lambda arr: _try_quantile(arr, 0.05),
        'q95': lambda arr: _try_quantile(arr, 0.95),
        'std': lambda arr: _try_metric_func(np.std, arr),
    }
    return metrics_func

def get_effect_size_func():
    effect_size_func = {
        'cohen': cohen_effect_size,
        'glass': glass_effect_size,
        'hedge': hedge_effect_size
    }
    return effect_size_func

def get_features_groups():
    return docs.parse_single_spot_features_groups()

def get_aggr_features_groups():
    return docs.parse_aggr_features_groups()

def aggr_feature_names_to_col_names_mapper():
    return docs.parse_aggr_features_column_names()
            
def single_spot_feature_names_to_col_names_mapper():
    return docs.single_spot_features_column_names()

def feature_names_to_col_names_mapper():
    return single_spot_feature_names_to_col_names_mapper()

def true_positive_feauture_inequality_direction_mapper():
    mapper = {}
    for group_name, feature_names in get_features_groups().items():
        if group_name.find('p-value') != -1:
            direction = 'max'
        else:
            direction = 'min'
        for feature_name in feature_names:
            mapper[f'{group_name}, {feature_name}'] = direction
    return mapper

def add_consecutive_spots_distance(df, zyx_voxel_size, suffix=''):
    coords_colnames = ['z', 'y', 'x']
    if suffix:
        coords_colnames = [f'{col}{suffix}' for col in coords_colnames]
    df_coords = df[coords_colnames]
    df_coords_diff = df_coords.rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0])
    df[f'consecutive_spots_distance{suffix}_voxel'] = np.linalg.norm(
        df_coords_diff.values, axis=1
    )
    df_coords_diff_physical_units = df_coords_diff*zyx_voxel_size
    df[f'consecutive_spots_distance{suffix}_um'] = np.linalg.norm(
        df_coords_diff_physical_units.values, axis=1
    )

def add_ttest_values(
        arr1: np.ndarray, arr2: np.ndarray, df: pd.DataFrame, 
        idx: Union[int, pd.Index], name: str='spot_vs_backgr',
        logger_func=printl
    ):
    try:
        tstat, pvalue = scipy.stats.ttest_ind(arr1, arr2, equal_var=False)
    except FloatingPointError as e:
        logger_func(
            '[WARNING]: FloatingPointError while performing t-test.'
        )
        tstat, pvalue = np.nan, np.nan
    df.at[idx, f'{name}_ttest_tstat'] = tstat
    df.at[idx, f'{name}_ttest_pvalue'] = pvalue

def add_distribution_metrics(
        arr, df, idx, col_name='*name', add_bkgr_corrected_metrics=False
    ):
    distribution_metrics_func = get_distribution_metrics_func()
    for name, func in distribution_metrics_func.items():
        _col_name = col_name.replace('*name', name)
        df.at[idx, _col_name] = func(arr)
    
    if not add_bkgr_corrected_metrics:
        return
    
    mean_col = col_name.replace('*name', 'mean')
    mean_foregr_value = df.at[idx, mean_col]
    
    name_idx = col_name.find("*name")
    bkgr_id = col_name[:name_idx].replace('spot_', '')
    bkgr_col = f'background_median_{bkgr_id}image'
    bkgr_value = df.at[idx, bkgr_col]
    bkgr_col_z = f'background_median_z_slice_{bkgr_id}image'
    bkgr_value_z = df.at[idx, bkgr_col_z]
    
    volume = df.at[idx, 'spot_mask_volume_voxel']
    
    mean_corr = mean_foregr_value - bkgr_value
    mean_corr_col = col_name.replace('*name', 'backgr_corrected_mean')
    df.at[idx, mean_corr_col] = mean_corr
    
    mean_corr_z = mean_foregr_value - bkgr_value_z
    mean_corr_col_z = col_name.replace('*name', 'backgr_z_slice_corrected_mean')
    df.at[idx, mean_corr_col_z] = mean_corr_z
    
    sum_corr = mean_corr*volume
    sum_corr_col = col_name.replace('*name', 'backgr_corrected_sum')
    df.at[idx, sum_corr_col] = sum_corr
    
    sum_corr_z = mean_corr_z*volume
    sum_corr_col_z = col_name.replace('*name', 'backgr_z_slice_corrected_sum')
    df.at[idx, sum_corr_col_z] = sum_corr_z
     
def add_effect_sizes(
        pos_arr, neg_arr, df, idx, name='spot_vs_backgr', 
        debug=False
    ):
    effect_size_func = get_effect_size_func()
    negative_name = name[8:]
    info = {}
    for eff_size_name, func in effect_size_func.items():
        result = _try_metric_func(func, pos_arr, neg_arr)
        if result is not np.nan:
            eff_size, negative_mean, negative_std = result
        else:
            eff_size, negative_mean, negative_std = np.nan, np.nan, np.nan
        col_name = f'{name}_effect_size_{eff_size_name}'
        df.at[idx, col_name] = eff_size
        negative_mean_colname = (
            f'{negative_name}_effect_size_{eff_size_name}_negative_mean'
        )
        df.at[idx, negative_mean_colname] = negative_mean
        negative_std_colname = (
            f'{negative_name}_effect_size_{eff_size_name}_negative_std'
        )
        df.at[idx, negative_std_colname] = negative_std
        if debug:
            info[eff_size_name] = (
                eff_size, np.mean(pos_arr), negative_mean, negative_std
            )
    if debug:
        print('')
        for eff_size_name, values in info.items():
            eff_size, pos_mean, negative_mean, negative_std = values
            print(f'Effect size {eff_size_name} = {eff_size}')
            print(f'Positive mean = {pos_mean}')
            print(f'Negative mean = {negative_mean}')
            print(f'Negative std = {negative_std}')
            print('-'*60)
        import pdb; pdb.set_trace()

def get_normalised_spot_ref_ch_intensities(
        normalised_spots_img_obj, normalised_ref_ch_img_obj,
        spheroid_mask, slice_global_to_local
    ):
    norm_spot_slice = (normalised_spots_img_obj[slice_global_to_local])
    norm_spot_slice_dt = norm_spot_slice
    norm_spot_intensities = norm_spot_slice_dt[spheroid_mask]

    norm_ref_ch_slice = (normalised_ref_ch_img_obj[slice_global_to_local])
    norm_ref_ch_slice_dt = norm_ref_ch_slice
    norm_ref_ch_intensities = norm_ref_ch_slice_dt[spheroid_mask]

    return norm_spot_intensities, norm_ref_ch_intensities

def add_additional_spotfit_features(df_spotfit):
    df_spotfit['Q_factor_yx'] = df_spotfit['A_fit']/df_spotfit['sigma_yx_mean_fit']
    df_spotfit['Q_factor_z'] = df_spotfit['A_fit']/df_spotfit['sigma_z_fit']
    return df_spotfit

def find_local_peaks(image, min_distance=1, footprint=None, labels=None):
    """Find local peaks in intensity image

    Parameters
    ----------
    image : (Y, X) or (Z, Y, X) numpy.ndarray
        Grayscale image where to detect the peaks. It can be 2D or 3D.
    min_distance : int or tuple of floats (one per axis of `image`), optional
        The minimal allowed distance separating peaks. To find the maximum 
        number of peaks, use min_distance=1. Pass a tuple of floats with one 
        value per axis of the image if you need different minimum distances 
        per axis of the image. This will result in only the brightest peak 
        per ellipsoid with `radii=min_distance` centered at peak being 
        returned. Default is 1
    footprint : numpy.ndarray of bools, optional
        If provided, footprint == 1 represents the local region within which 
        to search for peaks at every point in image. Default is None
    labels : numpy.ndarray of ints, optional
        If provided, each unique region labels == value represents a unique 
        region to search for peaks. Zero is reserved for background. 
        Default is None

    Returns
    -------
    (N, 2) or (N, 3) np.ndarray
        The coordinates of the peaks. This is a numpy array with `N` number of 
        rows, where `N` is the number of detected peaks, and 2 or 3 columns 
        with order (y, x) or (z, y, x) for 2D or 3D data, respectively.
    
    Notes
    -----
    This function uses `skimage.feature.peak_local_max` for the first step of 
    the detection. Since `footprint` is not 100% reliable in filtering 
    peaks that are at a minimum distance = `min_distance`, we perform a 
    second step where we ensure that only the brightest peak per ellipsoid is 
    returned.    
    
    See also
    --------
    `skimage.feature.peak_local_max <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max>`__
    """    
    if isinstance(min_distance, Number):
        min_distance = [min_distance]*image.ndim
    
    if footprint is None:
        zyx_radii_pxl = [val/2 for val in min_distance]
        footprint = transformations.get_local_spheroid_mask(
            zyx_radii_pxl
        )
    
    peaks_coords = skimage.feature.peak_local_max(
        image, 
        footprint=footprint, 
        labels=labels.astype('int32')
    )
    intensities = image[tuple(peaks_coords.transpose())]
    valid_peaks_coords = filters.filter_valid_points_min_distance(
        peaks_coords, min_distance, intensities=intensities
    )
    valid_peaks_coords = transformations.reshape_spots_coords_to_3D(
        valid_peaks_coords
    )
    return peaks_coords