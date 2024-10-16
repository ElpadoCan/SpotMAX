from logging import warning
from typing import Union, Literal
from numbers import Number
import warnings

import math

import numpy as np
import pandas as pd

import scipy.stats

import skimage.feature

from . import docs
from . import printl
from . import transformations
from . import filters
from . import _warnings
from . import core

def normalise_by_dist_transform_simple(
        spot_slice_z, dist_transf, backgr_vals_z_spot, debug=False
    ):
    norm_spot_slice_z = spot_slice_z*dist_transf
    backgr_mean = np.mean(backgr_vals_z_spot)
    norm_spot_slice_z[norm_spot_slice_z<backgr_mean] = backgr_mean
    return norm_spot_slice_z

def normalise_by_dist_transform_range(
        spot_slice_z, dist_transf, backgr_vals_z_spot, debug=False
    ):
    """Normalise the distance transform based on the distance from expected 
    value. 

    The idea is that intesities that are too high and far away from the center 
    should be corrected by the distance transform. On the other hand, if a 
    pixel is far but already at background level it doesn't need correction. 

    We do not allow corrected values below background mean, so these values 
    are set to background mean.

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
    backgr_val = np.mean(backgr_vals_z_spot)
    min_dist_transf_nonzero = np.min(dist_transf[np.nonzero(dist_transf)])
    expected_values = (1 + (dist_transf-min_dist_transf_nonzero))*backgr_val
    spot_slice_z_nonzero = spot_slice_z.copy()
    # Ensure that we don't divide by zeros
    spot_slice_z_nonzero[spot_slice_z==0] = 1E-15
    dist_from_expected_perc = (spot_slice_z-expected_values)/spot_slice_z_nonzero
    dist_transf_range = 1 - dist_transf
    dist_transf_correction = np.abs(dist_from_expected_perc*dist_transf_range)
    dist_transf_required = 1-np.sqrt(dist_transf_correction)
    dist_transf_required[dist_transf_required<0] = 0
    norm_spot_slice_z = spot_slice_z*dist_transf_required
    norm_spot_slice_z[norm_spot_slice_z<backgr_val] = backgr_val
    if debug:
        import pdb; pdb.set_trace()
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

def pooled_std_two_samples(sample_1: np.ndarray, sample_2: np.ndarray):
    # See https://en.wikipedia.org/wiki/Cohen%27s_d
    n1 = len(sample_1)
    n2 = len(sample_2)
    s1 = np.std(sample_1, ddof=1)
    s2 = np.std(sample_2, ddof=1)
    v1 = s1**2
    v2 = s2**2
    
    pooled_var = ((n1-1)*v1 + (n2-1)*v2)/(n1+n2-2)
    pooled_std = pooled_var**(0.5)

    return pooled_std

def cohen_effect_size(positive_sample, negative_sample, n_bootstraps=0):
    pooled_std = pooled_std_two_samples(positive_sample, negative_sample)
    
    # positive_std = np.std(positive_sample)
    # negative_std = np.std(negative_sample)
    # pooled_std = np.sqrt((np.square(positive_std)+np.square(negative_std))/2)

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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
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

def get_features_groups(
        category: Literal['spots', 'ref. channel objects']='spots'
    ):
    if category == 'spots':
        return docs.parse_single_spot_features_groups()

    if category == 'ref. channel objects':
        return docs.parse_ref_ch_featurs_groups()

def get_aggr_features_groups():
    return docs.parse_aggr_features_groups()

def aggr_feature_names_to_col_names_mapper():
    return docs.parse_aggr_features_column_names()
            
def single_spot_feature_names_to_col_names_mapper():
    return docs.single_spot_features_column_names()

def feature_names_to_col_names_mapper(
        category: Literal['spots', 'ref. channel objects'] = 'spots'
    ):
    if category == 'spots':
        return single_spot_feature_names_to_col_names_mapper()
    
    if category == 'ref. channel objects':
        return docs.ref_ch_features_column_names()

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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tstat, pvalue = scipy.stats.ttest_ind(arr1, arr2, equal_var=False)
    except FloatingPointError as e:
        logger_func(
            '[WARNING]: FloatingPointError while performing t-test.'
        )
        tstat, pvalue = np.nan, np.nan
    df.at[idx, f'{name}_ttest_tstat'] = tstat
    df.at[idx, f'{name}_ttest_pvalue'] = pvalue

def add_distribution_metrics(
        arr, df, idx, col_name='*name', add_bkgr_corrected_metrics=False, 
        logger_warning_report=None, logger_func=print, iter_idx=0
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
    
    try:
        bkgr_value = df.at[idx, bkgr_col]
    except Exception as err:
        return
    
    bkgr_col_z = f'background_median_z_slice_{bkgr_id}image'
    bkgr_value_z = df.at[idx, bkgr_col_z]
    
    volume = df.at[idx, 'spot_mask_volume_voxel']
    
    mean_corr = mean_foregr_value - bkgr_value
    mean_corr_col = col_name.replace('*name', 'backgr_corrected_mean')
    df.at[idx, mean_corr_col] = mean_corr
    
    spot_center_intens_col = f'spot_center_{bkgr_id}intensity' 
    spot_center_intens = df.at[idx, spot_center_intens_col]
    spot_bkgr_ratio_col = f'spot_center_{bkgr_id}intens_to_backgr_median_ratio' 
    if bkgr_value == 0:
        if iter_idx == 0:
            _warnings.warn_background_value_is_zero(
                logger_func, logger_warning_report=logger_warning_report
            )
        spot_bkgr_ratio_value = np.nan
    else:
        spot_bkgr_ratio_value = spot_center_intens/bkgr_value
        
    df.at[idx, spot_bkgr_ratio_col] = spot_bkgr_ratio_value
    
    mean_corr_z = mean_foregr_value - bkgr_value_z
    mean_corr_col_z = col_name.replace('*name', 'backgr_z_slice_corrected_mean')
    df.at[idx, mean_corr_col_z] = mean_corr_z
    
    spot_bkgr_z_ratio_col = (
        f'spot_center_{bkgr_id}intens_to_backgr_z_slice_median_ratio'
    ) 
    if bkgr_value_z == 0:
        if iter_idx == 0:
            _warnings.warn_background_value_is_zero(
                logger_func, logger_warning_report=logger_warning_report
            )
        spot_bkgr_z_ratio_value = np.nan
    else:
        spot_bkgr_z_ratio_value = spot_center_intens/bkgr_value_z
    df.at[idx, spot_bkgr_z_ratio_col] = spot_bkgr_z_ratio_value
    
    sum_corr = mean_corr*volume
    sum_corr_col = col_name.replace('*name', 'backgr_corrected_sum')
    df.at[idx, sum_corr_col] = sum_corr
    
    sum_corr_z = mean_corr_z*volume
    sum_corr_col_z = col_name.replace('*name', 'backgr_z_slice_corrected_sum')
    df.at[idx, sum_corr_col_z] = sum_corr_z
     
def add_effect_sizes(
        pos_arr, neg_arr, df, idx, name='spot_vs_backgr', 
        debug=False, logger_warning_report=None, 
        logger_func=print
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
        print('='*100)
        print(f'Name = {name}:\n')
        for eff_size_name, values in info.items():
            eff_size, pos_mean, negative_mean, negative_std = values
            print(f'  - Effect size {eff_size_name} = {eff_size}')
            print(f'  - Positive mean = {pos_mean}')
            print(f'  - Negative mean = {negative_mean}')
            print(f'  - Negative std = {negative_std}')
            print('-'*100)
        print('='*100)
        import pdb; pdb.set_trace()

def add_spot_localization_metrics(
        df, spot_id, zyx_center, obj_centroid, voxel_size=(1, 1, 1), 
        debug=False, logger_warning_report=None, logger_func=print
    ):
    dist_pxl = math.dist(zyx_center, obj_centroid)
    df.at[spot_id, 'spot_distance_from_obj_centroid_pixels'] = dist_pxl
    
    p_um = [c/ps for c, ps in zip(zyx_center, voxel_size)]
    q_um = [c/ps for c, ps in zip(obj_centroid, voxel_size)]
    
    dist_um = math.dist(p_um, q_um)
    df.at[spot_id, 'spot_distance_from_obj_centroid_um'] = dist_um

def add_missing_cells_to_df_agg_from_segm(df_agg, segm_data):
    missing_rows = [df_agg]
    for frame_i, lab in enumerate(segm_data):
        rp = skimage.measure.regionprops(lab)
        IDs = [obj.label for obj in rp]
        df_frame = df_agg.loc[frame_i]
        for ID in IDs:
            if ID in df_frame.index:
                continue
            
            new_idx = (frame_i, ID)
            missing_row = get_df_row_empty_vals(
                df_agg, integer_default=0, index=new_idx
            )
            missing_rows.append(missing_row)
    
    if len(missing_rows)>1:
        df_agg = pd.concat(missing_row)
    
    return df_agg.sort_index()

def add_missing_cols_from_src_df_agg(df_agg, src_df_agg):
    common_index = df_agg.index.intersection(src_df_agg.index)
    for col in src_df_agg.columns:
        if col in df_agg.columns:
            continue
        
        df_agg.loc[common_index, col] = src_df_agg.loc[common_index, col]
    
    return df_agg

def add_columns_from_acdc_output_file(df_agg, acdc_df):
    if acdc_df is None:
        return df_agg
    
    common_cols = df_agg.columns.intersection(acdc_df.columns)
    common_index = df_agg.index.intersection(acdc_df.index)
    
    df_agg.loc[common_index, common_cols] = (
        acdc_df.loc[common_index, common_cols]
    )
    return acdc_df

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

def find_local_peaks(
        image, min_distance=1, footprint=None, labels=None, debug=False
    ):
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
    
    if labels is not None and not np.any(labels):
        # No point in searching for spots, labels are empty
        return np.zeros((0, 2), dtype=np.int32)
    
    peaks_coords = skimage.feature.peak_local_max(
        image, 
        footprint=footprint, 
        labels=labels.astype('int32'),
        p_norm=2
    )
    intensities = image[tuple(peaks_coords.transpose())]
    valid_peaks_coords = filters.filter_valid_points_min_distance(
        peaks_coords, min_distance, intensities=intensities, 
        debug=debug
    )
    valid_peaks_coords = transformations.reshape_spots_coords_to_3D(
        valid_peaks_coords
    )
    
    if debug:
        from spotmax import _debug
        _debug.find_local_peaks(
            image, labels, peaks_coords, valid_peaks_coords, footprint
        )
        import pdb; pdb.set_trace()
    
    return valid_peaks_coords

def add_custom_combined_measurements(df, logger_func=print, **features_exprs):
    """Add custom combined measurement pandas.eval

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with standard features.
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    features_exprs : dict, options
        Dictionary of {column_name: expr} where `column_name = expr` will 
        be evaluated with pandas.eval
    """    
    for colname, expression in features_exprs.items():
        expr_to_eval = f'{colname} = {expression}'
        try:
            df = df.eval(expr_to_eval)
        except Exception as err:
            print('\n')
            logger_func(
                f'[WARNING]: could not add feature `{expr_to_eval}`. '
                'Might retry later. Skipping it for now.'
            )
    return df

def _init_df_ref_ch(ref_ch_rp):
    nrows = len(ref_ch_rp)
    index = [sub_obj.label for sub_obj in ref_ch_rp]
    col_names = list(docs.ref_ch_features_column_names().values())
    ncols = len(col_names)
    data = np.zeros((nrows, ncols))
    df = pd.DataFrame(data=data, columns=col_names, index=index)
    df.index.name = 'sub_obj_id'
    return df

def df_spots_to_aggregated(df_spots):
    aggregate_spots_feature_func = get_aggregating_spots_feature_func()
    
    name_to_func_mapper = {
        name:(col, func) for name, (col, func, _) 
        in aggregate_spots_feature_func.items() 
        if col in df_spots.columns
    }
    
    df_agg = (
        df_spots
        .reset_index()
        .groupby(['frame_i', 'Cell_ID'])
        .agg(**name_to_func_mapper)
    )
    return df_agg

def get_df_row_empty_vals(
        df, 
        as_df=True, 
        index=None, 
        integer_default=-1, 
        float_default=np.nan, 
        object_default=''
    ):
    empty_vals = []
    dtypes = []
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            val = integer_default
        elif pd.api.types.is_float_dtype(df[col]):
            val = float_default
        else:
            val = object_default
        if as_df:
            val = pd.Series(data=[val], name=col, dtype=df[col].dtype)
        empty_vals.append(val)
    
    if as_df:
        index_names = df.index.names
        empty_vals = pd.concat(empty_vals, axis=1)
        if index is not None:
            for name, val in zip(index_names, index):
                empty_vals[name] = val
            empty_vals = empty_vals.set_index(index_names)
        
    return empty_vals

def compute_spots_zyx_radii_from_params(params, return_voxel_size=False):
    metadata_params = params['METADATA']
    emission_wavelen = metadata_params['emWavelen']['loadedVal']
    num_aperture = metadata_params['numAperture']['loadedVal']
    physical_size_x = metadata_params['numAperture']['pixelWidth']
    physical_size_y = metadata_params['numAperture']['pixelHeight']
    physical_size_z = metadata_params['numAperture']['voxelDepth']
    z_resolution_limit_um = metadata_params['numAperture']['zResolutionLimit']
    yx_resolution_multiplier = (
        metadata_params['numAperture']['yxResolLimitMultiplier']
    )
    
    spots_zyx_radii_pixel, spots_zyx_radii_um = core.calcMinSpotSize(
        emission_wavelen, num_aperture, physical_size_x, 
        physical_size_y, physical_size_z, z_resolution_limit_um, 
        yx_resolution_multiplier
    )
    if return_voxel_size:
        zyx_voxel_size = (physical_size_z, physical_size_y, physical_size_x)
        return spots_zyx_radii_pixel, spots_zyx_radii_um, zyx_voxel_size
    else:
        return spots_zyx_radii_pixel, spots_zyx_radii_um

def nearest_point(points, point_idx):
    point = points[point_idx]
    diff = np.subtract(points, point)
    dist = np.linalg.norm(diff, axis=1)
    dist[point_idx] = np.inf
    min_idx = dist.argmin()
    nearest = points[min_idx]
    return nearest

def calc_distance_matrix(points, spacing=None, other_points=None):
    if other_points is None:
        other_points = points
        
    diff = points[:, np.newaxis] - other_points
    if spacing is not None:
        diff = diff/spacing
    dist_matrix = np.linalg.norm(diff, axis=2)
    return dist_matrix

def get_all_pairs_within_distance(
        points: np.ndarray, max_distance: float
    ):
    dist_matrix = calc_distance_matrix(points, spacing=max_distance)
    ii, jj = np.nonzero(dist_matrix <= 1)
    
    nondiag_mask = ii != jj
    ii = ii[nondiag_mask]
    jj = jj[nondiag_mask]
    
    paired_points = [
        np.row_stack((points[i], points[j])) for i, j in zip(ii, jj) if i<j
    ]
    return paired_points

def kurtosis_from_hist(bin_centers, counts):
    # see here https://stackoverflow.com/questions/54414462/how-can-i-calculate-the-kurtosis-of-already-binned-data
    total = np.sum(counts)
    mean = np.sum(counts * bin_centers) / total
    variance = np.sum(counts * (bin_centers - mean)**2) / total
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kurtosis = (
            np.sum(counts * (bin_centers - mean)**4) 
            / (variance**2 * total)
        )
    return kurtosis