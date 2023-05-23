import numpy as np
import scipy.stats

from . import printl

def normalise_by_dist_transform_simple(
        spot_slice_z, dist_transf, bakgr_vals_z_spots
    ):
    norm_spot_slice_z = spot_slice_z*dist_transf
    backgr_median = np.median(bakgr_vals_z_spots)
    norm_spot_slice_z[norm_spot_slice_z<backgr_median] = backgr_median
    return norm_spot_slice_z

def normalise_by_dist_transform_range(
        spot_slice_z, dist_transf, bakgr_vals_z_spots
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
    bakgr_vals_z_spots : np.ndarray
        Bacgrkound values
    
    Returns
    -------
    norm_spot_slice_z : np.ndarray, same shape as `spot_slice_z`
        Normalised `spot_slice_z`.
    
    Examples
    --------
    >>> import numpy as np
    >>> from spotmax import features
    >>> bakgr_vals_z_spots = np.array([0.4, 0.4, 0.4, 0.4])
    >>> dist_transf = np.array([0.25, 0.5, 0.75, 1.0])
    >>> spot_slice_z = np.array([2.5,0.5,3.4,0.7])
    >>> norm_spot_slice_z_range = features.normalise_by_dist_transform_range(
    ...    spot_slice_z, dist_transf, bakgr_vals_z_spots)
    >>> norm_spot_slice_z_range
    [0.51568652 0.5        1.85727514 0.7       ]
    """    
    backgr_median = np.median(bakgr_vals_z_spots)
    expected_values = (1 + (dist_transf-dist_transf.min()))*backgr_median
    dist_from_expected_perc = (spot_slice_z-expected_values)/spot_slice_z
    dist_transf_range = 1 - dist_transf
    dist_transf_correction = np.abs(dist_from_expected_perc*dist_transf_range)
    dist_tranf_required = 1-np.sqrt(dist_transf_correction)
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

    return eff_size

def cohen_effect_size(positive_sample, negative_sample, n_bootstraps=0):
    pooled_std = np.std(np.concatenate((positive_sample, negative_sample)))

    positive_mean = np.mean(positive_sample)
    negative_mean = np.mean(negative_sample)

    eff_size = (positive_mean-negative_mean)/pooled_std
    return eff_size

def hedge_effect_size(positive_sample, negative_sample, n_bootstraps=0):
    n1 = len(positive_sample)
    n2 = len(negative_sample)
    correction_factor = 1 - (3/((4*(n1-n2))-9))
    return cohen_effect_size(positive_sample, negative_sample)*correction_factor

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
        'num_spots_inside_ref_ch': ('is_spot_inside_ref_ch', 'sum', 0),
        'num_spots': ('x', 'count', 0),
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
        'q95': lambda arr: _try_quantile(arr, 0.95)
    }
    return metrics_func

def distrubution_metrics_names():
    metrics_names = {}
    for metric_name in get_distribution_metrics_func().keys():
        if metric_name.startswith('q'):
            key = f'{int(metric_name[1:])} percentile'
            metrics_names[key] = metric_name
        else:
            key = metric_name.capitalize()
            metrics_names[key] = metric_name
    return metrics_names

def spotfit_size_metrics_names():
    metrics_names = {
        'Radius x- direction': 'sigma_x_fit',
        'Radius y- direction': 'sigma_y_fit',
        'Radius z- direction': 'sigma_z_fit',
        'Mean radius xy- direction': 'sigma_yx_mean_fit',
        'Spot volume (voxel)': 'spheroid_vol_vox_fit',
    }
    return metrics_names

def spotfit_intensities_metrics_names():
    metrics_names = {
        'Total integral gauss. peak': 'total_integral_fit',
        'Foregr. integral gauss. peak': 'foreground_integral_fit',
        'Amplitude gauss. peak': 'A_fit',
        'Backgr. level gauss. peak': 'B_fit',
    }
    return metrics_names

def spotfit_gof_metrics_names():
    metrics_names = {
        'RMS error gauss. fit': 'RMSE_fit',
        'Normalised RMS error gauss. fit': 'NRMSE_fit',
        'F-norm. RMS error gauss. fit': 'F_NRMSE_fit'
    }
    return metrics_names

def get_effect_size_func():
    effect_size_func = {
        'cohen': cohen_effect_size,
        'glass': glass_effect_size,
        'hedge': hedge_effect_size
    }
    return effect_size_func

def get_features_groups():
    features_groups = {
        'Effect size (vs. backgr.)': 
            ['Glass', 'Cohen', 'Hedge'],
        'Effect size (vs. ref. ch.)': 
            ['Glass', 'Cohen', 'Hedge'],
        'Statistical test (vs. backgr.)': 
            ['t-statistic', 'p-value (t-test)'],
        'Statistical test (vs. ref. ch.)': 
            ['t-statistic', 'p-value (t-test)'],
        'Raw intens. metric': 
           list( distrubution_metrics_names().keys()),
        'Preprocessed intens. metric': 
            list(distrubution_metrics_names().keys()), 
        'Spotfit size metric':
            list(spotfit_size_metrics_names().keys()),
        'Spotfit intens. metric':
            list(spotfit_intensities_metrics_names().keys()),
        'Goodness-of-fit':
            list(spotfit_gof_metrics_names().keys()),
    }
    return features_groups

def _feature_types_to_col_endname_mapper():
    mapper = {
        'Glass': 'effect_size_glass',
        'Cohen': 'effect_size_cohen',
        'Hedge': 'effect_size_hedge',
        't-statistic': 'ttest_tstat',
        'p-value (t-test)': 'ttest_pvalue'
    }
    mapper = {
        **mapper,
        **distrubution_metrics_names(),
        **spotfit_size_metrics_names(),
        **spotfit_intensities_metrics_names(),
        **spotfit_gof_metrics_names(),
    }
    return mapper

def feature_names_to_col_names_mapper():
    mapper = {}
    types_to_col_endname_mapper = _feature_types_to_col_endname_mapper()
    for group_name, feature_names in get_features_groups().items():
        if group_name.find('backgr') != -1:
            prefix = 'spot_vs_backgr_'
            suffix = ''
        elif group_name.find('ref. channel') != -1:
            prefix = 'spot_vs_ref_ch_'
            suffix = ''
        elif group_name.find('Preprocessed') != -1:
            prefix = 'spot_preproc_'
            suffix = '_in_spot_minimumsize_vol'
        elif group_name.find('Raw') != -1:
            prefix = 'spot_raw_'
            suffix = '_in_spot_minimumsize_vol'
        else:
            prefix = ''
            suffix = ''
        for feature_name in feature_names:
            endname = types_to_col_endname_mapper[feature_name]
            colname = f'{prefix}{endname}{suffix}'
            mapper[f'{group_name}, {feature_name}'] = colname
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
    return df