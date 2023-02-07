import numpy as np
import scipy.stats

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

def get_aggregating_spots_feature_func():
    func = {
        'num_spots_inside_ref_ch': ('is_spot_inside_ref_ch', 'sum'),
        'num_spots': ('x', 'count'),
        'spotfit_sum_foregr_integral': ('I_foregr', 'sum'),
        'spotfit_sum_tot_integral': ('I_tot', 'sum'),
        'mean_sigma_z': ('sigma_z_fit', 'mean'),
        'mean_sigma_y': ('sigma_y_fit', 'mean'),
        'mean_sigma_x': ('sigma_x_fit', 'mean'),
        'std_sigma_z': ('sigma_z_fit', 'std'),
        'std_sigma_y': ('sigma_y_fit', 'std'),
        'std_sigma_x': ('sigma_x_fit', 'std'),
        'sum_A_fit': ('A_fit', 'sum'),
        'mean_B_fit': ('B_fit', 'mean'),
        'solution_found': ('solution_found', 'mean'),
        'mean_reduced_chisq': ('reduced_chisq', 'mean'),
        'combined_p_chisq': ('p_chisq', scipy.stats.combine_pvalues),
        'mean_RMSE': ('RMSE', 'mean'),
        'mean_NRMSE': ('NRMSE', 'mean'),
        'mean_F_NRMSE': ('F_NRMSE', 'mean'),
        'mean_ks': ('KS_stat', 'mean'),
        'combined_p_ks': ('p_KS', 'mean'),
        'mean_ks_null': ('null_ks_test', 'mean'),
        'mean_chisq_null': ('null_chisq_test', 'mean'),
        'mean_QC_passed': ('QC_passed', 'mean')
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

def get_distribution_metric_func():
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

def get_effect_size_func():
    effect_size_func = {
        'cohen': cohen_effect_size,
        'glass': glass_effect_size,
        'hedge': hedge_effect_size
    }
    return effect_size_func