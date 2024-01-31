from sklearn import neural_network
from tqdm import tqdm

import numpy as np
import pandas as pd

try:
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    import cupy as cp
    CUPY_INSTALLED = True
except Exception as e:
    CUPY_INSTALLED = False

import skimage.morphology
import skimage.filters
import skimage.measure

from . import GUI_INSTALLED
if GUI_INSTALLED:
    from cellacdc.plot import imshow

from . import error_up_str, printl
from . import config, transformations

import math
SQRT_2 = math.sqrt(2)

def remove_hot_pixels(image, logger_func=print, progress=True):
    is_3D = image.ndim == 3
    if is_3D:
        if progress:
            pbar = tqdm(total=len(image), ncols=100)
        filtered = image.copy()
        for z, img in enumerate(image):
            filtered[z] = skimage.morphology.opening(img)
            if progress:
                pbar.update()
        if progress:
            pbar.close()
    else:
        filtered[z] = skimage.morphology.opening(img)
    return filtered

def gaussian(image, sigma, use_gpu=False, logger_func=print):
    try:
        if len(sigma) > 1 and sigma[0] == 0:
            return image
    except Exception as err:
        pass
    
    try:
        if sigma == 0:
            return image
    except Exception as err:
        pass
    
    try:
        if len(sigma) == 0:
            sigma = sigma[0]
    except Exception as err:
        pass
    
    if CUPY_INSTALLED and use_gpu:
        try:
            image = cp.array(image, dtype=float)
            filtered = gpu_gaussian_filter(image, sigma)
            filtered = cp.asnumpy(filtered)
        except Exception as err:
            logger_func('*'*60)
            logger_func(err)
            logger_func(
                '[WARNING]: GPU acceleration of the gaussian filter failed. '
                f'Using CPU...{error_up_str}'
            )
            filtered = skimage.filters.gaussian(image, sigma=sigma)
    else:
        filtered = skimage.filters.gaussian(image, sigma=sigma)
    return filtered

def ridge(image, sigmas):
    filtered = skimage.filters.sato(image, sigmas=sigmas, black_ridges=False)
    return filtered

def DoG_spots(image, spots_zyx_radii_pxl, use_gpu=False, logger_func=print):
    spots_zyx_radii_pxl = np.array(spots_zyx_radii_pxl)
    if image.ndim == 2 and len(spots_zyx_radii_pxl) == 3:
        spots_zyx_radii_pxl = spots_zyx_radii_pxl[1:]
    
    sigma1 = spots_zyx_radii_pxl/(1+SQRT_2)
        
    blurred1 = gaussian(
        image, sigma1, use_gpu=use_gpu, logger_func=logger_func
    )
    
    sigma2 = SQRT_2*sigma1
    blurred2 = gaussian(
        image, sigma2, use_gpu=use_gpu, logger_func=logger_func
    )
    
    sharpened = blurred1 - blurred2
    
    out_range = (image.min(), image.max())
    sharp_rescaled = skimage.exposure.rescale_intensity(
        sharpened, out_range=out_range
    )
    return sharp_rescaled

def threshold(
        image, threshold_func, do_max_proj=False, logger_func=print
    ):
    if do_max_proj and image.ndim == 3:
        input_image = image.max(axis=0)
    else:
        input_image = image
    
    try:
        thresh_val = threshold_func(input_image)
    except Exception as e:
        logger_func(f'{e} ({threshold_func})')
        thresh_val = np.inf
    
    return image > thresh_val

def threshold_masked_by_obj(
        image, mask, threshold_func, do_max_proj=False, return_thresh_val=False
    ):
    if do_max_proj and image.ndim == 3:
        input_img = image.max(axis=0)
        mask = mask.max(axis=0)
    else:
        input_img = image
        
    masked = input_img[mask>0]
    try:
        thresh_val = threshold_func(masked)
        thresholded = image > thresh_val
    except Exception as err:
        thresh_val = np.nan
        thresholded = np.zeros(image.shape, dtype=bool)
    
    if return_thresh_val:
        return thresholded, thresh_val
    else:
        return thresholded

def _get_threshold_funcs(threshold_func=None, try_all=True):
    if threshold_func is None and try_all:
        threshold_funcs = {
            method:getattr(skimage.filters, method) for 
            method in config.skimageAutoThresholdMethods()
        }
    elif isinstance(threshold_func, str):
        threshold_funcs = {'custom': getattr(skimage.filters, threshold_func)}
    elif threshold_func is not None:
        threshold_funcs = {'custom': threshold_func}
    else:
        threshold_funcs = {}
    return threshold_funcs

def local_semantic_segmentation(
        image, lab, 
        threshold_func=None, 
        lineage_table=None, 
        return_image=False,
        nnet_model=None, 
        nnet_params=None, 
        nnet_input_data=None,
        do_max_proj=True, 
        clear_outside_objs=False, 
        ridge_filter_sigmas=0,
        return_only_output_mask=False, 
        do_try_all_thresholds=True,
        bioimageio_model=None,
        bioimageio_params=None,
        bioimageio_input_image=None
    ):
    # Get prediction mask by thresholding objects separately
    threshold_funcs = _get_threshold_funcs(
        threshold_func=threshold_func, try_all=do_try_all_thresholds
    )
    
    # Add neural network method if required (we just need the key for the loop)
    if nnet_model is not None:
        threshold_funcs['neural_network'] = None
    
    # Add bioimage io key if required
    if bioimageio_model is not None:
        threshold_funcs['bioimageio_model'] = None
    
    slicer = transformations.SliceImageFromSegmObject(
        lab, lineage_table=lineage_table
    )
    aggr_rp = skimage.measure.regionprops(lab)
    result = {}
    if return_image:
        result['input_image'] = np.zeros_like(image)
    
    if do_try_all_thresholds:
        pbar = tqdm(total=len(threshold_funcs), ncols=100)
    for method, thresh_func in threshold_funcs.items():
        labels = np.zeros_like(lab)
        for obj in aggr_rp:
            if lineage_table is not None:
                if lineage_table.at[obj.label, 'relationship'] == 'bud':
                    # Skip buds since they are aggregated with mother
                    continue
            
            spots_img_obj, lab_mask_lab, merged_obj_slice, bud_ID = (
                slicer.slice(image, obj)
            )
            obj_mask_lab = lab_mask_lab[merged_obj_slice]

            if method == 'neural_network' and nnet_input_data is not None:
                input_img, _, _, _ = (
                    slicer.slice(nnet_input_data, obj)
                )
            elif method == 'bioimageio_model':
                input_img, _, _, _ = (
                    slicer.slice(bioimageio_input_image, obj)
                )
            else:
                input_img = spots_img_obj
            
            if ridge_filter_sigmas:
                input_img = ridge(input_img, ridge_filter_sigmas)
            
            result['input_image'][merged_obj_slice] = input_img
            
            if method == 'neural_network':
                predict_mask_merged = nnet_model.segment(
                    input_img, **nnet_params['segment']
                )
            elif method == 'bioimageio_model':
                predict_mask_merged = bioimageio_model.segment(
                    input_img, **bioimageio_params['segment']
                )
            else:
                # Threshold
                predict_mask_merged = threshold_masked_by_obj(
                    input_img, obj_mask_lab, thresh_func, 
                    do_max_proj=do_max_proj
                )
                # predict_mask_merged[~(obj_mask_lab>0)] = False

            if clear_outside_objs:
                predict_mask_merged[~(obj_mask_lab>0)] = False
            
            # Iterate eventually merged (mother-bud) objects
            for obj_local in skimage.measure.regionprops(obj_mask_lab):  
                predict_mask_obj = np.logical_and(
                    predict_mask_merged[obj_local.slice], 
                    obj_local.image
                )
                id = obj_local.label
                labels[merged_obj_slice][obj_local.slice][predict_mask_obj] = id
        
        result[method] = labels.astype(np.int32)
        if do_try_all_thresholds:
            pbar.update()
    if do_try_all_thresholds:
        pbar.close()
    
    if return_only_output_mask:
        if nnet_model is not None:
            return result['neural_network']
        elif bioimageio_model is not None:
            return result['bioimageio_model']
        else:
            return result['custom']
    else:
        return result

def global_semantic_segmentation(
        image, lab, 
        lineage_table=None, 
        zyx_tolerance=None, 
        threshold_func='', 
        logger_func=print, 
        return_image=False,
        keep_input_shape=True, 
        nnet_model=None, 
        nnet_params=None,
        nnet_input_data=None, 
        ridge_filter_sigmas=0,
        return_only_output_mask=False, 
        do_try_all_thresholds=True,
        pre_aggregated=False,
        bioimageio_model=None,
        bioimageio_params=None,
        bioimageio_input_image=None
    ):    
    if image.ndim != 3 or image.ndim != 3:
        ndim = image.ndim
        raise TypeError(
            f'Input image has {ndim} dimensions. Only 2D and 3D is supported.'
        )
    
    threshold_funcs = _get_threshold_funcs(
        threshold_func=threshold_func, try_all=do_try_all_thresholds
    )
    
    if pre_aggregated:
        aggr_img = image
        aggregated_lab = lab
        aggr_transf_spots_nnet_img = nnet_input_data
    else:
        aggregated = transformations.aggregate_objs(
            image, lab, lineage_table=lineage_table, 
            zyx_tolerance=zyx_tolerance,
            additional_imgs_to_aggr=[nnet_input_data, bioimageio_input_image]
        )
        aggr_img, aggregated_lab, aggr_imgs = aggregated
        aggr_transf_spots_nnet_img = aggr_imgs[0]
        aggr_transf_spots_bioimageio_img = aggr_imgs[1]
    
    if ridge_filter_sigmas:
        aggr_img = ridge(aggr_img, ridge_filter_sigmas)
    
    # Thresholding
    result = {}
    for method, thresh_func in threshold_funcs.items():
        thresholded = threshold(
            aggr_img, thresh_func, logger_func=logger_func,
            do_max_proj=True
        )
        result[method] = thresholded
    
    # Neural network
    if nnet_model is not None:
        if aggr_transf_spots_nnet_img is None:
            nnet_input_img = aggr_img
        else:
            nnet_input_img = aggr_transf_spots_nnet_img

        nnet_labels = nnet_model.segment(
            nnet_input_img, **nnet_params['segment']
        )
        result['neural_network'] = nnet_labels
    
    if bioimageio_model is not None:
        bioimageio_labels = bioimageio_model.segment(
            aggr_transf_spots_bioimageio_img, **bioimageio_params['segment']
        )
        result['bioimageio_model'] = bioimageio_labels
    
    if keep_input_shape:
        reindexed_result = {}
        for method, aggr_segm in result.items():
            reindexed_result[method] = (
                transformations.index_aggregated_segm_into_input_lab(
                    lab, aggr_segm, aggregated_lab
            ))
        result = reindexed_result
        if return_image:
            deaggr_img = transformations.deaggregate_img(
                aggr_img, aggregated_lab, lab
            )
            input_image_dict = {'input_image': deaggr_img}
            result = {**input_image_dict, **result}
    elif return_image:
        input_image_dict = {'input_image': aggr_img}
        result = {**input_image_dict, **result}
    
    result = {key:np.squeeze(img) for key, img in result.items()}
    
    if return_only_output_mask:
        if nnet_model is not None:
            return result['neural_network']
        elif bioimageio_model is not None:
            return result['bioimageio_model']
        else:
            return result['custom']
    else:
        return result

def filter_largest_obj(mask_or_labels):
    lab = skimage.measure.label(mask_or_labels)
    positive_values = lab[lab > 0]
    counts = np.bincount(positive_values)
    
    if len(counts) == 0:
        if mask_or_labels.dtype == bool:
            return lab > 0 
        else:
            lab[lab>0] = mask_or_labels[lab>0]
            return lab
    
    largest_obj_id = np.argmax(counts)
    lab[lab != largest_obj_id] = 0
    if mask_or_labels.dtype == bool:
        return lab > 0
    lab[lab>0] = mask_or_labels[lab>0]
    return lab

def filter_largest_sub_obj_per_obj(mask_or_labels, lab):
    rp = skimage.measure.regionprops(lab)
    filtered = np.zeros_like(mask_or_labels)
    for obj in rp:
        obj_mask_to_filter = np.zeros_like(obj.image)
        mask_obj_sub_obj = np.logical_and(obj.image, mask_or_labels[obj.slice])
        obj_mask_to_filter[mask_obj_sub_obj] = True
        filtered_obj_mask = filter_largest_obj(obj_mask_to_filter)
        filtered[obj.slice][filtered_obj_mask] = obj.label
    return filtered

def _warn_feature_is_missing(missing_feature, logger_func):
    logger_func(f"\n{'='*60}")
    txt = (
        f'[WARNING]: The feature name "{missing_feature}" is not present '
        'in the table. It cannot be used for filtering spots at '
        f'this stage.{error_up_str}'
    )
    logger_func(txt)

def filter_spots_from_features_thresholds(
            df_features: pd.DataFrame, 
            features_thresholds: dict, is_spotfit=False,
            debug=False,
            logger_func=None
        ):
        """_summary_

        Parameters
        ----------
        df_features : pd.DataFrame
            Pandas DataFrame with 'spot_id' as index and the features as columns.
        features_thresholds : dict
            A dictionary of features and thresholds to use for filtering. The 
            keys are the feature names that mush coincide with one of the columns'
            names. The values are a tuple of `(min, max)` thresholds.
            For example, for filtering spots that have the t-statistic of the 
            t-test spot vs reference channel > 0 and the p-value < 0.025 
            (i.e. spots are significantly brighter than reference channel) 
            we pass the following dictionary:
            ```
            features_thresholds = {
                'spot_vs_ref_ch_ttest_pvalue': (None,0.025),
	            'spot_vs_ref_ch_ttest_tstat': (0, None)
            }
            ```
            where `None` indicates the absence of maximum or minimum.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame
        """      
        queries = []  
        for feature_name, thresholds in features_thresholds.items():
            if not is_spotfit and feature_name.endswith('_fit'):
                # Ignore _fit features if not spotfit
                continue
            if is_spotfit and not feature_name.endswith('_fit'):
                # Ignore non _fit features if spotfit
                continue
            if feature_name not in df_features.columns:
                # Warn and ignore missing features
                _warn_feature_is_missing(feature_name, logger_func)
                continue
            _min, _max = thresholds
            if _min is not None:
                queries.append(f'({feature_name} > {_min})')
            if _max is not None:
                queries.append(f'({feature_name} < {_max})')

        if not queries:
            return df_features
        
        query = ' & '.join(queries)

        return df_features.query(query)

def drop_spots_not_in_ref_ch(df, ref_ch_mask, local_peaks_coords):
    if ref_ch_mask is None:
        return df
    
    zz = local_peaks_coords[:,0]
    yy = local_peaks_coords[:,1]
    xx = local_peaks_coords[:,2]
    in_ref_ch_spots_mask = ref_ch_mask[zz, yy, xx] > 0
    return df[in_ref_ch_spots_mask]
