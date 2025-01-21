import traceback

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
from . import RATIO_ON_BKGR_TO_TOTAL_SPOT_MASK

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
        filtered = skimage.morphology.opening(image)
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
            return image
    except Exception as err:
        pass
    
    if image.ndim == 2:
        try:
            sigma = sigma[1:]
        except Exception as err:
            pass
    
    if CUPY_INSTALLED and use_gpu:
        try:
            image = cp.array(image, dtype=float)
            filtered = gpu_gaussian_filter(image, sigma)
            filtered = cp.asnumpy(filtered)
        except Exception as err:
            logger_func('*'*100)
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
    input_shape = image.shape
    filtered = skimage.filters.sato(
        np.squeeze(image), sigmas=sigmas, black_ridges=False
    ).reshape(input_shape)
    return filtered

def DoG_spots(
        image, spots_zyx_radii_pxl, use_gpu=False, logger_func=print, lab=None
    ):
    spots_zyx_radii_pxl = np.array(spots_zyx_radii_pxl)
    if image.ndim == 2 and len(spots_zyx_radii_pxl) == 3:
        spots_zyx_radii_pxl = spots_zyx_radii_pxl[1:]
    
    sigma1 = spots_zyx_radii_pxl/(1+SQRT_2)
    
    if 0 in sigma1:
        raise TypeError(
            f'Sharpening filter input sigmas cannot be 0. `zyx_sigma1 = {sigma1}`'
        )
    
    if image.ndim == 2:
        sigma1 = sigma1[0]
    
    blurred1 = gaussian(
        image, sigma1, use_gpu=use_gpu, logger_func=logger_func
    )
    
    sigma2 = SQRT_2*sigma1
    blurred2 = gaussian(
        image, sigma2, use_gpu=use_gpu, logger_func=logger_func
    )
    
    sharpened = blurred1 - blurred2
    
    if lab is None:
        out_range = (image.min(), image.max())
        in_range = 'image'
    else:
        lab_mask = lab > 0
        img_masked = image[lab_mask]
        out_range = (img_masked.min(), img_masked.max())
        sharp_img_masked = sharpened[lab_mask]
        in_range = (sharp_img_masked.min(), sharp_img_masked.max())
    sharp_rescaled = skimage.exposure.rescale_intensity(
        sharpened, in_range=in_range, out_range=out_range
    )
    
    return sharp_rescaled

def threshold(
        image, threshold_func, do_max_proj=False, logger_func=print, 
        mask=None
    ):
    if do_max_proj and image.ndim == 3:
        input_image = image.max(axis=0)
        if mask is not None:
            mask = mask.max(axis=0)
    else:
        input_image = image
    
    if mask is None:
        input_vals = input_image
    else:
        input_vals = input_image[mask]
    
    try:
        thresh_val = threshold_func(input_vals)
    except Exception as e:
        logger_func(f'{e} ({threshold_func})')
        thresh_val = np.inf
    
    return image > thresh_val

def clear_objs_outside_mask(mask_to_clear, clearing_mask):
    lab_to_clear = skimage.measure.label(mask_to_clear)
    rp = skimage.measure.regionprops(lab_to_clear)
    for sub_obj in rp:
        if np.any(clearing_mask[sub_obj.slice][sub_obj.image]):
            continue
        mask_to_clear[sub_obj.slice][sub_obj.image] = 0
    return mask_to_clear

def threshold_masked_by_obj(
        image, mask, threshold_func, do_max_proj=False, 
        return_thresh_val=False, use_mask=True,
    ):
    if do_max_proj and image.ndim == 3:
        input_img = image.max(axis=0)
        mask = mask.max(axis=0)
    else:
        input_img = image
    
    if use_mask:
        masked = input_img[mask>0]
    else:
        masked = input_img
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

def _get_semantic_segm_output(
        result, return_only_output_mask, nnet_model, return_nnet_prediction, 
        bioimageio_model, spotiflow_model
    ):
    if not return_only_output_mask:
        return result
    
    if nnet_model is not None:
        segm_out = result['neural_network']
    elif bioimageio_model is not None:
        segm_out = result['bioimageio_model']
    elif spotiflow_model is not None:
        segm_out = result['spotiflow']
    else:
        segm_out = result['custom']
    
    if return_nnet_prediction:
        nnet_pred_map = result['neural_network_prediciton']
        return segm_out, nnet_pred_map
    return segm_out

def local_semantic_segmentation(
        image, lab, 
        threshold_func=None, 
        zyx_tolerance=None, 
        lineage_table=None, 
        return_image=False,
        nnet_model=None, 
        nnet_params=None, 
        nnet_input_data=None,
        return_nnet_prediction=False,
        do_max_proj=True, 
        keep_objects_touching_lab_intact=False, 
        ridge_filter_sigmas=0,
        return_only_output_mask=False, 
        do_try_all_thresholds=True,
        bioimageio_model=None,
        bioimageio_params=None,
        bioimageio_input_image=None,
        spotiflow_model=None,
        spotiflow_params=None,
        spotiflow_input_image=None,
        min_mask_size=1
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
    
    # Add bioimage io key if required
    if spotiflow_model is not None:
        threshold_funcs['spotiflow'] = None
    
    if zyx_tolerance is None:
        zyx_tolerance = (1, 1, 1)
    
    slicer = transformations.SliceImageFromSegmObject(
        lab, lineage_table=lineage_table, zyx_tolerance=zyx_tolerance
    )
    rp = skimage.measure.regionprops(lab)
    result = {}
    if return_image:
        result['input_image'] = np.zeros_like(image)
    
    
    try:
        save_pred_map = nnet_params['init'].get('save_prediction_map')
    except Exception as err:
        save_pred_map = False
    
    return_nnet_prediction = return_nnet_prediction or save_pred_map
    
    if return_nnet_prediction:
        result['neural_network_prediciton'] = np.zeros(lab.shape)
    
    if do_try_all_thresholds:
        pbar = tqdm(total=len(threshold_funcs), ncols=100)
    for method, thresh_func in threshold_funcs.items():
        labels = np.zeros_like(lab)
        for obj in rp:
            if lineage_table is not None:
                try:
                    if lineage_table.at[obj.label, 'relationship'] == 'bud':
                        # Skip buds since they are aggregated with mother
                        continue
                except Exception as err:
                    printl(traceback.format_exc())
                    import pdb; pdb.set_trace()
            
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
            elif method == 'spotiflow':
                input_img, _, _, _ = (
                    slicer.slice(spotiflow_input_image, obj)
                )
            else:
                input_img = spots_img_obj
            
            if ridge_filter_sigmas:
                input_img = ridge(input_img, ridge_filter_sigmas)
            
            if return_image:
                result['input_image'][merged_obj_slice] = input_img
            
            if method == 'neural_network':
                nnet_params['segment']['return_pred'] = return_nnet_prediction
                nnet_result = nnet_model.segment(
                    input_img, **nnet_params['segment']
                )
                if return_nnet_prediction:
                    predict_mask_merged, nnet_pred_merged = nnet_result
                    result['neural_network_prediciton'][merged_obj_slice] = (
                        nnet_pred_merged
                    )
                else:
                    predict_mask_merged = nnet_result
            elif method == 'bioimageio_model':
                predict_mask_merged = bioimageio_model.segment(
                    input_img, **bioimageio_params['segment']
                )
            elif method == 'spotiflow':
                predict_mask_merged = spotiflow_model.segment(
                    input_img, **spotiflow_params['segment']
                )
            else:
                # Threshold
                predict_mask_merged = threshold_masked_by_obj(
                    input_img, obj_mask_lab, thresh_func, 
                    do_max_proj=do_max_proj, 
                    # use_mask=not keep_objects_touching_lab_intact
                )
            
            if not keep_objects_touching_lab_intact:
                predict_mask_merged[~(obj_mask_lab>0)] = False
            else:
                predict_mask_merged = clear_objs_outside_mask(
                    predict_mask_merged, obj_mask_lab
                )
            
            if bud_ID > 0:
                # Split object into mother and bud 
                predict_lab_merged = np.zeros(
                    predict_mask_merged.shape, dtype=int
                )
                moth_mask = obj_mask_lab == obj.label
                predict_moth_mask = np.zeros_like(predict_mask_merged)
                predict_moth_mask[moth_mask] = predict_mask_merged[moth_mask]
                
                # Label sub-objects in the mother and add them to labels
                predict_moth_lab = skimage.measure.label(predict_moth_mask)
                predict_lab_merged[moth_mask] = predict_moth_lab[moth_mask]
                
                # Label sub-objects in the bud and add them to labels
                bud_mask = obj_mask_lab == bud_ID
                predict_bud_mask = np.zeros_like(predict_mask_merged)
                predict_bud_mask[bud_mask] = predict_mask_merged[bud_mask]
                
                predict_bud_lab = skimage.measure.label(predict_bud_mask)
                predict_moth_rp = skimage.measure.regionprops(predict_moth_lab)
                max_sub_id_moth = max(
                    [obj.label for obj in predict_moth_rp], default=1
                )
                predict_bud_lab[predict_bud_mask] += max_sub_id_moth
                predict_lab_merged[bud_mask] = predict_bud_lab[bud_mask]
            else:
                predict_lab_merged = skimage.measure.label(predict_mask_merged)
            
            # Assign ID to sub-objets in predict_mask_merged depending on 
            # the most common ID they lie on
            local_labels = labels[merged_obj_slice]
            predict_rp = skimage.measure.regionprops(predict_lab_merged)
            for sub_obj in predict_rp:
                if sub_obj.area < min_mask_size:
                    continue
                IDs = obj_mask_lab[sub_obj.slice][sub_obj.image]
                IDs = IDs[IDs>0]
                IDs, counts = np.unique(IDs, return_counts=True)
                most_common_idx = np.argmax(counts)
                ID = IDs[most_common_idx]
                local_labels[sub_obj.slice][sub_obj.image] = ID
        
        # labels = filter_labels_by_size(labels, min_mask_size)

        result[method] = labels.astype(np.int32)
        if do_try_all_thresholds:
            pbar.update()
    if do_try_all_thresholds:
        pbar.close()
    
    out = _get_semantic_segm_output(
        result, return_only_output_mask, nnet_model, return_nnet_prediction, 
        bioimageio_model, spotiflow_model
    )
    return out

def global_semantic_segmentation(
        image, lab, 
        lineage_table=None, 
        zyx_tolerance=None, 
        threshold_func='', 
        logger_func=print, 
        return_image=False,
        keep_input_shape=True,
        keep_objects_touching_lab_intact=True,
        thresh_only_inside_objs_intens=True,
        nnet_model=None, 
        nnet_params=None,
        nnet_input_data=None, 
        return_nnet_prediction=False,
        ridge_filter_sigmas=0,
        return_only_output_mask=False, 
        do_try_all_thresholds=True,
        pre_aggregated=False,
        x_slice_idxs=None,
        bioimageio_model=None,
        bioimageio_params=None,
        bioimageio_input_image=None,
        spotiflow_model=None,
        spotiflow_params=None,
        spotiflow_input_image=None,
        min_mask_size=1
    ):    
    if image.ndim not in (2, 3):
        ndim = image.ndim
        raise TypeError(
            f'Input image has {ndim} dimensions. Only 2D and 3D is supported.'
        )
    
    threshold_funcs = _get_threshold_funcs(
        threshold_func=threshold_func, try_all=do_try_all_thresholds
    )
    
    if x_slice_idxs is None:
        pre_aggregated = False
    
    if pre_aggregated:
        aggr_img = image
        aggregated_lab = lab
        aggr_transf_spots_nnet_img = nnet_input_data
        aggr_transf_spots_bioimageio_img = bioimageio_input_image
        aggr_transf_spotiflow_img = spotiflow_input_image
    else:
        additional_imgs_to_aggr = (
            nnet_input_data, bioimageio_input_image, spotiflow_input_image
        )
        aggregated = transformations.aggregate_objs(
            image, lab, lineage_table=lineage_table, 
            zyx_tolerance=zyx_tolerance,
            additional_imgs_to_aggr=additional_imgs_to_aggr, 
            return_x_slice_idxs=True 
        )
        aggr_img, aggregated_lab, aggr_imgs, x_slice_idxs = aggregated
        aggr_transf_spots_nnet_img = aggr_imgs[0]
        aggr_transf_spots_bioimageio_img = aggr_imgs[1]
        aggr_transf_spotiflow_img = aggr_imgs[2]
    
    if ridge_filter_sigmas:
        aggr_img = ridge(aggr_img, ridge_filter_sigmas)
    
    try:
        save_pred_map = nnet_params['init'].get('save_prediction_map')
    except Exception as err:
        save_pred_map = False
    
    return_nnet_prediction = return_nnet_prediction or save_pred_map
    
    # Thresholding
    thresh_mask = None
    if thresh_only_inside_objs_intens:
        thresh_mask = aggregated_lab > 0
        
    result = {}
    for method, thresh_func in threshold_funcs.items():
        thresholded = threshold(
            aggr_img, thresh_func, logger_func=logger_func,
            do_max_proj=True, mask=thresh_mask
        )
        thresholded = filter_labels_by_size(thresholded, min_mask_size)
        result[method] = thresholded
    
    # Neural network
    if nnet_model is not None:
        if aggr_transf_spots_nnet_img is None:
            nnet_input_img = aggr_img
        else:
            nnet_input_img = aggr_transf_spots_nnet_img

        nnet_params['segment']['return_pred'] = return_nnet_prediction
        nnet_result = nnet_model.segment(
            nnet_input_img, **nnet_params['segment']
        )
        if return_nnet_prediction:
            nnet_labels, aggr_nnet_pred = nnet_result
        else:
            nnet_labels = nnet_result
        nnet_labels = filter_labels_by_size(nnet_labels, min_mask_size)
        result['neural_network'] = nnet_labels
    
    if bioimageio_model is not None:
        bioimageio_labels = bioimageio_model.segment(
            aggr_transf_spots_bioimageio_img, **bioimageio_params['segment']
        )
        bioimageio_labels = filter_labels_by_size(
            bioimageio_labels, min_mask_size
        )
        result['bioimageio_model'] = bioimageio_labels
    
    if spotiflow_model is not None:
        spotiflow_labels = spotiflow_model.segment(
            aggr_transf_spotiflow_img, **spotiflow_params['segment']
        )
        spotiflow_labels = filter_labels_by_size(
            spotiflow_labels, min_mask_size
        )
        result['spotiflow'] = spotiflow_labels
    
    if keep_input_shape:
        reindexed_result = {}
        for method, aggr_segm in result.items():
            keep_subobj_intact = keep_objects_touching_lab_intact
            reindexed_result[method] = (
                transformations.index_aggregated_segm_into_input_lab(
                    lab, aggr_segm, aggregated_lab, x_slice_idxs,
                    keep_objects_touching_lab_intact=keep_subobj_intact, 
                )
            )
        result = reindexed_result
        if return_image:
            deaggr_img = transformations.deaggregate_img(
                aggr_img, aggregated_lab, lab
            )
            input_image_dict = {'input_image': deaggr_img}
            result = {**input_image_dict, **result}
        if return_nnet_prediction:
            deaggr_nnet_pred = transformations.deaggregate_img(
                aggr_nnet_pred, aggregated_lab, lab
            )
            result['neural_network_prediciton'] = deaggr_nnet_pred
    else:
        if return_image:
            input_image_dict = {'input_image': aggr_img}
            result = {**input_image_dict, **result}
        if return_nnet_prediction:
            result['neural_network_prediciton'] = aggr_nnet_pred
    
    # result = {key:np.squeeze(img) for key, img in result.items()}
    out = _get_semantic_segm_output(
        result, return_only_output_mask, nnet_model, return_nnet_prediction, 
        bioimageio_model, spotiflow_model
    )
    return out

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
    logger_func(f"\n{'='*100}")
    txt = (
        f'[WARNING]: The feature name "{missing_feature}" is not present '
        'in the table. It cannot be used for filtering at '
        f'this stage.{error_up_str}'
    )
    logger_func(txt)

def filter_df_from_features_thresholds(
        df_features: pd.DataFrame, 
        features_thresholds: dict, 
        is_spotfit=False,
        debug=False,
        logger_func=print
    ):
    """Filter valid spots based on features ranges

    Parameters
    ----------
    df_features : pd.DataFrame
        Pandas DataFrame with 'spot_id' or ('Cell_ID', 'spot_id') as index and 
        the features as columns.
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
    is_spotfit : bool, optional
        If False, features ending with '_fit' will be ignored. Default is False
    debug : bool, optional
        If True, it can be used for debugging like printing additional 
        internal steps or visualize intermediate results.
    logger_func : callable, optional
        Function used to print or log process information. Default is print

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame
    """      
    queries = []  
    for f, (feature_name, thresholds) in enumerate(features_thresholds.items()):        
        close_parenthesis = False

        statements = []
        if feature_name.startswith('| '):
            feature_name = feature_name[2:]
            statements.append(' | ')
        elif feature_name.startswith('& '):
            feature_name = feature_name[2:]
            statements.append(' & ')
        elif f>0:
            statements.append(' & ')
            
        if feature_name.startswith('('):
            feature_name = feature_name[1:]
            statements.append('(')
        
        if feature_name.endswith(')'):
            feature_name = feature_name[:-1]
            close_parenthesis = True
        
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
        
        queries.extend(statements)
        
        _min, _max = thresholds
        _query = ''
        if _min is not None:
            _query = f'{feature_name} > {_min}'
        if _max is not None:
            _query = f'{_query} & {feature_name} < {_max}'
        
        _query = _query.strip(' & ')
        queries.append(f'({_query})')
        
        if close_parenthesis:
            queries.append(')')

    if not queries:
        return df_features
    
    query = ''.join(queries)
    query = query.strip().lstrip('&').lstrip('|')
    
    if 'do_not_drop' in df_features.columns:
        query = f'({query}) | (do_not_drop > 0)'
    
    # logger_func(f'Filtering with query = `{query}`')
    
    df_filtered = df_features.query(query)
    
    if debug:
        import pdb; pdb.set_trace()
    
    return df_filtered

def drop_spots_not_in_ref_ch(df, ref_ch_mask, local_peaks_coords):
    if ref_ch_mask is None:
        return df
    
    zz = local_peaks_coords[:,0]
    yy = local_peaks_coords[:,1]
    xx = local_peaks_coords[:,2]
    in_ref_ch_spots_mask = ref_ch_mask[zz, yy, xx] > 0
    if 'do_not_drop' in df.columns:
        in_ref_ch_spots_mask = (in_ref_ch_spots_mask) | (df['do_not_drop'] > 0)
    
    return df[in_ref_ch_spots_mask]

def filter_labels_by_size(labels, min_size):
    if min_size <= 1:
        return labels
    
    lab = skimage.measure.label(labels.astype(np.uint32))
    rp = skimage.measure.regionprops(lab)
    filtered_labels = labels.copy()
    for obj in rp:
        if obj.area >= min_size:
            continue
        
        filtered_labels[obj.slice][obj.image] = 0
    return filtered_labels

def filter_valid_points_min_distance(
        points: np.ndarray, min_distance: np.ndarray, intensities=None, 
        return_valid_points_mask=False, debug=False
    ):
    num_points = len(points)
    if intensities is not None:
        # Sort points by descending intensities
        sorting_idxs_descending = np.flip(intensities.argsort())
        points = points[sorting_idxs_descending]
        
    masked_points = np.ma.masked_array(
        data=points, 
        mask=np.zeros(points.shape, bool), 
        fill_value=-1
    )
    valid_points_mask = np.ones(num_points, dtype=bool)
    for i, point in enumerate(points):
        if not valid_points_mask[i]:
            # Skip points that have already been dropped
            continue
        points_ellipsoid = np.square((masked_points - point)/min_distance)
        points_too_close_mask = np.sum(points_ellipsoid, axis=1) < 1
        valid_points_mask[i+1:] = np.invert(points_too_close_mask[i+1:])
        
        # Mask dropped points to avoid computing the distance to them in 
        # future iterations
        masked_points.mask[points_too_close_mask] = True
    
    valid_points = points[valid_points_mask]
    
    if return_valid_points_mask:
        return valid_points, valid_points_mask
    
    return valid_points

def validate_spots_labels(spot_labels, lab):
    if spot_labels is None:
        return []
    
    invalid_IDs = []
    labels_rp = skimage.measure.regionprops(spot_labels)
    for spot_obj in labels_rp:
        tourching_IDs, counts = np.unique(
            lab[spot_obj.slice][spot_obj.image], return_counts=True
        )
        if tourching_IDs[0] != 0:
            continue
        
        count_ratio = counts[0]/spot_obj.area
        if count_ratio < RATIO_ON_BKGR_TO_TOTAL_SPOT_MASK:
            continue
        
        invalid_IDs.append(spot_obj.label)
    
    return invalid_IDs

def remove_object_IDs(lab, IDs):
    rp = skimage.measure.regionprops(lab)
    for obj in rp:
        if obj.label not in IDs:
            continue
        
        lab[obj.slice][obj.image] = 0
    return lab
        