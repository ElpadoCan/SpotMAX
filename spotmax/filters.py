from tqdm import tqdm

import numpy as np

try:
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    import cupy as cp
    CUPY_INSTALLED = True
except Exception as e:
    CUPY_INSTALLED = False

import skimage.morphology
import skimage.filters
import skimage.measure

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
    if CUPY_INSTALLED and use_gpu:
        try:
            image = cp.array(image, dtype=float)
            filtered = gpu_gaussian_filter(image, sigma)
            filtered = cp.asnumpy(filtered)
        except Exception as err:
            logger_func('*'*50)
            logger_func(err)
            logger_func(
                '[WARNING]: GPU acceleration of the gaussian filter failed. '
                f'Using CPU...{error_up_str}'
            )
            filtered = skimage.filters.gaussian(image, sigma=sigma)
    else:
        filtered = skimage.filters.gaussian(image, sigma=sigma)
    return filtered

def DoG_spots(image, spots_zyx_radii, use_gpu=False, logger_func=print):
    spots_zyx_radii = np.array(spots_zyx_radii)
    if image.ndim == 2 and len(spots_zyx_radii) == 3:
        spots_zyx_radii = spots_zyx_radii[1:]
    
    sigma1 = spots_zyx_radii/(1+SQRT_2)
        
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

def try_all_thresholds(image, logger_func=print):
    methods = config.skimageAutoThresholdMethods()
    result = {}
    if image.ndim == 3:
        input_image = image.max(axis=0)
    else:
        input_image = image
    for method in tqdm(methods, desc='Thresholding', ncols=100):
        threshold_func = getattr(skimage.filters, method)
        try:
            thresh_val = threshold_func(input_image)
        except Exception as err:
            print('')
            logger_func('*'*50)
            logger_func(f'[WARNING]: {err} ({method})')
            thresh_val = np.inf
        result[method] = image > thresh_val
    return result

def threshold(image, thresholding_method: str, logger_func=print):
    if image.ndim == 3:
        input_image = image.max(axis=0)
    else:
        input_image = image
    threshold_func = getattr(skimage.filters, thresholding_method)
    try:
        thresh_val = threshold_func(input_image)
    except Exception as e:
        logger_func(f'{e} ({thresholding_method})')
        thresh_val = np.inf
    return image > thresh_val

def local_semantic_segmentation(
        image, lab, threshold_func=None, lineage_table=None, return_image=False
    ):
    # Get prediction mask by thresholding objects separately
    if threshold_func is None:
        threshold_funcs = {
            method:getattr(skimage.filters, method) for 
            method in config.skimageAutoThresholdMethods()
        }
    elif isinstance(threshold_func, str):
        threshold_funcs = {'custom': getattr(skimage.filters, threshold_func)}
    else:
        threshold_funcs = {'custom': threshold_func}
    
    slicer = transformations.SliceImageFromSegmObject(
        lab, lineage_table=lineage_table
    )
    aggr_rp = skimage.measure.regionprops(lab)
    IDs = [obj.label for obj in aggr_rp]
    result = {}
    if return_image:
        result['input_image'] = image
    
    if threshold_func is None:
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

            # Threshold
            threshold_val = thresh_func(spots_img_obj.max(axis=0))
            predict_mask_merged = spots_img_obj > threshold_val
            
            import pdb; pdb.set_trace()
            # Iterate eventually merged (mother-bud) objects
            for obj_local in skimage.measure.regionprops(obj_mask_lab):  
                predict_mask_obj = predict_mask_merged[obj_local.slice].copy()
                labels[merged_obj_slice][obj_local.slice][predict_mask_obj] = (
                    obj.label
                )
        
        result[method] = labels.astype(np.int32)
        if threshold_func is None:
            pbar.update()
    if threshold_func is None:
        pbar.close()
    
    if threshold_func is not None:
        return result['custom']
    else:
        return result

def global_semantic_segmentation(
        image, lab, lineage_table=None, zyx_tolerance=None, 
        thresholding_method='', logger_func=print, return_image=False,
        keep_input_shape=True
    ):
    if image.ndim == 2:
        image = image[np.newaxis]
    
    if lab.ndim == 2 and image.ndim == 3:
        # Stack 2D lab into 3D z-stack
        lab = np.array([lab]*len(image))
    
    if image.ndim != 3:
        ndim = image.ndim
        raise TypeError(
            f'Input image has {ndim} dimensions. Only 2D and 3D is supported.'
        )
    
    aggr_spots_img, aggregated_lab = transformations.aggregate_objs(
        image, lab, lineage_table=lineage_table, zyx_tolerance=zyx_tolerance
    )
    if thresholding_method is not None:
        thresholded = threshold(
            aggr_spots_img, thresholding_method, logger_func=logger_func
        )
        result = {thresholding_method: thresholded}
    else:
        result = try_all_thresholds(aggr_spots_img, logger_func=print)
    
    if keep_input_shape:
        reindexed_result = {}
        for method, aggr_img in result.items():
            reindexed_result[method] = (
                transformations.index_aggregated_segm_into_input_image(
                    image, lab, aggr_img, aggregated_lab
            ))
        result = reindexed_result
        if return_image:
            input_image_dict = {'input_image': image}
            result = {**input_image_dict, **result}
    elif return_image:
        input_image_dict = {'input_image': aggr_spots_img}
        result = {**input_image_dict, **result}
    
    result = {key:np.squeeze(img) for key, img in result.items()}
    return result

def filter_largest_obj(mask_or_labels):
    lab = skimage.measure.label(mask_or_labels)
    positive_mask = lab > 0
    counts = np.bincount(positive_mask)
    largest_obj_id = np.argmax(counts)
    lab[lab != largest_obj_id] = 0
    if mask_or_labels.dtype == bool:
        return lab > 0
    return lab

def filter_largest_sub_obj_per_obj(mask_or_labels, lab):
    rp = skimage.measure.regionprops(lab)
    filtered = np.zeros_like(mask_or_labels)
    for obj in lab:
        obj_mask_to_filter = np.zeros_like(obj.image)
        mask_obj_sub_obj = np.logical_and(obj.image, mask_or_labels[obj.slice])
        obj_mask_to_filter[mask_obj_sub_obj] = True
        filtered_obj = filter_largest_obj(obj_mask_to_filter)
        filtered[obj.slice] = filtered_obj
    return filtered