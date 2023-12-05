from tqdm import tqdm

import numpy as np
import pandas as pd

import skimage.measure
import skimage.filters

from . import GUI_INSTALLED
if GUI_INSTALLED:
    from cellacdc.plot import imshow

from . import filters
from . import transformations
from . import printl
from . import ZYX_LOCAL_COLS, ZYX_LOCAL_EXPANDED_COLS
from . import features

distribution_metrics_func = features.get_distribution_metrics_func()
effect_size_func = features.get_effect_size_func()

def preprocess_image(
        image, 
        lab=None, 
        do_remove_hot_pixels=False,
        gauss_sigma=0.0,
        use_gpu=True, 
        return_lab=False,
        do_sharpen=False,
        spots_zyx_radii=None,
        logger_func=print
    ):
    _, image = transformations.reshape_lab_image_to_3D(lab, image)
        
    if do_remove_hot_pixels:
        image = filters.remove_hot_pixels(image)
    else:
        image = image
    
    if do_sharpen and spots_zyx_radii is not None:
        image = filters.DoG_spots(
            image, spots_zyx_radii, use_gpu=use_gpu, 
            logger_func=logger_func
        )
    elif gauss_sigma>0:
        image = filters.gaussian(
            image, gauss_sigma, use_gpu=use_gpu, logger_func=logger_func
        )
    else:
        image = image
    
    if return_lab:
        return image, lab
    else:
        return image


def ridge_filter(
        image, 
        lab=None, 
        do_remove_hot_pixels=False, 
        ridge_sigmas=0.0,
        logger_func=print
    ):
    _, image = transformations.reshape_lab_image_to_3D(lab, image)
        
    if do_remove_hot_pixels:
        image = filters.remove_hot_pixels(image)
    else:
        image = image
    
    if ridge_sigmas:
        image = filters.ridge(image, ridge_sigmas)
    else:
        image = image
    return image

def spots_semantic_segmentation(
        image, 
        lab=None,
        gauss_sigma=0.0,
        spots_zyx_radii=None, 
        do_sharpen=False, 
        do_remove_hot_pixels=False,
        lineage_table=None,
        do_aggregate=True,
        use_gpu=False,
        logger_func=print,
        thresholding_method=None,
        keep_input_shape=True,
        nnet_model=None,
        nnet_params=None,
        nnet_input_data=None,
        bioimageio_model=None,
        bioimageio_params=None,
        do_preprocess=True,
        do_try_all_thresholds=True,
        return_only_segm=False,
        pre_aggregated=False,
        raw_image=None
    ):  
    """Pipeline to perform semantic segmentation on the spots channel, 
    i.e., determine the areas where spot will be detected.

    Parameters
    ----------
    image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. Spots will be detected only inside each object. If None, 
        detection will be performed on the entire image. Default is None. 
    gauss_sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard deviations of 
        the Gaussian filter are given for each axis as a sequence, or as a 
        single number, in which case it is equal for all axes. If 0, no 
        gaussian filter is applied. Default is 0.0
    spots_zyx_radii : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in pixels. The values are used to determine the sigmas used in the 
        difference-of-gaussians filter that enhances spots-like structures. 
        If None, no filter is applied. Default is None
    do_sharpen : bool, optional
        If True and spots_zyx_radii is not None, applies a 
        difference-of-gaussians (DoG) filter before segmenting. This filter 
        enhances spots-like structures and it usually improves detection. 
        Default is False.
        For more details, see the parameter `Sharpen spots signal prior 
        detection` at the following webpage: 
        https://spotmax.readthedocs.io/parameters_description.html#pre-processing
    do_remove_hot_pixels : bool, optional
        If True, apply a grayscale morphological opening filter before 
        segmenting. Opening can remove small bright spots (i.e. “salt”, or 
        "hot pixels") and connect small dark cracks. Default is False
    lineage_table : pandas.DataFrame, optional
        Table containing parent-daughter relationships. Default is None
        For more details, see the parameter `Table with lineage info end name 
        or path` at the following webpage: 
        https://spotmax.readthedocs.io/parameters_description.html#file-paths-and-channels
    do_aggregate : bool, optional
        If True, perform segmentation on all the cells at once. Default is True
    use_gpu : bool, optional
        If True, some steps will run on the GPU, potentially speeding up the 
        computation. Default is False
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    thresholding_method : {'threshol_li', 'threshold_isodata', 'threshold_otsu',
        'threshold_minimum', 'threshold_triangle', 'threshold_mean',
        'threshold_yen'} or callable, optional
        Thresholding method used to obtain semantic segmentation masks of the 
        spots. If None and do_try_all_thresholds is True, the result of every 
        threshold method available is returned. Default is None
    keep_input_shape : bool, optional
        If True, return segmentation array with the same shape of the 
        input image. If False, output shape will depend on whether do_aggregate
        is True or False. Default is True
    nnet_model : Cell-ACDC segmentation model class, optional
        If not None, the output will include the key 'neural_network' with the 
        result of the segmentation using the neural network model. 
        Default is None
    nnet_params : dict with 'segment' key, optional
        Parameters used in the segment method of the nnet_model. Default is None
    nnet_input_data : numpy.ndarray or sequence of arrays, optional
        If not None, run the neural network on this data and not on the 
        pre-processed input image. Default is None
    bioimageio_model : Cell-ACDC implementation of any BioImage.IO model, optional
        If not None, the output will include the key 'bioimageio_model' with the 
        result of the segmentation using the BioImage.IO model. 
        Default is None
    bioimageio_params : _type_, optional
        Parameters used in the segment method of the bioimageio_model. 
        Default is None
    do_preprocess : bool, optional
        If True, pre-process image before segmentation using the filters 
        'remove hot pixels', 'gaussian', and 'sharpen spots' (if requested). 
        Default is True
    do_try_all_thresholds : bool, optional
        If True and thresholding_method is not None, the result of every 
        threshold method available is returned. Default is True
    return_only_segm : bool, optional
        If True, return only the result of the segmentation as numpy.ndarray 
        with the same shape as the input image. Default is False
    pre_aggregated : bool, optional
        If True and do_aggregate is True, run segmentation on the entire input 
        image without aggregating segmented objects. Default is False
    raw_image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        If not None, neural network and BioImage.IO models will segment 
        the raw image. Default is None

    Returns
    -------
    dict or numpy.ndarray
        If return_only_segm is True, the output will the the numpy.ndarray 
        with the segmentation result. 
        If thresholding_method is None and do_try_all_thresholds is True, 
        the output will be a dictionary with keys {'threshol_li', 
        'threshold_isodata', 'threshold_otsu', 'threshold_minimum', 
        'threshold_triangle', 'threshold_mean', 'threshold_yen'} and values 
        the result of each thresholding method. 
        If thresholding_method is not None, the output will be a dictionary 
        with key {'custom'} and value the result of applying the requested 
        thresholding_method. 
        If nnet_model is not None, the output dictionary will include the 
        'neural_network' key with value the result of running the nnet_model
        requested. 
        If bioimageio_model is not None, the output dictionary will include the 
        'bioimageio_model' key with value the result of running the bioimageio_model
        requested. 
        The output dictionary will also include the key 'input_image' with value 
        the pre-processed image. 
    """    
    if raw_image is None:
        raw_image = image.copy()
        
    if do_preprocess:
        image, lab = preprocess_image(
            image, 
            lab=lab, 
            do_remove_hot_pixels=do_remove_hot_pixels, 
            gauss_sigma=gauss_sigma,
            use_gpu=use_gpu, 
            return_lab=True,
            do_sharpen=do_sharpen,
            spots_zyx_radii=spots_zyx_radii,
            logger_func=logger_func
        )

    if lab is None:
        lab = np.ones(image.shape, dtype=np.uint8)
    
    if not np.any(lab):
        result = {
            'input_image': image,
            'Segmentation_data_is_empty': np.zeros(image.shape, dtype=np.uint8)
        }
        return result
    
    if nnet_model is not None and nnet_input_data is None:
        # Use raw image as input to neural network if nnet_input_data is None
        nnet_input_data = raw_image
    
    if do_aggregate:
        zyx_tolerance = transformations.get_expand_obj_delta_tolerance(
            spots_zyx_radii
        )
        result = filters.global_semantic_segmentation(
            image, lab, lineage_table=lineage_table, 
            zyx_tolerance=zyx_tolerance, 
            threshold_func=thresholding_method, 
            logger_func=logger_func, return_image=True,
            keep_input_shape=keep_input_shape,
            nnet_model=nnet_model, nnet_params=nnet_params,
            nnet_input_data=nnet_input_data,
            do_try_all_thresholds=do_try_all_thresholds,
            return_only_output_mask=return_only_segm,
            pre_aggregated=pre_aggregated,
            bioimageio_model=bioimageio_model,
            bioimageio_params=bioimageio_params,
            bioimageio_input_image=raw_image
        )
    else:
        result = filters.local_semantic_segmentation(
            image, lab, threshold_func=thresholding_method, 
            lineage_table=lineage_table, return_image=True,
            nnet_model=nnet_model, nnet_params=nnet_params,
            nnet_input_data=nnet_input_data,
            do_try_all_thresholds=do_try_all_thresholds,
            return_only_output_mask=return_only_segm,
            do_max_proj=True,
            bioimageio_model=bioimageio_model,
            bioimageio_params=bioimageio_params,
            bioimageio_input_image=raw_image
        )
    
    return result

def reference_channel_semantic_segm(
        image, 
        lab=None,
        gauss_sigma=0.0,
        keep_only_largest_obj=False,
        do_remove_hot_pixels=False,
        lineage_table=None,
        do_aggregate=True,
        use_gpu=False,
        logger_func=print,
        thresholding_method=None,
        ridge_filter_sigmas=0,
        keep_input_shape=True,
        do_preprocess=True,
        return_only_segm=False,
        do_try_all_thresholds=True,
        bioimageio_model=None,
        bioimageio_params=None,
        raw_image=None,
        pre_aggregated=False,
        show_progress=False
    ):    
    """Pipeline to segment the reference channel.

    Parameters
    ----------
    image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. Spots will be detected only inside each object. If None, 
        detection will be performed on the entire image. Default is None. 
    gauss_sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard deviations of 
        the Gaussian filter are given for each axis as a sequence, or as a 
        single number, in which case it is equal for all axes. If 0, no 
        gaussian filter is applied. Default is 0.0
    keep_only_largest_obj : bool, optional
        If True, keep only the largest object (determined by connected component
        labelling) per segmented object in lab. Default is False
    do_remove_hot_pixels : bool, optional
        If True, apply a grayscale morphological opening filter before 
        segmenting. Opening can remove small bright spots (i.e. “salt”, or 
        "hot pixels") and connect small dark cracks. Default is False
    lineage_table : pandas.DataFrame, optional
        Table containing parent-daughter relationships. Default is None
        For more details, see the parameter `Table with lineage info end name 
        or path` at the following webpage: 
        https://spotmax.readthedocs.io/parameters_description.html#file-paths-and-channels
    do_aggregate : bool, optional
        If True, perform segmentation on all the cells at once. Default is True
    use_gpu : bool, optional
        If True, some steps will run on the GPU, potentially speeding up the 
        computation. Default is False
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    thresholding_method : {'threshol_li', 'threshold_isodata', 'threshold_otsu',
        'threshold_minimum', 'threshold_triangle', 'threshold_mean',
        'threshold_yen'} or callable, optional
        Thresholding method used to obtain semantic segmentation masks of the 
        spots. If None and do_try_all_thresholds is True, the result of every 
        threshold method available is returned. Default is None
    ridge_filter_sigmas : scalar or sequence of scalars, optional
        Sigmas used as scales of filter. If not 0, filter the image with the 
        Sato tubeness filter. This filter can be used to detect continuous 
        ridges, e.g. mitochondrial network. Default is 0
    keep_input_shape : bool, optional
        If True, return segmentation array with the same shape of the 
        input image. If False, output shape will depend on whether do_aggregate
        is True or False. Default is True
    do_preprocess : bool, optional
        If True, pre-process image before segmentation using the filters 
        'remove hot pixels', 'gaussian', and 'sharpen spots' (if requested). 
        Default is True
    do_try_all_thresholds : bool, optional
        If True and thresholding_method is not None, the result of every 
        threshold method available is returned. Default is True
    return_only_segm : bool, optional
        If True, return only the result of the segmentation as numpy.ndarray 
        with the same shape as the input image. Default is False
    bioimageio_model : Cell-ACDC implementation of any BioImage.IO model, optional
        If not None, the output will include the key 'bioimageio_model' with the 
        result of the segmentation using the BioImage.IO model. 
        Default is None
    bioimageio_params : _type_, optional
        Parameters used in the segment method of the bioimageio_model. 
        Default is None
    pre_aggregated : bool, optional
        If True and do_aggregate is True, run segmentation on the entire input 
        image without aggregating segmented objects. Default is False
    raw_image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        If not None, neural network and BioImage.IO models will segment 
        the raw image. Default is None
    show_progress : bool, optional
        If True, display progressbars. Default is False

    Returns
    -------
    dict or numpy.ndarray
        If return_only_segm is True, the output will the the numpy.ndarray 
        with the segmentation result. 
        If thresholding_method is None and do_try_all_thresholds is True, 
        the output will be a dictionary with keys {'threshol_li', 
        'threshold_isodata', 'threshold_otsu', 'threshold_minimum', 
        'threshold_triangle', 'threshold_mean', 'threshold_yen'} and values 
        the result of each thresholding method. 
        If thresholding_method is not None, the output will be a dictionary 
        with key {'custom'} and value the result of applying the requested 
        thresholding_method. 
        If bioimageio_model is not None, the output dictionary will include the 
        'bioimageio_model' key with value the result of running the bioimageio_model
        requested. 
        The output dictionary will also include the key 'input_image' with value 
        the pre-processed image. 
    """    
    if raw_image is None:
        raw_image = image.copy()
        
    if do_preprocess:
        if show_progress:
            logger_func('Pre-processing image...')
        image, lab = preprocess_image(
            image, 
            lab=lab, 
            do_remove_hot_pixels=do_remove_hot_pixels, 
            gauss_sigma=gauss_sigma,
            use_gpu=use_gpu, 
            logger_func=logger_func,
            return_lab=True
        )
    
    if not np.any(lab):
        empty_segm = np.zeros(image.shape, dtype=np.uint8)
        if thresholding_method is not None or return_only_segm:
            return empty_segm
        else:
            result = {
                'input_image': image,
                'Segmentation_data_is_empty': empty_segm
            }
            return result
    
    if do_aggregate:
        result = filters.global_semantic_segmentation(
            image, lab, lineage_table=lineage_table, 
            threshold_func=thresholding_method, 
            logger_func=logger_func, return_image=True,
            keep_input_shape=keep_input_shape,
            ridge_filter_sigmas=ridge_filter_sigmas,
            return_only_output_mask=return_only_segm,
            do_try_all_thresholds=do_try_all_thresholds,
            pre_aggregated=pre_aggregated,
            bioimageio_model=bioimageio_model,
            bioimageio_params=bioimageio_params,
            bioimageio_input_image=raw_image
        )
    else:
        result = filters.local_semantic_segmentation(
            image, lab, threshold_func=thresholding_method, 
            lineage_table=lineage_table, return_image=True,
            do_max_proj=False, clear_outside_objs=True,
            ridge_filter_sigmas=ridge_filter_sigmas,
            return_only_output_mask=return_only_segm,
            do_try_all_thresholds=do_try_all_thresholds,
            bioimageio_model=bioimageio_model,
            bioimageio_params=bioimageio_params,
            bioimageio_input_image=raw_image
        )
    
    if not keep_only_largest_obj:
        return result
    
    if not np.any(lab):
        return result
    
    if not return_only_segm:
        input_image = result.pop('input_image')
        result = {
            key:filters.filter_largest_sub_obj_per_obj(img, lab) 
            for key, img in result.items()
        }
        result = {**{'input_image': input_image}, **result}
    else:
        result = filters.filter_largest_sub_obj_per_obj(result, lab)
    
    return result

def compute_spots_features(
        image, 
        zyx_coords, 
        spots_zyx_radii,
        sharp_image=None,
        lab=None, 
        do_remove_hot_pixels=False, 
        gauss_sigma=0.0, 
        optimise_with_edt=True,
        use_gpu=True, 
        logger_func=print
    ):
    min_size_spheroid_mask = transformations.get_local_spheroid_mask(
        spots_zyx_radii
    )
    delta_tol = transformations.get_expand_obj_delta_tolerance(spots_zyx_radii)
    
    if optimise_with_edt:
        dist_transform_spheroid = transformations.norm_distance_transform_edt(
            min_size_spheroid_mask
        )
    else:
        dist_transform_spheroid = None
    
    lab, raw_image = transformations.reshape_lab_image_to_3D(lab, image)
    if sharp_image is None:
        sharp_image = image
    _, sharp_image = transformations.reshape_lab_image_to_3D(lab, sharp_image)
        
    preproc_image = preprocess_image(
        raw_image, 
        lab=lab, 
        do_remove_hot_pixels=do_remove_hot_pixels, 
        gauss_sigma=gauss_sigma,
        use_gpu=use_gpu, 
        logger_func=logger_func
    )
    Z, Y, X = lab.shape
    rp = skimage.measure.regionprops(lab)
    pbar = tqdm(
        total=len(rp), ncols=100, desc='Computing features', position=0, 
        leave=False
    )
    dfs_features = []
    keys = []
    for obj_idx, obj in enumerate(rp):
        local_zyx_coords = transformations.to_local_zyx_coords(obj, zyx_coords)
        
        expanded_obj = transformations.get_expanded_obj_slice_image(
            obj, delta_tol, lab
        )
        obj_slice, obj_image, crop_obj_start = expanded_obj
        
        spots_img_obj = preproc_image[obj_slice]
        sharp_spots_img_obj = sharp_image[obj_slice]
        raw_spots_img_obj = raw_image[obj_slice]
        
        df_spots_coords = pd.DataFrame(
            columns=ZYX_LOCAL_COLS, data=local_zyx_coords
        )
        df_spots_coords['Cell_ID'] = obj.label
        df_spots_coords = df_spots_coords.set_index('Cell_ID')
        result = transformations.init_df_features(
            df_spots_coords, obj, crop_obj_start, spots_zyx_radii
        )
        df_obj_features, expanded_obj_coords = result
        if df_obj_features is None:
            continue
        
        # Increment spot_id with previous object
        if obj_idx > 0:
            last_spot_id = dfs_features[obj_idx-1].iloc[-1].name
            df_obj_features['spot_id'] += last_spot_id
        df_obj_features = df_obj_features.set_index('spot_id').sort_index()
        
        keys.append(obj.label)
        dfs_features.append(df_obj_features)
        
        obj_mask = obj_image
        result = transformations.get_spheroids_maks(
            local_zyx_coords, obj_mask.shape, 
            min_size_spheroid_mask=min_size_spheroid_mask, 
            zyx_radii_pxl=spots_zyx_radii
        )
        spheroids_mask, min_size_spheroid_mask = result
        backgr_mask = np.logical_and(obj_mask, ~spheroids_mask)
        
        # Calculate background metrics
        backgr_vals = sharp_spots_img_obj[backgr_mask]
        for name, func in distribution_metrics_func.items():
            df_obj_features.loc[:, f'background_{name}'] = func(backgr_vals)
        
        for row in df_obj_features.itertuples():
            spot_id = row.Index
            zyx_center = tuple(
                [getattr(row, col) for col in ZYX_LOCAL_EXPANDED_COLS]
            )
            slices = transformations.get_slices_local_into_global_3D_arr(
                zyx_center, spots_img_obj.shape, min_size_spheroid_mask.shape
            )
            slice_global_to_local, slice_crop_local = slices
            
            # Background values at spot z-slice
            backgr_mask_z_spot = backgr_mask[zyx_center[0]]
            sharp_spot_obj_z = sharp_spots_img_obj[zyx_center[0]]
            backgr_vals_z_spot = sharp_spot_obj_z[backgr_mask_z_spot]
            
            # Crop masks
            spheroid_mask = min_size_spheroid_mask[slice_crop_local]
            spot_slice = spots_img_obj[slice_global_to_local]

            # Get the sharp spot sliced
            sharp_spot_slice_z = sharp_spot_obj_z[slice_global_to_local[-2:]]
            
            if dist_transform_spheroid is None:
                # Do not optimise for high spot density
                sharp_spot_slice_z_transf = sharp_spot_slice_z
            else:
                dist_transf = dist_transform_spheroid[slice_crop_local]
                sharp_spot_slice_z_transf = (
                    transformations.normalise_spot_by_dist_transf(
                        sharp_spot_slice_z, dist_transf.max(axis=0),
                        backgr_vals_z_spot, how='range'
                ))
            
            # Get spot intensities
            spot_intensities = spot_slice[spheroid_mask]
            spheroid_mask_proj = spheroid_mask.max(axis=0)
            sharp_spot_intensities_z_edt = (
                sharp_spot_slice_z_transf[spheroid_mask_proj]
            )
            
            value = spots_img_obj[zyx_center]
            df_obj_features.at[spot_id, 'spot_preproc_intensity_at_center'] = value
            features.add_distribution_metrics(
                spot_intensities, df_obj_features, spot_id, 
                col_name='spot_preproc_*name_in_spot_minimumsize_vol'
            )
            
            raw_spot_intensities = (
                raw_spots_img_obj[slice_global_to_local][spheroid_mask]
            )
            value = raw_spots_img_obj[zyx_center]
            df_obj_features.at[spot_id, 'spot_raw_intensity_at_center'] = value

            features.add_distribution_metrics(
                raw_spot_intensities, df_obj_features, spot_id, 
                col_name='spot_raw_*name_in_spot_minimumsize_vol'
            )
            
            features.add_ttest_values(
                sharp_spot_intensities_z_edt, backgr_vals_z_spot, 
                df_obj_features, spot_id, name='spot_vs_backgr',
                logger_func=logger_func
            )
            
            features.add_effect_sizes(
                sharp_spot_intensities_z_edt, backgr_vals_z_spot, 
                df_obj_features, spot_id, name='spot_vs_backgr'
            )
    df_features = pd.concat(dfs_features, keys=keys, names=['Cell_ID'])
    return df_features

def spot_detection(
        image,
        spots_segmantic_segm=None,
        detection_method='peak_local_max',
        spot_footprint=None,
        spots_zyx_radii=None,
        return_spots_mask=False
    ):
    """_summary_

    Parameters
    ----------
    image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    spots_segmantic_segm : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        If not None and detection_method is 'peak_local_max', peaks will be 
        searched only where spots_segmantic_segm > 0. Default is None
    detection_method : {'peak_local_max', 'label_prediction_mask'}, optional
        Method used to detect the peaks. Default is 'peak_local_max'
        For more details, see the parameter `Spots detection method` at the 
        following webpage: 
        https://spotmax.readthedocs.io/parameters_description.html#spots-channel
    spot_footprint : numpy.ndarray of bools, optional
        If not None, only one peak is searched in the footprint at every point 
        in the image. Default is None
    spots_zyx_radii : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in pixels. The values are used to determine the spot footprint if 
        spot_footprint is not provided. Default is None
    return_spots_mask : bool, optional
        Used only if detection_method is 'label_prediction_mask'. 
        If True, the second element returned will be a list of region properties 
        (see scikit-image `skimage.measure.regionprops`) with an additional 
        attribute called `zyx_local_center`. Default is False

    Returns
    -------
    2-tuple ((N, 3) numpy.ndarray of ints, list of region properties or None)
        The first element is a (N, 3) array of integers where each row is the 
        (z, y, x) coordinates of one peak. The second element is either None 
        or a list of region properties with an additional 
        attribute called `zyx_local_center`.
    """    
    if spot_footprint is None and spots_zyx_radii is not None:
        zyx_radii_pxl = [val/2 for val in spots_zyx_radii]
        spot_footprint = transformations.get_local_spheroid_mask(
            zyx_radii_pxl
        )
    if spot_footprint is not None:
        spot_footprint = np.squeeze(spot_footprint)
    
    if spots_segmantic_segm is not None:
        spots_segmantic_segm = np.squeeze(spots_segmantic_segm.astype(int))
    
    spots_objs = None
    
    if detection_method == 'peak_local_max':
        spots_coords = skimage.feature.peak_local_max(
            np.squeeze(image), 
            footprint=spot_footprint, 
            labels=spots_segmantic_segm
        )
    elif detection_method == 'label_prediction_mask':
        prediction_lab = skimage.measure.label(spots_segmantic_segm>0)
        prediction_lab_rp = skimage.measure.regionprops(prediction_lab)
        num_spots = len(prediction_lab_rp)
        spots_coords = np.zeros((num_spots, 3), dtype=int)
        if return_spots_mask:
            spots_objs = []
        for s, spot_obj in enumerate(prediction_lab_rp):
            zyx_coords = tuple([round(c) for c in spot_obj.centroid])
            spots_coords[s] = zyx_coords
            if not return_spots_mask:
                continue
            zmin, ymin, xmin, _, _, _ = spot_obj.bbox
            spot_obj.zyx_local_center = (
                zyx_coords[0] - zmin,
                zyx_coords[1] - ymin,
                zyx_coords[2] - xmin
            )
            spots_objs.append(spot_obj)
    return spots_coords, spots_objs

def spots_filter_from_features(
        image, 
    ):
    for obj_idx, obj in enumerate(rp):
        expanded_obj = transformations.get_expanded_obj_slice_image(
            obj, delta_tol, lab
        )
        obj_slice, obj_image, crop_obj_start = expanded_obj

        local_spots_img = spots_img[obj_slice]
        local_sharp_spots_img = sharp_spots_img[obj_slice]

        result = transformations.init_df_features(
            df_spots_coords, obj, crop_obj_start, 
            self.metadata['zyxResolutionLimitPxl']
        )
        df_obj_spots_det, expanded_obj_coords = result
        if df_obj_spots_det is None:
            # 0 spots for this obj (object ID not present in index)
            s = f'  * Object ID {obj.label} = 0 --> 0 (0 iterations)'
            num_spots_filtered_log.append(s)
            continue
        
        # Increment spot_id with previous object
        df_obj_spots_det['spot_id'] += last_spot_id
        df_obj_spots_det = df_obj_spots_det.set_index('spot_id').sort_index()
        
        keys.append((frame_i, obj.label))
        num_spots_detected = len(df_obj_spots_det)
        last_spot_id += num_spots_detected
        
        dfs_spots_det.append(df_obj_spots_det)

        if ref_ch_mask_or_labels is not None:
            local_ref_ch_mask = ref_ch_mask_or_labels[obj_slice]>0
            local_ref_ch_mask = np.logical_and(local_ref_ch_mask, obj_image)
        else:
            local_ref_ch_mask = None

        if ref_ch_img is not None:
            local_ref_ch_img = ref_ch_img[obj_slice]
        else:
            local_ref_ch_img = None
        
        if raw_spots_img is not None:
            raw_spots_img_obj = raw_spots_img[obj_slice]

        df_obj_spots_gop = df_obj_spots_det.copy()
        if do_keep_spots_in_ref_ch:
            df_obj_spots_gop = self._drop_spots_not_in_ref_ch(
                df_obj_spots_gop, local_ref_ch_mask, expanded_obj_coords
            )
        
        debug = False
        i = 0
        while True:     
            num_spots_prev = len(df_obj_spots_gop)
            if num_spots_prev == 0:
                num_spots_filtered = 0
                break
            
            # if obj.label == 36:
            #     debug = True
            #     import pdb; pdb.set_trace()
            
            df_obj_spots_gop = self._compute_obj_spots_metrics(
                local_spots_img, 
                df_obj_spots_gop, 
                obj_image, 
                local_sharp_spots_img, 
                raw_spots_img_obj=raw_spots_img_obj,
                min_size_spheroid_mask=min_size_spheroid_mask, 
                dist_transform_spheroid=dist_transform_spheroid,
                ref_ch_mask_obj=local_ref_ch_mask, 
                ref_ch_img_obj=local_ref_ch_img,
                backgr_is_outside_ref_ch_mask=backgr_is_outside_ref_ch_mask,
                zyx_resolution_limit_pxl=zyx_resolution_limit_pxl,
                debug=debug
            )
            if i == 0:
                # Store metrics at first iteration
                df_obj_spots_det = df_obj_spots_gop.copy()
            
            # if self.debug and obj.label == 79:
            # from . import _debug
            # _debug._spots_filtering(
            #     local_spots_img, df_obj_spots_gop, obj, obj_image
            # )
            
            df_obj_spots_gop = filters.filter_spots_from_features_thresholds(
                df_obj_spots_gop, gop_filtering_thresholds,
                is_spotfit=False, debug=False,
                logger_func=self.logger.info
            )
            num_spots_filtered = len(df_obj_spots_gop)   

            if num_spots_filtered == num_spots_prev or num_spots_filtered == 0:
                # Number of filtered spots stopped decreasing --> stop loop
                break

            i += 1

        nsd, nsf = num_spots_detected, num_spots_filtered
        s = f'  * Object ID {obj.label} = {nsd} --> {nsf} ({i} iterations)'
        num_spots_filtered_log.append(s)

        dfs_spots_gop.append(df_obj_spots_gop)

        pbar.update()
    pbar.close()