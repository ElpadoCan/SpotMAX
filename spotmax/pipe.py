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
from . import ZYX_LOCAL_COLS, ZYX_LOCAL_EXPANDED_COLS, ZYX_GLOBAL_COLS
from . import features
from . import utils

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
        spots_zyx_radii_pxl=None,
        logger_func=print
    ):
    _, image = transformations.reshape_lab_image_to_3D(lab, image)
        
    if do_remove_hot_pixels:
        image = filters.remove_hot_pixels(image)
    else:
        image = image
    
    if do_sharpen and spots_zyx_radii_pxl is not None:
        image = filters.DoG_spots(
            image, spots_zyx_radii_pxl, use_gpu=use_gpu, 
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
        spots_zyx_radii_pxl=None, 
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
    spots_zyx_radii_pxl : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in pixels. The values are used to determine the sigmas used in the 
        difference-of-gaussians filter that enhances spots-like structures. 
        If None, no filter is applied. Default is None
    do_sharpen : bool, optional
        If True and spots_zyx_radii_pxl is not None, applies a 
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
    result : dict or numpy.ndarray
        If return_only_segm is True, the output will bre the the numpy.ndarray 
        with the segmentation result. 
        
        If thresholding_method is None and do_try_all_thresholds is True, 
        the output will be a dictionary with keys {'threshol_li', 
        'threshold_isodata', 'threshold_otsu', 'threshold_minimum', 
        'threshold_triangle', 'threshold_mean', 'threshold_yen'} and values 
        the result of each thresholding method. 
        
        If thresholding_method is not None, the output will be a dictionary 
        with one key {'custom'} and the result of applying the requested 
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
    lab, image = transformations.reshape_lab_image_to_3D(lab, image)
    
    if raw_image is None:
        raw_image = image.copy()
    else:
        _, raw_image = transformations.reshape_lab_image_to_3D(lab, raw_image)
        
    if do_preprocess:
        image, lab = preprocess_image(
            image, 
            lab=lab, 
            do_remove_hot_pixels=do_remove_hot_pixels, 
            gauss_sigma=gauss_sigma,
            use_gpu=use_gpu, 
            return_lab=True,
            do_sharpen=do_sharpen,
            spots_zyx_radii_pxl=spots_zyx_radii_pxl,
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
            spots_zyx_radii_pxl
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
        single cells. Default is None. 
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
    result : dict or numpy.ndarray
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

def _add_spot_vs_ref_location(ref_ch_mask, zyx_center, df, idx):
    is_spot_in_ref_ch = int(ref_ch_mask[zyx_center] > 0)
    df.at[idx, 'is_spot_inside_ref_ch'] = is_spot_in_ref_ch
    _, dist_2D_from_ref_ch = utils.nearest_nonzero(
        ref_ch_mask[zyx_center[0]], zyx_center[1], zyx_center[2]
    )
    df.at[idx, 'spot_dist_2D_from_ref_ch'] = dist_2D_from_ref_ch

def _compute_obj_spots_features(
        spots_img_obj, 
        df_obj_spots, 
        obj_mask, 
        sharp_spots_img_obj, 
        raw_spots_img_obj=None, 
        min_size_spheroid_mask=None, 
        dist_transform_spheroid=None,
        ref_ch_mask_obj=None, 
        ref_ch_img_obj=None, 
        zyx_resolution_limit_pxl=None, 
        get_backgr_from_inside_ref_ch_mask=False,
        logger_func=print,
        show_progress=True,
        debug=False
    ):
    """_summary_

    Parameters
    ----------
    spots_img_obj : (Z, Y, X) ndarray
        Spots' signal 3D z-stack image sliced at the segmentation object
        level. Note that this is the preprocessed image, i.e., after 
        gaussian filtering, but NOT after sharpening. Sharpening is used 
        only to improve detection. The first dimension must be 
        the number of z-slices.
    df_obj_spots : pandas.DataFrame
        Pandas DataFrame with `spot_id` as index.
    obj_mask : (Z, Y, X) ndarray of dtype bool
        Boolean mask of the segmentation object.
    sharp_spots_img_obj : (Z, Y, X) ndarray
        Spots' signal 3D z-stack image sliced at the segmentation object
        level. Note that this is the preprocessed image, i.e., after 
        gaussian filtering, sharpening etc. It is used to determine the 
        threshold for peak detection and for filtering against background. 
        The first dimension must be the number of z-slices.
    raw_spots_img_obj : (Z, Y, X) ndarray or None, optional
        Raw spots' signal 3D z-stack image sliced at the segmentation
        object level. Note that this is the raw, unprocessed signal. 
        The first dimension must be  the number of z-slices. 
        If None, the features from the raw signal will not be computed.
    min_size_spheroid_mask : (Z, Y, X) ndarray of dtype bool, optional
        The boolean mask of the smallest spot expected. Default is None. 
        This is pre-computed using the resolution limit equations and the 
        pixel size. If None, this will be computed from 
        `zyx_resolution_limit_pxl`.
    dist_transform_spheroid : (Z, Y, X) ndarray, optional
        A distance transform of the `min_size_spheroid_mask`. This will be 
        multiplied by the spots intensities to reduce the skewing effect of 
        neighbouring peaks. 
        It must have the same shape of `min_size_spheroid_mask`.
        If None, normalisation will not be performed.
    ref_ch_mask_obj : (Z, Y, X) ndarray of dtype bool or None, optional
        Boolean mask of the reference channel, e.g., obtained by 
        thresholding. The first dimension must be  the number of z-slices.
        If not None, it is used to compute background metrics, filter 
        and localise spots compared to the reference channel, etc.
    ref_ch_img_obj : (Z, Y, X) ndarray or None, optional
        Reference channel's signal 3D z-stack image sliced at the 
        segmentation object level. Note that this is the preprocessed image,
        i.e., after gaussian filtering, sharpening etc. 
        The first dimension must be the number of z-slices.
        If None, the features from the reference channel signal will not 
        be computed.
    get_backgr_from_inside_ref_ch_mask : bool, optional by default False
        If True, the background mask are made of the pixels that are inside 
        the segmented object but outside of the reference channel mask.
    zyx_resolution_limit_pxl : (z, y, x) tuple or None, optional
        Resolution limit in (z, y, x) direction in pixels. Default is None. 
        If `min_size_spheroid_mask` is None, this will be used to computed 
        the boolean mask of the smallest spot expected.
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    debug : bool, optional
        If True, displays intermediate results. Requires GUI libraries. 
        Default is False.
    """ 

    local_peaks_coords = df_obj_spots[ZYX_LOCAL_EXPANDED_COLS].to_numpy()
    result = transformations.get_spheroids_maks(
        local_peaks_coords, obj_mask.shape, 
        min_size_spheroid_mask=min_size_spheroid_mask, 
        zyx_radii_pxl=zyx_resolution_limit_pxl,
        debug=debug
    )
    spheroids_mask, min_size_spheroid_mask = result
    # if debug:
    #     from cellacdc.plot import imshow
    #     imshow(
    #         spheroids_mask, spots_img_obj, 
    #         points_coords=local_peaks_coords
    #     )
    #     import pdb; pdb.set_trace()

    # Check if spots_img needs to be normalised
    if get_backgr_from_inside_ref_ch_mask:
        backgr_mask = np.logical_and(ref_ch_mask_obj, ~spheroids_mask)
        normalised_result = transformations.normalise_img(
            ref_ch_img_obj, backgr_mask, raise_if_norm_zero=False
        )
        normalised_ref_ch_img_obj, ref_ch_norm_value = normalised_result
        df_obj_spots.loc[:, 'ref_ch_normalising_value'] = ref_ch_norm_value
        normalised_result = transformations.normalise_img(
            spots_img_obj, backgr_mask, raise_if_norm_zero=True
        )
        normalised_spots_img_obj, spots_norm_value = normalised_result
        df_obj_spots.loc[:, 'spots_normalising_value'] = spots_norm_value
    else:
        backgr_mask = np.logical_and(obj_mask, ~spheroids_mask)
        normalised_spots_img_obj = spots_img_obj
        normalised_ref_ch_img_obj = ref_ch_img_obj

    # Calculate background metrics
    backgr_vals = sharp_spots_img_obj[backgr_mask]
    for name, func in distribution_metrics_func.items():
        df_obj_spots.loc[:, f'background_{name}'] = func(backgr_vals)
    
    if raw_spots_img_obj is None:
        raw_spots_img_obj = spots_img_obj
    
    if show_progress:
        pbar_desc = 'Computing spots features'
        pbar = tqdm(
            total=len(df_obj_spots), ncols=100, desc=pbar_desc, position=3, 
            leave=False
        )
    
    spot_ids_to_drop = []
    for row in df_obj_spots.itertuples():
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
        
        if len(backgr_vals_z_spot) == 0:
            # This is most likely because the reference channel mask at 
            # center z-slice is smaller than the spot resulting 
            # in no background mask (since the background is outside of 
            # the spot but inside the ref. ch. mask) --> there is not 
            # enough ref. channel to consider this a valid spot.
            spot_ids_to_drop.append(spot_id)
            continue
        
        if debug:
            print('')
            zyx_local = tuple(
                [getattr(row, col) for col in ZYX_LOCAL_COLS]
            )
            zyx_global = tuple(
                [getattr(row, col) for col in ZYX_GLOBAL_COLS]
            )
            print(f'Local coordinates = {zyx_local}')
            print(f'Global coordinates = {zyx_global}')
            print(f'Spot raw intensity at center = {raw_spots_img_obj[zyx_center]}')
            from ._debug import _compute_obj_spots_metrics
            win = _compute_obj_spots_metrics(
                sharp_spot_obj_z, backgr_mask_z_spot, 
                spheroids_mask[zyx_center[0]], 
                zyx_center[1:], block=False
            )

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
        df_obj_spots.at[spot_id, 'spot_preproc_intensity_at_center'] = value
        features.add_distribution_metrics(
            spot_intensities, df_obj_spots, spot_id, 
            col_name='spot_preproc_*name_in_spot_minimumsize_vol'
        )
        
        if raw_spots_img_obj is None:
            raw_spot_intensities = spot_intensities
        else:
            raw_spot_intensities = (
                raw_spots_img_obj[slice_global_to_local][spheroid_mask]
            )
            value = raw_spots_img_obj[zyx_center]
            df_obj_spots.at[spot_id, 'spot_raw_intensity_at_center'] = value

            features.add_distribution_metrics(
                raw_spot_intensities, df_obj_spots, spot_id, 
                col_name='spot_raw_*name_in_spot_minimumsize_vol'
            )

        # When comparing to the background we use the sharpened image 
        # at the center z-slice of the spot
        features.add_ttest_values(
            sharp_spot_intensities_z_edt, backgr_vals_z_spot, 
            df_obj_spots, spot_id, name='spot_vs_backgr',
            logger_func=logger_func
        )
        
        features.add_effect_sizes(
            sharp_spot_intensities_z_edt, backgr_vals_z_spot, 
            df_obj_spots, spot_id, name='spot_vs_backgr',
            debug=debug
        )
        
        if ref_ch_img_obj is None:
            # Raw reference channel not present --> continue
            pbar.update()
            continue

        normalised_spot_intensities, normalised_ref_ch_intensities = (
            features.get_normalised_spot_ref_ch_intensities(
                normalised_spots_img_obj, normalised_ref_ch_img_obj,
                spheroid_mask, slice_global_to_local
            )
        )
        features.add_ttest_values(
            normalised_spot_intensities, normalised_ref_ch_intensities, 
            df_obj_spots, spot_id, name='spot_vs_ref_ch',
            logger_func=logger_func
        )
        features.add_effect_sizes(
            normalised_spot_intensities, normalised_ref_ch_intensities, 
            df_obj_spots, spot_id, name='spot_vs_ref_ch'
        )
        _add_spot_vs_ref_location(
            ref_ch_mask_obj, zyx_center, df_obj_spots, spot_id
        )                
        
        value = ref_ch_img_obj[zyx_center]
        df_obj_spots.at[spot_id, 'ref_ch_raw_intensity_at_center'] = value

        ref_ch_intensities = (
            ref_ch_img_obj[slice_global_to_local][spheroid_mask]
        )
        features.add_distribution_metrics(
            ref_ch_intensities, df_obj_spots, spot_id, 
            col_name='ref_ch_raw_*name_in_spot_minimumsize_vol'
        )
        if show_progress:
            pbar.update()
    if show_progress:
        pbar.close()
    if spot_ids_to_drop:
        df_obj_spots = df_obj_spots.drop(index=spot_ids_to_drop)
    return df_obj_spots

def compute_spots_features(
        image, 
        zyx_coords, 
        spots_zyx_radii_pxl,
        min_size_spheroid_mask=None,
        delta_tol=None,
        sharp_image=None,
        lab=None, 
        dist_transform_spheroid=None,
        do_remove_hot_pixels=False, 
        gauss_sigma=0.0, 
        optimise_with_edt=True,
        use_gpu=True, 
        logger_func=print
    ):
    if min_size_spheroid_mask is None:
        min_size_spheroid_mask = transformations.get_local_spheroid_mask(
            spots_zyx_radii_pxl
        )
    
    if delta_tol is None:
        delta_tol = transformations.get_expand_obj_delta_tolerance(
            spots_zyx_radii_pxl
        )
    
    if optimise_with_edt and dist_transform_spheroid is None:
        dist_transform_spheroid = transformations.norm_distance_transform_edt(
            min_size_spheroid_mask
        )
    
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
            df_spots_coords, obj, crop_obj_start, spots_zyx_radii_pxl
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
        obj_mask = obj_image
        df_obj_features = _compute_obj_spots_features(
            spots_img_obj, 
            df_obj_features, 
            obj_mask, 
            sharp_spots_img_obj, 
            raw_spots_img_obj=raw_spots_img_obj, 
            min_size_spheroid_mask=min_size_spheroid_mask, 
            dist_transform_spheroid=dist_transform_spheroid,
            ref_ch_mask_obj=None, 
            ref_ch_img_obj=None, 
            zyx_resolution_limit_pxl=spots_zyx_radii_pxl, 
            get_backgr_from_inside_ref_ch_mask=False,
            logger_func=logger_func,
            show_progress=True,
            debug=False
        )
        dfs_features.append(df_obj_features)
        pbar.update()
    pbar.close()
    df_features = pd.concat(dfs_features, keys=keys, names=['Cell_ID'])
    return df_features

def spot_detection(
        image,
        spots_segmantic_segm=None,
        detection_method='peak_local_max',
        spot_footprint=None,
        spots_zyx_radii_pxl=None,
        return_spots_mask=False,
        lab=None,
        return_df=False
    ):
    """Detect spots and return their coordinates

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
    spots_zyx_radii_pxl : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in pixels. The values are used to determine the spot footprint if 
        spot_footprint is not provided. Default is None
    return_spots_mask : bool, optional
        Used only if detection_method is 'label_prediction_mask'. 
        If True, the second element returned will be a list of region properties 
        (see scikit-image `skimage.measure.regionprops`) with an additional 
        attribute called `zyx_local_center`. Default is False
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. It will be used to create the pandas.DataFrame with 
        spots coordinates per object (if `return_df` is True).
        If None, it will be generated with one object covering the entire image. 
        Default is None.
    return_df : bool, optional
        If True, returns a pandas DataFrame. 
        Default is False

    Returns
    -------
    spots_coords : (N, 3) numpy.ndarray of ints
        (N, 3) array of integers where each row is the (z, y, x) coordinates 
        of one peak. Returned only if `return_df` is False
    
    df_coords : pandas.DataFrame with Cell_ID as index and columns 
        {'z', 'y', 'x'} with the detected spots coordinates.
        Returned only if `return_df` is True
    
    spots_objs : list of region properties or None
        List of region properties where each element has an additional attribute 
        called `zyx_local_center`. None if `return_spots_mask` is False.
    """        
    if spot_footprint is None and spots_zyx_radii_pxl is not None:
        zyx_radii_pxl = [val/2 for val in spots_zyx_radii_pxl]
        spot_footprint = transformations.get_local_spheroid_mask(
            zyx_radii_pxl
        )
    if spot_footprint is not None:
        spot_footprint = np.squeeze(spot_footprint)
    
    if spots_segmantic_segm is not None:
        spots_segmantic_segm = np.squeeze(spots_segmantic_segm.astype(int))
    else:
        spots_segmantic_segm = np.ones(np.squeeze(image).shape, int)
    
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
    
    if return_df:
        if lab is None:
            raise NameError(
                'With `return_df=True`, `lab` cannot be None.'
            )
        lab, _ = transformations.reshape_lab_image_to_3D(lab, image)
        df_coords = transformations.from_spots_coords_arr_to_df(
            spots_coords, lab
        )
        return df_coords, spots_objs
    else:
        return spots_coords, spots_objs

def spots_calc_features_and_filter(
        image, 
        spots_zyx_radii_pxl,
        df_spots_coords,
        frame_i=0,
        sharp_spots_image=None,
        lab=None,
        rp=None,
        gop_filtering_thresholds=None,
        delta_tol=None,   
        raw_image=None,
        ref_ch_mask_or_labels=None, 
        ref_ch_img=None,   
        keep_only_spots_in_ref_ch=False,
        min_size_spheroid_mask=None,
        optimise_for_high_spot_density=False,
        dist_transform_spheroid=None,
        get_backgr_from_inside_ref_ch_mask=False,
        show_progress=True,
        verbose=True,
        logger_func=print
    ):
    """Calculate spots features and filter valid spots based on 
    `gop_filtering_thresholds`.

    Parameters
    ----------
    image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    spots_zyx_radii_pxl : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in pixels. The values are used to build the ellipsoid mask centered at 
        each spot. The volume of the ellipsoid is then used for those aggregated 
        metrics like the mean intensity in the spot.
    df_spots_coords : pandas.DataFrame
        DataFrame with Cell_ID as index and the columns {'z', 'y', 'x'} which 
        are the coordinates of the spots in `image`. 
    frame_i : int, optional
        Frame index in timelapse data. Default is 0
    sharp_spots_image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        Optional image that was filtered to enhance the spots (e.g., using 
        spotmax.filters.DoG_spots). This image will be used for those features 
        that requires comparing the spot's signal to a reference signal 
        (background or reference channel). If None, `image` will be used 
        instead. Default is None
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. If None, it will be generated with one object covering 
        the entire image. Default is None.
    rp : list of skimage.measure.RegionProperties, optional
        If not None, list of properties of objects in `lab` as returned by 
        skimage.measure.regionprops(lab). If None, this will be computed 
        with `skimage.measure.regionprops(lab)`. Default is None
    gop_filtering_thresholds : dict of {'feature_name': (min_value, max_value)}, optional
        Features and their maximum and minimum values to filter valid spots. 
        A spot is valid when `feature_name` is greater than `min_value` and 
        lower than `max_value`. If a value is None it means there is no minimum 
        or maximum threshold. Default is None
    delta_tol : (z, y, x) sequence of floats, optional
        If not None, these values will be used to enlarge the segmented objects. 
        It will prevent clipping the spots masks for those spots whose intensities 
        bleed outside of the object (e.g., single cell). Default is None
    raw_image : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        Optional image to calculate features based on the raw image. The name 
        of these features will have the text '_raw_'. Default is None
    ref_ch_mask_or_labels : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Instance or semantic segmentation of the reference channel. If not None, 
        this is used to calculate the background intensity inside the segmented 
        object from `lab` but outside of the reference channel mask. 
        Default is None
    ref_ch_img : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray, optional
        Reference channel image. Default is None
    keep_only_spots_in_ref_ch : bool, optional
        If True, drops the spots that are outside of the reference channel mask. 
        Default is False
    min_size_spheroid_mask : (M, N) numpy.ndarray or (K, M, N) numpy.ndarray or bools, optional
        Ellipsoid mask used to calcualte those aggregated features like the 
        mean intensity in each spot. 
        If None, this will be created from `spots_zyx_radii_pxl`. 
        Default is None
    optimise_for_high_spot_density : bool, optional
        If True and `dist_transform_spheroid` is None, then `dist_transform_spheroid`
        will be initialized with the euclidean distance transform of 
        `min_size_spheroid_mask`.
    dist_transform_spheroid : (M, N) numpy.ndarray or (K, M, N) numpy.ndarray of floats, optional
        Optional probability map that will be multiplicated to each spot's 
        intensities. An example is the euclidean distance tranform 
        (normalised to the range 0-1). This is useful to reduce the influence 
        of bright neighbouring spots on dimmer spots since the intensities of the 
        bright spot can bleed into the edges of the dimmer spot skewing its 
        metrics like the mean intensity. 
        If None and `optimise_for_high_spot_density` is True, this will be 
        initialized with the euclidean distance transform of 
        `min_size_spheroid_mask`. Default is None
    get_backgr_from_inside_ref_ch_mask : bool, optional
        If True, the background will be determined from the pixels that are
        outside of the spots, but inside the reference channel mask. 
        Default is False
    show_progress : bool, optional
        If True, display progressbars. Default is False
    verbose : bool, optional
        If True, additional information text will be printed to the terminal. 
        Default is True
    logger_func : callable, optional
        Function used to print or log process information. Default is print

    Returns
    -------
    keys : list of 2-tuple (int, int) 
        List of keys that can be used to concatenate the 
        dataframes with 
        `pandas.concat(dfs_spots_gop, keys=keys, names=['frame_i', 'Cell_ID'])` 
    dfs_spots_det : list of pandas.DataFrames
        List of DataFrames with the features columns 
        for each frame and ID of the segmented objects in `lab` 
    dfs_spots_gop : list of pandas.DataFrames
        Same as `dfs_spots_det` but with only the valid spots 
    """    
    if verbose:
        print('')
        logger_func('Filtering valid spots...')
    
    if gop_filtering_thresholds is None:
        gop_filtering_thresholds = {}
    
    if lab is None:
        lab = np.zeros(image.shape, dtype=np.uint8)
    
    lab, image = transformations.reshape_lab_image_to_3D(lab, image)
    
    if rp is None:
        rp = skimage.measure.regionprops(lab)
    
    if delta_tol is None:
        delta_tol = transformations.get_expand_obj_delta_tolerance(
            spots_zyx_radii_pxl
        )
    
    if min_size_spheroid_mask is None:
        min_size_spheroid_mask = transformations.get_local_spheroid_mask(
            spots_zyx_radii_pxl
        )
    
    if optimise_for_high_spot_density and dist_transform_spheroid is None:
        dist_transform_spheroid = transformations.norm_distance_transform_edt(
            min_size_spheroid_mask
        )
    
    if sharp_spots_image is None:
        sharp_spots_image = image
    
    if show_progress:
        desc = 'Filtering spots'
        pbar = tqdm(
            total=len(rp), ncols=100, desc=desc, position=3, leave=False
        )
    
    keys = []
    dfs_spots_det = []
    dfs_spots_gop = []
    num_spots_filtered_log = []
    last_spot_id = 0
    for o, obj in enumerate(rp):
        expanded_obj = transformations.get_expanded_obj_slice_image(
            obj, delta_tol, lab
        )
        obj_slice, obj_image, crop_obj_start = expanded_obj

        local_spots_img = image[obj_slice]
        local_sharp_spots_img = sharp_spots_image[obj_slice]

        result = transformations.init_df_features(
            df_spots_coords, obj, crop_obj_start, spots_zyx_radii_pxl
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
        
        if raw_image is not None:
            raw_spots_img_obj = raw_image[obj_slice]

        df_obj_spots_gop = df_obj_spots_det.copy()
        if keep_only_spots_in_ref_ch:
            df_obj_spots_gop = filters.drop_spots_not_in_ref_ch(
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

            df_obj_spots_gop = _compute_obj_spots_features(
                local_spots_img, 
                df_obj_spots_gop, 
                obj_image, 
                local_sharp_spots_img, 
                raw_spots_img_obj=raw_spots_img_obj,
                min_size_spheroid_mask=min_size_spheroid_mask, 
                dist_transform_spheroid=dist_transform_spheroid,
                ref_ch_mask_obj=local_ref_ch_mask, 
                ref_ch_img_obj=local_ref_ch_img,
                get_backgr_from_inside_ref_ch_mask=get_backgr_from_inside_ref_ch_mask,
                zyx_resolution_limit_pxl=spots_zyx_radii_pxl,
                debug=debug
            )
            if i == 0:
                # Store metrics at first iteration
                dfs_spots_det[o] = df_obj_spots_gop.copy()
 
            # from . import _debug
            # _debug._spots_filtering(
            #     local_spots_img, df_obj_spots_gop, obj, obj_image
            # )
            
            df_obj_spots_gop = filters.filter_spots_from_features_thresholds(
                df_obj_spots_gop, gop_filtering_thresholds,
                is_spotfit=False, debug=False,
                logger_func=logger_func
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

        if show_progress:
            pbar.update()
    if show_progress:
        pbar.close()
    
    if verbose:
        print('')
        print('*'*60)
        info = '\n'.join(num_spots_filtered_log)
        logger_func(
            f'Number of spots after filtering valid spots:\n{info}'
        )
        print('-'*60)
        
    return keys, dfs_spots_det, dfs_spots_gop

def spotfit(
        kernel,
        spots_img, 
        df_spots, 
        zyx_voxel_size, 
        zyx_spot_min_vol_um,
        delta_tol=None,
        rp=None, 
        lab=None, 
        frame_i=0, 
        ref_ch_mask_or_labels=None, 
        use_gpu=False,
        show_progress=True,
        logger_func=print,
        xy_center_half_interval_val=0.1, 
        z_center_half_interval_val=0.2, 
        sigma_x_min_max_expr=('0.5', 'spotsize_yx_radius_pxl'),
        sigma_y_min_max_expr=('0.5', 'spotsize_yx_radius_pxl'),
        sigma_z_min_max_expr=('0.5', 'spotsize_z_radius_pxl'),
        A_min_max_expr=('0.0', 'spotsize_A_max'),
        B_min_max_expr=('spot_B_min', 'inf'),
    ):
    """Run spotFIT (fitting 3D gaussian curves) and get the related features

    Parameters
    ----------
    kernel : spotmax.core.SpotFIT
        Initialized SpoFIT class defined in spotmax.core.SpotFIT
    spots_img : (Y, X) numpy.ndarray or (Z, Y, X) numpy.ndarray
        Input 2D or 3D image.
    df_spots : pandas.DataFrame
        DataFrame with Cell_ID as index and the columns {'z', 'y', 'x'} which 
        are the coordinates of the spots in `spots_img` to fit. 
    zyx_voxel_size : sequence of 3 floats (z, y, x)
        Voxel size in μm/pixel
    zyx_spot_min_vol_um : (z, y, x) sequence of floats, optional
        Rough estimation of the expected spot radii in z, y, and x direction
        in μm. The values are used to build starting masks for the spotSIZE step.
        The spotSIZE step will determine the extent of each spot, i.e., the pixels 
        that will be the input for the fitting procedure.
    delta_tol : (z, y, x) sequence of floats, optional
        If not None, these values will be used to enlarge the segmented objects. 
        It will enable correct fitting of those spots whose intensities 
        bleed outside of the object (e.g., single cell). Default is None
    rp : list of skimage.measure.RegionProperties, optional
        If not None, list of properties of objects in `lab` as returned by 
        skimage.measure.regionprops(lab). If None, this will be computed 
        with `skimage.measure.regionprops(lab)`. Default is None
    lab : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Optional input segmentation image with the masks of the objects, i.e. 
        single cells. Default is None. 
    frame_i : int, optional
        Frame index in timelapse data. Default is 0
    ref_ch_mask_or_labels : (Y, X) numpy.ndarray of ints or (Z, Y, X) numpy.ndarray of ints, optional
        Instance or semantic segmentation of the reference channel. 
        Default is None
    use_gpu : bool, optional
        If True, some steps will run on the GPU, potentially speeding up the 
        computation. Default is False
    show_progress : bool, optional
        If True, display progressbars. Default is False
    logger_func : callable, optional
        Function used to print or log process information. Default is print
    xy_center_half_interval_val : float, optional
        Half interval width for bounds on x and y center coordinates during fit. 
        Default is 0.1
    z_center_half_interval_val : float, optional
        Half interval width for bounds on z center coordinate during fit. 
        Default is 0.2
    sigma_x_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `sigma_x` fitting paramter. The expression 
        can be any text that can be evaluated by `pandas.eval`. 
        Default is ('0.5', 'spotsize_yx_radius_pxl').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    sigma_y_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `sigma_y` fitting paramter. The expression 
        can be any text that can be evaluated by `pandas.eval`. 
        Default is ('0.5', 'spotsize_yx_radius_pxl').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
        Default is ('0.5', 'spotsize_yx_radius_pxl').
    sigma_z_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `sigma_z` fitting paramter. The expression 
        can be any text that can be evaluated by `pandas.eval`. 
        Default is ('0.5', 'spotsize_z_radius_pxl').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    A_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `A_fit` (peak amplitude) fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is ('0.0', 'spotsize_A_max').
        More details here: 
        https://pandas.pydata.org/docs/reference/api/pandas.eval.html
    B_min_max_expr : 2-tuple of str, optional
        Expressions to evaluate with `pandas.eval` to determine minimum and 
        maximum values for bounds on `B_fit` (background) fitting paramter. 
        The expression can be any text that can be evaluated by `pandas.eval`. 
        Default is ('spot_B_min', 'inf').
        More details here: 
        ttps://pandas.pydata.org/docs/reference/api/pandas.eval.html
        
    Returns
    -------
    keys : list of 2-tuple (int, int) 
        List of keys that can be used to concatenate the 
        dataframes with 
        `pandas.concat(dfs_spots_spotfit, keys=keys, names=['frame_i', 'Cell_ID'])`
    
    dfs_spots_spotfit : list of pandas.DataFrames
        List of DataFrames with additional spotFIT features columns 
        for each frame and ID of the segmented objects in `lab`
    """    
    if lab is None:
        lab = np.ones(spots_img.shape, dtype=np.uint8)

    if rp is None:
        rp = skimage.measure.regionprops(lab)
    
    if delta_tol is None:
        delta_tol = (0, 0, 0)
    
    dfs_spots_spotfit = []
    keys = []
    
    if show_progress:
        desc = 'Measuring spots'
        pbar = tqdm(
            total=len(rp), ncols=100, desc=desc, position=3, leave=False
        )
    for obj in rp:
        if obj.label not in df_spots.index:
            continue
        expanded_obj = transformations.get_expanded_obj(obj, delta_tol, lab)
        kernel.set_args(
            expanded_obj, 
            spots_img, 
            df_spots, 
            zyx_voxel_size, 
            zyx_spot_min_vol_um, 
            xy_center_half_interval_val=xy_center_half_interval_val, 
            z_center_half_interval_val=z_center_half_interval_val, 
            sigma_x_min_max_expr=sigma_x_min_max_expr,
            sigma_y_min_max_expr=sigma_y_min_max_expr,
            sigma_z_min_max_expr=sigma_z_min_max_expr,
            A_min_max_expr=A_min_max_expr,
            B_min_max_expr=B_min_max_expr,
            ref_ch_mask_or_labels=ref_ch_mask_or_labels,
            use_gpu=use_gpu, 
            logger_func=logger_func
        )
        kernel.fit()
        dfs_spots_spotfit.append(kernel.df_spotFIT_ID)
        keys.append((frame_i, obj.label))
        if show_progress:
            pbar.update()
    if show_progress:
        pbar.close()
    
    return keys, dfs_spots_spotfit