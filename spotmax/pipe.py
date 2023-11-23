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
            