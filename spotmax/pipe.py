import numpy as np

from . import filters, transformations, config, printl

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
        keep_input_shape=True
    ):  
    if lab is None:
        lab = np.ones(image.shape, dtype=np.uint8) 
    
    if image.ndim == 2:
        image = image[np.newaxis]
        
    if lab.ndim == 2 and image.ndim == 3:
        # Stack 2D lab into 3D z-stack
        lab = np.array([lab]*len(image))
    
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
    
    if not np.any(lab):
        result = {
            'input_image': image,
            'Segmentation_data_is_empty': np.zeros(image.shape, dtype=np.uint8)
        }
        return result
    
    if do_aggregate:
        result = filters.global_semantic_segmentation(
            image, lab, lineage_table=lineage_table, 
            zyx_tolerance=spots_zyx_radii, 
            thresholding_method=thresholding_method, 
            logger_func=logger_func, return_image=True,
            keep_input_shape=keep_input_shape
        )
    else:
        result = filters.local_semantic_segmentation(
            image, lab, threshold_func=thresholding_method, 
            lineage_table=lineage_table, return_image=True
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
        keep_input_shape=True
    ):
    result = spots_semantic_segmentation(
        image, 
        lab=lab,
        gauss_sigma=gauss_sigma,
        spots_zyx_radii=None, 
        do_sharpen=False, 
        do_remove_hot_pixels=do_remove_hot_pixels,
        lineage_table=lineage_table,
        do_aggregate=do_aggregate,
        use_gpu=use_gpu,
        logger_func=logger_func,
        thresholding_method=thresholding_method,
        keep_input_shape=keep_input_shape
    )
    if not keep_only_largest_obj:
        return result
    
    if not np.any(lab):
        return result
    
    if lab is None:
        lab = np.ones(image.shape, dtype=np.uint8) 
        
    if lab.ndim == 2 and image.ndim == 3:
        # Stack 2D lab into 3D z-stack
        lab = np.array([lab]*len(image))
    
    input_image = result.pop('input_image')
    result = {
        key:filters.filter_largest_sub_obj_per_obj(img, lab) 
        for key, img in result.items()
    }
    result = {**{'input_image': input_image}, **result}
    
    return result