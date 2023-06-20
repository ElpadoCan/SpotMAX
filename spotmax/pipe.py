import numpy as np

from . import filters, transformations, config

def spots_semantic_segmentation(
        raw_image, 
        lab=None,
        initial_sigma=0.0,
        spots_zyx_radii=None, 
        do_sharpen=False, 
        lineage_table=None,
        do_aggregate=True,
        use_gpu=False,
        logger_func=print,
        thresholding_method=None,
        keep_input_shape=True
    ):  
    if lab is None:
        lab = np.ones(raw_image.shape, dtype=np.uint8) 
    
    if raw_image.ndim == 2:
        raw_image = raw_image[np.newaxis]
        
    if lab.ndim == 2 and raw_image.ndim == 3:
        # Stack 2D lab into 3D z-stack
        lab = np.array([lab]*len(raw_image))
        
    if do_sharpen:
        image = filters.DoG_spots(
            raw_image, spots_zyx_radii, use_gpu=use_gpu, 
            logger_func=logger_func
        )
    elif initial_sigma>0:
        image = filters.gaussian(
            raw_image, initial_sigma, use_gpu=use_gpu, logger_func=logger_func
        )
    else:
        image = raw_image
    
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