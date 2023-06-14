import numpy as np

import skimage.filters

from . import filters

def spots_instance_segmentation(
        raw_image, 
        initial_sigma=0.0,
        spots_zyx_radii=None, 
        do_sharpen=False, 
        use_gpu=False,
        logger_func=print,
        thresholding_method=''
    ):   
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
    
    if thresholding_method:
        result = filters.threshold(
            image, thresholding_method, logger_func=logger_func
        )
    else:
        result = filters.try_all_thresholds(image, logger_func=print)
    return result