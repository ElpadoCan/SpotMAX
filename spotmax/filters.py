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

from . import error_up_str, printl
from . import config

import math
SQRT_2 = math.sqrt(2)

def remove_hot_pixels(image, logger_func=print):
    pbar = tqdm(total=len(image), ncols=100)
    for z, img in enumerate(image):
        image[z] = skimage.morphology.opening(img)
        pbar.update()
    pbar.close()
    return image

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
    
    sigma1 = 2*spots_zyx_radii/(1+SQRT_2)
        
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
    for method in tqdm(methods, ncols=100):
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