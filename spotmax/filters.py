from tqdm import tqdm

try:
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    import cupy as cp
    CUPY_INSTALLED = True
except Exception as e:
    CUPY_INSTALLED = False

import skimage.morphology
import skimage.filters

from . import error_up_str, printl

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
            image = cp.array(image)
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