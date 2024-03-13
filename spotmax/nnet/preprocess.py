import numpy as np
import h5py

from . import transform

def get_scale_factor(initial_size: float, final_size: float) -> float:
    """Function for calculating the scale factor for resizing images.

    Args:
        initial_size (float): Initial size of the image.
        final_size (float): Final size of the image.

    Returns:
        float: Scale factor.
    """
    return initial_size / final_size

def preprocess_data(
        experiment_paths: list,
        fn_pixel_size: int,
        fn_img_size: tuple,
        x_transformer: transform.ImageTransformer,
        y_transformer: transform.ImageTransformer
    ) -> tuple:
    """Function for reading and preprocessing data.

    Args:
        experiment_paths (list): List of paths to the experiments.
        fn_pixel_size (int): Pixel size of the images.
        fn_img_size (tuple): Size of the images.
        x_transformer (it.ImageTransformer): Image transformer for the images.
        y_transformer (it.ImageTransformer): Image transformer for the masks.

    Returns:
        tuple: Tuple with the images and masks.
    """

    X, y = [], []
    for exp in experiment_paths:
        with h5py.File(exp, 'r') as f:
            pixel_size = f["pixel_size"][()]
            scale = get_scale_factor(final_size=fn_pixel_size, initial_size=pixel_size)
            X.append(
                x_transformer.transform(
                    f["X"],
                    scale=scale,
                    final_size=fn_img_size)
            )
            y.append(
                y_transformer.transform(
                    f["y"],
                    scale=scale,
                    final_size=fn_img_size,
                    anti_aliasing=False,
                    order=0)
            )
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)