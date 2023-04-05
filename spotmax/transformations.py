import numpy as np

import scipy.ndimage

from . import utils

def get_expanded_obj_slice_image(obj, delta_expand, lab):
    Z, Y, X = lab.shape
    crop_obj_start = np.array([s.start for s in obj.slice]) - delta_expand
    crop_obj_start = np.clip(crop_obj_start, 0, None)

    crop_obj_stop = np.array([s.stop for s in obj.slice]) + delta_expand
    crop_obj_stop = np.clip(crop_obj_stop, None, (Z, Y, X))
    
    obj_slice = (
        slice(crop_obj_start[0], crop_obj_stop[0]), 
        slice(crop_obj_start[1], crop_obj_stop[1]),  
        slice(crop_obj_start[2], crop_obj_stop[2]), 
    )
    obj_lab = lab[obj_slice]
    obj_image = obj_lab==obj.label
    return obj_slice, obj_image, crop_obj_start

def get_expanded_obj(obj, delta_expand, lab):
    expanded_obj = utils._Dummy(name='ExpandedObject')
    expanded_results = get_expanded_obj_slice_image(
        obj, delta_expand, lab
    )
    obj_slice, obj_image, crop_obj_start = expanded_results
    expanded_obj.slice = obj_slice
    expanded_obj.label = obj.label
    expanded_obj.crop_obj_start = crop_obj_start
    expanded_obj.image = obj_image
    return expanded_obj

def expand_labels(label_image, distance=1, zyx_vox_size=None):
    distances, nearest_label_coords = scipy.ndimage.distance_transform_edt(
        label_image==0, return_indices=True, sampling=zyx_vox_size,
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out