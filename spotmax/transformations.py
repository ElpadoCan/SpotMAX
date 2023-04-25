import numpy as np

import scipy.ndimage
import skimage.measure

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

def get_aggregate_obj_slice(
        obj, max_h_top, max_height, max_h_bottom, max_d_fwd, max_depth, 
        max_d_back, img_data_shape, dx=0
    ):
    Z, Y, X = img_data_shape
    slice_w = obj.slice[2]
    x_left, x_right = slice_w.start-int(dx/2), slice_w.stop+int(dx/2)
    if x_left < 0:
        x_left = 0
    if x_right > X:
        x_right = X

    slice_w = slice(x_left, x_right)
    zmin, ymin, xmin, zmax, ymax, xmax = obj.bbox
    z, y = int(zmin+(zmax-zmin)/2), int(ymin+(ymax-ymin)/2)
    h_top = y - max_h_top
    if h_top < 0:
        # Object slicing would extend negative y
        h_top = 0
        h_bottom = max_height
    else:
        h_bottom = y + max_h_bottom
    
    if h_bottom > Y:
        # Object slicing would extend more than the img data Y
        h_bottom = Y
        h_top = h_bottom - max_height
    
    d_fwd = z - max_d_fwd
    if d_fwd < 0:
        # Object slicing would extend negative z
        d_fwd = 0
        d_top = max_depth
    else:
        # Object slicing would extend more than the img data Z
        d_top = z + max_d_back
    
    if d_top > Z:
        d_top = Z
        d_fwd = d_top - max_depth

    obj_slice = (
        slice(d_fwd, d_top), slice(h_top, h_bottom), slice_w
    )
    return obj_slice

def aggregate_objs(img_data, lab, zyx_tolerance=None):
    # Add tolerance based on resolution limit
    if zyx_tolerance is not None:
        dz, dy, dx = [int(2*np.ceil(dd)) for dd in zyx_tolerance]
    else:
        dz, dy, dx = 0, 0, 0

    # Get max height and total width
    rp_merged = skimage.measure.regionprops(lab)
    tot_width = 0
    max_height = 0
    max_depth = 0
    for obj in rp_merged:
        d, h, w = obj.image.shape
        d, h, w = d+dz, h+dy, w+dx
        if h > max_height:
            max_height = h
        if d > max_depth:
            max_depth = d
        tot_width += w

    Z, Y, X = lab.shape
    if max_depth > Z:
        max_depth = Z
    if max_height > Y:
        max_height = Y

    # Aggregate data horizontally by slicing object centered at 
    # centroid and using largest object as slicing box
    aggr_shape = (max_depth, max_height, tot_width)
    max_h_top = int(max_height/2)
    max_h_bottom = max_height-max_h_top
    max_d_fwd = int(max_depth/2)
    max_d_back = max_depth-max_d_fwd
    aggregated_img = np.zeros(aggr_shape, dtype=img_data.dtype)
    aggregated_img[:] = aggregated_img.min()
    aggregated_lab = np.zeros(aggr_shape, dtype=lab.dtype)
    last_w = 0
    excess_width = 0
    for obj in rp_merged:
        w = obj.image.shape[-1] + dx
        obj_slice = get_aggregate_obj_slice(
            obj, max_h_top, max_height, max_h_bottom, max_d_fwd, max_depth, 
            max_d_back, img_data.shape, dx=dx
        )
        obj_width = obj_slice[-1].stop - obj_slice[-1].start
        excess_width += w - obj_width
        aggregated_img[:, :, last_w:last_w+obj_width] = img_data[obj_slice]
        obj_lab = lab[obj_slice].copy()
        obj_lab[obj_lab != obj.label] = 0
        aggregated_lab[:, :, last_w:last_w+obj_width] = obj_lab
        last_w += w
    if excess_width > 0:
        # Trim excess width result of adding dx to all objects
        aggregated_img = aggregated_img[..., :-excess_width]
        aggregated_lab = aggregated_lab[..., :-excess_width]
    return aggregated_img, aggregated_lab
