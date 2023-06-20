import numpy as np

import scipy.ndimage
import skimage.measure

from . import utils, rng

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

def _aggregate_objs(img_data, lab, zyx_tolerance=None):
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

def _merge_moth_bud(lineage_table, lab, return_bud_images=False):
    if lineage_table is None:
        if return_bud_images:
            return lab, {}
        else:
            return lab

    df_buds = lineage_table[lineage_table.relationship == 'bud']
    moth_IDs = df_buds['relative_ID'].unique()
    df_buds = df_buds.reset_index().set_index('relative_ID')
    if len(moth_IDs) == 0:
        if return_bud_images:
            return lab, {}
        else:
            return lab
    
    lab_merged = lab.copy()
    if return_bud_images:
        bud_images = {}
    for mothID in moth_IDs:
        budID = df_buds.at[mothID, 'Cell_ID']
        lab_merged[lab==budID] = mothID
        if return_bud_images:
            moth_bud_image = np.zeros(lab_merged.shape, dtype=np.uint8)
            moth_bud_image[lab==budID] = 1
            moth_bud_image[lab==mothID] = 1
            moth_bud_obj = skimage.measure.regionprops(moth_bud_image)[0]
            moth_bud_image[lab==mothID] = 0
            bud_image = moth_bud_image[moth_bud_obj.slice] > 0
            bud_images[mothID] = {
                'image': bud_image, 'budID': budID
            }
    if return_bud_images:
        return lab_merged, bud_images
    else:
        return lab_merged

def _separate_moth_buds(lab_merged, bud_images):
    rp = skimage.measure.regionprops(lab_merged)
    for obj in rp:
        if obj.label not in bud_images:
            continue
        bud_info = bud_images.get(obj.label)
        budID = bud_info['budID']
        bud_image = bud_info['image']
        lab_merged[obj.slice][bud_image] = budID
    return lab_merged

def aggregate_objs(
        img_data, lab, zyx_tolerance=None, return_bud_images=True, 
        lineage_table=None
    ):
    lab_merged, bud_images = _merge_moth_bud(
        lineage_table, lab, return_bud_images=return_bud_images
    )
    aggregated_img, aggregated_lab = _aggregate_objs(
        img_data, lab_merged, zyx_tolerance=zyx_tolerance
    )
    aggregated_lab = _separate_moth_buds(
        aggregated_lab, bud_images
    )
    return aggregated_img, aggregated_lab

class SliceImageFromSegmObject:
    def __init__(self, lab, lineage_table=None):
        self._lab = lab
        self._lineage_df = lineage_table
    
    def _get_obj_mask(self, obj):
        lab_obj_image = self._lab == obj.label
        
        if self._lineage_df is None:
            return lab_obj_image, -1
        
        cc_stage = self._lineage_df.at[obj.label, 'cell_cycle_stage']
        if cc_stage == 'G1':
            return lab_obj_image, -1
        
        # Merge mother and daughter when in S phase
        rel_ID = self._lineage_df.at[obj.label, 'relative_ID']
        lab_obj_image = np.logical_or(
            self._lab == obj.label, self._lab == rel_ID
        )
        
        return lab_obj_image, rel_ID
    
    def slice(self, image, obj):
        lab_mask, bud_ID = self._get_obj_mask(obj)
        lab_mask_rp = skimage.measure.regionprops(lab_mask.astype(np.uint8))
        lab_mask_obj = lab_mask_rp[0]
        img_local = image[lab_mask_obj.slice]
        backgr_vals = img_local[~lab_mask_obj.image]
        if backgr_vals.size == 0:
            return img_local, lab_mask_obj.image, bud_ID
        
        backgr_mean = backgr_vals.mean()
        backgr_mean = backgr_mean if backgr_mean>=0 else 0
        backgr_std = backgr_vals.std()/3
        # gamma_shape = np.square(backgr_mean/backgr_std)
        # gamma_scale = np.square(backgr_std)/backgr_mean
        # img_backgr = rng.gamma(
        #     gamma_shape, gamma_scale, size=lab_mask_obj.image.shape
        # )
        img_backgr = rng.normal(
            backgr_mean, backgr_std, size=lab_mask_obj.image.shape
        )
        np.clip(img_backgr, 0, 1, out=img_backgr)

        img_backgr[lab_mask_obj.image] = img_local[lab_mask_obj.image]

        return img_backgr, lab_mask_obj.image, bud_ID

def crop_from_segm_data_info(segm_data, delta_tolerance):
    if segm_data.ndim != 4:
        ndim = segm_data.ndim
        raise TypeError(
            f'Input segmentation data has {ndim} dimensions. Only 4D data allowed. '
            'Make sure to reshape your input data to shape `(Time, Z-slices, Y, X)`.'
        )
    
    T, Z, Y, X = segm_data.shape
    
    segm_time_proj = np.any(segm_data, axis=0).astype(np.uint8)
    segm_time_proj_obj = skimage.measure.regionprops(segm_time_proj)[0]

    # Store cropping coordinates to save correct spots coordinates
    crop_to_global_coords = np.array([
        s.start for s in segm_time_proj_obj.slice
    ]) 
    crop_to_global_coords = crop_to_global_coords - delta_tolerance
    crop_to_global_coords = np.clip(crop_to_global_coords, 0, None)

    crop_stop_coords = np.array([
        s.stop for s in segm_time_proj_obj.slice
    ]) 
    crop_stop_coords = crop_stop_coords + delta_tolerance
    crop_stop_coords = np.clip(crop_stop_coords, None, (Z, Y, X))

    # Build (z,y,x) cropping slices
    z_start, y_start, x_start = crop_to_global_coords        
    z_stop, y_stop, x_stop = crop_stop_coords  
    segm_slice = (
        slice(None), slice(z_start, z_stop), 
        slice(y_start, y_stop), slice(x_start, x_stop)
    )

    pad_widths = []
    for _slice, D in zip(segm_slice, (T, Z, Y, X)):
        _pad_width = [0, 0]
        if _slice.start is not None:
            _pad_width[0] = _slice.start
        if _slice.stop is not None:
            _pad_width[1] = D - _slice.stop
        pad_widths.append(tuple(_pad_width))

    return segm_slice, pad_widths, crop_to_global_coords

def index_aggregated_segm_into_input_image(
        image, lab, aggregated_segm, aggregated_lab
    ):
    segm_lab = np.zeros_like(lab)
    rp = skimage.measure.regionprops(lab)
    obj_idxs = {obj.label:obj for obj in rp}
    aggr_rp = skimage.measure.regionprops(aggregated_lab)
    aggr_segm_labels = aggregated_segm.copy().astype(np.uint32)
    aggr_segm_labels[aggregated_lab == 0] = 0
    aggr_obj_origin = {}
    for aggr_obj in aggr_rp:
        mask = np.logical_and(aggr_obj.image, aggr_segm_labels[aggr_obj.slice])
        aggr_segm_labels[aggr_obj.slice][mask] = aggr_obj.label
        aggr_obj_origin[aggr_obj.label] = aggr_obj.bbox[:3]
    aggr_segm_rp = skimage.measure.regionprops(aggr_segm_labels)
    for aggr_segm_obj in aggr_segm_rp:
        obj = obj_idxs[aggr_segm_obj.label]
        z0, y0, x0 = aggr_obj_origin[aggr_segm_obj.label]
        local_coords = aggr_segm_obj.coords - (z0, y0, x0)
        global_coords = local_coords + obj.bbox[:3]
        zz, yy, xx = global_coords[:,0], global_coords[:,1], global_coords[:,2]
        segm_lab[zz, yy, xx] = obj.label
    return segm_lab