import os
import numpy as np

from . import printl

def _peak_local_max(
        folder_name, local_sharp_spots_img, footprint, labels, cell_ID,
        threshold_val, df_obj_spots_gop=None, df_obj_spots_det=None, 
        view=True, save=False
    ):
    if df_obj_spots_det is not None:
        printl(df_obj_spots_det)
    if save:
        from . import data_path
        test_data_path = os.path.join(data_path, folder_name)
        np.save(
            os.path.join(test_data_path, 'local_sharp_spots_img.npy'),
            local_sharp_spots_img
        )
        np.save(
            os.path.join(test_data_path, 'footprint.npy'),
            footprint
        )
        np.save(
            os.path.join(test_data_path, 'labels.npy'),
            labels
        )
    if not view:
        return
    
    if df_obj_spots_gop is not None:
        zyx_cols = ['z_local', 'y_local', 'x_local']
        points_coords = df_obj_spots_gop[zyx_cols].to_numpy()
        data_cols = [
            'spot_vs_backgr_effect_size_hedge',
            'spot_vs_backgr_effect_size_cohen',
            'spot_vs_backgr_effect_size_glass'
        ]
        points_data = df_obj_spots_gop[data_cols].reset_index()
    else:
        points_coords = None
        points_data = None

    from acdctools.plot import imshow
    printl(threshold_val, cell_ID)
    imshow(
        local_sharp_spots_img, 
        local_sharp_spots_img>threshold_val,
        labels, footprint, 
        points_coords=points_coords, 
        points_data=points_data
    )
    import pdb; pdb.set_trace()

def _spots_filtering(local_spots_img, df_obj_spots_gop, obj, obj_image):
    print(f'Cell ID = {obj.label}')
    from acdctools.plot import imshow
    zyx_cols = ['z_local_expanded', 'y_local_expanded', 'x_local_expanded']
    points_coords = df_obj_spots_gop[zyx_cols].to_numpy()
    data_cols = [
        'spot_vs_backgr_effect_size_hedge',
        'spot_vs_backgr_effect_size_cohen',
        'spot_vs_backgr_effect_size_glass'
    ]
    points_data = df_obj_spots_gop[data_cols].reset_index()
    zyx_cols.extend(data_cols)
    printl(df_obj_spots_gop[zyx_cols])
    imshow(
        local_spots_img, obj_image, obj.image,
        points_coords=points_coords, 
        points_data=points_data
    )
    import pdb; pdb.set_trace()