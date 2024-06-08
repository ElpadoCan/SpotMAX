import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from . import printl
from . import ZYX_AGGR_COLS, ZYX_LOCAL_COLS

def _gui_autotune_f1_score(to_debug):
    (method, thresholded, input_image, zz_true, yy_true, 
    xx_true, zz_false, yy_false, xx_false,
    positive_area, f1_score, worker) = to_debug
    
    printl(
        f'{method = }\n'
        f'{zz_true = }\n'
        f'{yy_true = }\n'
        f'{xx_true = }\n'
        f'{zz_false = }\n'
        f'{yy_false = }\n'
        f'{xx_false = }\n'
        f'{positive_area = }\n'
        f'{f1_score = }\n'
    )
    from cellacdc.plot import imshow
    points_coords = np.column_stack((zz_true, yy_true, xx_true))
    imshow(input_image, thresholded, points_coords=points_coords)

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

    from cellacdc.plot import imshow
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
    from cellacdc.plot import imshow
    zyx_cols = ['z_local_expanded', 'y_local_expanded', 'x_local_expanded']
    points_coords = df_obj_spots_gop[zyx_cols].to_numpy()
    data_cols = [
        'spot_vs_backgr_effect_size_hedge',
        'spot_vs_backgr_effect_size_cohen',
        'spot_vs_backgr_effect_size_glass'
    ]
    points_data = (
        df_obj_spots_gop[data_cols]
        .reset_index()
    )
    zyx_cols.extend(data_cols)
    printl(
        df_obj_spots_gop[zyx_cols]
        .sort_values('spot_vs_backgr_effect_size_glass', ascending=False)
    )
    imshow(
        (local_spots_img/local_spots_img.max()*255).astype(np.uint8), 
        obj_image.astype(np.uint8), 
        obj.image.astype(np.uint8),
        points_coords=points_coords, 
        points_data=points_data
    )
    import pdb; pdb.set_trace()

def find_local_peaks(
        image, labels, peaks_coords, valid_peaks_coords, footprint
    ):
    from cellacdc.plot import imshow
    
    df_peaks = pd.DataFrame(data=peaks_coords, columns=['z', 'y', 'x'])
    df_valid = pd.DataFrame(data=valid_peaks_coords, columns=['z', 'y', 'x'])
    
    imshow(image, labels, points_coords_df=df_valid)
    
    import pdb; pdb.set_trace()

def _spots_detection(
        aggregated_lab, labels, aggr_spots_img, df_spots_coords, ID=None
    ):
    from cellacdc.plot import imshow
    if ID is None:
        imshow(
            aggregated_lab, 
            labels, 
            aggr_spots_img,
            points_coords=df_spots_coords[ZYX_AGGR_COLS].to_numpy()
        )
        import pdb; pdb.set_trace()
        return
        
    zz, yy, xx = np.nonzero(aggregated_lab == ID)
    zmin, ymin, xmin = zz.min(), yy.min(), xx.min()
    zmax, ymax, xmax = zz.max(), yy.max(), xx.max()
    bbox_slice = (
        slice(zmin, zmax+1), 
        slice(ymin, ymax+1),
        slice(xmin, xmax+1),
    )
    points_coords = (
        df_spots_coords.loc[ID][ZYX_LOCAL_COLS].to_numpy()
    )
    imshow(
        aggregated_lab[bbox_slice], 
        labels[bbox_slice], 
        aggr_spots_img[bbox_slice],
        points_coords=points_coords
    )
    import pdb; pdb.set_trace()

def _compute_obj_spots_metrics(
        sharp_spot_obj_z, backgr_mask_z_spot, spheroids_mask, yx_center, 
        local_spot_bkgr_mask_z, ID=1, block=True
    ):
    from cellacdc.plot import imshow
    y, x = yx_center
    points_coords = np.array([[y, x]])
    win = imshow(
        sharp_spot_obj_z, 
        backgr_mask_z_spot.astype(np.uint16)*ID,
        spheroids_mask,
        local_spot_bkgr_mask_z,
        points_coords=points_coords,
        block=block, 
        annotate_labels_idxs=[1], 
        axis_titles=[
            'Spot z-slice intensity img', 'Background mask', 'Spots masks', 
            'Local spot background mask'
        ]
    )

def _spotfit_fit(
        gauss3Dmodel, spots_img, fit_coeffs, num_spots_s, 
        num_coeffs, z, y, x, s_data, spots_centers, ID, fit_ids,
        init_guess_s, low_limit, high_limit, fit_idx
    ):
    _shape = (num_spots_s, num_coeffs)
    B_fit = fit_coeffs[-1]
    B_guess = init_guess_s[-1]
    B_min = low_limit[-1]
    B_max = high_limit[-1]
    lstsq_x = fit_coeffs[:-1]
    lstsq_x = lstsq_x.reshape(_shape)
    init_guess_s_2D = init_guess_s[:-1].reshape(_shape)
    low_bounds_2D = low_limit[:-1].reshape(_shape)
    high_bounds_2D = high_limit[:-1].reshape(_shape)
    print('\n\n\n')
    print(f'Cell ID = {ID}')
    print(f'{fit_ids = }')
    iterable = zip(lstsq_x, init_guess_s_2D, low_bounds_2D, high_bounds_2D)
    points_coords = []
    for _x, _init, _l, _h in iterable:
        points_coords.append(_x[:3])
        print('Centers solution: ', _x[:3])
        print('Centers init guess: ', _init[:3])
        print('Centers low bound: ', _l[:3])
        print('Centers high bound: ', _h[:3])
        print('')
        print('Sigma solution: ', _x[3:6])
        print('Sigma init guess: ', _init[3:6])
        print('Sigma low bound: ', _l[3:6])
        print('Sigma high bound: ', _h[3:6])
        print('')
        print('A, B solution: ', _x[6], B_fit)
        print('A, B init guess: ', _init[6], B_guess)
        print('A, B low bound: ', _l[6], B_min)
        print('A, B high bound: ', _h[6], B_max)
        print('')
        print('')
    img = spots_img
    from cellacdc.plot import imshow
    # 3D gaussian evaluated on the entire image
    V_fit = np.zeros_like(spots_img)
    zz, yy, xx = np.nonzero(V_fit==0)
    V_fit[zz, yy, xx] = gauss3Dmodel(
        zz, yy, xx, fit_coeffs, num_spots_s, num_coeffs, 0
    )

    fit_data = gauss3Dmodel(
        z, y, x, fit_coeffs, num_spots_s, num_coeffs, 0
    )
    input_data = img[z, y, x]
    
    square_res = np.square(input_data-fit_data)
    SSE = np.sum(square_res)
    RMSE = np.sqrt(SSE/len(z))
    
    printl(f'{RMSE = }')

    img_fit = np.zeros_like(img)
    img_fit[z,y,x] = fit_data
    img_s = np.zeros_like(img)
    img_s[z,y,x] = s_data
    y_intens = img_s.max(axis=(0, 1))
    y_intens = y_intens[y_intens!=0]
    y_gauss = img_fit.max(axis=(0, 1))
    y_gauss = y_gauss[y_gauss!=0]

    fig, ax = plt.subplots(1,3)
    ax[0].sharex(ax[1])
    ax[0].sharey(ax[1])
    ax[0].imshow(img.max(axis=0))
    _, yyc, xxc = np.array(spots_centers[fit_idx]).T
    ax[0].plot(xxc, yyc, 'r.')
    ax[1].imshow(V_fit.max(axis=0))
    ax[1].plot(xxc, yyc, 'r.')
    ax[2].scatter(range(len(y_intens)), y_intens)
    ax[2].plot(range(len(y_gauss)), y_gauss, c='r')
    plt.show()
    
    imshow(img, V_fit)

    import pdb; pdb.set_trace()

def _spotfit_quality_control(QC_limit, all_gof_metrics):
    fig, ax = plt.subplots(2,4)
    ax = ax.flatten()

    sns.histplot(x=all_gof_metrics[:,0], ax=ax[0])
    sns.boxplot(x=all_gof_metrics[:,0], ax=ax[4])
    ax[0].set_title('Reduced chisquare')

    sns.histplot(x=all_gof_metrics[:,2], ax=ax[1])
    sns.boxplot(x=all_gof_metrics[:,2], ax=ax[5])
    ax[1].set_title('RMSE')

    sns.histplot(x=all_gof_metrics[:,5], ax=ax[2])
    sns.boxplot(x=all_gof_metrics[:,5], ax=ax[6])
    ax[2].axvline(QC_limit, color='r', linestyle='--')
    ax[6].axvline(QC_limit, color='r', linestyle='--')
    ax[2].set_title('NMRSE')

    sns.histplot(x=all_gof_metrics[:,6], ax=ax[3])
    sns.boxplot(x=all_gof_metrics[:,6], ax=ax[7])
    ax[3].set_title('F_NRMSE')

    plt.show()

    import pdb; pdb.set_trace()

def _threshold_spots_img(spots_img):
    import skimage.filters
    from cellacdc.plot import imshow
    threshold_func_names = (
        'threshold_li', 
        'threshold_otsu', 
        'threshold_triangle', 
        'threshold_yen', 
        'threshold_isodata', 
        'threshold_minimum'
    )
    all_thresholded = []
    for func_name in threshold_func_names:
        thresh_func = getattr(skimage.filters, func_name)
        input_img = spots_img.max(axis=0)
        thresh_val = thresh_func(input_img)
        prediction_mask = spots_img>thresh_val
        all_thresholded.append(prediction_mask)
        printl(func_name, thresh_val, input_img.max())
        imshow(input_img, prediction_mask)
    imshow(spots_img, *all_thresholded, axis_titles=('image', *threshold_func_names))
    import pdb; pdb.set_trace()

def _gui_autotune_compute_features(to_debug):
    (result, zz_true, yy_true, xx_true, 
     zz_false, yy_false, xx_false, worker) = to_debug
    from cellacdc.plot import imshow
    points_coords = np.column_stack((zz_true, yy_true, xx_true))
    imshow(result, points_coords=points_coords)
    