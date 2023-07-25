import numpy as np
import pandas as pd

from . import pipe
from . import printl

class TuneKernel:
    def __init__(self):
        self._image_data = None
        self._segm_data = None
        self._ref_ch_data = None
    
    def set_kwargs(self, kwargs):
        self._kwargs = kwargs
    
    def set_tzyx_true_spots_coords(self, tzyx_coords, crop_to_global_coords):
        self._tzyx_true_spots_coords = tzyx_coords
        self._true_spots_coords_df = pd.DataFrame(
            columns=['frame_i', 'z_global', 'y_global', 'x_global'], 
            data=tzyx_coords
        )
        self._true_spots_coords_df[['z', 'y', 'x']] = (
            self._true_spots_coords_df[['z_global', 'y_global', 'x_global']]
            - crop_to_global_coords
        )
    
    def true_spots_coords_df(self):
        return self._true_spots_coords_df

    def set_tzyx_false_spots_coords(self, tzyx_coords, crop_to_global_coords):
        if len(tzyx_coords) == 0:
            tzyx_coords = None
        self._tzyx_false_spots_coords = tzyx_coords
        self._false_spots_coords_df = pd.DataFrame(
            columns=['frame_i', 'z_global', 'y_global', 'x_global'], 
            data=tzyx_coords
        )
        self._false_spots_coords_df[['z', 'y', 'x']] = (
            self._false_spots_coords_df[['z_global', 'y_global', 'x_global']]
            - crop_to_global_coords
        )
    
    def false_spots_coords_df(self):
        return self._false_spots_coords_df
    
    def ref_ch_endname(self):
        return self._kwargs.get('ref_ch_endname', '')
    
    def set_image_data(self, image_data):
        self._image_data = image_data
    
    def set_segm_data(self, segm_data):
        self._segm_data = segm_data
    
    def set_ref_ch_data(self, ref_ch_data):
        self._ref_ch_data = ref_ch_data
    
    def ref_ch_data(self):
        return self._ref_ch_data

    def segm_data(self):
        return self._segm_data
    
    def image_data(self):
        return self._image_data
    
    def preprocess_image(self, image):
        keys = [
            'do_remove_hot_pixels', 'gauss_sigma', 'use_gpu'
        ]
        kwargs = {key:self._kwargs[key] for key in keys}
        spots_img = pipe.initial_gaussian_filter(image, **kwargs)
        return spots_img
    
    def find_best_threshold_method(self, **kwargs):
        emitDebug = kwargs.get('emitDebug')
        
        false_coords_df = self.false_spots_coords_df().set_index('frame_i')
        true_coords_df = self.true_spots_coords_df()
        f1_scores = []
        positive_areas = []
        methods = []
        frames = []
        for frame_i, true_df in true_coords_df.groupby('frame_i'):
            frames.append(frame_i)
            image = self._image_data[frame_i]
            keys = [
                'lab', 'gauss_sigma', 'spots_zyx_radii', 'do_sharpen',
                'do_remove_hot_pixels', 'lineage_table', 'do_aggregate', 
                'use_gpu'
            ]
            kwargs = {key:self._kwargs[key] for key in keys}
            result = pipe.spots_semantic_segmentation(
                image, keep_input_shape=True, **kwargs
            )
            zz_true = true_df['z'].to_list()
            yy_true = true_df['y'].to_list()
            xx_true = true_df['x'].to_list()
            try:
                zz_false = false_coords_df.loc[frame_i, 'z'].to_list()
                yy_false = false_coords_df.loc[frame_i, 'y'].to_list()
                xx_false = false_coords_df.loc[frame_i, 'x'].to_list()
            except Exception as e:
                zz_false, yy_false, xx_false = [], [], []
            
            for method, thresholded in result.items():
                if method == 'input_image':
                    continue
                true_mask = thresholded[zz_true, yy_true, xx_true]
                false_mask = thresholded[zz_false, yy_false, xx_false]
                tp = np.count_nonzero(true_mask)
                fn = len(true_mask) - tp
                positive_area = np.count_nonzero(thresholded)
                tn = np.count_nonzero(false_mask)
                fp = len(false_mask) - tn
                f1_score = tp/(tp + ((fp+fn)/2))
                f1_scores.append(f1_score)
                methods.append(method)
                positive_areas.append(positive_area)
                
                # input_image = result['input_image']
                # to_debug = (
                #     method, thresholded, input_image, zz_true, yy_true, 
                #     xx_true, zz_false, yy_false, xx_false, tp, fn, tn, fp, 
                #     positive_area, f1_score
                # )
                # emitDebug(to_debug)
            
        df_score = pd.DataFrame({
            'threshold_method': methods,
            'f1_score': f1_scores,
            'positive_area': positive_areas
        }).sort_values(['f1_score', 'positive_area'], ascending=[False, True])
        
        best_method = df_score.iloc[0]['threshold_method']
        return best_method
                
    
    def run(self, logger_func=printl, **kwargs):
        emitDebug = kwargs.get('emitDebug')
        logger_func('Determining optimal thresholding method...')
        self.best_threshold_method = self.find_best_threshold_method(
            emitDebug=emitDebug
        )
        df_spots = self.compute_spots_features()