import numpy as np
import pandas as pd

import cellacdc.myutils

from . import pipe
from . import metrics
from . import printl
from . import filters
from . import ZYX_GLOBAL_COLS

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
        self._image_data = cellacdc.myutils.img_to_float(image_data)
    
    def set_segm_data(self, segm_data):
        self._segm_data = segm_data
    
    def set_ref_ch_data(self, ref_ch_data):
        self._ref_ch_data = cellacdc.myutils.img_to_float(ref_ch_data)
    
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
        spots_img = pipe.preprocess_image(image, **kwargs)
        return spots_img
    
    def input_kwargs(self):
        return self._kwargs
    
    def _iter_frames(self):
        false_coords_df = self.false_spots_coords_df().set_index('frame_i')
        true_coords_df = self.true_spots_coords_df()
        for frame_i, true_df in true_coords_df.groupby('frame_i'):
            keys = [
                'lab', 'gauss_sigma', 'spots_zyx_radii', 'do_sharpen',
                'do_remove_hot_pixels', 'lineage_table', 'do_aggregate', 
                'use_gpu'
            ]
            input_kwargs = {key:self._kwargs[key] for key in keys}
            zz_true = true_df['z'].to_list()
            yy_true = true_df['y'].to_list()
            xx_true = true_df['x'].to_list()
            try:
                zz_false = false_coords_df.loc[frame_i, 'z'].to_list()
                yy_false = false_coords_df.loc[frame_i, 'y'].to_list()
                xx_false = false_coords_df.loc[frame_i, 'x'].to_list()
            except Exception as e:
                zz_false, yy_false, xx_false = [], [], []
            yield (
                input_kwargs, zz_true, yy_true, xx_true, zz_false, 
                yy_false, xx_false
            )
    
    def find_best_threshold_method(self, **kwargs):
        emitDebug = kwargs.get('emitDebug')

        f1_scores = []
        positive_areas = []
        methods = []
        frames = []
        for frame_i, inputs in enumerate(self._iter_frames()):
            (segm_kwargs, zz_true, yy_true, xx_true, 
            zz_false, yy_false, xx_false) = inputs
            
            image = self._image_data[frame_i]
            result = pipe.spots_semantic_segmentation(
                image, keep_input_shape=True, **segm_kwargs
            )
            frames.append(frame_i)
            
            for method, thresholded in result.items():
                if method == 'input_image':
                    continue
                true_mask = thresholded[zz_true, yy_true, xx_true]
                false_mask = thresholded[zz_false, yy_false, xx_false]
                f1_score = metrics.semantic_segm_f1_score(true_mask, false_mask)
                positive_area = np.count_nonzero(thresholded)
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
    
    def find_features_range(self, **kwargs):
        emitDebug = kwargs.get('emitDebug')
        
        dfs = []
        frames = []
        for frame_i, inputs in enumerate(self._iter_frames()):
            (input_kwargs, zz_true, yy_true, xx_true, 
            zz_false, yy_false, xx_false) = inputs 
            
            image = self._image_data[frame_i] 
            zyx_coords_true = np.column_stack(
                (zz_true, yy_true, xx_true)
            ).astype(int)
            zyx_coords_false = np.column_stack(
                (zz_false, yy_false, xx_false)
            ).astype(int)
            zyx_coords = np.concatenate(
                (zyx_coords_true, zyx_coords_false), axis=0
            )
            
            kwargs_keys = [
                'lab', 'do_remove_hot_pixels', 'gauss_sigma', 'use_gpu', 
                'use_gpu'
            ]
            features_kwargs = {key:input_kwargs[key] for key in kwargs_keys}
            spots_zyx_radii = input_kwargs['spots_zyx_radii']
            
            if input_kwargs['do_sharpen']:
                sharp_image = filters.DoG_spots(
                    image, spots_zyx_radii, use_gpu=input_kwargs['use_gpu']
                )
            else:
                sharp_image = None
            features_kwargs['sharp_image'] = sharp_image
            
            df_features_frame = pipe.compute_spots_features(
                image, 
                zyx_coords, 
                spots_zyx_radii,
                **features_kwargs
            )
            
            true_idx = [tuple(row) for row in zyx_coords_true]
            false_idx = [tuple(row) for row in zyx_coords_false]
            
            df_features_frame = (
                df_features_frame
                .reset_index()
                .set_index(ZYX_GLOBAL_COLS)
            )
            
            df_features_frame_true = df_features_frame.loc[true_idx]
            df_features_frame_false = df_features_frame.loc[false_idx]
            
            df_features_frame_true = (
                df_features_frame_true
                .reset_index()
                .set_index(['Cell_ID', 'spot_id'])
            )
            df_features_frame_false = (
                df_features_frame_false
                .reset_index()
                .set_index(['Cell_ID', 'spot_id'])
            )
            
            frames.append((frame_i, 'true_spot'))
            dfs.append(df_features_frame_true)
            
            frames.append((frame_i, 'false_spot'))
            dfs.append(df_features_frame_false)
        
        features_range = self.input_kwargs()['tune_features_range']
        df_features = pd.concat(dfs, keys=frames, names=['frame_i', 'category'])
        
        if not features_range:
            return df_features
        printl(df_features)
        printl(features_range)
        
    
    def run(self, logger_func=printl, **kwargs):
        emitDebug = kwargs.get('emitDebug')
        logger_func('Determining optimal thresholding method...')
        self.best_threshold_method = self.find_best_threshold_method(
            emitDebug=emitDebug
        )
        
        logger_func('Determining optimal features range...')
        df_features = self.find_features_range(emitDebug=emitDebug)