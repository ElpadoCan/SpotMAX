import os
import shutil

import sys
import subprocess
import traceback

from tqdm import tqdm

import numpy as np
import pandas as pd

import cellacdc.myutils
from spotmax.nnet import preprocess

from . import pipe
from . import scores
from . import printl
from . import filters
from . import features
from . import io
from . import utils
from . import ZYX_GLOBAL_COLS

class TuneKernel:
    def __init__(self):
        self.init_input_data()
    
    def set_kwargs(self, kwargs):
        self._kwargs = kwargs
    
    def set_crop_to_global_coords(self, pos_foldername, crop_to_global_coords):
        self._crop_to_global_coords[pos_foldername] = crop_to_global_coords
    
    def set_tzyx_true_spots_df_coords(self, pos_tzyx_df_coords):
        self._true_spots_coords_df = pos_tzyx_df_coords
    
    def true_spots_coords_df(self):
        return self._true_spots_coords_df

    def set_tzyx_false_spots_df_coords(self, pos_tzyx_df_coords):
        if len(pos_tzyx_df_coords) == 0:
            pos_tzyx_df_coords = pd.DataFrame(
                columns=['Position_n', 'frame_i']
            )
        self._false_spots_coords_df = pos_tzyx_df_coords
    
    def crop_to_global_coords(self, pos_foldername):
        return self._crop_to_global_coords[pos_foldername]
    
    def false_spots_coords_df(self):
        return self._false_spots_coords_df

    def kwargs(self):
        return self._kwargs

    def set_image_data(self, pos_foldername, image_data):
        self._image_data[pos_foldername] = (
            cellacdc.myutils.img_to_float(image_data)
        )
    
    def set_segm_data(self, pos_foldername, segm_data):
        self._segm_data[pos_foldername] = segm_data
    
    def segm_data(self):
        return self._segm_data
    
    def init_input_data(self):
        self._image_data = {}
        self._segm_data = {}
        self._crop_to_global_coords = {}
        self._images_paths = {}
        self._basenames = {}
    
    def set_ini_filepath(self, ini_filepath):
        self._ini_filepath = ini_filepath
    
    def set_images_path(self, pos_foldername, images_path, basename):
        self._images_paths[pos_foldername] = images_path
        self._basenames[pos_foldername] = basename
        self._exp_folderpath = os.path.dirname(os.path.dirname(images_path))
    
    def exp_folderpath(self):
        return self._exp_folderpath
    
    def images_paths(self):
        return self._images_paths
    
    def ini_filepath(self):
        return self._ini_filepath
    
    def image_data(self):
        return self._image_data
    
    def input_kwargs(self):
        return self._kwargs
    
    def _iter_frames(self, to_crop=False):
        grouping_cols = ['Position_n', 'frame_i']
        false_coords_df = self.false_spots_coords_df().set_index(grouping_cols)        
        true_coords_df = self.true_spots_coords_df()
        grouped = true_coords_df.groupby(grouping_cols)
        for idx, true_df in grouped:
            keys = [
                'lab', 'gauss_sigma', 'spots_zyx_radii_pxl', 'do_sharpen',
                'do_remove_hot_pixels', 'lineage_table', 'do_aggregate', 
                'use_gpu'
            ]
            pos_foldername, frame_i = idx
            input_kwargs = {key:self._kwargs[key] for key in keys}
            to_global_coords = self.crop_to_global_coords(pos_foldername)
            if to_crop:
                zz_true = (true_df['z'] - to_global_coords[0]).to_list()
                yy_true = (true_df['y'] - to_global_coords[1]).to_list()
                xx_true = (true_df['x'] - to_global_coords[2]).to_list()
            else:
                zz_true = (true_df['z']).to_list()
                yy_true = (true_df['y']).to_list()
                xx_true = (true_df['x']).to_list()
            try:
                if to_crop:
                    zz_false = (
                        false_coords_df.loc[[idx], 'z'] - to_global_coords[0]
                    ).to_list()
                    yy_false = (
                        false_coords_df.loc[[idx], 'y'] - to_global_coords[1]
                    ).to_list()
                    xx_false = (
                        false_coords_df.loc[[idx], 'x'] - to_global_coords[2]
                    ).to_list()
                else:
                    zz_false = (false_coords_df.loc[[idx], 'z']).to_list()
                    yy_false = (false_coords_df.loc[[idx], 'y']).to_list()
                    xx_false = (false_coords_df.loc[[idx], 'x']).to_list()
            except Exception as e:
                zz_false, yy_false, xx_false = [], [], []
            out = (
                input_kwargs, zz_true, yy_true, xx_true, 
                zz_false, yy_false, xx_false
            )
            yield idx, out
    
    def find_best_threshold_method(self, **kwargs):
        emitDebug = kwargs.get('emitDebug')
        logger_func = kwargs.get('logger_func', print)

        f1_scores = []
        recall_scores = []
        positive_areas = []
        methods = []
        keys = []
        for idx, inputs in self._iter_frames(to_crop=True):
            (segm_kwargs, zz_true, yy_true, xx_true, 
            zz_false, yy_false, xx_false) = inputs
            
            pos_folder, frame_i = idx
            
            image = self.image_data()[pos_folder][frame_i]
            segm_kwargs['lab'] = self.segm_data()[pos_folder][frame_i]
            result = pipe.spots_semantic_segmentation(
                image, keep_input_shape=True, **segm_kwargs
            )
            pbar_method = tqdm(total=len(result), ncols=100)
            for method, thresholded in result.items():
                if method == 'input_image':
                    pbar_method.update()
                    continue
                true_mask = thresholded[zz_true, yy_true, xx_true]
                false_mask = thresholded[zz_false, yy_false, xx_false]
                f1_score = scores.semantic_segm_f1_score(true_mask, false_mask)
                positive_area = np.count_nonzero(thresholded)
                f1_scores.append(f1_score)
                recall = scores.semantic_segm_recall(true_mask)
                recall_scores.append(recall)
                methods.append(method)
                positive_areas.append(positive_area)
                keys.append(idx)
                pbar_method.update()
                
                # input_image = result['input_image']
                # to_debug = (
                #     method, thresholded, input_image, zz_true, yy_true, 
                #     xx_true, zz_false, yy_false, xx_false, 
                #     positive_area, f1_score
                # )
                # emitDebug(to_debug)
            pbar_method.close()
        df_scores = pd.DataFrame({
            'threshold_method': methods,
            'f1_score': f1_scores,
            'positive_area': positive_areas,
            'recall': recall_scores
        })
        
        sort_by = ['recall', 'f1_score', 'positive_area']
        ascending=[False, False, True]
        df_score = df_scores.groupby('threshold_method').agg(
            recall=('recall', 'min'),
            f1_score=('f1_score', 'mean'),
            positive_area=('positive_area', 'median')
        ).sort_values(sort_by, ascending=ascending)
        
        logger_func(f'Thresholding methods score:\n{df_score}')
        
        best_method = df_score.iloc[0].name
        return best_method
    
    def _cleanup_analysis_files(self):
        run_number = io.get_run_number_from_ini_filepath(self.ini_filepath())
        for images_path in self._analysed_images_paths:
            pos_path = os.path.dirname(images_path)
            spotmax_out_folder = os.path.join(pos_path, 'spotMAX_output')
            io.remove_run_number_spotmax_out_files(
                run_number, spotmax_out_folder
            )
            for file in utils.listdir(spotmax_out_folder):
                if file.startswith('_temp_autotune_coords'):
                    try:
                        os.remove(os.path.join(spotmax_out_folder, file))
                    except Exception as err:
                        pass

        try:
            shutil.rmtree(os.path.dirname(self.ini_filepath()))
        except Exception as err:
            pass
        
    def _setup_configparser(self, images_paths_to_analyse, logger_func=print):
        self._analysed_images_paths = images_paths_to_analyse
        cp = io.read_ini(self.ini_filepath())
        cp = io.set_out_files_extension(cp, '.h5')
        cp = io.add_use_default_values_to_configparser(cp)
        cp = io.disable_saving_masks_configparser(cp)
        cp = io.add_folders_to_analyse_to_configparser(
            cp, images_paths_to_analyse
        )
        
        run_nums = io.get_existing_run_nums(
            self.exp_folderpath(), logger_func=logger_func
        )
        new_run_num = max(run_nums, default=0) + 1
        cp = io.add_run_number_to_configparser(cp, new_run_num)
        cp = io.add_spots_coordinates_endname_to_configparser(
            cp, '_temp_autotune_coords.csv'
        )
        cp = io.add_text_to_append_to_configparser(
            cp, 'temp_autotune_coords'
        )
        
        io.write_to_ini(cp, self.ini_filepath())
    
    def _load_analysis_df_spots(self):
        dfs = []
        keys = []
        for pos_foldername, images_path in self.images_paths().items():
            pos_folderpath = os.path.dirname(images_path)
            spotmax_out_path = os.path.join(pos_folderpath, 'spotMAX_output')
            printl(utils.listdir(spotmax_out_path))
            valid_spots_filename = None
            spotfit_filename = None
            for file in utils.listdir(spotmax_out_path):
                if file.endswith('1_valid_spots_temp_autotune_coords.h5'):
                    valid_spots_filename = file
                elif file.endswith('2_spotfit_temp_autotune_coords.csv'):
                    spotfit_filename = file
                    break
            
            file_to_load = spotfit_filename
            if file_to_load is None:
                file_to_load = valid_spots_filename
                
            keys.append(pos_foldername)
            dfs.append(io.load_spots_table(spotmax_out_path, file_to_load))
             
        df_analysis = pd.concat(dfs, keys=keys, names=['Position_n'])
        return df_analysis
    
    def _run_analysis(self, df_spots_coords, logger_func=print):
        from . import _process
        
        df_spots_coords['do_not_drop'] = 1
        
        images_paths_to_analyse = []
        for pos_foldername, images_path in self.images_paths().items():
            try:
                df_spots_coords_pos = df_spots_coords.loc[pos_foldername]
            except KeyError as err:
                continue
            
            images_paths_to_analyse.append(images_path)
            basename = self._basenames[pos_foldername]
            temp_csv_path = os.path.join(
                images_path, f'{basename}temp_autotune_coords.csv'
            )
            df_spots_coords_pos.to_csv(temp_csv_path)
        
        self._setup_configparser(
            images_paths_to_analyse, logger_func=logger_func
        )
        
        command = f'spotmax, -p, {self.ini_filepath()}'
        # command = r'python, spotmax\test.py'
        command_format = command.replace(',', '')
        logger_func(f'SpotMAX analysis started with command `{command_format}`')
        args = [sys.executable, _process.__file__, '-c', command]
        subprocess.run(args)
    
        df_spots_analysis = self._load_analysis_df_spots()
        return df_spots_analysis
    
    def _init_df_features(self, df_spots_coords_input, df_spots_det):
        pos_zyx_cols = ['Position_n', 'z', 'y', 'x']
        df_spots_det_zyx = (
            df_spots_det.reset_index().set_index(pos_zyx_cols)
        )
        df_spots_coords_input_zyx = (
            df_spots_coords_input.reset_index().set_index(pos_zyx_cols)
        )
        df_spots_det_zyx['category'] = df_spots_coords_input_zyx['category']
        
        out_index = ['Position_n', 'frame_i', 'Cell_ID']
        df_features = (
            df_spots_det_zyx.reset_index().set_index(out_index)
        )
        return df_features
    
    def find_features_range(self, **kwargs):
        emitDebug = kwargs.get('emitDebug')
        logger_func = kwargs.get('logger_func', print)
        
        dfs = []
        keys = []
        for idx, inputs in self._iter_frames():
            (input_kwargs, zz_true, yy_true, xx_true, 
            zz_false, yy_false, xx_false) = inputs 
            
            pos_folder, frame_i = idx
            
            df_true = pd.DataFrame({
                'z': zz_true, 
                'y': yy_true, 
                'x': xx_true
            })
            df_false = pd.DataFrame({
                'z': zz_false, 
                'y': yy_false, 
                'x': xx_false
            })
            dfs.append(df_true)
            keys.append((*idx, 'true_spot'))
            
            dfs.append(df_false)
            keys.append((*idx, 'false_spot'))        
        
        df_spots_coords_input = pd.concat(
            dfs, keys=keys, names=['Position_n', 'frame_i', 'category']
        )
        try:
            df_spots_analysis = self._run_analysis(
                df_spots_coords_input, logger_func=logger_func
            )   
        except Exception as err:
            raise err
        finally:
            self._cleanup_analysis_files()
        
        df_features = self._init_df_features(
            df_spots_coords_input, df_spots_analysis
        )
        
        features_range = self.input_kwargs()['tune_features_range']
        # df_features = self.to_global_coords(df_features)
        
        if not features_range:
            return df_features, features_range
        
        df_features_tp = df_features[df_features.category == 'true_spot']
        df_features_fp = df_features[df_features.category == 'false_spot']
        
        to_col_mapper = features.feature_names_to_col_names_mapper()
        inequality_direction_mapper = (
            features.true_positive_feauture_inequality_direction_mapper()
        )
        for feature_name in features_range.keys():
            inequality_dir = inequality_direction_mapper[feature_name]
            col_name = to_col_mapper[feature_name]
            if inequality_dir == 'max':
                maximum = df_features_tp[col_name].max()
                minimum = None
                if not df_features_fp.empty:
                    minimum = df_features_fp[col_name].min()
            else:
                minimum = df_features_tp[col_name].min()
                maximum = None
                if not df_features_fp.empty:
                    maximum = df_features_fp[col_name].max()
            features_range[feature_name][0] = minimum
            features_range[feature_name][1] = maximum
        
        return df_features, features_range
        
    def run(self, logger_func=print, **kwargs):
        emitDebug = kwargs.get('emitDebug')
        logger_func('Determining optimal thresholding method...')
        best_threshold_method = self.find_best_threshold_method(
            emitDebug=emitDebug, logger_func=logger_func
        )
        
        logger_func('Determining optimal features range...')
        df_features, features_range = self.find_features_range(
            emitDebug=emitDebug, logger_func=logger_func
        )
        result = TuneResult(df_features, features_range, best_threshold_method)
        return result

class TuneResult:
    def __init__(self, df_features, features_range, best_threshold_method):
        self.df_features = df_features
        self.features_range = features_range
        self.threshold_method = best_threshold_method