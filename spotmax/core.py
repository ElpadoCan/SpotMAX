import os
import sys
import warnings

import pandas as pd
import numpy as np

import cv2

from tqdm import tqdm

import skimage.morphology
import skimage.measure
import skimage.transform
import skimage.filters

import acdctools.io
try:
    from acdctools.widgets import imshow
except Exception as e:
    pass

from . import utils, rng
from . import issues_url, printl, io

class _DataLoader:
    def __init__(self, debug=False, log=print):
        self.debug = debug
        self.log = log
    
    def get_data_from_images_path(
            self, images_path: os.PathLike, spots_ch_endname: str, 
            ref_ch_endname: str, segm_endname: str, ref_ch_segm_endname: str,
            lineage_table_endname: str
        ):
        data = self._load_data_from_images_path(
            images_path, spots_ch_endname, ref_ch_endname, segm_endname, 
            ref_ch_segm_endname, lineage_table_endname
        )
        data = self._reshape_data(data, self.metadata)
        data = self._add_regionprops(data)
        data = self._initialize_dataframes(data)
        return data
    
    def _load_data_from_images_path(
            self, images_path: os.PathLike, spots_ch_endname: str, 
            ref_ch_endname: str, segm_endname: str, ref_ch_segm_endname: str,
            lineage_table_endname: str
        ):
        channels = {
            spots_ch_endname: 'spots_ch', 
            ref_ch_endname: 'ref_ch', 
            segm_endname: 'segm',
            ref_ch_segm_endname: 'ref_ch_segm'
        }
        data = {}
        for channel, key in channels.items():
            if not channel:
                continue
            ch_path = acdctools.io.get_filepath_from_channel_name(
                images_path, os.path.basename(channel)
            )
            self.log(f'Loading "{channel}" channel from "{ch_path}"...')
            to_float = channel == 'spots_ch' or channel == 'ref_ch'
            ch_data, ch_dtype = io.load_image_data(
                ch_path, to_float=to_float, return_dtype=True
            )
            data[f'{key}.dtype'] = ch_dtype
            data[key] = ch_data

        ch_key = 'spots_ch' if 'spots_ch' in data else 'ref_ch'
        data_shape = data[ch_key].shape
        if 'segm' not in data:
            # Use entire image as segm_data  
            segm_data = np.ones(data_shape, dtype=np.uint8)
            data['segm'] = segm_data
        elif data['segm'].ndim < len(data_shape):
            # Stack the 2D segm into z-slices
            if len(data_shape) == 4:
                # Timelapse data, stack on second axis (T, Z, Y, X)
                SizeZ = data_shape[1]
                data['segm'] = np.stack([data['segm']]*SizeZ, axis=1)
            else:
                # Snapshot data, stack on first axis (Z, Y, X)
                SizeZ = data_shape[0]
                data['segm'] = np.stack([data['segm']]*SizeZ, axis=0)

        if not lineage_table_endname:
            return

        # Load lineage table
        table_path = acdctools.io.get_filepath_from_endname(
            images_path, os.path.basename(lineage_table_endname), ext='.csv'
        )
        self.log(
            f'Loading "{lineage_table_endname}" channel from "{table_path}"...'
        )
        data['lineage_table'] = pd.read_csv(table_path)

        return data
    
    def _add_regionprops(self, data):
        data['segm_rp'] = [
            skimage.measure.regionprops(data['segm'][frame_i]) 
            for frame_i in range(len(data['segm']))
        ]
        return data

    def _reshape_data(self, data, metadata: dict):
        SizeZ = metadata['SizeZ']
        arr_keys = ('spots_ch', 'ref_ch', 'ref_ch_segm', 'segm')
        for key in arr_keys:
            if key not in data:
                continue
            ch_data = data[key]
            if SizeZ > 1:
                # Data is already 4D
                continue
            
            if ch_data.ndim == 2:
                # Data is 2D. Add new axis. T axis will be added later
                data[key] = data[key][np.newaxis]
            elif ch_data.ndim == 3:
                # Data is 3D timelpase. Add axis for z-slice
                data[key] = data[key][:, np.newaxis]

        SizeT = metadata['SizeT']
        if SizeT > 1:
            # Data is already time-lapse --> do not reshape
            return data

        for key in arr_keys:
            if key not in data:
                continue
            data[key] = data[key][np.newaxis]
        
        if 'lineage_table' not in data:
            return data
        
        table = data['lineage_table']
        if 'frame_i' not in table.columns:
            table['frame_i'] = 0
        
        table = table.set_index(['frame_i', 'Cell_ID'])
        data['lineage_table'] = table
        return data
    
    def _initialize_dataframes(self, data):
        segm_data = data['segm']
        frame_idxs = []
        IDs = []
        for frame_i in range(len(segm_data)):
            rp = data['segm_rp'][frame_i]
            for obj in rp:
                IDs.append(obj.label)
                frame_idxs.append(frame_i)
        df_data = {'frame_i': frame_idxs, 'Cell_ID': IDs}
        df_cells = pd.DataFrame(df_data).set_index(['frame_i', 'Cell_ID'])
        return data

class _ParamsParser(_DataLoader):
    def __init__(self, debug=False, is_cli=True, log=print):
        super().__init__(debug=debug, log=log)
        self.debug = debug
        self.is_cli = is_cli
    
    @utils.exception_handler_cli
    def init_params(self, params_path, metadata_csv_path=''):
        params_path = utils.check_cli_params_path(params_path)
        
        self._params = io.config.analysisInputsParams(params_path)
        if metadata_csv_path:
            self._params = io.add_metadata_from_csv(
                metadata_csv_path, self._params)
        
        params_folder_path = os.path.dirname(params_path)
        params_file_name = os.path.splitext(os.path.basename(params_path))[0]
        self.ini_params_filename = f'{params_file_name}.ini'
        self.ini_params_file_path = os.path.join(
            params_folder_path, self.ini_params_filename
        )
        if params_path.endswith('.csv'):
            if os.path.exists(self.ini_params_file_path):       
                proceed = self._ask_user_save_ini_from_csv(
                    self.ini_params_file_path)
                if not proceed:
                    self.logger.info(
                        'spotMAX execution stopped by the user. '
                        'No option to save .ini file was selected.'
                    )
                    self.quit()
                    return
            io.writeConfigINI(
                params=self._params, ini_path=self.ini_params_file_path
            )
            
            self.logger.info(
                f'New parameters file created: "{self.ini_params_file_path}"'
            )
        
        self.check_metadata()
        self.check_missing_params()
        self.cast_loaded_values_dtypes()
        self.set_abs_exp_paths()
        self.set_metadata()
    
    def _ask_user_save_ini_from_csv(self, ini_filepath):
        filename = os.path.basename(ini_filepath)
        ini_folderpath = os.path.dirname(ini_filepath)
        options = (
            'Overwrite existing file', 'Save with a new name..'
        )
        question = 'What do you want to do'
        txt = (
            f'[WARNING]: spotMAX would like to save the parameters in the file '
            f'"{filename}" (see full path below). '
            'However, this file already exists.\n\n'
            f'File path: "{ini_filepath}"'
        )
        answer = io.get_user_input(question, options=options, info_txt=txt)
        if not answer:
            return False
        if answer == 'Overwrite existing file':
            return True
        elif answer == 'Save with a new name..':
            new_filename = acdctools.io.get_filename_cli(
                question='Insert a new filename for the .ini parameters file'
            )
            if new_filename is None:
                return False
            if not new_filename.endswith('.ini'):
                new_filename = f'{new_filename}.ini'

            self.ini_params_file_path = os.path.join(
                ini_folderpath, new_filename
            )      
            return True

    def _ask_user_multiple_run_nums(self, run_nums):
        new_run_num = max(run_nums)+1
        options = (
            'Choose run number to overwrite', f'Save as new run number {new_run_num}'
        )
        question = 'What do you want to do'
        txt = (
            '[WARNING]: All or some of the experiments present in the loaded '
            'folder have already been analysed before '
            f'(run numbers presents are {run_nums})'
        )
        answer = io.get_user_input(question, options=options, info_txt=txt)
        if answer == options[1]:
            return new_run_num
        elif answer == options[0]:
            options = [f'Run number {r}' for r in run_nums]
            question = 'Which run number do you want to overwrite?'
            new_run_num_txt = io.get_user_input(question, options=options)
            new_run_num = int(new_run_num_txt[11:])
            return new_run_num
    
    @utils.exception_handler_cli
    def set_abs_exp_paths(self):
        SECTION = 'File paths and channels'
        ANCHOR = 'filePathsToAnalyse'
        loaded_exp_paths = self._params[SECTION][ANCHOR]['loadedVal']
        self.exp_paths_list = []
        for exp_path in loaded_exp_paths:
            is_single_pos = False
            if io.is_pos_path(exp_path):
                pos_path = exp_path
                pos_foldername = os.path.basename(exp_path)
                exp_path = os.path.dirname(pos_path)
                exp_paths = (
                    {exp_path: {'pos_foldernames': [pos_foldername]}}
                )
                is_single_pos = True
            elif io.is_images_path(exp_path):
                pos_path = os.path.dirname(exp_path)
                pos_foldername = os.path.basename(pos_path)
                exp_path = os.path.dirname(os.path.dirname(pos_path))
                exp_paths = (
                    {exp_path: {'pos_foldernames': [pos_foldername]}}
                )
                is_single_pos = True
            
            # Scan and determine run numbers
            pathScanner = io.expFolderScanner(exp_path)
            pathScanner.getExpPaths(exp_path)
            pathScanner.infoExpPaths(pathScanner.expPaths)
            run_nums = list(pathScanner.paths.keys())
            is_multi_run = False
            if len(run_nums) > 1:
                run_number = self._ask_user_multiple_run_nums(run_nums)
                if run_number is None:
                    self.logger.info(
                        'spotMAX stopped by the user. Run number was not provided.'
                    )
                    self.quit()
                if is_single_pos:
                    exp_paths['run_number'] = run_number
                else:
                    exp_paths = {}
                    for run_num, run_num_info in pathScanner.paths.items():
                        for exp_path, exp_info in run_num_info.items():
                            if exp_path in exp_paths:
                                continue
                            exp_paths[exp_path] = {
                                'pos_foldernames': exp_info['posFoldernames'],
                                'run_number': run_number
                            }
            else:
                exp_paths = {}
                for exp_path, exp_info in pathScanner.paths[run_nums[0]].items():
                    if not exp_info['numPosSpotCounted'] > 0:
                        run_number = 1
                    else:
                        run_number = self._ask_user_multiple_run_nums(run_nums)
                    exp_paths[exp_path] = {
                        'pos_foldernames': exp_info['posFoldernames'],
                        'run_number': run_number
                    }
            self.exp_paths_list.append(exp_paths)
        self.set_channel_names()
    
    def set_channel_names(self):
        SECTION = 'File paths and channels'
        section_params = self._params[SECTION]
        spots_ch_endname = section_params['spotsEndName'].get('loadedVal')
        ref_ch_endname = section_params['refChEndName'].get('loadedVal')
        segm_endname = section_params['segmEndName'].get('loadedVal')
        ref_ch_segm_endname = section_params['refChSegmEndName'].get('loadedVal')
        lineage_table_endname = section_params['lineageTableEndName'].get('loadedVal')
        if self.exp_paths_list:
            for i in range(len(self.exp_paths_list)):
                for exp_path in list(self.exp_paths_list[i].keys()):
                    exp_info = self.exp_paths_list[i][exp_path]
                    exp_info['spotsEndName'] = spots_ch_endname
                    exp_info['refChEndName'] = ref_ch_endname
                    exp_info['segmEndName'] = segm_endname
                    exp_info['refChSegmEndName'] = ref_ch_segm_endname
                    exp_info['lineageTableEndName'] = lineage_table_endname
                    self.exp_paths_list[i][exp_path] = exp_info
        else:
            self.single_path_info = {
                'spots_ch_filepath': spots_ch_endname,
                'ref_ch_filepath': ref_ch_endname,
                'segm_filepath': segm_endname,
                'ref_ch_segm_filepath': ref_ch_segm_endname,
                'lineage_table_filepath': lineage_table_endname
            }
        
    def _add_resolution_limit_metadata(self, metadata):
        emission_wavelen = metadata['emWavelen']
        num_aperture = metadata['numAperture']
        physical_size_x = metadata['pixelWidth']
        physical_size_y = metadata['pixelHeight']
        physical_size_z = metadata['voxelDepth']
        z_resolution_limit_um = metadata['zResolutionLimit']
        yx_resolution_multiplier = metadata['yxResolLimitMultiplier']
        zyx_resolution_limit_pxl, zyx_resolution_limit_um = calcMinSpotSize(
            emission_wavelen, num_aperture, physical_size_x, 
            physical_size_y, physical_size_z, z_resolution_limit_um, 
            yx_resolution_multiplier
        )
        metadata['zyxResolutionLimitPxl'] = zyx_resolution_limit_pxl
        metadata['zyxResolutionLimitUm'] = zyx_resolution_limit_um

    @utils.exception_handler_cli
    def set_metadata(self):
        SECTION = 'METADATA'
        self.metadata = {}
        for anchor, options in self._params[SECTION].items():
            dtype_conveter = options.get('dtype')
            if dtype_conveter is None:
                continue
            self.metadata[anchor] = dtype_conveter(options['loadedVal'])
        
        self._add_resolution_limit_metadata(self.metadata)

    @utils.exception_handler_cli
    def _get_missing_metadata(self):
        SECTION = 'METADATA'
        missing_metadata = []
        for param_name, options in self._params[SECTION].items():
            dtype_converter = options.get('dtype')
            if dtype_converter is None:
                continue
            metadata_value = options.get('loadedVal')
            if metadata_value is None:
                missing_metadata.append(options['desc'])
                continue
            try:
                dtype_converter(metadata_value)
            except Exception as e:
                missing_metadata.append(options['desc'])
        return missing_metadata

    def check_metadata(self):
        missing_metadata = self._get_missing_metadata()
        if not missing_metadata:
            return

        missing_metadata_str = [f'    * {v}' for v in missing_metadata]
        missing_metadata_format = '\n'.join(missing_metadata_str)
        print('*'*40)
        err_msg = (
            f'The parameters file "{self.ini_params_filename}" is missing '
            'the following REQUIRED metadata:\n\n'
            f'{missing_metadata_format}\n\n'
            'Add them to the file (see path below) '
            'at the [METADATA] section. If you do not have timelapse data and\n'
            'the "Analyse until frame number" is missing you need to\n'
            'to write "Analyse until frame number = 1".'
            f'Parameters file path: "{self.ini_params_file_path}"\n'
        )
        self.logger.info(err_msg)
        if self.is_cli:
            print('*'*40)
            self.logger.info(
                'spotMAX execution aborted because some metadata are missing. '
                'See details above.'
            )
            self.quit()
        else:
            raise FileNotFoundError('Metadata missing. See details above')
    
    def _get_missing_params(self):
        missing_params = []
        for section_name, anchors in self._params.items():
            if section_name == 'METADATA':
                continue
            for anchor, options in anchors.items():
                dtype_converter = options.get('dtype')
                if dtype_converter is None:
                    continue
                value = options.get('loadedVal')
                default_val = options.get('initialVal')
                if value is None or value == '':
                    missing_param = (
                        section_name, options['desc'], default_val, anchor)
                    missing_params.append(missing_param)
                    continue
                # Try to check that type casting works
                try:
                    dtype_converter(value)
                except Exception as e:
                    missing_param = (
                        section_name, options['desc'], default_val, anchor
                    )
                    missing_params.append(missing_param)
                    continue
        return missing_params
    
    def _set_default_val_params(self, missing_params):
        for param in missing_params:
            section_name, _, default_val, anchor = param
            self._params[section_name][anchor]['loadedVal'] = default_val
    
    def _ask_user_input_missing_params(self, missing_params):
        question = (
            'Do you want to continue with default value for the missing parameters?'
        )
        options = (
            'Yes, use default values', 'No, stop process', 
            'Display default values'
        )
        answer = io.get_user_input(
            question, options=options, default_option='No'
        )
        if answer == 'No, stop process' or answer == None:
            return False
        elif answer == 'Yes, use default values':
            self._set_default_val_params(missing_params)
            return True
        else:
            print('')
            missing_params_str = [
                f'    * {param[1]} (section: [{param[0]}]) = {param[2]}' 
                for param in missing_params
            ]
            missing_params_format = '\n'.join(missing_params_str)
            self.logger.info(
                f'Default values:\n\n{missing_params_format}'
            )
            print('-'*50)

    @utils.exception_handler_cli
    def check_missing_params(self):
        missing_params = self._get_missing_params()
        if not missing_params:
            return
        
        cannot_continue = False
        missing_params_desc = {param[1]:param[2] for param in missing_params}
        if 'Experiment folder path(s) to analyse' in missing_params_desc:
            # Experiment folder path is missing --> continue only if 
            # either spots or reference channel are proper file paths
            spots_ch_path = missing_params_desc.get(
                'Spots channel end name or path', ''
            )
            ref_ch_path = missing_params_desc.get(
                'Reference channel end name or path', ''
            )
            cannot_continue = not (
                os.path.exists(spots_ch_path) or os.path.exists(ref_ch_path)
            )           
        
        missing_params_str = [
            f'    * {param[1]} (section: [{param[0]}])' 
            for param in missing_params
        ]
        missing_params_format = '\n'.join(missing_params_str)
        print('*'*40)
        err_msg = (
            f'The parameters file "{self.ini_params_filename}" is missing '
            'the following parameters:\n\n'
            f'{missing_params_format}\n\n'
        )
        
        if cannot_continue:
            err_msg = (f'{err_msg}'
                'Add them to the file (see path below) '
                'at the right section (shown in parethensis above).\n'
                'Note that you MUST provide at least one of the file/folder '
                'paths.\n\n'
                f'Parameters file path: "{self.ini_params_file_path}"\n'
            )
            self.logger.info(err_msg)
            if self.is_cli:
                print('*'*40)
                self.logger.info(
                    'spotMAX execution aborted because some parameters are missing. '
                    'See details above.'
                )
                self.quit()
            else:
                raise FileNotFoundError('Metadata missing. See details above')
        else:
            err_msg = (f'{err_msg}'
                'You can add them to the file (see path below) '
                'at the right section (shown in parethensis above), or continue '
                'with default values.\n'
                f'Parameters file path: "{self.ini_params_file_path}"\n'
            )
            self.logger.info(err_msg)
            proceed = self._ask_user_input_missing_params(missing_params)
            if not proceed:
                self.logger.info(
                    'spotMAX execution stopped by the user. '
                    'Some parameters are missing'
                )
                self.quit()
                return

    def cast_loaded_values_dtypes(self):
        for section_name in list(self._params.keys()):
            anchor_names = list(self._params[section_name].keys())
            for anchor_name in anchor_names:
                to_dtype = self._params[section_name][anchor_name].get('dtype')
                if to_dtype is None:
                    continue
                value = self._params[section_name][anchor_name]['loadedVal']
                value = to_dtype(value)
                self._params[section_name][anchor_name]['loadedVal'] = value

class Kernel(_ParamsParser):
    def __init__(self, debug=False, is_cli=True):
        self.logger, self.log_path, self.logs_path = utils.setupLogger('cli')
        super().__init__(debug=debug, is_cli=is_cli, log=self.logger.info)
        self.debug = debug
        self.is_cli = is_cli

    def _preprocess(self, image_data):
        SECTION = 'Pre-processing'
        ANCHOR = 'gaussSigma'
        options = self._params[SECTION][ANCHOR]
        sigma = options.get('loadedVal')
        if sigma is None:
            return image_data
        
        if sigma == 0:
            return image_data

        print('')
        self.logger.info(f'Applying a gaussian filter with sigma = {sigma}...')
        filtered_data = skimage.filters.gaussian(image_data, sigma=sigma)
        return filtered_data
    
    def _sharpen_spots(self, spots_img, metadata):
        sigmas = metadata['zyxResolutionLimitPxl']
        blurred = skimage.filters.gaussian(spots_img, sigma=sigmas)
        sharpened = spots_img - blurred
        imshow(spots_img, blurred, sharpened)
        return sharpened
    
    def _get_obj_mask(self, lab, obj, lineage_table):
        lab_mask_ID = lab == obj.label
        
        if lineage_table is None:
            return lab_mask_ID, -1
        
        cc_stage = lineage_table.at[obj.label, 'cell_cycle_stage']
        if cc_stage == 'G1':
            return lab_mask_ID, -1
        
        # Merge mother and daughter when in S phase
        rel_ID = lineage_table.at[obj.label, 'relative_ID']
        lab_mask_ID = np.logical_or(lab == obj.label, lab == rel_ID)
        
        return lab_mask_ID, rel_ID
    
    def _filter_largest_obj(self, mask_or_labels):
        lab = skimage.measure.label(mask_or_labels)
        positive_mask = lab > 0
        counts = np.bincount(positive_mask)
        largest_obj_id = np.argmax(counts)
        lab[lab != largest_obj_id] = 0
        if mask_or_labels.dtype == bool:
            return lab > 0
        return lab
    
    def _extract_img_from_segm_obj(self, image, lab, obj, lineage_table):
        lab_mask, bud_ID = self._get_obj_mask(lab, obj, lineage_table)
        lab_mask_rp = skimage.measure.regionprops(lab_mask.astype(np.uint8))
        lab_mask_obj = lab_mask_rp[0]
        ref_ch_img_local = image[lab_mask_obj.slice]
        bkgr_vals = ref_ch_img_local[~lab_mask_obj.image]
        if bkgr_vals.size == 0:
            return ref_ch_img_local, lab_mask, bud_ID
        
        bkgr_mean = bkgr_vals.mean()
        bkgr_std = bkgr_vals.std()
        gamma_shape = np.square(bkgr_mean/bkgr_std)
        gamma_scale = np.square(bkgr_std)/bkgr_mean
        ref_ch_img_bkgr = rng.gamma(
            gamma_shape, gamma_scale, size=lab_mask_obj.image.shape
        )

        ref_ch_img_bkgr[lab_mask_obj.image] = ref_ch_img_local[lab_mask_obj.image]

        return ref_ch_img_bkgr, lab_mask_obj.image, bud_ID
    
    def add_ref_ch_num_features(
            self, df_cells, frame_i, ID, ref_ch_mask_local
        ):
        vol_voxels = np.count_nonzero(ref_ch_mask_local)
        df_cells.at[(frame_i, ID), 'ref_ch_vol_vox'] = vol_voxels

        rp = skimage.measure.regionprops(ref_ch_mask_local.astype(np.uint16))
        num_fragments = len(rp)
        df_cells.at[(frame_i, ID), 'ref_ch_num_fragments'] = num_fragments

        return df_cells
    
    def _segment_ref_ch(
            self, ref_ch_img, lab, lab_rp, df_cells, lineage_table, 
            threshold_func, frame_i, keep_only_largest_obj, ref_ch_segm, 
            thresh_val=None, verbose=True
        ):
        if verbose:
            self.logger.info('Segmenting reference channel...')
        IDs = [obj.label for obj in lab_rp]
        for obj in tqdm(lab_rp, ncols=100):
            if lineage_table is not None:
                if lineage_table.at[obj.label, 'relationship'] == 'bud':
                    # Skip buds since they are aggregated with mother
                    continue
            
            # Get the image for thresholding (either object or mother-bud)
            ref_ch_img_local, obj_mask, bud_ID = self._extract_img_from_segm_obj(
                ref_ch_img, lab, obj, lineage_table
            )

            # Compute threshold value if not aggregate object (threshold value 
            # computed before on the aggregated image)
            if thresh_val is None:
                thresh_val = threshold_func(ref_ch_img_local.max(axis=0))
            
            # Store threshold value
            df_idx = (frame_i, obj.label)
            df_cells.at[df_idx, 'ref_ch_threshold_value'] = thresh_val
            if bud_ID > 0:
                bud_idx = (frame_i, bud_ID)
                df_cells.at[bud_idx, 'ref_ch_threshold_value'] = thresh_val

            # Threshold
            ref_mask_local = ref_ch_img_local > thresh_val
            ref_mask_local[~obj_mask] = False

            if bud_ID > 0:
                bud_obj = lab_rp[IDs.index(bud_ID)]
                objs = [obj, bud_obj]
            else:
                objs = [obj]

            for obj in objs:
                ref_ch_mask = np.zeros_like(obj.image)
                local_slice = tuple([slice(0,d) for d in obj.image.shape])
                ref_ch_mask[local_slice] = ref_mask_local[local_slice]
                ref_ch_mask[~obj.image] = False
                if keep_only_largest_obj:
                    ref_ch_mask = self._filter_largest_obj(ref_ch_mask)

                # Add numerical features
                df_cells = self.add_ref_ch_num_features(
                    df_cells, frame_i, obj.label, ref_ch_mask
                )

                ref_ch_segm[obj.slice][ref_ch_mask] = obj.label

        return ref_ch_segm, df_cells
    
    def ref_ch_to_physical_units(self, df_cells, metadata):
        vox_to_um3_factor = (
            self.metadata['pixelWidth']
            *self.metadata['pixelHeight']
            *self.metadata['voxelDepth']
        )
        df_cells['ref_ch_vol_um3'] = df_cells['ref_ch_vol_vox']*vox_to_um3_factor
        return df_cells

    @utils.exception_handler_cli
    def segment_ref_ch(
            self, ref_ch_img, threshold_method='threshold_otsu', lab_rp=None, 
            lab=None, lineage_table=None, keep_only_largest_obj=False, 
            do_aggregate_objs=False, df_cells=None, frame_i=0, 
            verbose=True
        ):
        if lab is None:
            lab = np.ones(ref_ch_img.shape, dtype=np.uint8)
            lineage_table = None
        
        if lab_rp is None:
            lab_rp = skimage.measure.regionprops(lab)
        
        if df_cells is None:
            IDs = [obj.label for obj in lab_rp]
            df_data = {'frame_i': [frame_i]*len(IDs), 'Cell_ID': IDs}
            df_cells = pd.DataFrame(df_data).set_index(['frame_i', 'Cell_ID'])
        
        df_cells['ref_ch_threshold_value'] = np.nan

        ref_ch_segm = np.zeros_like(lab)

        if isinstance(threshold_method, str):
            threshold_func = getattr(skimage.filters, threshold_method)
        else:
            threshold_func = threshold_method

        if do_aggregate_objs:
            aggr_ref_ch_img = self.aggregate_objs(
                ref_ch_img, lab, lineage_table=lineage_table
            )
            thresh_val = threshold_func(aggr_ref_ch_img.max(axis=0))
        else:
            thresh_val = None
        
        ref_ch_segm, df_cells = self._segment_ref_ch(
            ref_ch_img, lab, lab_rp, df_cells, lineage_table, 
            threshold_func, frame_i, keep_only_largest_obj, ref_ch_segm, 
            thresh_val=thresh_val, verbose=verbose
        )

        return ref_ch_segm, df_cells
    
    def aggregate_objs(self, img_data, lab, lineage_table=None):
        if lineage_table is not None:
            # Check if moth-buds are to be merged before aggregation
            lab = lab.copy()
            df_buds = lineage_table[lineage_table.relationship == 'bud']
            moth_IDs = df_buds['relative_ID'].unique()
            df_buds = df_buds.reset_index().set_index('relative_ID')
            if len(moth_IDs) > 0:
                lab = lab.copy()
                for mothID in moth_IDs:
                    budID = df_buds.at[mothID, 'Cell_ID']
                    lab[lab==budID] = mothID

        # Get max height and total width
        rp = skimage.measure.regionprops(lab)
        tot_width = 0
        max_height = 0
        for obj in rp:
            h, w = obj.image.shape[-2:]
            if h > max_height:
                max_height = h
            tot_width += w

        # Aggregate data horizontally
        aggr_shape = (img_data.shape[0], max_height, tot_width)
        aggregated_img = np.zeros(aggr_shape, dtype=img_data.dtype)
        aggregated_img[:] = img_data.min()
        last_w = 0
        for obj in rp:
            h, w = obj.image.shape[-2:]
            aggregated_img[:, :h, last_w:last_w+w] = img_data[obj.slice]
            last_w += w
        return aggregated_img

    @utils.exception_handler_cli
    def spot_detection(
            self, spots_img, ref_ch_img=None, ref_ch_mask_or_labels=None, 
            frame_i=0, lab=None, rp=None, raw_spots_img=None
        ):
        if lab is None:
            lab = np.ones(spots_img.shape, dtype=np.uint8)
        
        if rp is None:
            rp = skimage.measure.regionprops(lab)

    @utils.exception_handler_cli
    def _run_from_images_path(
            self, images_path, spots_ch_endname: str='', ref_ch_endname: str='', 
            segm_endname: str='', ref_ch_segm_endname: str='', 
            lineage_table_endname: str=''
        ):
        data = self.get_data_from_images_path(
            images_path, spots_ch_endname, ref_ch_endname, segm_endname, 
            ref_ch_segm_endname, lineage_table_endname
        )
        do_segment_ref_ch = (
            self._params['Reference channel']['segmRefCh']['loadedVal']
        )
        do_aggregate_objs = (
            self._params['Pre-processing']['aggregate']['loadedVal']
        )
        ref_ch_data = data.get('ref_ch')
        segm_rp = data.get('segm_rp')
        segm_data = data.get('segm')
        df_cells = data.get('df_cells')
        ref_ch_segm_data = data.get('ref_ch_segm')
        acdc_df = data.get('lineage_table')
        
        if ref_ch_data is not None and do_segment_ref_ch:
            self.logger.info('Segmenting reference channel...')
            SECTION = 'Reference channel'
            ref_ch_threshold_method = (
                self._params[SECTION]['refChThresholdFunc']['loadedVal']
            )
            is_ref_ch_single_obj = (
                self._params[SECTION]['refChSingleObj']['loadedVal']
            )
            ref_ch_segm_data = np.zeros(ref_ch_data.shape, dtype=np.uint16)
            stopFrameNum = self.metadata['stopFrameNum']
            for frame_i in tqdm(range(stopFrameNum), ncols=100):
                if acdc_df is not None:
                    lineage_table = acdc_df.loc[frame_i]
                else:
                    lineage_table = None
                lab_rp = segm_rp[frame_i]
                ref_ch_img = ref_ch_data[frame_i]
                ref_ch_img = self._preprocess(ref_ch_img)
                lab = segm_data[frame_i]
                ref_ch_lab, df_cells = self.segment_ref_ch(
                    ref_ch_img, lab_rp=lab_rp, lab=lab, 
                    threshold_method=ref_ch_threshold_method, 
                    keep_only_largest_obj=is_ref_ch_single_obj,
                    df_cells=df_cells, frame_i=frame_i, 
                    do_aggregate_objs=do_aggregate_objs,
                    lineage_table=lineage_table,
                    verbose=False
                )
                ref_ch_segm_data[frame_i] = ref_ch_lab

            df_cells = self.ref_ch_to_physical_units(df_cells, self.metadata)

            data['df_cells'] = df_cells
            data['ref_ch_segm'] = ref_ch_segm_data
        
        if 'spots_ch' not in data:
            # Spot detection not required
            return
        
        spots_data = data.get('spots_ch')
        stopFrameNum = self.metadata['stopFrameNum']
        do_sharpen_spots = (
            self._params['Pre-processing']['sharpenSpots']['loadedVal']
        )
        SECTION = 'Reference channel'
        do_filter_spots_vs_ref_ch = (
            self._params[SECTION]['filterPeaksInsideRef']['loadedVal']
        )
        do_keep_spots_in_ref_ch = (
            self._params[SECTION]['keepPeaksInsideRef']['loadedVal']
        )
        how_filter_spots_vs_ref_ch = (
            self._params[SECTION]['filterPeaksInsideRefMethod']['loadedVal']
        )

        for frame_i in tqdm(range(stopFrameNum), ncols=100):
            raw_spots_img = spots_data[frame_i]
            filtered_spots_img = self._preprocess(raw_spots_img)
            if do_sharpen_spots:
                filtered_spots_img = self._sharpen_spots(
                    filtered_spots_img, self.metadata
                )
            lab = segm_data[frame_i]
            rp = segm_rp[frame_i]
            if ref_ch_data is not None:
                ref_ch_img = ref_ch_data[frame_i]
            else:
                ref_ch_img = None
            self.spot_detection(
                filtered_spots_img, ref_ch_img=ref_ch_img, frame_i=frame_i, 
                df_cells=df_cells, ref_ch_mask_or_labels=ref_ch_segm_data,
                lab=lab, rp=rp, raw_spots_img=raw_spots_img
            )

    @utils.exception_handler_cli
    def _run_exp_paths(self, exp_paths):
        """Run spotMAX analysis from a dictionary of Cell-ACDC style experiment 
        paths

        Parameters
        ----------
        exp_paths : dict
            Dictionary where the keys are the experiment paths containing the 
            Position folders with the following values: `run_number`, `pos_foldernames`
            `spotsEndName`, `refChEndName`, `segmEndName`, `refChSegmEndName`,
            and `lineageTableEndName`.

            NOTE: This dictionary is computed in the `set_abs_exp_paths` method.
        """        
        for exp_path, exp_info in exp_paths.items():
            exp_path = utils.get_abspath(exp_path)
            run_number = exp_info['run_number']
            pos_foldernames = exp_info['pos_foldernames']  
            spots_ch_endname = exp_info['spotsEndName'] 
            ref_ch_endname = exp_info['refChEndName']
            segm_endname = exp_info['segmEndName']
            ref_ch_segm_endname = exp_info['refChSegmEndName']
            lineage_table_endname = exp_info['lineageTableEndName']
            for pos in pos_foldernames:
                pos_path = os.path.join(exp_path, pos)
                images_path = os.path.join(pos_path, 'Images')
                self._run_from_images_path(
                    images_path, 
                    spots_ch_endname=spots_ch_endname, 
                    ref_ch_endname=ref_ch_endname, 
                    segm_endname=segm_endname,
                    ref_ch_segm_endname=ref_ch_segm_endname, 
                    lineage_table_endname=lineage_table_endname
                )               

    @utils.exception_handler_cli
    def _run_single_path(self, single_path_info):
        pass

    @utils.exception_handler_cli
    def run(self, parser_args):
        params_path = parser_args['params']
        metadata_csv_path = parser_args['metadata']
        self.init_params(params_path, metadata_csv_path=metadata_csv_path)
        if self.exp_paths_list:
            for exp_paths in self.exp_paths_list:
                self._run_exp_paths(exp_paths)
        else:
            self._run_single_path(self.single_path_info)
            
    def quit(self, error=None):
        if not self.is_cli and error is not None:
            raise error

        self.logger.info('='*50)
        if error is not None:
            self.logger.info(f'[ERROR]: {error}')
            self.logger.info('^'*50)
            err_msg = (
                'spotMAX aborted due to **error**. '
                'More details above or in the folowing log file:\n\n'
                f'{self.log_path}\n\n'
                'If you cannot solve it, you can report this error by opening '
                'an issue on our '
                'GitHub page at the following link:\n\n'
                f'{issues_url}\n\n'
                'Please **send the log file** when reporting a bug, thanks!'
            )
            self.logger.info(err_msg)
        else:
            self.logger.info(
                'spotMAX command line-interface closed. '
                f'{utils.get_salute_string()}'
            )
        self.logger.info('='*50)
        exit()

def eucl_dist_point_2Dyx(points, all_others):
    """
    Given 2D array of [y, x] coordinates points and all_others return the
    [y, x] coordinates of the two points (one from points and one from all_others)
    that have the absolute minimum distance
    """
    # Compute 3D array where each ith row of each kth page is the element-wise
    # difference between kth row of points and ith row in all_others array.
    # (i.e. diff[k,i] = points[k] - all_others[i])
    diff = points[:, np.newaxis] - all_others
    # Compute 2D array of distances where
    # dist[i, j] = euclidean dist (points[i],all_others[j])
    dist = np.linalg.norm(diff, axis=2)
    # Compute i, j indexes of the absolute minimum distance
    i, j = np.unravel_index(dist.argmin(), dist.shape)
    nearest_point = all_others[j]
    point = points[i]
    min_dist = dist.min()
    return min_dist, nearest_point

def rotationalVolume(obj, PhysicalSizeY=1, PhysicalSizeX=1):
    vox_to_fl = float(PhysicalSizeY)*(float(PhysicalSizeX)**2)
    rotate_ID_img = skimage.transform.rotate(
        obj.image.astype(np.uint8), -(obj.orientation*180/np.pi),
        resize=True, order=3, preserve_range=True
    )
    radii = np.sum(rotate_ID_img, axis=1)/2
    vol_vox = np.sum(np.pi*(radii**2))
    if vox_to_fl is not None:
        return vol_vox, float(vol_vox*vox_to_fl)
    else:
        return vol_vox, vol_vox

def calcMinSpotSize(
        emWavelen, NA, physicalSizeX, physicalSizeY, physicalSizeZ,
        zResolutionLimit_um, yxResolMultiplier
    ):
    try:
        airyRadius_nm = (1.22 * emWavelen)/(2*NA)
        airyRadius_um = airyRadius_nm*1E-3
        yxMinSize_um = airyRadius_um*yxResolMultiplier
        xMinSize_pxl = yxMinSize_um/physicalSizeX
        yMinSize_pxl = yxMinSize_um/physicalSizeY
        zMinSize_pxl = zResolutionLimit_um/physicalSizeZ
        zyxMinSize_pxl = (zMinSize_pxl, yMinSize_pxl, xMinSize_pxl)
        zyxMinSize_um = (zResolutionLimit_um, yxMinSize_um, yxMinSize_um)
        return zyxMinSize_pxl, zyxMinSize_um
    except ZeroDivisionError as e:
        # warnings.warn(str(e), RuntimeWarning)
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

def skeletonize(dataToSkel, is_zstack=False):
    skeleton = skimage.morphology.skeletonize(dataToSkel)
    skeletonCoords = {'all': np.nonzero(skeleton)}
    if is_zstack:
        for z, skel in enumerate(skeleton):
            skeletonCoords[z] = np.nonzero(skel)
    return skeletonCoords

def objContours(obj):
    contours, _ = cv2.findContours(
        obj.image.astype(np.uint8),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )
    min_y, min_x, _, _ = obj.bbox
    contours_li = []
    for cont in contours:
        cont = np.squeeze(cont, axis=1)
        cont = np.vstack((cont, cont[0]))
        cont += [min_x, min_y]
        contours_li.append(cont)
    return contours_li

def findContours(dataToCont, is_zstack=False):
    contCoords = {'proj': {}}
    if is_zstack:
        for z, img in enumerate(dataToCont):
            lab = skimage.measure.label(img)
            rp = skimage.measure.regionprops(lab)
            allObjContours = {}
            for obj in rp:
                contours_li = objContours(obj)
                allObjContours[obj.label] = contours_li
            contCoords[z] = allObjContours
        dataToCont2D = dataToCont.max(axis=0)
    else:
        dataToCont2D = dataToCont.max(axis=0)

    lab = skimage.measure.label(dataToCont2D)
    rp = skimage.measure.regionprops(lab)
    for obj in rp:
        contours_li = objContours(obj)
        contCoords['proj'][obj.label] = contours_li
    return contCoords
