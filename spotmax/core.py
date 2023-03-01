import os
import sys
import shutil
import traceback

from typing import Union
from tqdm import tqdm
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import scipy.stats
import scipy.ndimage

import cv2

import skimage.morphology
import skimage.measure
import skimage.transform
import skimage.filters
import skimage.feature
from scipy.special import erf

import acdctools.io
import acdctools.utils
import acdctools.measure

from . import GUI_INSTALLED, error_up_str, error_down_str

if GUI_INSTALLED:
    from acdctools.plot import imshow
    import matplotlib.pyplot as plt
    import matplotlib

try:
    import numba
    from numba import njit, prange
    NUMBA_INSTALLED = True
except Exception as e:
    NUMBA_INSTALLED = False
    from .utils import njit_replacement as njit
    prange = range

from . import utils, rng, base_lineage_table_values
from . import issues_url, printl, io, features, config

np.seterr(all='raise')

distribution_metrics_func = features.get_distribution_metric_func()
effect_size_func = features.get_effect_size_func()
aggregate_spots_feature_func = features.get_aggregating_spots_feature_func()

dfs_filenames = {
    'spots_detection': '*rn*_0_detected_spots*desc*.h5',
    'spots_gop': '*rn*_1_valid_spots*desc*.h5',
    'spots_spotfit': '*rn*_2_spotfit*desc*.h5'
}

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
        data = self._initialize_df_agg(data)
        return data
    
    def _critical_channel_not_found(self, channel, channel_path):
        self.logger.info(
            f'{error_down_str}'
            f'The channel {channel} was not found. If you are trying to load '
            'a channel without an extension make sure that one of the following '
            'channels exists:\n\n'
            f'   * {channel}.tif\n'
            f'   * {channel}.h5\n'
            f'   * {channel}_aligned.h5\n'
            f'   * {channel}_aligned.npz\n'
            f'   * {channel}.npy\n'
            f'   * {channel}.npz\n\n'
            'Alternatively, provide the extension to the channel name in the '
            '.ini configuration file.\n'
        )
        error = FileNotFoundError(f'The channel "{channel}" does not exist')
        self.logger.info(f'[ERROR]: {error}{error_up_str}')       
        self.quit()
    
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
            if not os.path.exists(ch_path):
                self._critical_channel_not_found(channel, ch_path)
                return

            self.log(f'Loading "{channel}" channel from "{ch_path}"...')
            import pdb; pdb.set_trace()
            to_float = key == 'spots_ch' or key == 'ref_ch'
            ch_data, ch_dtype = io.load_image_data(
                ch_path, to_float=to_float, return_dtype=True
            )
            data[f'{key}.dtype'] = ch_dtype
            data[key] = ch_data

        ch_key = 'spots_ch' if 'spots_ch' in data else 'ref_ch'
        data_shape = data[ch_key].shape
        data['is_segm_3D'] = True
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
                data['is_segm_3D'] = False
            else:
                # Snapshot data, stack on first axis (Z, Y, X)
                SizeZ = data_shape[0]
                data['segm'] = np.stack([data['segm']]*SizeZ, axis=0)
                data['is_segm_3D'] = False

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
    
    def _initialize_df_agg(self, data):
        segm_data = data['segm']
        frame_idxs = []
        IDs = []
        for frame_i in range(len(segm_data)):
            rp = data['segm_rp'][frame_i]
            for obj in rp:
                IDs.append(obj.label)
                frame_idxs.append(frame_i)
        df_data = {'frame_i': frame_idxs, 'Cell_ID': IDs}
        df_agg = pd.DataFrame(df_data).set_index(['frame_i', 'Cell_ID'])
        df_agg['analysis_datetime'] = pd.Timestamp.now()
        data['df_agg'] = df_agg

        if 'lineage_table' not in data:
            return data
        
        idx_segm = data['df_agg'].index
        idx_acdc_df = data['lineage_table'].index
        idx_both = idx_segm.intersection(idx_acdc_df)
        for col_name, value in base_lineage_table_values.items():
            data['df_agg'][col_name] = value
            data['df_agg'].loc[idx_both, col_name] = (
                data['lineage_table'].loc[idx_both, col_name]
            )
        return data

class _ParamsParser(_DataLoader):
    def __init__(self, debug=False, is_cli=True, log=print):
        super().__init__(debug=debug, log=log)
        self.debug = debug
        self.is_cli = is_cli
    
    def _check_report_filepath(
            self, report_folderpath, params_path, report_filename='', 
            force_default=False
        ):
        if report_folderpath and not os.path.isdir(report_folderpath):
            raise FileNotFoundError(
                'The provided path to the final report does not exist or '
                f'is not a folder path. Path: "{report_folderpath}"'
            )   

        if report_folderpath and report_filename:
            # User provided both folder path and filename for the report file
            report_filepath = os.path.join(report_folderpath, report_filename)
            return report_filepath
        
        report_filepath = self.get_default_report_filepath(params_path)
        if report_folderpath or force_default:
            # User provided folder path in .ini or as argument but not the filename
            return report_filepath
        
        report_rel_filepath = io.get_relpath(report_filepath)
        if report_filepath == report_rel_filepath:
            report_filepath_option = report_rel_filepath
        else:
            report_filepath_option = f'...{os.sep}{report_rel_filepath}'
        default_option = 'Save report to default path'
        options = (
            default_option, 'Save report to..', 'Do not save report'
        )
        info_txt = (
            'spotMAX can save a final report with a summary of warnings '
            'and errors raised during the analysis.\n\n'
            f'Default report path: "{report_filepath_option}"'
        )
        question = 'Where do you want to save the report'
        answer = io.get_user_input(
            question, options=options, info_txt=info_txt, 
            logger=self.logger.info, default_option=default_option
        )
        if answer is None:
            return
        if answer == default_option:
            return report_filepath
        if answer == 'Do not save report':
            return 'do_not_save'
        
        report_folderpath = acdctools.io.get_filename_cli(
            question='Insert the folder path where to save the report',
            check_exists=True, is_path=True
        )
        if report_folderpath is None:
            return
        report_filename = os.path.basename(report_filepath)
        report_filepath = os.path.join(report_folderpath, report_filename)
        
        return report_filepath

    def _check_exists_report_file(
            self, report_filepath, params_path, force_default=False
        ):
        report_default_filepath = self.get_default_report_filepath(params_path)
        report_default_filename = os.path.basename(report_default_filepath)

        if not os.path.exists(report_filepath) or force_default:
            return report_default_filepath
        
        new_report_filepath, txt = acdctools.path.newfilepath(report_filepath)
        new_report_filename = os.path.basename(new_report_filepath)

        
        
        default_option = f'Save with default filename "{report_default_filename}"'
        options = (
            default_option, f'Append "{txt}" to filename', 'Save as..', 
            'Do not save report'
        )
        info_txt = (
            'The provided report file already exists.\n\n'
            f'Report file path: "{report_filepath}"'
        )
        question = 'How do you want to proceed'
        answer = io.get_user_input(
            question, options=options, info_txt=info_txt, 
            logger=self.logger.info, default_option=default_option,
            format_vertical=True
        )
        if answer is None:
            return
        if answer == default_option:
            return report_default_filepath
        if answer == options[1]:
            return new_report_filepath
        if answer == 'Do not save report':
            return 'do_not_save'
        
        new_report_filename = acdctools.io.get_filename_cli(
            question='Write a filename for the report file',
            check_exists=False, is_path=False
        )
        if new_report_filename is None:
            return
        if not new_report_filepath.endswith('.rst'):
            new_report_filepath = f'{new_report_filepath}.rst'
        
        folder_path = os.path.dirname(report_filepath)
        new_report_filepath = os.path.join(folder_path, new_report_filename)
        return new_report_filepath


    def _check_numba_num_threads(self, num_threads, force_default=False):
        max_num_threads = numba.config.NUMBA_NUM_THREADS
        default_option = str(int(max_num_threads/2))
        if force_default:
            return int(default_option)
        options = [str(n) for n in range(1,max_num_threads+1)]
        info_txt = (
            'spotMAX can perform some of the analysis steps considerably faster '
            'through parallelisation across the available CPU threads.\n'
            'However, you might want to limit the amount of resources used.'
        )
        question = 'How many threads should spotMAX use'
        if num_threads<0 or num_threads>max_num_threads:
            num_threads = io.get_user_input(
                question, options=options, info_txt=info_txt, 
                logger=self.logger.info, default_option=default_option
            )
            if num_threads is None:
                return
            else:
                num_threads = int(num_threads)
        return num_threads

    def _check_raise_on_critical(self, force_default=False):
        info_txt = (
            'spotMAX default behaviour is to NOT stop the analysis process '
            'if a critical error is raised, but to continue with the analysis '
            'of the next folder.'
        )
        question = 'Do you want to stop the analysis process on critical error'
        default_option = 'no'
        if force_default:
            return False
        
        answer = io.get_user_input(
            question, options=None, info_txt=info_txt, 
            logger=self.logger.info, default_option=default_option
        )
        if answer is None:
            return
        elif answer == default_option:
            return False
        else:
            return True
    
    def add_parser_args_to_params_ini_file(self, parser_args, params_path):        
        configPars = config.ConfigParser()
        configPars.read(params_path, encoding="utf-8")
        SECTION = 'Configuration'
        if SECTION not in configPars.sections():
            configPars[SECTION] = {}
        
        config_default_params = config._configuration_params()
        for anchor, options in config_default_params.items():
            arg_name = options['parser_arg']
            value = parser_args[arg_name]
            configPars[SECTION][options['desc']] = str(value)
        
        with open(params_path, 'w', encoding="utf-8") as file:
            configPars.write(file)

    def _add_missing_args_from_params_ini_file(self, parser_args, params_path):
        if not params_path.endswith('.ini'):
            return parser_args

        configPars = config.ConfigParser()
        configPars.read(params_path, encoding="utf-8")
        SECTION = 'Configuration'
        if SECTION not in configPars.sections():
            return parser_args
        
        config_default_params = config._configuration_params()
        for anchor, options in config_default_params.items():
            option = configPars.get(SECTION, options['desc'], fallback=None)
            if option is None or not option:
                continue
            dtype_converter = options['dtype']
            value = dtype_converter(option)
            parser_args[options['parser_arg']] = value
            if anchor == 'raiseOnCritical':
                parser_args['raise_on_critical_present'] = True
        return parser_args
    
    @utils.exception_handler_cli
    def check_parsed_arguments(self, parser_args):
        params_path = parser_args['params']
        params_path = utils.check_cli_file_path(params_path)

        parser_args = self._add_missing_args_from_params_ini_file(
            parser_args, params_path
        )

        force_default = parser_args['force_default_values']

        metadata_path = parser_args['metadata']
        if metadata_path:
            metadata_path = utils.check_cli_file_path(
                metadata_path, desc='metadata'
            )
            parser_args['metadata'] = metadata_path
        
        disable_final_report = parser_args['disable_final_report']
        report_folderpath = parser_args['report_folderpath']

        if not disable_final_report:
            report_filepath = self._check_report_filepath(
                report_folderpath, params_path, force_default=force_default,
                report_filename=parser_args['report_filename']
            )
            report_filepath = self._check_exists_report_file(
                report_filepath, params_path, force_default=force_default
            )
            if report_filepath is None:
                self.logger.info(
                    'spotMAX execution stopped by the user. '
                    'Report filepath was not provided.'
                )
                self.quit()
                return

            if report_filepath == 'do_not_save':
                parser_args['disable_final_report'] = True
            parser_args['report_folderpath'] = os.path.dirname(report_filepath)
            parser_args['report_filename'] = os.path.basename(report_filepath)

        if NUMBA_INSTALLED:
            num_threads = int(parser_args['num_threads'])
            num_threads = self._check_numba_num_threads(
                num_threads, force_default=force_default
            )
            if num_threads is None:
                self.logger.info(
                    'spotMAX execution stopped by the user. '
                    'Number of threads was not provided.'
                )
                self.quit()
                return
            parser_args['num_threads'] = num_threads
        else:
            self.logger.info(
                '[WARNING]: numba not installed. '
                'Consider installing it with `pip install numba`. '
                'It will speed up analysis if you need to compute the spots size.'
            )
        
        raise_on_critical = parser_args['raise_on_critical']
        raise_on_critical_present = parser_args.get(
            'raise_on_critical_present', False
        )
        if not raise_on_critical and not raise_on_critical_present:
            raise_on_critical = self._check_raise_on_critical(
                force_default=force_default
            )
            if raise_on_critical is None:
                self.logger.info(
                    'spotMAX execution stopped by the user. '
                    '"Raise of critical" parameter was not provided.'
                )
                self.quit()
                return
            parser_args['raise_on_critical'] = raise_on_critical

        return parser_args

    @utils.exception_handler_cli
    def init_params(self, params_path, metadata_csv_path=''):        
        self._params = config.analysisInputsParams(params_path)
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
                    self.ini_params_file_path
                )
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
        proceed = self.check_paths_exist()
        if not proceed:
            return False
        self.set_abs_exp_paths()
        self.set_metadata()
        return True
    
    def _ask_user_save_ini_from_csv(self, ini_filepath):
        filename = os.path.basename(ini_filepath)
        ini_folderpath = os.path.dirname(ini_filepath)
        options = (
            'Overwrite existing file', 'Append number to the end', 
            'Save with a new name..'
        )
        default_option = 'Append number to the end'
        question = 'What do you want to do'
        txt = (
            f'[WARNING]: spotMAX would like to save the parameters in the file '
            f'"{filename}" (see full path below). '
            'However, this file already exists.\n\n'
            f'File path: "{ini_filepath}"'
        )
        if self._force_default:
            self.logger.info('*'*50)
            self.logger.info(txt)
            io._log_forced_default(default_option, self.logger.info)
            answer = default_option
        else:
            answer = io.get_user_input(
                question, options=options, info_txt=txt, 
                logger=self.logger.info, default_option=default_option
            )
        if not answer:
            return False
        if answer == 'Overwrite existing file':
            return True
        elif answer == 'Append number to the end':
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

    def _ask_user_run_num_exists(self, user_run_num, run_nums):
        default_option = f'Overwrite existing run number {user_run_num}'
        options = ('Choose a different run number', default_option )
        question = 'What do you want to do'
        txt = (
            f'[WARNING]: The requested run number {user_run_num} already exists! '
            f'(run numbers presents are {run_nums})'
        )
        if self._force_default:
            self.logger.info('*'*50)
            self.logger.info(txt)
            io._log_forced_default(default_option, self.logger.info)
            return user_run_num
        answer = io.get_user_input(
            question, options=options, info_txt=txt, logger=self.logger.info
        )
        if answer is None:
            return
        if answer == default_option:
            return user_run_num
        
        question = 'Insert an integer greater than 0 for the run number'
        user_run_num = io.get_user_input(question, dtype='uint')
        return user_run_num

    def _ask_user_multiple_run_nums(self, run_nums):
        new_run_num = max(run_nums)+1
        default_option = f'Save as new run number {new_run_num}'
        options = ('Choose run number to overwrite', default_option )
        question = 'What do you want to do'
        txt = (
            '[WARNING]: All or some of the experiments present in the loaded '
            'folder have already been analysed before '
            f'(run numbers presents are {run_nums})'
        )
        if self._force_default:
            self.logger.info('*'*50)
            self.logger.info(txt)
            io._log_forced_default(default_option, self.logger.info)
            return new_run_num

        answer = io.get_user_input(
            question, options=options, info_txt=txt, logger=self.logger.info
        )
        if answer == options[1]:
            return new_run_num
        elif answer == options[0]:
            options = [f'Run number {r}' for r in run_nums]
            question = 'Which run number do you want to overwrite?'
            new_run_num_txt = io.get_user_input(question, options=options)
            new_run_num = int(new_run_num_txt[11:])
            return new_run_num
    
    def _store_run_number(self, run_number, pathScannerPaths, exp_paths):
        if exp_paths:
            for exp_path in list(exp_paths.keys()):
                exp_paths[exp_path]['run_number'] = run_number
        else:
            exp_paths = {}
            for run_num, run_num_info in pathScannerPaths.items():
                for exp_path, exp_info in run_num_info.items():
                    if exp_path in exp_paths:
                        continue
                    exp_paths[exp_path] = {
                        'pos_foldernames': exp_info['posFoldernames'],
                        'run_number': run_number
                    }
        # Store in .ini file
        configPars = config.ConfigParser()
        configPars.read(self.ini_params_file_path, encoding="utf-8")
        SECTION = 'File paths and channels'
        if SECTION not in configPars.sections():
            configPars[SECTION] = {}
        ANCHOR = 'runNumber'
        option = self._params[SECTION][ANCHOR]['desc']
        configPars[SECTION][option] = str(run_number)

        with open(self.ini_params_file_path, 'w', encoding="utf-8") as file:
            configPars.write(file)


    @utils.exception_handler_cli
    def set_abs_exp_paths(self):
        SECTION = 'File paths and channels'
        ANCHOR = 'filePathsToAnalyse'
        loaded_exp_paths = self._params[SECTION][ANCHOR]['loadedVal']
        user_run_number = self._params[SECTION]['runNumber'].get('loadedVal')
        self.exp_paths_list = []
        for exp_path in loaded_exp_paths:
            if io.is_pos_path(exp_path):
                pos_path = exp_path
                pos_foldername = os.path.basename(exp_path)
                exp_path = os.path.dirname(pos_path)
                exp_paths = (
                    {exp_path: {'pos_foldernames': [pos_foldername]}}
                )
            elif io.is_images_path(exp_path):
                pos_path = os.path.dirname(exp_path)
                pos_foldername = os.path.basename(pos_path)
                exp_path = os.path.dirname(os.path.dirname(pos_path))
                exp_paths = (
                    {exp_path: {'pos_foldernames': [pos_foldername]}}
                )
            else:
                exp_paths = {}
            
            # Scan and determine run numbers
            pathScanner = io.expFolderScanner(exp_path)
            pathScanner.getExpPaths(exp_path)
            pathScanner.infoExpPaths(pathScanner.expPaths)
            run_nums = sorted([int(r) for r in pathScanner.paths.keys()])

            if len(run_nums) > 1 and user_run_number is None:
                # Multiple run numbers detected
                run_number = self._ask_user_multiple_run_nums(run_nums)
                if run_number is None:
                    self.logger.info(
                        'spotMAX stopped by the user. Run number was not provided.'
                    )
                    self.quit()
            elif user_run_number is None:
                # Single run number --> we still need to check if already exists
                ask_run_number = False
                for exp_path, exp_info in pathScanner.paths[run_nums[0]].items():
                    if exp_info['numPosSpotCounted'] > 0:
                        ask_run_number = True
                        break
                else:
                    run_number = 1
                
                if ask_run_number:
                    run_number = self._ask_user_multiple_run_nums(run_nums)
                    if run_number is None:
                        self.logger.info(
                            'spotMAX stopped by the user.'
                            'Run number was not provided.'
                        )
                        self.quit()
            elif user_run_number is not None:
                # Check that user run number is not already existing
                if user_run_number in run_nums:
                    run_num_info = pathScanner.paths[user_run_number]
                    ask_run_number = False
                    for exp_path, exp_info in run_num_info.items():
                        if exp_info['numPosSpotCounted'] > 0:
                            ask_run_number = True
                            break
                    
                    if ask_run_number:
                        user_run_number = self._ask_user_run_num_exists(
                            user_run_number, run_nums
                        )
                        if user_run_number is None:
                            self.logger.info(
                                'spotMAX stopped by the user.'
                                'Run number was not provided.'
                            )
                            self.quit()
                    
                run_number = user_run_number
            self._store_run_number(run_number, pathScanner.paths, exp_paths)
            self.exp_paths_list.append(exp_paths)
        self.set_channel_names()
    
    def set_channel_names(self):
        SECTION = 'File paths and channels'
        section_params = self._params[SECTION]
        spots_ch_endname = section_params['spotsEndName'].get('loadedVal')
        ref_ch_endname = section_params['refChEndName'].get('loadedVal')
        segm_endname = section_params['segmEndName'].get('loadedVal')
        ref_ch_segm_endname = section_params['refChSegmEndName'].get('loadedVal')
        lineage_table_endname = section_params['lineageTableEndName'].get(
            'loadedVal'
        )
        text_to_append = section_params['textToAppend'].get(
            'loadedVal', ''
        )
        if self.exp_paths_list:
            for i in range(len(self.exp_paths_list)):
                for exp_path in list(self.exp_paths_list[i].keys()):
                    exp_info = self.exp_paths_list[i][exp_path]
                    exp_info['spotsEndName'] = spots_ch_endname
                    exp_info['refChEndName'] = ref_ch_endname
                    exp_info['segmEndName'] = segm_endname
                    exp_info['refChSegmEndName'] = ref_ch_segm_endname
                    exp_info['lineageTableEndName'] = lineage_table_endname
                    exp_info['textToAppend'] = text_to_append
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
        metadata['zyxVoxelSize'] = (
            physical_size_z, physical_size_y, physical_size_x
        )
        z_resolution_limit_um = metadata['zResolutionLimit']
        yx_resolution_multiplier = metadata['yxResolLimitMultiplier']
        zyx_resolution_limit_pxl, zyx_resolution_limit_um = calcMinSpotSize(
            emission_wavelen, num_aperture, physical_size_x, 
            physical_size_y, physical_size_z, z_resolution_limit_um, 
            yx_resolution_multiplier
        )
        metadata['zyxResolutionLimitPxl'] = zyx_resolution_limit_pxl
        metadata['zyxResolutionLimitUm'] = zyx_resolution_limit_um

    def _add_physical_units_conversion_factors(self, metadata):
        PhysicalSizeX = metadata.get('pixelWidth', 1)
        PhysicalSizeY = metadata.get('pixelHeight', 1)
        PhysicalSizeZ = metadata.get('voxelDepth', 1)
        pxl_to_um2 = PhysicalSizeY*PhysicalSizeX
        vox_to_um3 = PhysicalSizeY*PhysicalSizeX*PhysicalSizeZ
        vox_to_fl_rot = float(PhysicalSizeY)*(float(PhysicalSizeX)**2)
        metadata['vox_to_um3_factor'] = vox_to_um3
        metadata['pxl_to_um2_factor'] = pxl_to_um2
        metadata['vox_to_fl_rot_factor'] = vox_to_fl_rot

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
        self._add_physical_units_conversion_factors(self.metadata)

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
        print('*'*50)
        err_msg = (
            f'The parameters file "{self.ini_params_filename}" is missing '
            'the following REQUIRED metadata:\n\n'
            f'{missing_metadata_format}\n\n'
            'Add them to the file (see path below) '
            'at the [METADATA] section. If you do not have timelapse data and\n'
            'the "Analyse until frame number" is missing you need to\n'
            'to write "Analyse until frame number = 1".\n\n'
            f'Parameters file path: "{self.ini_params_file_path}"\n'
        )
        self.logger.info(err_msg)
        if self.is_cli:
            print('*'*50)
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
            if anchor == 'runNumber':
                # We do not force any run number, this will be determined later.
                continue
            self._params[section_name][anchor]['loadedVal'] = default_val
    
    def _get_default_values_params(self, missing_params):
        default_values_format = []
        for param in missing_params:
            section_name, desc, default_val, anchor = param
            if anchor == 'runNumber':
                default_val = (
                    '1 for never analysed data. '
                    'Determined later for previously analysed data.'
                )
            if not default_val:
                default_val = 'Empty text --> Ignored.'
            s = f'    * {desc} (section: [{section_name}]) = {default_val}' 
            default_values_format.append(s)
        default_values_format = '\n'.join(default_values_format)
        return default_values_format

    def _ask_user_input_missing_params(self, missing_params, info_txt):
        question = (
            'Do you want to continue with default value for the missing parameters?'
        )
        options = (
            'Yes, use default values', 'No, stop process', 
            'Display default values'
        )
        if self._force_default:
            self.logger.info('*'*50)
            self.logger.info(info_txt)
            io._log_forced_default(options[0], self.logger.info)
            return True
        
        while True:
            answer = io.get_user_input(
                question, options=options, info_txt=info_txt, 
                logger=self.logger.info
            )
            if answer == 'No, stop process' or answer == None:
                return False
            elif answer == 'Yes, use default values':
                self._set_default_val_params(missing_params)
                return True
            else:
                print('')
                default_values_format = self._get_default_values_params(
                    missing_params
                )
                self.logger.info(
                    f'Default values:\n\n{default_values_format}'
                )
                print('-'*50)
                info_txt = ''
    
    def _check_correlated_missing_ref_ch_params(self, missing_params):
        missing_ref_ch_msg = ''
        missing_params_desc = {param[1]:param[2] for param in missing_params}
        if 'Reference channel end name or path' not in missing_params_desc:
            return missing_ref_ch_msg
        
        # Reference channel end name is missing, check that it is not required
        for anchor, options in self._params['Reference channel'].items():
            value = options['loadedVal']
            if not isinstance(value, bool):
                continue
            if value:
                # At least one option suggests tha ref. channel is required.
                break
        else:
            return missing_ref_ch_msg

        missing_ref_ch_msg = (
            '[ERROR]: You requested to use the reference channel for the analysis '
            'but the entry "Reference channel end name or path" is missing in the '
            '.ini params file.\n\n'
        )

        return missing_ref_ch_msg
    
    def _check_missing_exp_folder(self, missing_params):
        missing_exp_folder_msg = ''
        missing_params_desc = {param[1]:param[2] for param in missing_params}
        if 'Experiment folder path(s) to analyse' not in missing_params_desc:
            return missing_exp_folder_msg
        
        # Experiment folder path is missing --> continue only if 
        # either spots or reference channel are proper file paths
        spots_ch_path = missing_params_desc.get(
            'Spots channel end name or path', ''
        )
        ref_ch_path = missing_params_desc.get(
            'Reference channel end name or path', ''
        )
        is_critical = not (
            os.path.exists(spots_ch_path) or os.path.exists(ref_ch_path)
        )   
        if not is_critical:
            return missing_exp_folder_msg
        
        missing_exp_folder_msg = (
            '[ERROR]: Neither the "Spots channel end name" nor the '
            '"Reference channel end name or path" are present in the .ini params file.\n\n'
        )
        return missing_exp_folder_msg    

    @utils.exception_handler_cli
    def check_missing_params(self):
        missing_params = self._get_missing_params()
        if not missing_params:
            return
        
        missing_exp_folder_msg = self._check_missing_exp_folder(missing_params)
        missing_ref_ch_msg = self._check_correlated_missing_ref_ch_params(
            missing_params
        )

        is_missing_critical = (
            missing_exp_folder_msg or missing_ref_ch_msg
        )

        missing_params_str = [
            f'    * {param[1]} (section: [{param[0]}])' 
            for param in missing_params
        ]
        missing_params_format = '\n'.join(missing_params_str)
        print('*'*50)
        err_msg = (
            f'[WARNING]: The configuration file "{self.ini_params_filename}" is missing '
            'the following parameters:\n\n'
            f'{missing_params_format}\n\n'
        )
        
        if is_missing_critical:
            err_msg = (f'{err_msg}'
                'Add them to the file (see path below) '
                'at the right section (shown in parethensis above).\n\n'
                f'{missing_exp_folder_msg}'
                f'{missing_ref_ch_msg}'
                f'Parameters file path: "{self.ini_params_file_path}"\n'
            )
            self.logger.info(err_msg)
            if self.is_cli:
                print('*'*50)
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
                'with default values.\n\n'
                f'Parameters file path: "{self.ini_params_file_path}"\n'
            )
            proceed = self._ask_user_input_missing_params(
                missing_params, info_txt=err_msg
            )
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
                try:
                    value = to_dtype(value)
                except Exception as e:
                    value = None
                self._params[section_name][anchor_name]['loadedVal'] = value
    
    def check_paths_exist(self):
        SECTION = 'File paths and channels'
        ANCHOR = 'filePathsToAnalyse'
        loaded_exp_paths = self._params[SECTION][ANCHOR]['loadedVal']
        for exp_path in loaded_exp_paths:
            if not os.path.exists(exp_path):
                self.logger.info('='*50)
                txt = (
                    '[ERROR]: The provided experiment path does not exist: '
                    f'{exp_path}{error_up_str}'
                )
                self.logger.info(txt)
                self.logger.info('spotMAX aborted due to ERROR. See above more details.')
                return False
            if not os.path.isdir(exp_path):
                self.logger.info('='*50)
                txt = (
                    '[ERROR]: The provided experiment path is not a folder: '
                    f'{exp_path}{error_up_str}'
                )
                self.logger.info(txt)
                self.logger.info('spotMAX aborted due to ERROR. See above more details.')
                return False
        return True

class _GaussianModel:
    def __init__(self, nfev=0):
        pass

    @staticmethod
    @njit(parallel=True)
    def jac_gauss3D(coeffs, data, z, y, x, num_spots, num_coeffs, const=0):
        # Gradient ((m,n) Jacobian matrix):
        # grad[i,j] = derivative of f[i] wrt coeffs[j]
        # e.g. m data points with n coeffs --> grad with m rows and n col
        grad = np.empty((len(z), num_coeffs*num_spots))
        ns = np.arange(0,num_coeffs*num_spots,num_coeffs)
        for i in prange(num_spots):
            n = ns[i]
            coeffs_i = coeffs[n:n+num_coeffs]
            z0, y0, x0, sz, sy, sx, A, B = coeffs_i
            # Center rotation around peak center
            zc = z - z0
            yc = y - y0
            xc = x - x0
            # Build 3D gaussian by multiplying each 1D gaussian function
            gauss_x = np.exp(-(xc**2)/(2*(sx**2)))
            gauss_y = np.exp(-(yc**2)/(2*(sy**2)))
            gauss_z = np.exp(-(zc**2)/(2*(sz**2)))
            f_x = 1/(sx*np.sqrt(2*np.pi))
            f_y = 1/(sy*np.sqrt(2*np.pi))
            f_z = 1/(sz*np.sqrt(2*np.pi))
            g = gauss_x*gauss_y*gauss_z
            f = f_x*f_y*f_z
            fg = f*g

            # Partial derivatives
            d_g_sz = g * zc**2 / (sz**3)
            d_f_sz = A/(np.sqrt(2*np.pi)*(sz**2))
            d_fg_sz = g*d_f_sz + f*d_g_sz

            d_g_sy = g * yc**2 / (sy**2)
            d_f_sy = -A/(np.sqrt(2*np.pi)*(sy**2))
            d_fg_sy = g*d_f_sz + f*d_g_sz

            d_g_sx = g * xc**2 / (sx**2)
            d_f_sx = A/(np.sqrt(2*np.pi)*(sx**2))
            d_fg_sx = g*d_f_sz + f*d_g_sz

            # Gradient array
            grad[:,n] = A*fg * zc / (sz**2) # wrt zc
            grad[:,n+1] = A*fg * yc / (sy**2) # wrt yc
            grad[:,n+2] = A*fg * xc / (sx**2) # wrt xc
            grad[:,n+3] = d_fg_sz # wrt sz
            grad[:,n+4] = d_fg_sy # wrt sy
            grad[:,n+5] = d_fg_sx # wrt sx
            grad[:,n+6] = fg # wrt A
        grad[:,-1] = np.ones(len(x)) # wrt B
        return -grad

    @staticmethod
    @njit(parallel=False)
    def _gauss3D(z, y, x, coeffs, num_spots, num_coeffs, const):
        model = np.zeros(len(z))
        n = 0
        B = coeffs[-1]
        for i in range(num_spots):
            coeffs_i = coeffs[n:n+num_coeffs]
            z0, y0, x0, sz, sy, sx, A = coeffs_i
            # Center rotation around peak center
            zc = z - z0
            yc = y - y0
            xc = x - x0
            # Build 3D gaussian by multiplying each 1D gaussian function
            gauss_x = np.exp(-(xc**2)/(2*(sx**2)))
            gauss_y = np.exp(-(yc**2)/(2*(sy**2)))
            gauss_z = np.exp(-(zc**2)/(2*(sz**2)))
            model += A*gauss_x*gauss_y*gauss_z
            n += num_coeffs
        return model + const + B

    def gaussian_3D(self, z, y, x, coeffs, B=0):
        """Non-NUMBA version of the model"""
        z0, y0, x0, sz, sy, sx, A = coeffs
        # Center rotation around peak center
        zc = z - z0
        yc = y - y0
        xc = x - x0
        # Build 3D gaussian by multiplying each 1D gaussian function
        gauss_x = np.exp(-(xc**2)/(2*(sx**2)))
        gauss_y = np.exp(-(yc**2)/(2*(sy**2)))
        gauss_z = np.exp(-(zc**2)/(2*(sz**2)))
        return A*gauss_x*gauss_y*gauss_z

    def compute_const(self, z, y, x, const_coeffs):
        const = 0
        for const_c in const_coeffs:
            const += self.gaussian_3D(z, y, x, const_c)
        return const

    def residuals(self, coeffs, data, z, y, x, num_spots, num_coeffs, const=0):
        self.pbar.update(1)
        f = self._gauss3D
        return data - f(z, y, x, coeffs, num_spots, num_coeffs, const)

    def goodness_of_fit(self, y_obs, y_model, ddof, is_linear_regr=False):
        # Degree of freedom
        N = len(y_obs)
        dof = N-ddof

        # Reduced chi square
        try:
            # Normalize to sum 1
            y_obs_chi = y_obs/y_obs.sum()
            y_model_chi = y_model/y_model.sum()
            chisq, p_chisq = scipy.stats.chisquare(
                y_obs_chi, y_model_chi, ddof=ddof
            )
            reduced_chisq = chisq/dof
        except:
            chisq = 0
            p_chisq = 1
            reduced_chisq = 0
            print('WARNING: error calculating chisquare')

        # Sum of squared errors
        SSE = np.sum(np.square(y_obs-y_model))
        # Total sum of squares
        y_mean = y_obs.mean()
        SST = np.sum(np.square(y_obs-y_mean))
        # NOTE: R-square is valid ONLY for linear regressions
        R_sq = 1 - (SSE/SST)
        # Adjusted R squared
        adj_Rsq = 1 - (((1-R_sq)*(N-1))/(N-ddof-1))

        # Root mean squared error (a.k.a "standard error of the regression")
        RMSE = np.sqrt(SSE/dof)
        # Normalized mean squared error
        NRMSE = RMSE/y_mean
        # Frank relative NRMSE (i.e. NRMSE normalized to 0,1
        # with 1 being perfect fit)
        F_NRMSE = 2/(1+np.exp(NRMSE))

        # Kolmogorov–Smirnov test
        ks, p_ks = scipy.stats.ks_2samp(y_obs, y_model)
        if is_linear_regr:
            return (reduced_chisq, p_chisq, R_sq, RMSE, ks, p_ks, adj_Rsq,
                    NRMSE, F_NRMSE)
        else:
            return reduced_chisq, p_chisq, RMSE, ks, p_ks, NRMSE, F_NRMSE

    def get_bounds_init_guess(self, num_spots_s, num_coeffs, fit_ids,
                              fit_idx, spots_centers, spots_3D_lab_ID,
                              spots_rp, spots_radii_pxl, spots_img,
                              spots_Bs_guess, spots_B_mins):

        low_limit = np.zeros(num_spots_s*num_coeffs+1)
        high_limit = np.zeros(num_spots_s*num_coeffs+1)
        init_guess_s = np.zeros(num_spots_s*num_coeffs+1)
        n = 0
        # center bounds limit
        xy_cbl = 0.2
        z_cbl = 0.1
        # Sigma bound limit multiplier
        s_f = 3
        _pi_f = np.sqrt(2*np.pi)
        max_s_z = spots_radii_pxl[:,0].max()
        max_s_yx = spots_radii_pxl[:,1].max()
        B_min = min([spots_B_mins[i] for i in fit_idx])
        A_max = max([spots_img[spots_3D_lab_ID==obj.label].sum()
                     for obj in spots_rp])+1
        for i, id in zip(fit_idx, fit_ids):
            z0, y0, x0 = spots_centers[i]
            c, b, a = spots_radii_pxl[i]
            B_guess = spots_Bs_guess[i]
            spot_mask = spots_3D_lab_ID == id
            raw_vals = spots_img[spot_mask]
            # A_min = np.sum(raw_vals-raw_vals.min())
            A_guess = np.sum(raw_vals)/num_spots_s
            # z0, y0, x0, sz, sy, sx, A = coeffs
            low_lim = np.array([z0-z_cbl, y0-xy_cbl, x0-xy_cbl,
                                 0.5, 0.5, 0.5, 0])
            high_lim = np.array([z0+z_cbl, y0+xy_cbl, x0+xy_cbl,
                                 max_s_z, max_s_yx, max_s_yx, A_max])
            guess = np.array([z0, y0, x0, c, b, a, A_guess])
            low_limit[n:n+num_coeffs] = low_lim
            high_limit[n:n+num_coeffs] = high_lim
            init_guess_s[n:n+num_coeffs] = guess
            n += num_coeffs
        low_limit[-1] = B_min
        high_limit[-1] = np.inf
        init_guess_s[-1] = B_guess
        bounds = (low_limit, high_limit)
        return bounds, init_guess_s

    def integrate(self, zyx_center, zyx_sigmas, A, B,
                  sum_obs=0, lower_bounds=None, upper_bounds=None,
                  verbose=0):
        """Integrate Gaussian peaks with erf function.

        Parameters
        ----------
        zyx_center : (3,) ndarray
            [zc, yc, xc] ndarray centre coordinates of the peak
        zyx_sigmas : (3,) ndarray
            [zs, ys, xs] ndarray sigmas of the peak.
        A : float
            Amplitude of the peak
        B : float
            Background level of the peak
        lower_bounds : ndarray
            [z, y, x] lower bounds of the integration volume. If None the
            lower bounds will be equal to -1.96*zyx_sigmas (95%)
        upper_bounds : ndarray
            [z, y, x] upper bounds of the integration volume. If None the
            upper bounds will be equal to 1.96*zyx_sigmas (95%)
        sum_obs: float
            Printed alongside with the returned I_tot is verbose==3. Used for
            debugging to check that sum_obs and I_tot are in the same order
            of magnitude.


        Returns
        -------
        I_tot: float
            Result of the total integration.
        I_foregr: float
            Result of foregroung integration (i.e. background subtracted).

        """
        # Center gaussian to peak center coords
        if lower_bounds is None:
            # Use 95% of peak as integration volume
            zyx_c1 = -1.96 * zyx_sigmas
        else:
            zyx_c1 = lower_bounds - zyx_center
        if upper_bounds is None:
            zyx_c2 = 1.96 * zyx_sigmas
        else:
            zyx_c2 = upper_bounds - zyx_center

        # Substitute variable x --> t to apply erf
        t_z1, t_y1, t_x1 = zyx_c1 / (np.sqrt(2)*zyx_sigmas)
        t_z2, t_y2, t_x2 = zyx_c2 / (np.sqrt(2)*zyx_sigmas)
        s_tz, s_ty, s_tx = (zyx_sigmas) * np.sqrt(np.pi/2)
        D_erf_z = erf(t_z2)-erf(t_z1)
        D_erf_y = erf(t_y2)-erf(t_y1)
        D_erf_x = erf(t_x2)-erf(t_x1)
        I_foregr = A * (s_tz*s_ty*s_tx) * (D_erf_z*D_erf_y*D_erf_x)
        I_tot = I_foregr + (B*np.prod(zyx_c2-zyx_c1, axis=0))
        if verbose==3:
            print('--------------')
            print(f'Total integral result, observed sum = {I_tot}, {sum_obs}')
            print(f'Foregroung integral values: {I_foregr}')
            print('--------------')
        return I_tot, I_foregr

class spheroid:
    def __init__(self, V_ch):
        self.V_ch = V_ch
        self.V_shape = V_ch.shape
        Z, Y, X = self.V_shape

    def calc_semiax_len(self, i, zyx_vox_dim, zyx_resolution):
        zvd, yvd, xvd = zyx_vox_dim
        zr, yr, xr = zyx_resolution
        xys = yr + (yvd*i)  # a radius in real units
        zs = zr + (yvd*i)  # c radius in real units
        self.xys = xys
        self.zs = zs
        a = xys/yvd  # a radius in pixels (xy direction)
        c = zs/zvd  # c radius in pixels (z direction)
        return a, c

    def get_backgr_vals(self, zyx_c, semiax_len, V, spot_id):
        spot_surf_mask, spot_filled_mask = self.get_sph_surf_mask(
                                                    semiax_len,
                                                    zyx_c,
                                                    self.V_shape,
                                                    return_filled_mask=True)
        surf_pixels = V[spot_surf_mask]
        surf_mean = np.mean(surf_pixels)
        return surf_mean, spot_filled_mask

    def get_sph_surf_mask(self, semiax_len, zyx_center, V_shape,
                          return_filled_mask=False):
        """
        Generate a spheroid surface mask array that can be used to index a 3D array.
        ogrid is given by
        Z, Y, X = V.shape
        z, y, x = np.ogrid[0:Z, 0:Y, 0:X]

        The spheroid is generated by logical_xor between two spheroids that have
        1 pixel difference between their axis lengths
        """
        a, c = semiax_len
        # Outer full mask
        s_outer = self.get_local_spot_mask(semiax_len)
        a_inner = a-1
        # Make sure than c_inner is never exactly 0
        c_inner = c-1 if c-1 != 0 else c-1+1E-15
        # Inner full mask with same shape as outer mask
        s_inner = self.get_local_spot_mask((a_inner, c_inner),
                                            ogrid_bounds=semiax_len)
        # Surface mask (difference between outer and inner)
        spot_surf_mask = np.logical_xor(s_outer, s_inner)
        # Insert local mask into global
        spot_mask = self.get_global_spot_mask(spot_surf_mask, zyx_center,
                                                              semiax_len)
        if return_filled_mask:
            spot_mask_filled = self.get_global_spot_mask(
                                         s_outer, zyx_center, semiax_len)
            return spot_mask, spot_mask_filled
        else:
            return spot_mask

    def calc_mean_int(self, i, semiax_len, zyx_centers, V):
        V_shape = self.V_shape
        intens = [np.mean(V[self.get_sph_surf_mask(semiax_len,
                                                   zyx_c, V_shape)])
                                                   for zyx_c in zyx_centers]
        return intens

    def filled_mask_from_um(self, zyx_vox_dim, sph_z_um, sph_xy_um, zyx_center):
        zc, yc, xc = zyx_center
        z_vd, y_vd, x_vd = zyx_vox_dim
        a = sph_xy_um/y_vd
        c = sph_z_um/z_vd
        local_mask = self.get_local_spot_mask((a, c))
        spot_mask = self.get_global_spot_mask(local_mask, zyx_center, (a, c))
        return spot_mask

    def intersect2D(self, a, b):
        """
        Return intersecting rows between two 2D arrays 'a' and 'b'
        """
        tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
        return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]

    def get_local_spot_mask(self, semiax_len, ogrid_bounds=None,
                            return_center=False):
        a, c = semiax_len
        if ogrid_bounds is None:
            a_int = int(np.ceil(a))
            c_int = int(np.ceil(c))
        else:
            o_yx, o_z = ogrid_bounds
            a_int = int(np.ceil(o_yx))
            c_int = int(np.ceil(o_z))
        # Generate a sparse meshgrid to evaluate 3D spheroid mask
        z, y, x = np.ogrid[-c_int:c_int+1, -a_int:a_int+1, -a_int:a_int+1]
        # 3D spheroid equation
        mask_s = (x**2 + y**2)/(a**2) + z**2/(c**2) <= 1
        if return_center:
            return mask_s, None
        else:
            return mask_s

    def get_global_spot_mask(self, local_spot_mask, zyx_center, semiax_len,
                             additional_local_arr=None):
        spot_mask = np.zeros(self.V_shape, local_spot_mask.dtype)
        if additional_local_arr is not None:
            additional_global_arr = np.zeros(self.V_shape,
                                              additional_local_arr.dtype)
        else:
            additional_global_arr = None
        Z, Y, X = self.V_shape
        spot_mask, spot_mask_2 = self.index_local_into_global_mask(
                                 spot_mask, local_spot_mask,
                                 zyx_center, semiax_len, Z, Y, X,
                                 additional_global_arr=additional_global_arr,
                                 additional_local_arr=additional_local_arr
        )
        if additional_local_arr is not None:
            return spot_mask, spot_mask_2
        else:
            return spot_mask

    def get_slice_G_to_L(self, semiax_len, zyx_c, Z, Y, X):
        a, c = semiax_len
        a_int = int(np.ceil(a))
        c_int = int(np.ceil(c))
        zc, yc, xc = zyx_c

        z_min = zc-c_int
        z_max = zc+c_int+1
        z_min_crop, z_max_crop = None, None
        y_min_crop, y_max_crop = None, None
        x_min_crop, x_max_crop = None, None

        # Check z size and crop if needed
        if z_min < 0:
            z_min_crop = abs(z_min)
            z_min = 0
        if z_max > Z:
            z_max_crop = Z-z_max
            z_max = Z

        # Check y size and crop if needed
        y_min = yc-a_int
        y_max = yc+a_int+1
        if y_min < 0:
            y_min_crop = abs(y_min)
            y_min = 0
        if y_max > Y:
            y_max_crop = Y-y_max
            y_max = Y

        # Check x size and crop if needed
        x_min = xc-a_int
        x_max = xc+a_int+1
        if x_min < 0:
            x_min_crop = abs(x_min)
            x_min = 0
        if x_max > X:
            x_max_crop = X-x_max
            x_max = X

        slice_G_to_L = (slice(z_min,z_max),
                        slice(y_min,y_max),
                        slice(x_min,x_max))
        slice_crop = (slice(z_min_crop,z_max_crop),
                      slice(y_min_crop,y_max_crop),
                      slice(x_min_crop,x_max_crop))
        return slice_G_to_L, slice_crop


    def index_local_into_global_mask(self, global_mask, local_mask, zyx_c,
                                       semiax_len, Z, Y, X,
                                       additional_global_arr=None,
                                       additional_local_arr=None,
                                       do_sum=False, return_slice=False):
        """
        Insert local spot mask (which has shape = size of the spot)
        into global mask (which has shape = shape of V_spots).
        If the size of the local spot exceeds the bounds of V_spots it is
        cropped before being inserted.
        """
        slice_G_to_L, slice_crop = self.get_slice_G_to_L(
                                                     semiax_len, zyx_c, Z, Y, X)

        cropped_mask = local_mask[slice_crop].copy()

        if additional_local_arr is not None:
            cropped_mask_2 = additional_local_arr[slice_crop].copy()

        try:
            if do_sum:
                global_mask[slice_G_to_L] += cropped_mask
            else:
                global_mask[slice_G_to_L][cropped_mask] = True
            if additional_local_arr is not None:
                additional_global_arr[slice_G_to_L] = cropped_mask_2
        except:
            traceback.print_exc()
            print(Z, Y, X)
            print(zyx_c)
            print(slice_G_to_L)
            print(slice_crop)
            print(cropped_mask.shape)
            import pdb; pdb.set_trace()
        if additional_local_arr is not None:
            if return_slice:
                return (global_mask, additional_global_arr,
                        slice_G_to_L, slice_crop)
            else:
                return global_mask, additional_global_arr
        else:
            if return_slice:
                return global_mask, None, slice_G_to_L, slice_crop
            else:
                return global_mask, None

    def insert_grown_spot_id(self, grow_step_i, id, zyx_vox_dim,
                             zyx_resolution, zyx_c, spots_3D_lab):
        a, c = self.calc_semiax_len(grow_step_i, zyx_vox_dim,
                                    zyx_resolution)
        semiax_len = (np.ceil(a), np.ceil(c))
        local_spot_mask = self.get_local_spot_mask(semiax_len)
        Z, Y, X = self.V_shape

        slice_G_to_L, slice_crop = self.get_slice_G_to_L(
                                    semiax_len, zyx_c, Z, Y, X)
        cropped_mask = local_spot_mask[slice_crop]
        # Avoid spot overwriting existing spot
        cropped_mask[spots_3D_lab[slice_G_to_L] != 0] = False
        spots_3D_lab[slice_G_to_L][cropped_mask] = id
        return spots_3D_lab

    def get_spots_mask(self, i, zyx_vox_dim, zyx_resolution, zyx_centers,
                       method='min_spheroid', dtype=bool, ids=[]):
        if method == 'min_spheroid':
            Z, Y, X = self.V_shape
            # Calc spheroid semiaxis lengths in pixels (c: z, a: x and y)
            semiax_len = self.calc_semiax_len(i, zyx_vox_dim, zyx_resolution)
            local_spot_mask = self.get_local_spot_mask(semiax_len)
            # Pre-allocate arrays
            spots_mask = np.zeros(self.V_shape, dtype)
            temp_mask = np.zeros(self.V_shape, bool)
            # Insert local spot masks into global mask
            in_pbar = tqdm(
                desc='Building spots mask', total=len(zyx_centers),
                unit=' spot', leave=False, position=4, ncols=100
            )
            for i, zyx_c in enumerate(zyx_centers):
                (temp_mask, _, slice_G_to_L,
                slice_crop) = self.index_local_into_global_mask(
                                                 temp_mask, local_spot_mask,
                                                 zyx_c, semiax_len, Z, Y, X,
                                                 return_slice=True
                )
                if dtype == bool:
                    spots_mask = np.logical_or(spots_mask, temp_mask)
                elif dtype == np.uint16:
                    cropped_mask = local_spot_mask[slice_crop]
                    spots_mask[slice_G_to_L][cropped_mask] = ids[i]
                in_pbar.update(1)
            in_pbar.close()
        elif method == 'unsharp_mask':
            # result = unsharp_mask(self.V_ch, radius=10, amount=5,
            #                       preserve_range=True)
            blurred = skimage.filter.gaussian(self.V_ch, sigma=3)
            sharp = self.V_ch - blurred
            th = skimage.filters.threshold_isodata(sharp.max(axis=0))
            spots_mask = sharp > th
        return spots_mask

    def calc_foregr_sum(self, j, V_spots, min_int, spot_filled_mask):
        return np.sum(V_spots[spot_filled_mask] - min_int)

    def calc_mNeon_mKate_sum(self, V_spots, V_ref, mNeon_norm, mKate_norm,
                                   spot_filled_mask):
        V_mNeon_norm = V_spots[spot_filled_mask]/mNeon_norm
        V_ref_norm = V_ref[spot_filled_mask]/mKate_norm
        return np.sum(V_mNeon_norm-V_ref_norm)

    def volume(self):
        return np.pi*(self.xys**2)*self.zs*4/3

    def eval_grow_cond(self, semiax_len, zyx_centers, num_spots, grow_prev, V,
                       min_int, count_iter, verb=False):
        V_shape = self.V_shape
        grow = [False]*num_spots
        # Iterate each peak
        for b, (zyx_c, g1) in enumerate(zip(zyx_centers, grow_prev)):
            # Check if growing should continue (g1=True in grow_prev)
            if g1:
                sph_surf_mask, spot_filled_mask = self.get_sph_surf_mask(
                                                       semiax_len,
                                                       zyx_c, V_shape,
                                                       return_filled_mask=True)
                surf_pixels = V[sph_surf_mask]
                surf_mean = np.mean(surf_pixels)
                # Check if the current spheroid hit another peak
                zz, yy, xx = zyx_centers[:,0], zyx_centers[:,1], zyx_centers[:,2]
                num_zyx_c = np.count_nonzero(spot_filled_mask[zz, yy, xx])
                hit_neigh = num_zyx_c > 1
                if not hit_neigh:
                    cond = surf_mean > min_int or count_iter>20
                    grow[b] = cond
        return grow

class _spotFIT(spheroid):
    def __init__(self, debug=False):
        self.debug = debug

    def set_args(
            self, obj, spots_img, df_spots, zyx_vox_size, zyx_spot_min_vol_um, 
            verbose=0, inspect=0, ref_ch_mask_or_labels=None
        ):
        self.spots_img_local = spots_img[obj.slice]
        super().__init__(self.spots_img_local)
        ID = obj.label
        self.ID = obj.label
        self.df_spots_ID = df_spots.loc[ID].copy()
        self.zyx_vox_size = zyx_vox_size
        min_z, min_y, min_x, _, _, _ = obj.bbox
        self.obj_bbox_lower = (min_z, min_y, min_x)
        self.obj_image = obj.image
        self.zyx_spot_min_vol_um = zyx_spot_min_vol_um
        if ref_ch_mask_or_labels is not None:
            self.ref_ch_mask_local = ref_ch_mask_or_labels[obj.slice] > 0
        else:
            self.ref_ch_mask_local = None
        self.verbose = verbose
        self.inspect = inspect
        # z0, y0, x0, sz, sy, sx, A = coeffs; B added as one coeff
        self.num_coeffs = 7
        self._tol = 1e-10

    def fit(self):
        verbose = self.verbose
        inspect = self.inspect
        df_spots_ID = self.df_spots_ID

        if verbose > 0:
            print('')
            print('Segmenting spots...')
        self.spotSIZE()

        if verbose > 0:
            print('')
            print('Computing intersections...')
        self.compute_neigh_intersect()

        if verbose > 0:
            print('')
            print('Fitting 3D gaussians...')
        self._fit()

        if verbose > 0:
            print('')
            print('Running quality control...')
        self._quality_control()

        if self.fit_again_idx:
            if verbose > 0:
                print('')
                print('Attempting to fit again spots that '
                      'did not pass quality control...')
            self._fit_again()

        if verbose > 0:
            print('')
            print('Fitting process done.')

        _df_spotFIT = (self._df_spotFIT
                        .reset_index()
                        .drop(['intersecting_idx', 'neigh_idx',
                               's', 'neigh_ids'], axis=1)
                        .set_index('id')
                       )
        _df_spotFIT.index.names = ['spot_id']

        df_spots_ID = self.df_spots_ID

        self.df_spotFIT_ID = df_spots_ID.join(_df_spotFIT, how='outer')
        self.df_spotFIT_ID.index.names = ['spot_id']

        if verbose > 1:
            print('Summary results:')
            print(_df_spotFIT)
            if 'vox mNeon (uint8)' in self.df_spotFIT_ID.columns:
                cols = ['vox mNeon (uint8)', '|abs| mNeon (uint8)',
                        'I_tot', 'I_foregr', 'sigma_y_fit']
            else:
                cols = ['vox_spot', '|abs|_spot',
                        'I_tot', 'I_foregr', 'sigma_y_fit']
            print(self.df_spotFIT_ID[cols])

    def spotSIZE(self):
        df_spots_ID = self.df_spots_ID
        spots_img_denoise = skimage.filters.gaussian(self.spots_img_local, 0.8)
        min_z, min_y, min_x = self.obj_bbox_lower
        zyx_vox_dim = self.zyx_vox_size
        zyx_spot_min_vol_um = self.zyx_spot_min_vol_um
        obj_image = self.obj_image
        ref_ch_img_local = self.ref_ch_mask_local

        # Build spot mask and get background values
        num_spots = len(df_spots_ID)
        self.num_spots = num_spots
        spots_centers = df_spots_ID[['z', 'y', 'x']].to_numpy()
        spots_centers -= [min_z, min_y, min_x]
        self.spots_centers = spots_centers
        spots_mask = self.get_spots_mask(
            0, zyx_vox_dim, zyx_spot_min_vol_um, spots_centers
        )
        if ref_ch_img_local is None:
            backgr_mask = np.logical_and(obj_image, ~spots_mask)
        else:
            backgr_mask = np.logical_and(ref_ch_img_local, ~spots_mask)

        backgr_vals = spots_img_denoise[backgr_mask]
        backgr_mean = backgr_vals.mean()
        backgr_std = backgr_vals.std()

        self.backgr_mean = backgr_mean
        self.backgr_std = backgr_std

        limit = backgr_mean + 3*backgr_std

        # Build seeds mask for the expansion process
        self.spot_ids = df_spots_ID.index.to_list()
        seed_size = np.array(zyx_spot_min_vol_um)/2
        spots_seeds = self.get_spots_mask(0, zyx_vox_dim, seed_size,
                                         spots_centers, dtype=np.uint16,
                                         ids=self.spot_ids)
        spots_3D_lab = np.zeros_like(spots_seeds)

        # Start expanding the labels
        zs, ys, xs = seed_size
        zvd, yvd, _ = zyx_vox_dim
        stop_grow_info = [] # list of (stop_id, stop_mask, stop_slice)
        stop_grow_ids = []
        max_i = 10
        max_size = max_i*yvd
        self.spots_yx_size_um = [ys+max_size]*num_spots
        self.spots_z_size_um = [zs+max_size]*num_spots
        self.spots_yx_size_pxl = [(ys+max_size)/yvd]*num_spots
        self.spots_z_size_pxl = [(zs+max_size)/zvd]*num_spots
        expanding_steps = [0]*num_spots
        self.Bs_guess = [0]*num_spots
        _spot_surf_5ps = [0]*num_spots
        _spot_surf_means = [0]*num_spots
        _spot_surf_stds = [0]*num_spots
        _spot_B_mins = [0]*num_spots
        for i in range(max_i+1):
            # Note that expanded_labels has id from df_spots_ID
            expanded_labels = expand_labels(
                spots_seeds, distance=yvd*(i+1), zyx_vox_size=zyx_vox_dim
            )

            # Replace expanded labels with the stopped growing ones.
            for stop_id, stop_mask, stop_slice in stop_grow_info:
                expanded_labels[expanded_labels==stop_id] = 0
                expanded_labels[stop_slice][stop_mask] = stop_id

            # Iterate spots to determine which ones should stop growing
            spots_rp = skimage.measure.regionprops(expanded_labels)
            for o, s_obj in enumerate(spots_rp):
                id = s_obj.label
                # Skip spots where we stopped growing
                if id in stop_grow_ids:
                    continue
                exanped_spot_mask = expanded_labels[s_obj.slice]==id
                spot_mask = spots_seeds[s_obj.slice]==id
                local_spot_surf_mask = np.logical_xor(
                                             exanped_spot_mask, spot_mask
                )
                surf_vals = spots_img_denoise[s_obj.slice][local_spot_surf_mask]
                surf_mean = surf_vals.mean()
                # print('---------------')
                # print(f'ID {id} surface mean, backgr = {surf_mean}, {limit}')


                if surf_mean <= limit or i == max_i:
                    # NOTE: we use i+1 in order to include the pixels that
                    # are <= to the limit
                    stop_grow_info.append((id, s_obj.image, s_obj.slice))
                    stop_grow_ids.append(id)
                    self.spots_yx_size_um[o] = ys+yvd*(i+1)
                    self.spots_z_size_um[o] = zs+yvd*(i+1)
                    self.spots_yx_size_pxl[o] = (ys+yvd*(i+1))/yvd
                    self.spots_z_size_pxl[o] = (zs+yvd*(i+1))/zvd
                    # Insert grown spot into spots lab used for fitting
                    c_idx = self.spot_ids.index(id)
                    zyx_c = spots_centers[c_idx]
                    spots_3D_lab = self.insert_grown_spot_id(
                                                     i+1, id, zyx_vox_dim,
                                                     zyx_spot_min_vol_um,
                                                     zyx_c, spots_3D_lab)
                    raw_spot_surf_vals = (self.spots_img_local
                                           [s_obj.slice]
                                           [local_spot_surf_mask])
                    self.Bs_guess[o] = np.median(raw_spot_surf_vals)
                    _spot_surf_5ps[o] = np.quantile(raw_spot_surf_vals, 0.05)
                    _mean = raw_spot_surf_vals.mean()
                    _spot_surf_means[o] = _mean
                    _std = raw_spot_surf_vals.std()
                    _spot_surf_stds[o] = _std
                    _spot_B_mins[o] = _mean-3*_std

            # print(stop_grow_ids)
            # print(f'Current step = {(i+1)}')
            # print(len(stop_grow_ids), num_spots)

            # Stop loop if all spots have stopped growing
            if len(stop_grow_ids) == num_spots:
                break

        self.spots_radii_pxl = np.column_stack(
                                        (self.spots_z_size_pxl,
                                         self.spots_yx_size_pxl,
                                         self.spots_yx_size_pxl)
        )

        self.df_spots_ID['spotsize_yx_radius_um'] = self.spots_yx_size_um
        self.df_spots_ID['spotsize_z_radius_um'] = self.spots_z_size_um
        self.df_spots_ID['spotsize_yx_radius_pxl'] = self.spots_yx_size_pxl
        self.df_spots_ID['spotsize_z_radius_pxl'] = self.spots_z_size_pxl
        self.df_spots_ID['spotsize_limit'] = [limit]*num_spots

        self.df_spots_ID['spot_surf_50p'] = self.Bs_guess
        self.df_spots_ID['spot_surf_5p'] = _spot_surf_5ps
        self.df_spots_ID['spot_surf_mean'] = _spot_surf_means
        self.df_spots_ID['spot_surf_std'] = _spot_surf_stds
        self.df_spots_ID['spot_B_min'] = _spot_B_mins

        # Used as a lower bound for B parameter in spotfit
        self.B_mins = _spot_B_mins

        self.spots_3D_lab_ID = spots_3D_lab

    def _fit(self):
        verbose = self.verbose
        if verbose > 1:
            print('')
            print('===============')
        t0_opt = time.perf_counter()
        num_spots = self.num_spots
        df_intersect = self.df_intersect
        spots_centers = self.spots_centers
        spots_radii_pxl = self.spots_radii_pxl
        spots_Bs_guess = self.Bs_guess
        spots_B_mins = self.B_mins
        spots_3D_lab_ID = self.spots_3D_lab_ID
        spots_img = self.spots_img_local
        num_coeffs = self.num_coeffs
        inspect = self.inspect
        spots_rp = self.spots_rp

        init_guess_li = [None]*num_spots
        fitted_coeffs = [[] for _ in range(num_spots)]
        Bs_fitted = [0]*num_spots
        all_intersect_fitted_bool = [0]*num_spots
        solution_found_li = [0]*num_spots
        iter = zip(df_intersect.index,
                   df_intersect['id'],
                   df_intersect['intersecting_idx'],
                   df_intersect['neigh_idx'])
        for count, (s, s_id, intersect_idx, neigh_idx) in enumerate(iter):
            # Get the fitted coeffs of the intersecting peaks
            intersect_coeffs = [fitted_coeffs[i] for i in intersect_idx]
            if verbose > 2:
                print('-----------')
                print(f'Current spot idx: {s}')
                print(f'Neighbours indices of current spot: {intersect_idx}')
            all_intersect_fitted = all(intersect_coeffs)
            if all_intersect_fitted:
                if verbose > 2:
                    print(f'Fully fitted spot idx: {s}')
                all_intersect_fitted_bool[s] = True
                pbar = tqdm(
                    desc=f'Spot done {count+1}/{num_spots}', total=4, 
                    unit=' fev', position=2, leave=False, ncols=100
                )
                pbar.update(1)
                pbar.close()
                if verbose > 2:
                    print('-----------')
                continue
            if verbose > 2:
                print(f'Intersect. coeffs: {intersect_coeffs}')
            # Set coeffs of already fitted neighbours as model constants
            non_inters_neigh_idx = [s for s in neigh_idx
                                    if s not in intersect_idx
            ]
            if verbose > 2:
                print(f'Fitted bool: {all_intersect_fitted_bool}')
                print(f'Non-intersecting neighbours idx: {non_inters_neigh_idx}')
            neigh_fitted_coeffs = [
                        fitted_coeffs[i] for i in non_inters_neigh_idx
                        if all_intersect_fitted_bool[i]
            ]
            neigh_fitted_idx = [i for i in non_inters_neigh_idx
                                        if all_intersect_fitted_bool[i]]
            if verbose > 2:
                print('All-neighbours-fitted coeffs (model constants): '
                      f'{neigh_fitted_coeffs}')
            # Use not completely fitted neigh coeffs as initial guess
            not_all_intersect_fitted_coeffs = [
                                           fitted_coeffs[i]
                                           for i in intersect_idx
                                           if not all_intersect_fitted_bool[i]]
            if verbose > 2:
                print('Not-all-neighbours-fitted coeffs (model initial guess): '
                      f'{not_all_intersect_fitted_coeffs}')

            # Fit n intersecting spots as sum of n gaussian + model constants
            fit_idx = intersect_idx
            if verbose > 2:
                print(f'Fitting spot idx: {fit_idx}, '
                      f'with centers {spots_centers}')

            # Fit multipeaks
            fit_spots_lab = np.zeros(spots_3D_lab_ID.shape, bool)
            fit_ids = []
            num_spots_s = len(fit_idx)
            for i in fit_idx:
                fit_id = self.df_intersect.at[i, 'id']
                fit_ids.append(fit_id)
                fit_spots_lab[spots_3D_lab_ID==fit_id] = True
            z, y, x = np.nonzero(fit_spots_lab)
            s_data = self.spots_img_local[z,y,x]
            model = _GaussianModel(100*len(z))

            # Get constants
            if neigh_fitted_idx:
                const = model.compute_const(z,y,x, neigh_fitted_coeffs)
            else:
                const = 0
            # test this https://cars9.uchicago.edu/software/python/lmfit/examples/example_reduce_fcn.html#sphx-glr-examples-example-reduce-fcn-py
            bounds, init_guess_s = model.get_bounds_init_guess(
                                             num_spots_s, num_coeffs,
                                             fit_ids, fit_idx, spots_centers,
                                             spots_3D_lab_ID, spots_rp,
                                             spots_radii_pxl, spots_img,
                                             spots_Bs_guess, spots_B_mins
            )
            # bar_f = '{desc:<25}{percentage:3.0f}%|{bar:40}{r_bar}'
            model.pbar = tqdm(desc=f'Fitting spot {s} ({count+1}/{num_spots})',
                              total=100*len(z), unit=' fev',
                              position=5, leave=False, ncols=100)
            try:
                leastsq_result = scipy.optimize.least_squares(
                    model.residuals, init_guess_s,
                    args=(s_data, z, y, x, num_spots_s, num_coeffs),
                    # jac=model.jac_gauss3D,
                    kwargs={'const': const},
                    loss='linear', f_scale=0.1,
                    bounds=bounds, ftol=self._tol,
                    xtol=self._tol, gtol=self._tol
                )
            except:
                traceback.print_exc()
                _shape = (num_spots_s, num_coeffs)
                B_fit = leastsq_result.x[-1]
                B_guess = init_guess_s[-1]
                B_min = bounds[0][-1]
                B_max = bounds[1][-1]
                lstsq_x = leastsq_result.x[:-1]
                lstsq_x = lstsq_x.reshape(_shape)
                init_guess_s_2D = init_guess_s[:-1].reshape(_shape)
                low_bounds_2D = bounds[0][:-1].reshape(_shape)
                high_bounds_2D = bounds[1][:-1].reshape(_shape)
                print('')
                print(self.ID)
                print(fit_ids)
                for _x, _init, _l, _h in zip(lstsq_x, init_guess_s_2D,
                                             low_bounds_2D, high_bounds_2D):
                    print('')
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
                import pdb; pdb.set_trace()

            # model.pbar.update(100*len(z)-model.pbar.n)
            model.pbar.close()

            if inspect > 2:
            # if 1 in fit_ids and self.ID == 1:
                # sum(z0, y0, x0, sz, sy, sx, A), B = coeffs
                _shape = (num_spots_s, num_coeffs)
                B_fit = leastsq_result.x[-1]
                B_guess = init_guess_s[-1]
                B_min = bounds[0][-1]
                B_max = bounds[1][-1]
                lstsq_x = leastsq_result.x[:-1]
                lstsq_x = lstsq_x.reshape(_shape)
                init_guess_s_2D = init_guess_s[:-1].reshape(_shape)
                low_bounds_2D = bounds[0][:-1].reshape(_shape)
                high_bounds_2D = bounds[1][:-1].reshape(_shape)
                print('')
                print(self.ID)
                print(fit_ids)
                for _x, _init, _l, _h in zip(lstsq_x, init_guess_s_2D,
                                             low_bounds_2D, high_bounds_2D):
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
                import pdb; pdb.set_trace()
                matplotlib.use('TkAgg')
                fig, ax = plt.subplots(1,3)
                img = self.spots_img_local
                # 3D gaussian evaluated on the entire image
                V_fit = np.zeros_like(self.spots_img_local)
                zz, yy, xx = np.nonzero(V_fit==0)
                V_fit[zz, yy, xx] = model._gauss3D(
                                       zz, yy, xx, leastsq_result.x,
                                       num_spots_s, num_coeffs, 0)

                fit_data = model._gauss3D(z, y, x, leastsq_result.x,
                                          num_spots_s, num_coeffs, 0)

                img_fit = np.zeros_like(img)
                img_fit[z,y,x] = fit_data
                img_s = np.zeros_like(img)
                img_s[z,y,x] = s_data
                y_intens = img_s.max(axis=(0, 1))
                y_intens = y_intens[y_intens!=0]
                y_gauss = img_fit.max(axis=(0, 1))
                y_gauss = y_gauss[y_gauss!=0]
                ax[0].imshow(img.max(axis=0))
                _, yyc, xxc = np.array(spots_centers[fit_idx]).T
                ax[0].plot(xxc, yyc, 'r.')
                ax[1].imshow(V_fit.max(axis=0))
                ax[1].plot(xxc, yyc, 'r.')
                ax[2].scatter(range(len(y_intens)), y_intens)
                ax[2].plot(range(len(y_gauss)), y_gauss, c='r')
                plt.show()
                matplotlib.use('Agg')

            _shape = (num_spots_s, num_coeffs)
            B_fit = leastsq_result.x[-1]
            B_guess = init_guess_s[-1]
            lstsq_x = leastsq_result.x[:-1]
            lstsq_x = lstsq_x.reshape(_shape)
            init_guess_s_2D = init_guess_s[:-1].reshape(_shape)
            # print(f'Fitted coeffs: {lstsq_x}')
            # Store already fitted peaks
            for i, s_fit in enumerate(fit_idx):
                fitted_coeffs[s_fit] = list(lstsq_x[i])
                init_guess_li[s_fit] = list(init_guess_s_2D[i])
                Bs_fitted[s_fit] = B_fit
                solution_found_li[s_fit] = leastsq_result.success
            # Check if now the fitted spots are fully fitted
            all_intersect_fitted = all([True if fitted_coeffs[i] else False
                                         for i in intersect_idx])
            if all_intersect_fitted:
                if verbose > 2:
                    print(f'Fully fitted spot idx: {s}')
                all_intersect_fitted_bool[s] = True
            if verbose == 2:
                print('-----------')

        self.model = model
        self.fitted_coeffs = fitted_coeffs
        self.Bs_fitted = Bs_fitted
        self.init_guess_li = init_guess_li
        self.solution_found_li = solution_found_li

        t1_opt = time.perf_counter()
        exec_time = t1_opt-t0_opt
        exec_time_delta = timedelta(seconds=exec_time)
        if verbose > 1:
            print('')
            print(f'Fitting process done in {exec_time_delta} HH:mm:ss')

    def compute_neigh_intersect(self):
        inspect = self.inspect
        verbose = self.verbose
        zyx_vox_dim = self.zyx_vox_size
        zvd, yvd, _ = zyx_vox_dim
        spots_3D_lab_ID = self.spots_3D_lab_ID
        spots_3D_lab_ID_connect = skimage.measure.label(spots_3D_lab_ID>0)
        self.spots_3D_lab_ID_connect = spots_3D_lab_ID_connect
        spots_rp = skimage.measure.regionprops(spots_3D_lab_ID)
        self.spots_rp = spots_rp
        # Get intersect ids by expanding each single object by 2 pixels
        all_intersect_idx = []
        all_neigh_idx = []
        obj_ids = []
        num_intersect = []
        num_neigh = []
        all_neigh_ids = []
        for s, s_obj in enumerate(spots_rp):
            spot_3D_lab = np.zeros_like(spots_3D_lab_ID)
            spot_3D_lab[s_obj.slice][s_obj.image] = s_obj.label
            spot_3D_mask = spot_3D_lab>0
            expanded_spot_3D = expand_labels(
                spot_3D_lab, distance=yvd*2, zyx_vox_size=zyx_vox_dim
            )
            spot_surf_mask = np.logical_xor(expanded_spot_3D>0, spot_3D_mask)
            intersect_ids = np.unique(spots_3D_lab_ID[spot_surf_mask])
            intersect_idx = [self.spot_ids.index(id)
                             for id in intersect_ids if id!=0]
            intersect_idx.append(s)
            all_intersect_idx.append(intersect_idx)
            num_intersect.append(len(intersect_idx))

            # Get neigh idx by indexing the spots labels with the
            # connected component mask
            obj_id = np.unique(spots_3D_lab_ID_connect[spot_3D_mask])[-1]
            obj_ids.append(obj_id)
            obj_mask = np.zeros_like(spot_3D_mask)
            obj_mask[spots_3D_lab_ID_connect == obj_id] = True
            neigh_ids = np.unique(spots_3D_lab_ID[obj_mask])
            neigh_ids = [id for id in neigh_ids if id!=0]
            neigh_idx = [self.spot_ids.index(id) for id in neigh_ids]
            all_neigh_idx.append(neigh_idx)
            all_neigh_ids.append(neigh_ids)
            num_neigh.append(len(neigh_idx))


        self.df_intersect = (pd.DataFrame({
                                      'id': self.spot_ids,
                                      'obj_id': obj_ids,
                                      'num_intersect': num_intersect,
                                      'num_neigh': num_neigh,
                                      'intersecting_idx': all_intersect_idx,
                                      'neigh_idx': all_neigh_idx,
                                      'neigh_ids': all_neigh_ids})
                                      .sort_values('num_intersect')
                        )
        self.df_intersect.index.name = 's'




        if verbose > 1:
            print('Intersections info:')
            print(self.df_intersect)
            print('')

        if inspect > 1:
            imshow(self.spots_img_local, spots_3D_lab_ID, spots_3D_lab_ID_connect)

    def _quality_control(self):
        """
        Calculate goodness_of_fit metrics for each spot
        and determine which peaks should be fitted again
        """
        df_spotFIT = (self.df_intersect
                                 .reset_index()
                                 .set_index(['obj_id', 's']))
        df_spotFIT['QC_passed'] = 0
        df_spotFIT['null_ks_test'] = 0
        df_spotFIT['null_chisq_test'] = 0
        df_spotFIT['solution_found'] = 0

        self._df_spotFIT = df_spotFIT
        verbose = self.verbose
        inspect = self.inspect
        spots_3D_lab_ID = self.spots_3D_lab_ID
        spots_3D_lab_ID_connect = self.spots_3D_lab_ID_connect
        fitted_coeffs = self.fitted_coeffs
        init_guess_li = self.init_guess_li
        Bs_fitted = self.Bs_fitted
        solution_found_li = self.solution_found_li
        num_coeffs = self.num_coeffs
        model = self.model
        img = self.spots_img_local

        all_gof_metrics = np.zeros((self.num_spots, 7))
        self.fit_again_idx = []
        for obj_id, df_obj in df_spotFIT.groupby(level=0):
            obj_s_idxs = df_obj['neigh_idx'].iloc[0]
            # Iterate single spots
            for s in obj_s_idxs:
                s_id = df_obj.at[(obj_id, s), 'id']
                s_intersect_idx = df_obj.at[(obj_id, s), 'intersecting_idx']
                z_s, y_s, x_s = np.nonzero(spots_3D_lab_ID==s_id)

                # Compute fit data
                B_fit = Bs_fitted[s]
                s_coeffs = fitted_coeffs[s]
                s_fit_data = model.gaussian_3D(z_s, y_s, x_s, s_coeffs, B=B_fit)
                for n_s in obj_s_idxs:
                    neigh_coeffs = fitted_coeffs[n_s]
                    s_fit_data += model.gaussian_3D(z_s, y_s, x_s, neigh_coeffs)

                # Goodness of fit
                ddof = num_coeffs
                s_data = img[z_s, y_s, x_s]
                (reduced_chisq, p_chisq, RMSE, ks, p_ks, NRMSE,
                F_NRMSE) = model.goodness_of_fit(s_data, s_fit_data, ddof)

                all_gof_metrics[s] = [reduced_chisq, p_chisq, RMSE,
                                      ks, p_ks, NRMSE, F_NRMSE]

                if inspect > 2:
                # if True:
                # if s_id==3 and self.ID==5:
                    print('')
                    print('----------------------------')
                    print(f'Spot data max = {s_data.max():.3f}, '
                          f'spot fit max = {s_fit_data.max():.3f}')
                    print(f'Intersecting idx = {s_intersect_idx}')
                    print(f'Neighbours idx = {obj_s_idxs}')
                    print('Spot idx =', s)
                    print(f'Reduced chisquare = {reduced_chisq:.3f}, '
                          f'p = {p_chisq:.4f}')
                    print(f'KS stat = {ks:.3f}, p = {p_ks:.4f}')
                    # print(f'R_sq = {R_sq:.3f}, Adj. R-sq = {adj_Rsq:.3f}')
                    print(f'RMSE = {RMSE:.3f}')
                    print(f'NRMSE = {NRMSE:.3f}')
                    print(f'F_NRMSE = {F_NRMSE:.3f}')

                    # Initial guess
                    (z0_guess, y0_guess, x0_guess,
                    sz_guess, sy_guess, sx_guess,
                    A_guess) = init_guess_li[s]

                    # Fitted coeffs
                    (z0_fit, y0_fit, x0_fit,
                    sz_fit, sy_fit, sx_fit,
                    A_fit) = fitted_coeffs[s]

                    print('----------------------------')
                    print(f'Init guess center = ({z0_guess:.2f}, '
                                               f'{y0_guess:.2f}, '
                                               f'{x0_guess:.2f})')
                    print(f'Fit center =        ({z0_fit:.2f}, '
                                               f'{y0_fit:.2f}, '
                                               f'{x0_fit:.2f})')
                    print('')
                    print(f'Init guess sigmas = ({sz_guess:.2f}, '
                                               f'{sy_guess:.2f}, '
                                               f'{sx_guess:.2f})')
                    print(f'Sigmas fit        = ({sz_fit:.2f}, '
                                               f'{sy_fit:.2f}, '
                                               f'{sx_fit:.2f})')
                    print('')
                    print(f'A, B init guess   = ({A_guess:.3f}, '
                                               f'{np.nan})')
                    print(f'A, B fit          = ({A_fit:.3f}, '
                                               f'{B_fit:.3f})')
                    print('----------------------------')


                    matplotlib.use('TkAgg')
                    fig, ax = plt.subplots(1,3, figsize=[18,9])

                    img_s = np.zeros_like(img)
                    img_s[z_s, y_s, x_s] = s_data

                    img_s_fit = np.zeros_like(img)
                    img_s_fit[z_s, y_s, x_s] = s_fit_data

                    y_intens = img[int(z0_guess), int(y0_guess)]
                    # y_intens = y_intens[y_intens!=0]

                    y_gauss = img_s_fit[int(z0_guess), int(y0_guess)]
                    x_gauss = [i for i, yg in enumerate(y_gauss) if yg != 0]
                    y_gauss = y_gauss[y_gauss!=0]


                    ax[0].imshow(img.max(axis=0), vmax=img.max())
                    ax[1].imshow(img_s_fit.max(axis=0), vmax=img.max())
                    ax[2].scatter(range(len(y_intens)), y_intens)
                    ax[2].plot(x_gauss, y_gauss, c='r')
                    # ax[2].scatter(range(len(y_intens)), y_intens)
                    # ax[2].plot(range(len(y_gauss)), y_gauss, c='r')

                    # l = x_obj.min()
                    # b = y_obj.min()
                    #
                    # r = x_obj.max()
                    # t = y_obj.max()
                    #
                    # ax[0].set_xlim((l-2, r+2))
                    # ax[0].set_ylim((t+2, b-2))
                    #
                    # ax[1].set_xlim((l-2, r+2))
                    # ax[1].set_ylim((t+2, b-2))

                    plt.show()
                    matplotlib.use('Agg')

        # Automatic outliers detection
        NRMSEs = all_gof_metrics[:,5]
        Q1, Q3 = np.quantile(NRMSEs, q=(0.25, 0.75))
        IQR = Q3-Q1
        self.QC_limit = Q3 + (1.5*IQR)

        if False:
            matplotlib.use('TkAgg')
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
            ax[2].axvline(self.QC_limit, color='r', linestyle='--')
            ax[6].axvline(self.QC_limit, color='r', linestyle='--')
            ax[2].set_title('NMRSE')

            sns.histplot(x=all_gof_metrics[:,6], ax=ax[3])
            sns.boxplot(x=all_gof_metrics[:,6], ax=ax[7])
            ax[3].set_title('F_NRMSE')

            plt.show()
            matplotlib.use('Agg')

        # Given QC_limit determine which spots should be fitted again
        for obj_id, df_obj in df_spotFIT.groupby(level=0):
            obj_s_idxs = df_obj['neigh_idx'].iloc[0]
            # Iterate single spots
            for s in obj_s_idxs:
                gof_metrics = all_gof_metrics[s]

                (reduced_chisq, p_chisq, RMSE,
                ks, p_ks, NRMSE, F_NRMSE) = gof_metrics

                # Initial guess
                (z0_guess, y0_guess, x0_guess,
                sz_guess, sy_guess, sx_guess,
                A_guess) = init_guess_li[s]

                # Fitted coeffs
                B_fit = Bs_fitted[s]
                (z0_fit, y0_fit, x0_fit,
                sz_fit, sy_fit, sx_fit,
                A_fit) = fitted_coeffs[s]

                # Solution found
                solution_found = solution_found_li[s]

                # Store s idx of badly fitted peaks
                num_s_in_obj = len(obj_s_idxs)
                s_intersect_idx = df_obj.at[(obj_id, s), 'intersecting_idx']
                num_intersect_s = len(s_intersect_idx)
                if NRMSE > self.QC_limit and num_intersect_s < num_s_in_obj:
                    if verbose > 2:
                        print('')
                        print(f'Fit spot idx {s} again.')
                        print('----------------------------')
                    self.fit_again_idx.append(s)
                    continue

                # Store properties of good peaks
                zyx_c = np.abs(np.array([z0_fit, y0_fit, x0_fit]))
                zyx_sigmas = np.abs(np.array([sz_fit, sy_fit, sx_fit]))

                I_tot, I_foregr = model.integrate(
                    zyx_c, zyx_sigmas, A_fit, B_fit,
                    lower_bounds=None, upper_bounds=None
                )

                gof_metrics = (reduced_chisq, p_chisq,
                               ks, p_ks, RMSE, NRMSE, F_NRMSE)

                self.store_metrics_good_spots(obj_id, s, fitted_coeffs[s],
                                              I_tot, I_foregr, gof_metrics,
                                              solution_found, B_fit)

                if verbose > 1:
                    print('')
                    print(f'Sigmas fit = ({sz_fit:.3f}, {sy_fit:.3f}, {sx_fit:.3f})')
                    print(f'A fit = {A_fit:.3f}, B fit = {B_fit:.3f}')
                    print('Total integral result, fit sum, observed sum = '
                          f'{I_tot:.3f}, {s_fit_data.sum():.3f}, {s_data.sum():.3f}')
                    print(f'Foregroung integral value: {I_foregr:.3f}')
                    print('----------------------------')

    def _fit_again(self):
        fit_again_idx = self.fit_again_idx
        df_intersect_fit_again = (
                               self.df_intersect
                               .loc[fit_again_idx]
                               .sort_values(by='num_intersect')
                               .reset_index()
                               .set_index(['obj_id', 's'])
        )

        bad_fit_idx = fit_again_idx.copy()
        num_spots = len(df_intersect_fit_again)
        num_coeffs = self.num_coeffs
        model = self.model
        spots_3D_lab_ID = self.spots_3D_lab_ID
        spots_centers = self.spots_centers
        spots_radii_pxl = self.spots_radii_pxl
        spots_Bs_guess = self.Bs_guess
        spots_B_mins = self.B_mins
        fitted_coeffs = self.fitted_coeffs
        init_guess_li = self.init_guess_li
        img = self.spots_img_local
        verbose = self.verbose
        inspect = self.inspect
        spots_rp = self.spots_rp

        # Iterate each badly fitted spot and fit individually again
        for count, (obj_id, s) in enumerate(df_intersect_fit_again.index):
            neigh_idx = df_intersect_fit_again.loc[(obj_id, s)]['neigh_idx']
            s_id = df_intersect_fit_again.loc[(obj_id, s)]['id']
            s_intersect_idx = df_intersect_fit_again.at[(obj_id, s),
                                                        'intersecting_idx']
            good_neigh_idx = [s for s in neigh_idx if s not in bad_fit_idx]

            z_s, y_s, x_s = np.nonzero(spots_3D_lab_ID==s_id)

            # Constants from good neigh idx
            const_coeffs = [fitted_coeffs[good_s] for good_s in good_neigh_idx]
            const = model.compute_const(z_s, y_s, x_s, const_coeffs)

            # Bounds and initial guess
            num_spots_s = 1
            bounds, init_guess_s = model.get_bounds_init_guess(
                                         num_spots_s, num_coeffs,
                                         [s_id], [s], spots_centers,
                                         spots_3D_lab_ID, spots_rp,
                                         spots_radii_pxl, img,
                                         spots_Bs_guess, spots_B_mins
            )

            # Fit with constants
            s_data = img[z_s, y_s, x_s]
            model.pbar = tqdm(desc=f'Fitting spot {s} ({count+1}/{num_spots})',
                                  total=100*len(z_s), unit=' fev',
                                  position=4, leave=False, ncols=100)
            leastsq_result = scipy.optimize.least_squares(
                model.residuals, init_guess_s,
                args=(s_data, z_s, y_s, x_s, num_spots_s, num_coeffs),
                # jac=model.jac_gauss3D,
                kwargs={'const': const},
                loss='linear', f_scale=0.1,
                bounds=bounds, ftol=self._tol,
                xtol=self._tol, gtol=self._tol
            )
            model.pbar.close()

            # Goodness of fit
            ddof = num_coeffs
            s_fit_data =  model._gauss3D(z_s, y_s, x_s,
                                         leastsq_result.x,
                                         1, num_coeffs, const)
            (reduced_chisq, p_chisq, RMSE, ks, p_ks,
            NRMSE, F_NRMSE) = model.goodness_of_fit(s_data, s_fit_data, ddof)

            # Initial guess
            (z0_guess, y0_guess, x0_guess,
            sz_guess, sy_guess, sx_guess,
            A_guess) = init_guess_li[s]

            # Fitted coeffs
            (z0_fit, y0_fit, x0_fit,
            sz_fit, sy_fit, sx_fit,
            A_fit, B_fit) = leastsq_result.x


            zyx_c = np.abs(np.array([z0_fit, y0_fit, x0_fit]))
            zyx_sigmas = np.abs(np.array([sz_fit, sy_fit, sx_fit]))

            I_tot, I_foregr = model.integrate(
                            zyx_c, zyx_sigmas, A_fit, B_fit,
                            lower_bounds=None, upper_bounds=None
            )

            gof_metrics = (reduced_chisq, p_chisq,
                           ks, p_ks, RMSE, NRMSE, F_NRMSE)

            self.store_metrics_good_spots(
                                     obj_id, s, leastsq_result.x[:-1],
                                     I_tot, I_foregr, gof_metrics,
                                     leastsq_result.success, B_fit=B_fit
            )

            if inspect > 2:
            # if True:
                print('')
                print('----------------------------')
                if NRMSE > self.QC_limit:
                    print('Quality control NOT passed!')
                else:
                    print('Quality control passed!')
                print(f'Spot data max = {s_data.max():.3f}, '
                      f'spot fit max = {s_fit_data.max():.3f}')
                print(f'Intersecting idx = {s_intersect_idx}')
                print(f'Neighbours idx = {neigh_idx}')
                print('Spot idx =', s)
                print(f'Reduced chisquare = {reduced_chisq:.3f}, '
                      f'p = {p_chisq:.4f}')
                print(f'KS stat = {ks:.3f}, p = {p_ks:.4f}')
                # print(f'R_sq = {R_sq:.3f}, Adj. R-sq = {adj_Rsq:.3f}')
                print(f'RMSE = {RMSE:.3f}')
                print(f'NRMSE = {NRMSE:.3f}')
                print(f'F_NRMSE = {F_NRMSE:.3f}')
                print('')
                print(f'Sigmas fit = ({sz_fit:.3f}, {sy_fit:.3f}, {sx_fit:.3f})')
                print(f'A fit = {A_fit:.3f}, B fit = {B_fit:.3f}')
                print('Total integral result, fit sum, observed sum = '
                      f'{I_tot:.3f}, {s_fit_data.sum():.3f}, {s_data.sum():.3f}')
                print(f'Foregroung integral value: {I_foregr:.3f}')
                print('----------------------------')


                matplotlib.use('TkAgg')
                fig, ax = plt.subplots(1,3, figsize=[18,9])

                img_s = np.zeros_like(img)
                img_s[z_s, y_s, x_s] = s_data

                img_s_fit = np.zeros_like(img)
                img_s_fit[z_s, y_s, x_s] = s_fit_data

                y_intens = img_s.max(axis=0)[int(y0_guess)]
                y_intens = y_intens[y_intens!=0]

                y_gauss = img_s_fit.max(axis=0)[int(y0_guess)]
                y_gauss = y_gauss[y_gauss!=0]

                ax[0].imshow(img.max(axis=0), vmax=img.max())
                ax[1].imshow(img_s_fit.max(axis=0), vmax=img.max())
                ax[2].scatter(range(len(y_intens)), y_intens)
                ax[2].plot(range(len(y_gauss)), y_gauss, c='r')

                l = x_s.min()
                b = y_s.min()

                r = x_s.max()
                t = y_s.max()

                ax[0].set_xlim((l-2, r+2))
                ax[0].set_ylim((t+2, b-2))

                ax[1].set_xlim((l-2, r+2))
                ax[1].set_ylim((t+2, b-2))

                plt.show()
                matplotlib.use('Agg')

    def store_metrics_good_spots(self, obj_id, s, fitted_coeffs_s,
                                 I_tot, I_foregr, gof_metrics,
                                 solution_found, B_fit):

        (z0_fit, y0_fit, x0_fit,
        sz_fit, sy_fit, sx_fit,
        A_fit) = fitted_coeffs_s

        min_z, min_y, min_x = self.obj_bbox_lower

        self._df_spotFIT.at[(obj_id, s), 'z_fit'] = z0_fit+min_z
        self._df_spotFIT.at[(obj_id, s), 'y_fit'] = y0_fit+min_y
        self._df_spotFIT.at[(obj_id, s), 'x_fit'] = x0_fit+min_x

        # self._df_spotFIT.at[(obj_id, s), 'AoB_fit'] = A_fit/B_fit

        self._df_spotFIT.at[(obj_id, s), 'sigma_z_fit'] = abs(sz_fit)
        self._df_spotFIT.at[(obj_id, s), 'sigma_y_fit'] = abs(sy_fit)
        self._df_spotFIT.at[(obj_id, s), 'sigma_x_fit'] = abs(sx_fit)
        self._df_spotFIT.at[(obj_id, s),
                            'sigma_yx_mean'] = (abs(sy_fit)+abs(sx_fit))/2

        _vol = 4/3*np.pi*abs(sz_fit)*abs(sy_fit)*abs(sx_fit)
        self._df_spotFIT.at[(obj_id, s), 'spotfit_vol_vox'] = _vol

        self._df_spotFIT.at[(obj_id, s), 'A_fit'] = A_fit
        self._df_spotFIT.at[(obj_id, s), 'B_fit'] = B_fit

        self._df_spotFIT.at[(obj_id, s), 'I_tot'] = I_tot
        self._df_spotFIT.at[(obj_id, s), 'I_foregr'] = I_foregr

        (reduced_chisq, p_chisq,
        ks, p_ks, RMSE, NRMSE, F_NRMSE) = gof_metrics

        self._df_spotFIT.at[(obj_id, s), 'reduced_chisq'] = reduced_chisq
        self._df_spotFIT.at[(obj_id, s), 'p_chisq'] = p_chisq

        self._df_spotFIT.at[(obj_id, s), 'KS_stat'] = ks
        self._df_spotFIT.at[(obj_id, s), 'p_KS'] = p_ks

        self._df_spotFIT.at[(obj_id, s), 'RMSE'] = RMSE
        self._df_spotFIT.at[(obj_id, s), 'NRMSE'] = NRMSE
        self._df_spotFIT.at[(obj_id, s), 'F_NRMSE'] = F_NRMSE

        QC_passed = int(NRMSE<self.QC_limit)
        self._df_spotFIT.at[(obj_id, s), 'QC_passed'] = QC_passed

        self._df_spotFIT.at[(obj_id, s), 'null_ks_test'] = int(p_ks > 0.05)
        self._df_spotFIT.at[(obj_id, s), 'null_chisq_test'] = int(p_chisq > 0.05)

        self._df_spotFIT.at[(obj_id, s), 'solution_found'] = int(solution_found)

class Kernel(_ParamsParser):
    def __init__(self, debug=False, is_cli=True):
        self.logger, self.log_path, self.logs_path = utils.setupLogger('cli')
        super().__init__(debug=debug, is_cli=is_cli, log=self.logger.info)
        self.debug = debug
        self.is_batch_mode = False
        self.is_cli = is_cli
        self._force_close_on_critical = False
        self._SpotFit = _spotFIT()

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
        out_range = (spots_img.min(), spots_img.max())
        rescaled = skimage.exposure.rescale_intensity(
            sharpened, out_range=out_range
        )
        return rescaled
    
    def _get_obj_mask(self, lab, obj, lineage_table):
        lab_obj_image = lab == obj.label
        
        if lineage_table is None:
            return lab_obj_image, -1
        
        cc_stage = lineage_table.at[obj.label, 'cell_cycle_stage']
        if cc_stage == 'G1':
            return lab_obj_image, -1
        
        # Merge mother and daughter when in S phase
        rel_ID = lineage_table.at[obj.label, 'relative_ID']
        lab_obj_image = np.logical_or(lab == obj.label, lab == rel_ID)
        
        return lab_obj_image, rel_ID
    
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
        img_local = image[lab_mask_obj.slice]
        backgr_vals = img_local[~lab_mask_obj.image]
        if backgr_vals.size == 0:
            return img_local, lab_mask, bud_ID
        
        backgr_mean = backgr_vals.mean()
        backgr_mean = backgr_mean if backgr_mean>=0 else 0
        backgr_std = backgr_vals.std()
        gamma_shape = np.square(backgr_mean/backgr_std)
        gamma_scale = np.square(backgr_std)/backgr_mean
        img_backgr = rng.gamma(
            gamma_shape, gamma_scale, size=lab_mask_obj.image.shape
        )

        img_backgr[lab_mask_obj.image] = img_local[lab_mask_obj.image]

        return img_backgr, lab_mask_obj.image, bud_ID
    
    def _add_aggregated_ref_ch_features(
            self, df_agg, frame_i, ID, ref_ch_mask_local, vox_to_um3=None
        ):
        vol_voxels = np.count_nonzero(ref_ch_mask_local)
        df_agg.at[(frame_i, ID), 'ref_ch_vol_vox'] = vol_voxels
        if vox_to_um3 is not None:
            df_agg.at[(frame_i, ID), 'ref_ch_vol_fl'] = vol_voxels*vox_to_um3

        rp = skimage.measure.regionprops(ref_ch_mask_local.astype(np.uint16))
        num_fragments = len(rp)
        df_agg.at[(frame_i, ID), 'ref_ch_num_fragments'] = num_fragments

        return df_agg
    
    def _segment_ref_ch(
            self, ref_ch_img, lab, lab_rp, df_agg, lineage_table, 
            threshold_func, frame_i, keep_only_largest_obj, ref_ch_segm, 
            vox_to_um3=None, thresh_val=None, verbose=True
        ):
        if verbose:
            print('')
            self.logger.info('Segmenting reference channel...')
        IDs = [obj.label for obj in lab_rp]
        desc = 'Segmenting reference channel'
        pbar = tqdm(
            total=len(lab_rp), ncols=100, desc=desc, position=3, 
            leave=False
        )
        for obj in lab_rp:
            if lineage_table is not None:
                if lineage_table.at[obj.label, 'relationship'] == 'bud':
                    # Skip buds since they are aggregated with mother
                    pbar.update()
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
            df_agg.at[df_idx, 'ref_ch_threshold_value'] = thresh_val
            if bud_ID > 0:
                bud_idx = (frame_i, bud_ID)
                df_agg.at[bud_idx, 'ref_ch_threshold_value'] = thresh_val

            # Threshold
            ref_mask_local = ref_ch_img_local > thresh_val
            ref_mask_local[~obj_mask] = False

            if bud_ID > 0:
                bud_obj = lab_rp[IDs.index(bud_ID)]
                objs = [obj, bud_obj]
            else:
                objs = [obj]

            # Iterate eventually merged (mother-bud) objects
            for obj in objs:
                ref_ch_mask = np.zeros_like(obj.image)
                local_slice = tuple([slice(0,d) for d in obj.image.shape])
                ref_ch_mask[local_slice] = ref_mask_local[local_slice]
                ref_ch_mask[~obj.image] = False
                if keep_only_largest_obj:
                    ref_ch_mask = self._filter_largest_obj(ref_ch_mask)

                # Add numerical features
                df_agg = self._add_aggregated_ref_ch_features(
                    df_agg, frame_i, obj.label, ref_ch_mask, 
                    vox_to_um3=vox_to_um3
                )

                ref_ch_segm[obj.slice][ref_ch_mask] = obj.label
            
            pbar.update()
        pbar.close()

        return ref_ch_segm, df_agg
    
    def ref_ch_to_physical_units(self, df_agg, metadata):
        vox_to_um3_factor = metadata['vox_to_um3_factor']
        df_agg['ref_ch_vol_um3'] = df_agg['ref_ch_vol_vox']*vox_to_um3_factor
        return df_agg

    @utils.exception_handler_cli
    def segment_ref_ch(
            self, ref_ch_img, threshold_method='threshold_otsu', lab_rp=None, 
            lab=None, lineage_table=None, keep_only_largest_obj=False, 
            do_aggregate_objs=False, df_agg=None, frame_i=0, 
            vox_to_um3=None, verbose=True
        ):
        if lab is None:
            lab = np.ones(ref_ch_img.shape, dtype=np.uint8)
            lineage_table = None
        
        if lab_rp is None:
            lab_rp = skimage.measure.regionprops(lab)
        
        if df_agg is None:
            IDs = [obj.label for obj in lab_rp]
            df_data = {'frame_i': [frame_i]*len(IDs), 'Cell_ID': IDs}
            df_agg = pd.DataFrame(df_data).set_index(['frame_i', 'Cell_ID'])
        
        df_agg['ref_ch_threshold_value'] = np.nan

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
        
        ref_ch_segm, df_agg = self._segment_ref_ch(
            ref_ch_img, lab, lab_rp, df_agg, lineage_table, 
            threshold_func, frame_i, keep_only_largest_obj, ref_ch_segm, 
            thresh_val=thresh_val, vox_to_um3=vox_to_um3, verbose=verbose
        )

        return ref_ch_segm, df_agg
    
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
    
    def _get_local_spheroid_mask(self, zyx_radii_pxl):
        zr, yr, xr = zyx_radii_pxl
        wh, d = int(np.ceil(yr)), int(np.ceil(zr))

        # Generate a sparse meshgrid to evaluate 3D spheroid mask
        z, y, x = np.ogrid[-d:d+1, -wh:wh+1, -wh:wh+1]

        # 3D spheroid equation
        mask = (x**2 + y**2)/(yr**2) + z**2/(zr**2) <= 1

        return mask
    
    def _distance_transform_edt(self, mask):
        edt = scipy.ndimage.distance_transform_edt(mask)
        edt = edt/edt.max()
        return edt
    
    def _raise_norm_value_zero(self):
        print('')
        self.logger.info(
            '[ERROR]: Skipping Position, see error below. '
            f'More details in the final report.{error_up_str}'
        )
        raise FloatingPointError(
            'normalising value for the reference channel is zero.'
        )
    
    def _warn_norm_value_zero(self):
        warning_txt = (
            'normalising value for the spots channel is zero.'
        )
        print('')
        self.logger.info(f'[WARNING]: {warning_txt}{error_up_str}')
        self.log_warning_report(warning_txt)

    def _normalise_img(
            self, img: np.ndarray, norm_mask: np.ndarray, 
            method='median', raise_if_norm_zero=True
        ):
        values = img[norm_mask]
        if method == 'median':
            norm_value = np.median(values)
        else:
            norm_value = 1

        if norm_value == 0:
            if raise_if_norm_zero:
                self._raise_norm_value_zero()
            else:
                _norm_value = 1E-15
                self._warn_norm_value_zero()
        else:
            _norm_value = norm_value
        norm_img = img/_norm_value
        return norm_img, norm_value

    @utils.exception_handler_cli
    def spots_detection(
            self, spots_img, zyx_resolution_limit_pxl, spots_img_detect=None,
            raw_spots_img=None, ref_ch_img=None, ref_ch_mask_or_labels=None, 
            frame_i=0, lab=None, rp=None, do_filter_spots_vs_ref_ch=False, 
            df_agg=None, do_keep_spots_in_ref_ch=False, 
            gop_filtering_thresholds=None, dist_transform_spheroid=None,
            detection_method='peak_local_max', prediction_method='Thresholding',
            threshold_method='threshold_otsu', do_aggregate_objs=False,
            lineage_table=None, min_size_spheroid_mask=None, verbose=True,
            spot_footprint=None, dfs_lists=None
        ):
        if spots_img.ndim == 2:
            spots_img = spots_img[np.newaxis]
            if lab is not None:
                lab = lab[np.newaxis]
                rp = skimage.measure.regionprops(lab)
            if ref_ch_img is not None:
                ref_ch_img = ref_ch_img[np.newaxis]
            if ref_ch_mask_or_labels is not None:
                ref_ch_mask_or_labels = ref_ch_mask_or_labels[np.newaxis]
            if raw_spots_img is not None:
                raw_spots_img = raw_spots_img[np.newaxis]
            if min_size_spheroid_mask is not None:
                min_size_spheroid_mask = min_size_spheroid_mask[np.newaxis]
            if spot_footprint is not None:
                spot_footprint = spot_footprint[np.newaxis]

        if spots_img_detect is None:
            spots_img_detect = spots_img

        if lab is None:
            lab = np.ones(spots_img.shape, dtype=np.uint8)
        
        if rp is None:
            rp = skimage.measure.regionprops(lab)
        
        if df_agg is None:
            IDs = [obj.label for obj in rp]
            df_data = {'frame_i': [frame_i]*len(IDs), 'Cell_ID': IDs}
            df_agg = pd.DataFrame(df_data).set_index(['frame_i', 'Cell_ID'])
        
        prediction_args = self._get_spot_prediction_args(
            spots_img_detect, lab, prediction_method, threshold_method, 
            do_aggregate_objs, lineage_table=lineage_table
        )
        threshold_val, threshold_func, prediction_mask = prediction_args
        
        df_spots_det, df_spots_gop = self._spots_detection(
            spots_img, spots_img_detect, ref_ch_img, ref_ch_mask_or_labels, 
            do_filter_spots_vs_ref_ch, lab, rp, frame_i, detection_method,
            zyx_resolution_limit_pxl, spot_footprint=spot_footprint,
            min_size_spheroid_mask=min_size_spheroid_mask,
            threshold_val=threshold_val, verbose=verbose, 
            threshold_func=threshold_func, dfs_lists=dfs_lists,
            raw_spots_img=raw_spots_img, prediction_mask=prediction_mask,
            do_keep_spots_in_ref_ch=do_keep_spots_in_ref_ch, 
            gop_filtering_thresholds=gop_filtering_thresholds,
            lineage_table=lineage_table, 
            dist_transform_spheroid=dist_transform_spheroid
        )
        if df_spots_det is not None:
            dfs_segm_obj = self._add_aggregated_spots_features(
                df_spots_det, df_spots_gop, df_agg
            )
            return df_spots_det, df_spots_gop, *dfs_segm_obj
    
    def _get_spot_prediction_args(
            self, spots_img, lab, prediction_method, threshold_method, 
            do_aggregate_objs, lineage_table=None
        ):
        threshold_val = None
        threshold_func = None
        prediction_mask = None
        if prediction_method == 'Thresholding':
            if isinstance(threshold_method, str):
                threshold_func = getattr(skimage.filters, threshold_method)
            else:
                threshold_func = threshold_method
            
            if do_aggregate_objs:
                aggr_spots_img = self.aggregate_objs(
                    spots_img, lab, lineage_table=lineage_table
                )
                threshold_val = threshold_func(aggr_spots_img.max(axis=0))
        elif prediction_method == 'Neural network':
            pass
        
        return threshold_val, threshold_func, prediction_mask
    
    def _spots_detection(
            self, spots_img, spots_img_detect, ref_ch_img, ref_ch_mask_or_labels, 
            do_filter_spots_vs_ref_ch, lab, rp, frame_i, detection_method,
            zyx_resolution_limit_pxl, dfs_lists=None,
            threshold_val=None, verbose=False, threshold_func=None,
            spot_footprint=None, min_size_spheroid_mask=None,
            raw_spots_img=None, gop_filtering_thresholds=None, 
            do_keep_spots_in_ref_ch=False, prediction_mask=None,
            lineage_table=None, dist_transform_spheroid=None
        ):
        if verbose:
            print('')
            self.logger.info('Detecting and filtering valid spots...')
        
        if dfs_lists is None:
            dfs_spots_det = []
            dfs_spots_gop = []
            keys = []
        else:
            dfs_spots_det = dfs_lists['dfs_spots_detection']
            dfs_spots_gop = dfs_lists['dfs_spots_gop_test']
            keys = dfs_lists['keys']

        desc = 'Detecting spots'
        pbar = tqdm(
            total=len(rp), ncols=100, desc=desc, position=3, leave=False
        )
        for obj in rp:
            local_spots_img = spots_img[obj.slice]
            local_spots_img_detect = spots_img_detect[obj.slice]
            if threshold_val is None and prediction_mask is None:
                lab_single_obj_mask, budID = self._get_obj_mask(
                    lab, obj, lineage_table
                )
                lab_single_obj_mask_rp = skimage.measure.regionprops(
                    lab_single_obj_mask.astype(np.uint8)
                )
                thresh_input_img = spots_img_detect[lab_single_obj_mask_rp[0].slice]
                threshold_val = threshold_func(thresh_input_img)
            
            if detection_method == 'peak_local_max':
                if spot_footprint is None:
                    zyx_radii_pxl = [val/2 for val in zyx_resolution_limit_pxl]
                    footprint = self._get_local_spheroid_mask(zyx_radii_pxl)
                else:
                    footprint = spot_footprint
                
                if prediction_mask is None:
                    labels = obj.image.astype(np.uint8)
                else:
                    threshold_val=None
                    local_spots_mask = prediction_mask[obj.slice]
                    labels = np.logical_and(obj.image, local_spots_mask)
                
                local_peaks_coords = skimage.feature.peak_local_max(
                    local_spots_img_detect, threshold_abs=threshold_val, 
                    footprint=footprint, labels=labels, 
                    p_norm=2
                )
            else:
                if prediction_mask is None:
                    local_spots_mask = local_spots_img_detect > threshold_val
                else:
                    local_spots_mask = prediction_mask[obj.slice]

                local_spots_lab = skimage.measure.label(local_spots_mask)
                local_spots_rp = skimage.measure.regionprops(local_spots_lab)
                num_spots = len(local_spots_rp)
                local_peaks_coords = np.zeros((len(num_spots, 3)))
                for s, spot_obj in enumerate(local_spots_rp):
                    local_peaks_coords[s] = spot_obj.centroid

            # Store coordinates after detection
            zyx_local_to_global = [s.start for s in obj.slice]
            global_peaks_coords = local_peaks_coords + zyx_local_to_global
            num_spots = len(global_peaks_coords)
            df_obj_spots_det = pd.DataFrame({
                'spot_id': np.arange(1, num_spots+1),
                'z': global_peaks_coords[:,0],
                'y': global_peaks_coords[:,1],
                'x': global_peaks_coords[:,2],
                'z_local': local_peaks_coords[:,0],
                'y_local': local_peaks_coords[:,1],
                'x_local': local_peaks_coords[:,2],
            }).set_index('spot_id')
            dfs_spots_det.append(df_obj_spots_det)
            keys.append((frame_i, obj.label))

            if ref_ch_mask_or_labels is not None:
                local_ref_ch_mask = ref_ch_mask_or_labels[obj.slice]>0
                local_ref_ch_mask = np.logical_and(local_ref_ch_mask, obj.image)
            else:
                local_ref_ch_mask = None

            # Filter according to goodness-of-peak test
            # CONTINUE FROM HERE
            if ref_ch_img is not None:
                local_ref_ch_img = ref_ch_img[obj.slice]
            else:
                local_ref_ch_img = None
            
            if raw_spots_img is not None:
                raw_spots_img_obj = raw_spots_img[obj.slice]

            df_obj_spots_gop = df_obj_spots_det.copy()
            if do_keep_spots_in_ref_ch:
                df_obj_spots_gop = self._drop_spots_not_in_ref_ch(
                    df_obj_spots_gop, local_ref_ch_mask, local_peaks_coords
                )

            print('')
            self.logger.info(f'Number of spots detected = {num_spots}')
            self.logger.info('Iterating goodness-of-peak test...')
            
            i = 0
            while True:     
                num_spots_prev = len(df_obj_spots_gop)      
                df_obj_spots_gop = self._compute_obj_spots_metrics(
                    local_spots_img, df_obj_spots_gop, obj.image, 
                    local_peaks_coords, local_spots_img_detect, 
                    raw_spots_img_obj=raw_spots_img_obj,
                    min_size_spheroid_mask=min_size_spheroid_mask, 
                    dist_transform_spheroid=dist_transform_spheroid,
                    ref_ch_mask_obj=local_ref_ch_mask, 
                    ref_ch_img_obj=local_ref_ch_img,
                    do_filter_spots_vs_ref_ch=do_filter_spots_vs_ref_ch,
                    zyx_resolution_limit_pxl=zyx_resolution_limit_pxl,
                    verbose=verbose                    
                )
                if i == 0:
                    # Store metrics at first iteration
                    df_obj_spots_det = df_obj_spots_gop.copy()
                
                df_obj_spots_gop = self.filter_spots(
                    df_obj_spots_gop, gop_filtering_thresholds
                )
                num_spots_current = len(df_obj_spots_gop)
                
                if num_spots_current == num_spots_prev or num_spots_current == 0:
                    # Number of filtered spots stopped decreasing --> stop loop
                    break

                i += 1

            print('')
            self.logger.info(
                f'Number of valid spots after {i+1} iterations = {num_spots_current}'
            )

            dfs_spots_gop.append(df_obj_spots_gop)
            pbar.update()
        pbar.close()
        
        if dfs_lists is None:
            names = ['frame_i', 'Cell_ID', 'spot_id']
            df_spots_det = pd.concat(dfs_spots_det, keys=keys, names=names)
            df_spots_gop = pd.concat(dfs_spots_gop, keys=keys, names=names)
            return df_spots_det, df_spots_gop
        else:
            return None, None
    
    def _add_aggregated_spots_features(
            self, df_spots_det: pd.DataFrame, df_spots_gop: pd.DataFrame, 
            df_agg: pd.DataFrame, df_spots_fit: pd.DataFrame=None
        ):
        func = {
            name:(col, aggFunc) for name, (col, aggFunc, _) 
            in aggregate_spots_feature_func.items() 
            if col in df_spots_det.columns
        }
        df_agg_det = (
            df_spots_det.reset_index().groupby(['frame_i', 'Cell_ID'])
            .agg(**func)
        )
        df_agg_det = df_agg_det.join(df_agg, how='left')
        df_agg_gop = (
            df_spots_gop.reset_index().groupby(['frame_i', 'Cell_ID'])
            .agg(**func)
        )
        df_agg_gop = df_agg_gop.join(df_agg, how='left')
        if df_spots_fit is not None:
            spotfit_func = {
                name:(col, aggFunc) for name, (col, aggFunc, _) 
                in aggregate_spots_feature_func.items() 
                if col in df_spots_fit.columns
            }
            df_agg_spotfit = (
                df_spots_fit.reset_index().groupby(['frame_i', 'Cell_ID'])
                .agg(**spotfit_func)
            )
            df_agg_spotfit = df_agg_spotfit.join(df_agg, how='left')
        else:
            df_agg_spotfit = None

        df_agg_det = self._add_missing_cells_df_agg(df_agg, df_agg_det)
        df_agg_gop = self._add_missing_cells_df_agg(df_agg, df_agg_gop)
        if df_agg_spotfit is not None:
            df_agg_spotfit = self._add_missing_cells_df_agg(df_agg, df_agg_spotfit)

        return df_agg_det, df_agg_gop, df_agg_spotfit
    
    @utils.exception_handler_cli
    def measure_spots_spotfit(
            self, spots_img, df_spots, zyx_voxel_size, zyx_spot_min_vol_um,
            rp=None, dfs_lists=None, lab=None, frame_i=0, 
            ref_ch_mask_or_labels=None
        ):
        if spots_img.ndim == 2:
            spots_img = spots_img[np.newaxis]
            if lab is not None:
                lab = lab[np.newaxis]
                rp = skimage.measure.regionprops(lab)
        
        if lab is None:
            lab = np.ones(spots_img.shape, dtype=np.uint8)

        if rp is None:
            lab = np.ones(spots_img.shape, dtype=np.uint8)
            rp = skimage.measure.regionprops(lab)
        
        if dfs_lists is None:
            dfs_spots_spotfit = []
            keys = []
        else:
            dfs_spots_spotfit = dfs_lists['dfs_spots_spotfit']
            keys = dfs_lists['spotfit_keys']
        
        desc = 'Measuring spots'
        pbar = tqdm(len(rp), ncols=100, desc=desc, position=3, leave=False)
        for obj in rp:
            self._SpotFit.set_args(
                obj, spots_img, df_spots, zyx_voxel_size, zyx_spot_min_vol_um, 
                ref_ch_mask_or_labels=ref_ch_mask_or_labels
            )
            self._SpotFit.fit()
            dfs_spots_spotfit.append(self._SpotFit.df_spotFIT_ID)
            keys.append((frame_i, obj.label))
            pbar.update()
        pbar.close()
        
    def _get_obj_spheroids_mask(
            self, zyx_coords, mask_shape, min_size_spheroid_mask=None, 
            zyx_radii_pxl=None
        ):
        mask = np.zeros(mask_shape, dtype=bool)
        if min_size_spheroid_mask is None:
            min_size_spheroid_mask = self._get_local_spheroid_mask(zyx_radii_pxl)

        for zyx_center in zyx_coords:
            slice_global_to_local, slice_crop_local = (
                utils.get_slices_local_into_global_3D_arr(
                    zyx_center, mask_shape, min_size_spheroid_mask.shape
                )
            )
            local_mask = min_size_spheroid_mask[slice_crop_local]
            mask[slice_global_to_local][local_mask] = True
        return mask, min_size_spheroid_mask
    
    def _get_spot_intensities(
            self, spots_img, zyx_center, local_spot_mask
        ):
        # Get the spot intensities
        slice_global_to_local, slice_crop_local = (
            utils.get_slices_local_into_global_3D_arr(
                zyx_center, spots_img.shape, local_spot_mask.shape
            )
        )
        spot_mask = local_spot_mask[slice_crop_local]
        spot_intensities = spots_img[slice_global_to_local][spot_mask]
        return spot_intensities
    
    def _drop_spots_not_in_ref_ch(self, df, ref_ch_mask, local_peaks_coords):
        if ref_ch_mask is None:
            return df
        
        zz = local_peaks_coords[:,0]
        yy = local_peaks_coords[:,1]
        xx = local_peaks_coords[:,2]
        in_ref_ch_spots_mask = ref_ch_mask[zz, yy, xx]
        return df[in_ref_ch_spots_mask]

    def _add_ttest_values(
            self, arr1: np.ndarray, arr2: np.ndarray, df: pd.DataFrame, 
            idx: Union[int, pd.Index], name: str='spot_vs_backgr'
        ):
        tstat, pvalue = scipy.stats.ttest_ind(arr1, arr2, equal_var=False)
        df.at[idx, f'{name}_ttest_tstat'] = tstat
        df.at[idx, f'{name}_ttest_pvalue'] = pvalue

    def _add_distribution_metrics(self, arr, df, idx, col_name='*name'):
        for name, func in distribution_metrics_func.items():
            _col_name = col_name.replace('*name', name)
            df.at[idx, _col_name] = func(arr)
        
    def _add_effect_sizes(self, pos_arr, neg_arr, df, idx, name='spot_vs_backgr'):
        for eff_size_name, func in effect_size_func.items():
            eff_size = features._try_metric_func(func, pos_arr, neg_arr)
            col_name = f'{name}_effect_size_{eff_size_name}'
            df.at[idx, col_name] = eff_size

    def _add_spot_vs_ref_location(self, ref_ch_mask, zyx_center, df, idx):
        is_spot_in_ref_ch = int(ref_ch_mask[zyx_center] > 0)
        df.at[idx, 'is_spot_inside_ref_ch'] = is_spot_in_ref_ch
        _, dist_2D_from_ref_ch = utils.nearest_nonzero(
            ref_ch_mask[zyx_center[0]], zyx_center[1], zyx_center[2]
        )
        df.at[idx, 'spot_dist_2D_from_ref_ch'] = dist_2D_from_ref_ch

    def _get_normalised_spot_ref_ch_intensities(
            self, normalised_spots_img_obj, normalised_ref_ch_img_obj,
            spheroid_mask, slice_global_to_local, dist_transf
        ):
        norm_spot_slice = (normalised_spots_img_obj[slice_global_to_local])
        norm_spot_slice_dt = norm_spot_slice*dist_transf
        norm_spot_intensities = norm_spot_slice_dt[spheroid_mask]

        norm_ref_ch_slice = (normalised_ref_ch_img_obj[slice_global_to_local])
        norm_ref_ch_slice_dt = norm_ref_ch_slice*dist_transf
        norm_ref_ch_intensities = norm_ref_ch_slice_dt[spheroid_mask]

        return norm_spot_intensities, norm_ref_ch_intensities


    # @acdctools.utils.exec_time
    def _compute_obj_spots_metrics(
            self, spots_img_obj, df_obj_spots, obj_mask, local_peaks_coords, 
            spots_img_detect_obj, raw_spots_img_obj=None, 
            min_size_spheroid_mask=None, dist_transform_spheroid=None,
            ref_ch_mask_obj=None, ref_ch_img_obj=None, 
            zyx_resolution_limit_pxl=None, 
            do_filter_spots_vs_ref_ch=False,
            verbose=False
        ):
        """_summary_

        Parameters
        ----------
        spots_img_obj : (Z, Y, X) ndarray
            Spots' signal 3D z-stack image sliced at the segmentation object
            level. Note that this is the preprocessed image, i.e., after 
            gaussian filtering, but NOT after sharpening. Sharpening is used 
            only to improve detection. The first dimension must be 
            the number of z-slices.
        df_obj_spots : pandas.DataFrame
            Pandas DataFrame with `spot_id` as index.
        obj_mask : (Z, Y, X) ndarray of dtype bool
            Boolean mask of the segmentation object contaning both ('z', 'y', 'x') 
            global peaks coordinates and ('z_local', 'y_local', 'z_local') 
            peaks coordinates in the segmentation objects' frame of reference.
        local_peaks_coords : (n, 3) ndarray
            (n, 3) array of (z,y,x) coordinates of the peaks in the segmentation
            object's frame of reference (i.e., "local").
        spots_img_detect_obj : (Z, Y, X) ndarray
            Spots' signal 3D z-stack image sliced at the segmentation object
            level. Note that this is the preprocessed image, i.e., after 
            gaussian filtering, sharpening etc. It is used to determine the 
            threshold for peak detection and for filtering against background. 
            The first dimension must be the number of z-slices.
        raw_spots_img_obj : (Z, Y, X) ndarray or None, optional
            Raw spots' signal 3D z-stack image sliced at the segmentation
            object level. Note that this is the raw, unprocessed signal. 
            The first dimension must be  the number of z-slices. 
            If None, the features from the raw signal will not be computed.
        min_size_spheroid_mask : (Z, Y, X) ndarray of dtype bool, optional
            The boolean mask of the smallest spot expected, by default None. 
            This is pre-computed using the resolution limit equations and the 
            pixel size. If None, this will be computed from 
            `zyx_resolution_limit_pxl`.
        dist_transform_spheroid : (Z, Y, X) ndarray, optional
            A distance transform of the `min_size_spheroid_mask`. This will be 
            multiplied by the spots intensities to reduce the skewing effect of 
            neighbouring peaks. 
            It must have the same shape of `min_size_spheroid_mask`
        ref_ch_mask_obj : (Z, Y, X) ndarray of dtype bool or None, optional
            Boolean mask of the reference channel, e.g., obtained by 
            thresholding. The first dimension must be  the number of z-slices.
            If not None, it is used to compute background metrics, filter 
            and localise spots compared to the reference channel, etc.
        ref_ch_img_obj : (Z, Y, X) ndarray or None, optional
            Reference channel's signal 3D z-stack image sliced at the 
            segmentation object level. Note that this is the preprocessed image,
            i.e., after gaussian filtering, sharpening etc. 
            The first dimension must be the number of z-slices.
            If None, the features from the reference channel signal will not 
            be computed.
        do_filter_spots_vs_ref_ch : bool, optional by default False
            Filter spots by comparing to the reference channel
        zyx_resolution_limit_pxl : (z, y, x) tuple or None, optional
            Resolution limit in (z, y, x) direction in pixels, by default None. 
            If `min_size_spheroid_mask` is None, this will be used to computed 
            the boolean mask of the smallest spot expected.
        verbose : bool, optional
            Log additional information of the current step, by default False
        """        
        if verbose:
            print('')
            self.logger.info('Computing spots features...')
        
        spheroids_mask, min_size_spheroid_mask = self._get_obj_spheroids_mask(
            local_peaks_coords, obj_mask.shape, 
            min_size_spheroid_mask=min_size_spheroid_mask, 
            zyx_radii_pxl=zyx_resolution_limit_pxl
        )

        if dist_transform_spheroid is None:
            dist_transform_spheroid = min_size_spheroid_mask

        # Check if spots_img needs to be normalised
        if do_filter_spots_vs_ref_ch:
            backgr_mask = np.logical_and(ref_ch_mask_obj, ~spheroids_mask)
            normalised_ref_ch_img_obj, ref_ch_norm_value = self._normalise_img(
                ref_ch_img_obj, backgr_mask, raise_if_norm_zero=False
            )
            df_obj_spots['ref_ch_normalising_value'] = ref_ch_norm_value
            normalised_spots_img_obj, spots_norm_value = self._normalise_img(
                spots_img_obj, backgr_mask, raise_if_norm_zero=True
            )
            df_obj_spots['spots_normalising_value'] = spots_norm_value
        else:
            backgr_mask = np.logical_and(obj_mask, ~spheroids_mask)

        # Calculate background metrics
        backgr_vals = spots_img_obj[backgr_mask]
        for name, func in distribution_metrics_func.items():
            df_obj_spots[f'background_{name}'] = func(backgr_vals)
        
        if raw_spots_img_obj is None:
            raw_spots_img_obj = spots_img_obj

        pbar_desc = 'Computing spots features'
        pbar = tqdm(
            total=len(df_obj_spots), ncols=100, desc=pbar_desc, position=3, 
            leave=False
        )
        for row in df_obj_spots.itertuples():
            spot_id = row.Index
            zyx_center = (row.z_local, row.y_local, row.x_local)

            slices = utils.get_slices_local_into_global_3D_arr(
                zyx_center, spots_img_obj.shape, min_size_spheroid_mask.shape
            )
            slice_global_to_local, slice_crop_local = slices
            spheroid_mask = min_size_spheroid_mask[slice_crop_local]
            dist_transf = dist_transform_spheroid[slice_crop_local]
            spot_slice = spots_img_obj[slice_global_to_local]
            spot_slice_detect = spot_slice*dist_transf

            # Add metrics from spot_img (which could be filtered or not)
            spot_intensities = spot_slice[spheroid_mask]
            spot_intensities_detect = spot_slice_detect[spheroid_mask]

            value = spots_img_obj[zyx_center]
            df_obj_spots.at[spot_id, 'spot_preproc_intensity_at_center'] = value
            self._add_distribution_metrics(
                spot_intensities, df_obj_spots, spot_id, 
                col_name='spot_preproc_*name_in_spot_minimumsize_vol'
            )
            
            if raw_spots_img_obj is None:
                raw_spot_intensities = spot_intensities
            else:
                raw_spot_intensities = (
                    raw_spots_img_obj[slice_global_to_local][spheroid_mask]
                )
                value = raw_spots_img_obj[zyx_center]
                df_obj_spots.at[spot_id, 'spot_raw_intensity_at_center'] = value

                self._add_distribution_metrics(
                    raw_spot_intensities, df_obj_spots, spot_id, 
                    col_name='spot_raw_*name_in_spot_minimumsize_vol'
                )

            self._add_ttest_values(
                spot_intensities_detect, backgr_vals, df_obj_spots, spot_id, 
                name='spot_vs_backgr'
            )

            self._add_effect_sizes(
                spot_intensities_detect, backgr_vals, df_obj_spots, spot_id, 
                name='spot_vs_backgr'
            )

            if do_filter_spots_vs_ref_ch:
                normalised_spot_intensities, normalised_ref_ch_intensities = (
                    self._get_normalised_spot_ref_ch_intensities(
                        normalised_spots_img_obj, normalised_ref_ch_img_obj,
                        spheroid_mask, slice_global_to_local, dist_transf
                    )
                )
                self._add_ttest_values(
                    normalised_spot_intensities, normalised_ref_ch_intensities, 
                    df_obj_spots, spot_id, name='spot_vs_ref_ch'
                )
                self._add_effect_sizes(
                    normalised_spot_intensities, normalised_ref_ch_intensities, 
                    df_obj_spots, spot_id, name='spot_vs_ref_ch'
                )

            if ref_ch_mask_obj is not None:
                self._add_spot_vs_ref_location(
                    ref_ch_mask_obj, zyx_center, df_obj_spots, spot_id
                )

            if ref_ch_img_obj is None:
                # Raw reference channel not present --> continue
                pbar.update()
                continue
            
            value = ref_ch_img_obj[zyx_center]
            df_obj_spots.at[spot_id, 'ref_ch_raw_intensity_at_center'] = value

            ref_ch_intensities = (
                ref_ch_img_obj[slice_global_to_local][spheroid_mask]
            )
            self._add_distribution_metrics(
                ref_ch_intensities, df_obj_spots, spot_id, 
                col_name='ref_ch_raw_*name_in_spot_minimumsize_vol'
            )
            pbar.update()
        pbar.close()
        return df_obj_spots
    
    @utils.exception_handler_cli
    def filter_spots(self, df: pd.DataFrame, features_thresholds: dict):
        """_summary_

        Parameters
        ----------
        df : pd.DataFrame
            Pandas DataFrame with 'spot_id' as index and the features as columns.
        features_thresholds : dict
            A dictionary of features and thresholds to use for filtering. The 
            keys are the feature names that mush coincide with one of the columns'
            names. The values are a tuple of `(min, max)` thresholds.
            For example, for filtering spots that have the t-statistic of the 
            t-test spot vs reference channel > 0 and the p-value < 0.025 
            (i.e. spots are significantly brighter than reference channel) 
            we pass the following dictionary:
            ```
            features_thresholds = {
                'spot_vs_ref_ch_ttest_pvalue': (None,0.025),
	            'spot_vs_ref_ch_ttest_tstat': (0, None)
            }
            ```
            where `None` indicates the absence of maximum or minimum.

        Returns
        -------
        pd.DataFrame
            The filtered DataFrame
        """      
        queries = []  
        for feature_name, thresholds in features_thresholds.items():
            if feature_name not in df.columns:
                self._critical_feature_is_missing(feature_name, df)
            _min, _max = thresholds
            if _min is not None:
                queries.append(f'({feature_name} > {_min})')
            if _max is not None:
                queries.append(f'({feature_name} < {_max})')
        query = ' & '.join(queries)
        return df.query(query)
    
    def _critical_feature_is_missing(self, missing_feature, df):
        format_colums = [f'    * {col}' for col in df.columns]
        format_colums = '\n'.join(format_colums)
        self.logger.info('='*50)
        txt = (
            f'[ERROR]: The feature name {missing_feature} is not present in the table.\n\n'
            f'Available features are:\n\n{format_colums}{error_up_str}'
        )
        self.logger.info(txt)
        self.logger.info('spotMAX aborted due to ERROR. See above more details.')
        self.quit()
    
    def _add_segm_obj_features_from_labels(
            self, df_agg, lab, rp, metadata, frame_i=0, is_segm_3D=False
        ):
        if np.all(lab):
            # Segmentation was not present and it was initialized to whole image
            # There are no features to add
            return
        pxl_to_um2 = metadata.get('pxl_to_um2_factor', 1)
        vox_to_um3 = metadata.get('vox_to_um3_factor', 1)
        vox_to_fl_rot = metadata.get('vox_to_fl_rot_factor', 1)
        for obj in rp:
            idx = (frame_i, obj.label)
            cell_area_pxl = obj.area
            cell_area_um2 = cell_area_pxl*pxl_to_um2
            cell_vol_vox, cell_vol_fl = acdctools.measure.rotational_volume(
                obj, vox_to_fl=vox_to_fl_rot
            )
            df_agg.at[idx, 'cell_area_pxl'] = cell_area_pxl
            df_agg.at[idx, 'cell_area_um2'] = cell_area_um2
            df_agg.at[idx, 'cell_vol_vox'] = cell_vol_vox
            df_agg.at[idx, 'cell_vol_fl'] = cell_vol_fl
            if is_segm_3D:
                cell_vol_vox_3D = cell_area_pxl
                cell_vol_fl_3D = cell_area_pxl*vox_to_um3
                df_agg.at[idx, 'cell_vol_vox_3D'] = cell_vol_vox_3D
                df_agg.at[idx, 'cell_vol_fl_3D'] = cell_vol_fl_3D
        return df_agg

    def _add_spotfit_feautres_to_df_spots_gop(df_spots_fit, df_spots_gop):
        idx = df_spots_fit.index
        for col in df_spots_gop.columns:
            if col in df_spots_fit.columns:
                continue
            
            df_spots_fit[col] = np.nan
            df_spots_fit.loc[idx, col] = df_spots_gop.loc[idx, col]
    
    def _filter_spots_by_size(
            df_spots_fit: pd.DataFrame, spotfit_minsize, spotfit_maxsize
        ):
        queries = []
        if spotfit_minsize > 0:
            queries.append(f'(sigma_yx_mean >= {spotfit_minsize})')
        
        if spotfit_maxsize > 0:
            queries.append(f'(sigma_yx_mean <= {spotfit_maxsize})')
        
        if not queries:
            return df_spots_fit
        
        query = ' & '.join(queries)
        df_spots_fit = df_spots_fit.query(query)
        return df_spots_fit
    
    def init_report(self, params_path, report_filepath):
        report_filepath = io.get_abspath(report_filepath)
        self.logger.info(
            f'Initializing report (it will be saved to "{report_filepath}")...'
        )
        self._report = {
            'datetime_started': datetime.now(), 'params_path': params_path,
            'pos_info': {}, 'report_filepath': report_filepath
        }
    
    def get_default_report_filepath(self, params_path):
        folder_path = os.path.dirname(params_path)
        params_filename = os.path.basename(params_path)
        report_filename = params_filename.replace('.ini', '_spotMAX_report.rst')
        save_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        report_filename = f'{save_datetime}_{report_filename}'
        report_filepath = os.path.join(folder_path, report_filename)
        return report_filepath
    
    def save_report(self):
        if not hasattr(self, '_report'):
            return
        
        datetime_stopped = datetime.now()
        title = 'spotMAX analysis report'
        _line_title = '*'*len(title)
        title = f'{_line_title}\n{title}\n{_line_title}'
        report_formatted = (
            f'{title}\n\n'
            f'Analysis started on: {self._report["datetime_started"]}\n'
            f'Analysis ended on: {datetime_stopped}\n'
            f'Log file: "{self.log_path}"\n\n'
            f'Parameters file: "{self._report["params_path"]}"'
        )
        pos_txt = None
        for pos_path, info in self._report['pos_info'].items():
            subtitle = (
                f'The Position "{pos_path}" raised the following '
                'ERRORS and WARNINGS:'
            )
            underline_subtitle = '#'*len(subtitle)
            subtitle = f'{subtitle}\n{underline_subtitle}'
            errors = [f'* [ERROR]: {e}' for e in info['errors']]
            errors = '\n'.join(errors)
        
            _warnings = [f'* [WARNING]: {w}' for w in info['warnings']]
            _warnings = '\n'.join(_warnings)

            pos_txt = f'{subtitle}\n\n{errors}\n{_warnings}'
            end_of_pos = '-'*80
            report_formatted = f'{report_formatted}{pos_txt}\n{end_of_pos}\n\n'
        if pos_txt is None:
            report_formatted = (
                f'{report_formatted}\n\nNo errors or warnings to report.'
            )
        else:
            report_formatted = (
                f'{report_formatted}\n'
                'If you need help understanding the errors, feel free to '
                'open an issue on our GitHub page at the follwing link: '
                f'"{issues_url}"\n\n'
                'Please **send the log file** when opening an issue, thanks!\n\n'
                f'Log file path: "{self.log_path}"'
            )
        
        report_filepath = self._report['report_filepath']
        with open(report_filepath, 'w') as rst:
            rst.write(report_formatted) 
        self.logger.info('#'*50)
        self.logger.info(
            f'Final report saved to "{report_filepath}"'
        )
        self.logger.info('#'*50)

    def log_warning_report(self, warning_txt):
        if self._current_pos_path not in self._report['pos_info']:
            self._report['pos_info'][self._current_pos_path] = {
                'errors': [], 'warnings': []
            }
        self._report['pos_info'][self._current_pos_path]['warnings'].append(
            warning_txt
        )

    def log_exception_report(self, error, traceback_str=''):
        if self._force_close_on_critical:
            self.quit(error)
        else:
            if self._current_pos_path not in self._report['pos_info']:
                self._report['pos_info'][self._current_pos_path] = {
                    'errors': [], 'warnings': []
                }
            self._report['pos_info'][self._current_pos_path]['errors'].append(
                error
            )

    @utils.handle_log_exception_cli
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
        df_agg = data.get('df_agg')
        ref_ch_segm_data = data.get('ref_ch_segm')
        acdc_df = data.get('lineage_table')

        stopFrameNum = self.metadata['stopFrameNum']

        desc = 'Adding single-segmentation object features'
        pbar = tqdm(
            total=stopFrameNum, ncols=100, desc=desc, position=2, leave=False
        )
        for frame_i in range(stopFrameNum):
            lab = segm_data[frame_i]
            rp = segm_rp[frame_i]
            if acdc_df is not None:
                lineage_table = acdc_df.loc[frame_i]
            else:
                lineage_table = None
            df_agg = self._add_segm_obj_features_from_labels(
                df_agg, lab, rp, is_segm_3D=data['is_segm_3D'], 
                frame_i=frame_i, metadata=self.metadata
            )
            pbar.update()
        pbar.close()
        
        if ref_ch_data is not None and do_segment_ref_ch:
            print('')
            self.logger.info('Segmenting reference channel...')
            SECTION = 'Reference channel'
            ref_ch_threshold_method = (
                self._params[SECTION]['refChThresholdFunc']['loadedVal']
            )
            is_ref_ch_single_obj = (
                self._params[SECTION]['refChSingleObj']['loadedVal']
            )
            vox_to_um3 = self.metadata.get('vox_to_um3_factor', 1)
            ref_ch_segm_data = np.zeros(ref_ch_data.shape, dtype=np.uint16)
            desc = 'Frames completed (segm. ref. ch.)'
            pbar = tqdm(
                total=stopFrameNum, ncols=100, desc=desc, position=2, 
                leave=False
            )
            for frame_i in range(stopFrameNum):
                if acdc_df is not None:
                    lineage_table = acdc_df.loc[frame_i]
                else:
                    lineage_table = None
                lab_rp = segm_rp[frame_i]
                ref_ch_img = ref_ch_data[frame_i]
                ref_ch_img = self._preprocess(ref_ch_img)
                lab = segm_data[frame_i]
                ref_ch_lab, df_agg = self.segment_ref_ch(
                    ref_ch_img, lab_rp=lab_rp, lab=lab, 
                    threshold_method=ref_ch_threshold_method, 
                    keep_only_largest_obj=is_ref_ch_single_obj,
                    df_agg=df_agg, frame_i=frame_i, 
                    do_aggregate_objs=do_aggregate_objs,
                    lineage_table=lineage_table, vox_to_um3=vox_to_um3,
                    verbose=False
                )
                ref_ch_segm_data[frame_i] = ref_ch_lab
                pbar.update()
            pbar.close()

            df_agg = self.ref_ch_to_physical_units(df_agg, self.metadata)

            data['df_agg'] = df_agg
            data['ref_ch_segm'] = ref_ch_segm_data
        
        if 'spots_ch' not in data:
            dfs = {'agg_detection': data['df_agg']}
            return dfs
        
        spots_data = data.get('spots_ch')
        zyx_resolution_limit_pxl = self.metadata['zyxResolutionLimitPxl']
        min_size_spheroid_mask = self._get_local_spheroid_mask(
            zyx_resolution_limit_pxl
        )
        edt_spheroid = self._distance_transform_edt(min_size_spheroid_mask)

        # Get footprint passed to peak_local_max --> use half the radius
        # since spots can overlap by the radius according to resol limit
        spot_footprint = self._get_local_spheroid_mask(
            [val/2 for val in zyx_resolution_limit_pxl]
        )
        
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

        SECTION = 'Spots channel'
        gop_filtering_thresholds = (
            self._params[SECTION]['gopThresholds']['loadedVal']
        )
        prediction_method = (
            self._params[SECTION]['spotPredictionMethod']['loadedVal']
        )
        threshold_method = (
            self._params[SECTION]['spotThresholdFunc']['loadedVal']
        )
        detection_method = (
            self._params[SECTION]['spotDetectionMethod']['loadedVal']
        )
        do_spotfit = self._params[SECTION]['doSpotFit']['loadedVal']
        spotfit_minsize = self._params[SECTION]['minSpotSize']['loadedVal']
        spotfit_maxsize = self._params[SECTION]['maxSpotSize']['loadedVal']
        dfs_lists = {
            'dfs_spots_detection': [], 'dfs_spots_gop_test': [], 'keys': []
        }
        if do_spotfit:
            dfs_lists['dfs_spots_spotfit'] = []
            dfs_lists['spotfit_keys'] = []

        desc = 'Frames completed (spot detection)'
        pbar = tqdm(
            total=stopFrameNum, ncols=100, desc=desc, position=2, leave=False
        )
        for frame_i in range(stopFrameNum):
            raw_spots_img = spots_data[frame_i]
            filtered_spots_img = self._preprocess(raw_spots_img)
            if do_sharpen_spots:
                sharp_spots_img = self._sharpen_spots(
                    filtered_spots_img, self.metadata
                )
            lab = segm_data[frame_i]
            rp = segm_rp[frame_i]
            if ref_ch_data is not None:
                ref_ch_img = ref_ch_data[frame_i]
                filtered_ref_ch_img = self._preprocess(ref_ch_img)
            else:
                ref_ch_img = None
            if ref_ch_segm_data is not None:
                ref_ch_mask_or_labels = ref_ch_segm_data[frame_i]
            else:
                ref_ch_mask_or_labels = None
            if acdc_df is not None:
                lineage_table = acdc_df.loc[frame_i]
            else:
                lineage_table = None
            self.spots_detection(
                filtered_spots_img, zyx_resolution_limit_pxl, 
                spots_img_detect=sharp_spots_img,
                ref_ch_img=filtered_ref_ch_img, 
                frame_i=frame_i, lab=lab, rp=rp,
                ref_ch_mask_or_labels=ref_ch_mask_or_labels, 
                df_agg=df_agg,
                raw_spots_img=raw_spots_img, 
                dfs_lists=dfs_lists,
                min_size_spheroid_mask=min_size_spheroid_mask,
                dist_transform_spheroid=edt_spheroid,
                spot_footprint=spot_footprint,
                do_filter_spots_vs_ref_ch=do_filter_spots_vs_ref_ch,
                do_keep_spots_in_ref_ch=do_keep_spots_in_ref_ch,
                gop_filtering_thresholds=gop_filtering_thresholds,
                prediction_method=prediction_method,
                threshold_method=threshold_method,
                detection_method=detection_method,
                do_aggregate_objs=do_aggregate_objs,
                lineage_table=lineage_table,
                verbose=False
            )
            pbar.update()
        pbar.close()

        names = ['frame_i', 'Cell_ID', 'spot_id']
        keys = dfs_lists['keys']
        df_spots_det = pd.concat(
            dfs_lists['dfs_spots_detection'], keys=keys, names=names
        )
        df_spots_gop = pd.concat(
            dfs_lists['dfs_spots_gop_test'], keys=keys, names=names
        )

        if do_spotfit:
            zyx_spot_min_vol_um = self.metadata['zyxResolutionLimitUm']
            zyx_voxel_size = self.metadata['zyxVoxelSize']
            desc = 'Measuring spots (spotFIT)'
            pbar = tqdm(
                total=stopFrameNum, ncols=100, desc=desc, position=2, 
                leave=False
            )
            for frame_i in range(stopFrameNum):
                raw_spots_img = spots_data[frame_i]
                if ref_ch_segm_data is not None:
                    ref_ch_mask_or_labels = ref_ch_segm_data[frame_i]
                else:
                    ref_ch_mask_or_labels = None

                df_spots_frame = df_spots_gop.loc[frame_i]
                self.measure_spots_spotfit(
                    raw_spots_img, df_spots_frame, zyx_voxel_size, 
                    zyx_spot_min_vol_um, rp=rp, dfs_lists=dfs_lists,
                    ref_ch_mask_or_labels=ref_ch_mask_or_labels,
                    frame_i=frame_i
                )
                pbar.update()
            pbar.close()
            
            keys = dfs_lists['spotfit_keys']
            df_spots_fit = pd.concat(
                dfs_lists['dfs_spots_spotfit'], keys=keys, names=names
            )
            self._add_spotfit_feautres_to_df_spots_gop(
                df_spots_fit, df_spots_gop
            )
            df_spots_fit = self._filter_spots_by_size(
                df_spots_fit, spotfit_minsize, spotfit_maxsize
            )
        else:
            df_spots_fit = None
        
        dfs_agg = self._add_aggregated_spots_features(
            df_spots_det, df_spots_gop, df_agg, df_spots_fit=df_spots_fit
        )
        df_agg_det, df_agg_gop, df_agg_spotfit = dfs_agg

        dfs = {
            'spots_detection': df_spots_det,
            'spots_gop': df_spots_gop,
            'spots_spotfit': df_spots_fit,
            'agg_detection': df_agg_det,
            'agg_gop': df_agg_gop,
            'agg_spotfit': df_agg_spotfit
        }
        return dfs

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
        desc = 'Experiments completed'
        pbar_exp = tqdm(total=len(exp_paths), ncols=100, desc=desc, position=0)  
        for exp_path, exp_info in exp_paths.items():
            exp_path = utils.io.get_abspath(exp_path)
            exp_foldername = os.path.basename(exp_path)
            exp_parent_foldername = os.path.basename(os.path.dirname(exp_path))
            run_number = exp_info['run_number']
            pos_foldernames = exp_info['pos_foldernames']  
            spots_ch_endname = exp_info['spotsEndName'] 
            ref_ch_endname = exp_info['refChEndName']
            segm_endname = exp_info['segmEndName']
            ref_ch_segm_endname = exp_info['refChSegmEndName']
            lineage_table_endname = exp_info['lineageTableEndName']
            text_to_append = exp_info['textToAppend']
            desc = 'Experiments completed'
            pbar_pos = tqdm(total=len(exp_paths), ncols=100, desc=desc, position=1) 
            for pos in pos_foldernames:
                print('')
                pos_path = os.path.join(exp_path, pos)
                rel_path = os.path.join(
                    exp_parent_foldername, exp_foldername, pos
                )
                self.logger.info(f'Analysing "...{os.sep}{rel_path}"...')
                images_path = os.path.join(pos_path, 'Images')
                self._current_pos_path = pos_path
                dfs = self._run_from_images_path(
                    images_path, 
                    spots_ch_endname=spots_ch_endname, 
                    ref_ch_endname=ref_ch_endname, 
                    segm_endname=segm_endname,
                    ref_ch_segm_endname=ref_ch_segm_endname, 
                    lineage_table_endname=lineage_table_endname
                )      
                if dfs is None:
                    # Error raised, logged and dfs is None
                    continue
                self.save_dfs(
                    pos_path, dfs, run_number=run_number, 
                    text_to_append=text_to_append
                )
                pbar_pos.update()
            pbar_pos.close()
            pbar_exp.update()
        pbar_exp.close()
    
    def _add_missing_cells_df_agg(self, df_agg_src, df_agg_dst):
        missing_idx_df_agg_dst = df_agg_src.index.difference(df_agg_dst.index)
        default_src_values = {
            col:df_agg_src.at[idx, col] for col in df_agg_src.columns 
            for idx in missing_idx_df_agg_dst
        }
        df_agg_dst = df_agg_dst.reindex(df_agg_src.index, fill_value=0)
        default_dst_values = {}
        for col in df_agg_dst.columns:
            if col not in aggregate_spots_feature_func:
                continue
            default_dst_values[col] = aggregate_spots_feature_func[col][2]
        
        default_values = {**default_src_values, **default_dst_values}
        cols = default_values.keys()
        vals = default_values.values()

        df_agg_dst.loc[missing_idx_df_agg_dst, cols] = vals
        return df_agg_dst

    def save_dfs(self, folder_path, dfs, run_number=1, text_to_append=''):
        spotmax_out_path = os.path.join(folder_path, 'spotMAX_output')
        if not os.path.exists(spotmax_out_path):
            os.mkdir(spotmax_out_path)
        
        analysis_inputs_filepath = os.path.join(
            spotmax_out_path, f'{run_number}_analysis_parameters.ini'
        )
        shutil.copy2(self.ini_params_file_path, analysis_inputs_filepath)

        if text_to_append and not text_to_append.startswith('_'):
            text_to_append = f'_{text_to_append}'

        for key, filename in dfs_filenames.items():
            filename = filename.replace('*rn*', str(run_number))
            filename = filename.replace('*desc*', text_to_append)
            df_spots = dfs.get(key, None)
            h5_filename = filename

            if df_spots is not None:
                io.save_df_to_hdf(df_spots, spotmax_out_path, h5_filename)
            
            agg_filename = h5_filename.replace('.h5', '_aggregated.csv')
            agg_key = key.replace('spots', 'agg')
            df_agg = dfs.get(agg_key, None)

            if df_agg is not None:
                df_agg.to_csv(os.path.join(spotmax_out_path, agg_filename))

    @utils.exception_handler_cli
    def _run_single_path(self, single_path_info):
        pass

    @utils.exception_handler_cli
    def run(
            self, params_path: os.PathLike, 
            metadata_csv_path: os.PathLike='',
            num_numba_threads: int=-1, 
            force_default_values: bool=False, 
            force_close_on_critical: bool=False, 
            disable_final_report=False,
            report_filepath='',
            parser_args=None
        ):
        self._force_default = force_default_values
        self._force_close_on_critical = force_close_on_critical
        if NUMBA_INSTALLED and num_numba_threads > 0:
            numba.set_num_threads(num_numba_threads)
        proceed = self.init_params(
            params_path, metadata_csv_path=metadata_csv_path
        )
        if not proceed:
            self.quit()
            return

        if parser_args is not None:
            params_path = self.ini_params_file_path
            self.add_parser_args_to_params_ini_file(parser_args, params_path)
        
        is_report_enabled = not disable_final_report
        if is_report_enabled and report_filepath:
            self.init_report(self.ini_params_file_path, report_filepath)
        
        if self.exp_paths_list:
            self.is_batch_mode = True
            for exp_paths in self.exp_paths_list:
                self._run_exp_paths(exp_paths)
            self.save_report()
        else:
            self._run_single_path(self.single_path_info)
        self.quit()
            
    def quit(self, error=None):
        if not self.is_cli and error is not None:
            raise error

        self.logger.info('='*50)
        if error is not None:
            self.logger.exception(traceback.format_exc())
            print('-'*60)
            self.logger.info(f'[ERROR]: {error}{error_up_str}')
            err_msg = (
                'spotMAX aborted due to **error**. '
                'More details above or in the following log file:\n\n'
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

def expand_labels(label_image, distance=1, zyx_vox_size=None):
    """Expand labels in label image by ``distance`` pixels without overlapping.
    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).
    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Euclidean distance in pixels by which to grow the labels. Default is one.
    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged
    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.
    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.
    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])
    Labels will not overwrite each other:
    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])
    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.
    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """
    if zyx_vox_size is None:
        zyx_vox_size = [1]*label_image.ndim

    distances, nearest_label_coords = scipy.ndimage.distance_transform_edt(
        label_image == 0, return_indices=True, sampling=zyx_vox_size,
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out

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
