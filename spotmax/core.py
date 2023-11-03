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

import cellacdc.io
import cellacdc.myutils as acdc_myutils
import cellacdc.measure

from . import GUI_INSTALLED, error_up_str, error_down_str
from . import exception_handler_cli, handle_log_exception_cli

if GUI_INSTALLED:
    from cellacdc.plot import imshow
    import matplotlib.pyplot as plt
    import matplotlib

NUMBA_INSTALLED = False
try:
    import numba
    from numba import njit, prange
    NUMBA_INSTALLED = True
except Exception as e:
    from .utils import njit_replacement as njit
    prange = range

from . import utils, rng, base_lineage_table_values
from . import issues_url, printl, io, features, config
from . import transformations
from . import filters
from . import pipe
from . import ZYX_GLOBAL_COLS, ZYX_LOCAL_COLS, ZYX_AGGR_COLS
from . import ZYX_LOCAL_EXPANDED_COLS, ZYX_FIT_COLS, ZYX_RESOL_COLS
from . import BASE_COLUMNS
from . import DFs_FILENAMES

import math
SQRT_2 = math.sqrt(2)

np.seterr(divide='warn', invalid='warn')

CHANNELS_KEYS = (
    'spots_ch', 'ref_ch', 'ref_ch_segm', 'spots_ch_segm', 'segm',
    'transformed_spots_ch'
)
distribution_metrics_func = features.get_distribution_metrics_func()
effect_size_func = features.get_effect_size_func()
aggregate_spots_feature_func = features.get_aggregating_spots_feature_func()

class _DataLoader:
    def __init__(self, debug=False, log=print):
        self.debug = debug
        self.log = log
    
    def get_data_from_images_path(
            self, 
            images_path: os.PathLike, 
            spots_ch_endname: str, 
            ref_ch_endname: str, 
            segm_endname: str, 
            spots_ch_segm_endname: str,
            ref_ch_segm_endname: str,
            lineage_table_endname: str,
            transformed_spots_ch_nnet=None,
        ):
        data = self._load_data_from_images_path(
            images_path, spots_ch_endname, ref_ch_endname, segm_endname, 
            spots_ch_segm_endname, ref_ch_segm_endname, lineage_table_endname
        )
        if transformed_spots_ch_nnet is not None:
            data['transformed_spots_ch'] = transformed_spots_ch_nnet
        data = self._reshape_data(data, self.metadata)
        data = self._crop_based_on_segm_data(data)
        data = self._add_regionprops(data)
        data = self._initialize_df_agg(data)
        return data
    
    def _critical_channel_not_found(
            self, channel, images_path, searched_ext=None
        ):
        ext = os.path.splitext(channel)[-1]
        if ext:
            searched_files = f'   * _{channel}\n'
        elif searched_ext is not None:
            searched_files = f'   * _{channel}{searched_ext}\n'
        else:
            searched_files = (
                f'   * _{channel}.tif\n'
                f'   * _{channel}.h5\n'
                f'   * _{channel}_aligned.h5\n'
                f'   * _{channel}_aligned.npz\n'
                f'   * _{channel}.npy\n'
                f'   * _{channel}.npz\n'
            )
        self.logger.info(
            f'{error_down_str}\n'
            f'The file ending with "{channel}" was not found. If you are trying to load '
            'a channel without an extension make sure that one of the following '
            'channels exists:\n\n'
            f'{searched_files}\n'
            'Alternatively, provide the extension to the channel name in the '
            '.ini configuration file.\n'
        )
        error_msg = (
            f'The file ending with "{channel}" does not exist in the folder "{images_path}"'
        )
        error = FileNotFoundError(error_msg)
        self.logger.info(f'[ERROR]: {error}{error_up_str}')       
        raise error
    
    def _load_data_from_images_path(
            self, images_path: os.PathLike, 
            spots_ch_endname: str, 
            ref_ch_endname: str, 
            segm_endname: str, 
            spots_ch_segm_endname: str, 
            ref_ch_segm_endname: str,
            lineage_table_endname: str
        ):
        channels = {
            spots_ch_endname: 'spots_ch', 
            ref_ch_endname: 'ref_ch', 
            segm_endname: 'segm',
            spots_ch_segm_endname: 'spots_ch_segm',
            ref_ch_segm_endname: 'ref_ch_segm'
        }
        data = {'basename': io.get_basename(utils.listdir(images_path))}
        for channel, key in channels.items():
            if not channel:
                continue
            ch_path = cellacdc.io.get_filepath_from_channel_name(
                images_path, os.path.basename(channel)
            )
            if not os.path.exists(ch_path):
                self._critical_channel_not_found(channel, images_path)
                return

            self.log(f'Loading "{channel}" channel from "{ch_path}"...')
            to_float = key == 'spots_ch' or key == 'ref_ch'
            ch_data, ch_dtype = io.load_image_data(
                ch_path, to_float=to_float, return_dtype=True
            )
            data[f'{key}.dtype'] = ch_dtype
            data[key] = ch_data
            data[f'{key}.shape'] = ch_data.shape
            data[f'{key}.channel_name'] = channel

        data = self._init_reshape_segm_data(data)

        if not lineage_table_endname:
            return data

        # Load lineage table
        table_path = cellacdc.io.get_filepath_from_endname(
            images_path, os.path.basename(lineage_table_endname), ext='.csv'
        )
        if table_path is None:
            self._critical_channel_not_found(
                lineage_table_endname, images_path, searched_ext='.csv'
            )
            return
        self.log(
            f'Loading "{lineage_table_endname}" lineage table from "{table_path}"...'
        )
        data['lineage_table'] = pd.read_csv(table_path)

        return data
    
    def _add_regionprops(self, data):
        data['segm_rp'] = [
            skimage.measure.regionprops(data['segm'][frame_i]) 
            for frame_i in range(len(data['segm']))
        ]
        return data

    def _init_reshape_segm_data(self, data):
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
        return data
    
    def _reshape_data(self, data, metadata: dict):
        SizeZ = metadata['SizeZ']
        for key in CHANNELS_KEYS:
            if key not in data:
                continue
            ch_data = data[key]
            if SizeZ > 1:
                # Data is already 3D z-stacks. T axis will be added later
                continue
            
            if ch_data.ndim == 2:
                # Data is 2D. Add new axis. T axis will be added later
                data[key] = data[key][np.newaxis]
            elif ch_data.ndim == 3:
                # Data is 3D timelpase. Add axis for z-slice
                data[key] = data[key][:, np.newaxis]
            
            data[f'{key}.shape'] = data[key].shape

        if 'lineage_table' in data:
            table = data['lineage_table']
            if 'frame_i' not in table.columns:
                table['frame_i'] = 0
            table = table.set_index(['frame_i', 'Cell_ID'])
            data['lineage_table'] = table
        
        SizeT = metadata['SizeT']
        if SizeT > 1:
            # Data is already time-lapse --> check that it is true and do not 
            # reshape
            for key in CHANNELS_KEYS:
                if key not in data:
                    continue
                if key == 'segm':
                    # Do not check segm data length since it can be shorter
                    continue
                arr_data = data[key]
                if arr_data.shape[0] != SizeT:
                    D0 = arr_data.shape[0]
                    channel = data[f'{key}.channel_name']
                    raise TypeError(
                        f'The first dimension of the channel "{channel}" is {D0}'
                        f' (shape is {arr_data.shape}), but the "Number of frames '
                        f'(SizeT)" in the configuration file is {SizeT}. '
                        'Please double check that, thanks!'
                    )
                if arr_data.ndim != 4:
                    channel = data[f'{key}.channel_name']
                    raise TypeError(
                        f'The shape of the image data from channel "{channel}" '
                        f'is {arr_data.shape}, which means that either "Number '
                        'of frames (SizeT)" and/or "Number of z-slices (SizeZ)" '
                        'in the configuration file are wrong. '
                        'Please double check that, thanks!'
                    )
            return data

        for key in CHANNELS_KEYS:
            if key not in data:
                continue
            # Add axis for Time
            data[key] = data[key][np.newaxis]
            data[f'{key}.shape'] = data[key].shape
        
        return data

    def _crop_based_on_segm_data(self, data):
        segm_data = data['segm']
        if not np.any(segm_data):
            return data
        
        crop_info = transformations.crop_from_segm_data_info(
            segm_data, self.metadata['deltaTolerance'],
            lineage_table=data.get('lineage_table')
        )
        segm_slice, pad_widths, crop_to_global_coords = crop_info
        data['crop_to_global_coords'] = crop_to_global_coords
        data['pad_width'] = pad_widths
        
        # Crop images
        for key in CHANNELS_KEYS:
            if key not in data:
                continue
            data[key] = data[key][segm_slice].copy()

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
    
    def _setup_logger_file_handler(self, log_folder_path):
        if not os.path.isdir(log_folder_path):
            raise FileNotFoundError(
                'The provided path to the log does not exist or '
                f'is not a folder path. Path: "{log_folder_path}"'
            ) 
        
        if self.logs_path == os.path.normpath(log_folder_path):
            return

        self.logs_path = os.path.normpath(log_folder_path)
        log_filename = os.path.basename(self.log_path)
        new_log_path = os.path.join(log_folder_path, log_filename)
        shutil.copy(self.log_path, new_log_path)
        utils.store_current_log_file_path(new_log_path)
        self.log_path = new_log_path

        # Copy log file and add new handler
        self.logger.removeHandler(self.logger._file_handler)

        file_handler = utils.logger_file_handler(new_log_path, mode='a')
        self.logger._file_handler = file_handler
        self.logger.addHandler(file_handler)

        self.logger.info(f'Log file moved to "{self.log_path}"')
    
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
        
        report_folderpath = cellacdc.io.get_filename_cli(
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
        
        new_report_filepath, txt = cellacdc.path.newfilepath(report_filepath)
        new_report_filename = os.path.basename(new_report_filepath)

        default_option = f'Save with new, default filename "{report_default_filename}"'
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
        
        new_report_filename = cellacdc.io.get_filename_cli(
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
        if num_threads>0 and num_threads<=max_num_threads:
            return num_threads

        if num_threads > max_num_threads:
            print('')
            self.logger.info(
                '[WARNING]: Max number of numba threads on this machine is '
                f'{max_num_threads}. Ignoring provided number ({num_threads}) '
                f'and using the maximum allowed ({max_num_threads}).'
            )
            return max_num_threads
        
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
            parser_func = options.get('parser')
            if parser_func is not None:
                value = parser_func(value)
            else:
                value = str(value)
            check_path_to_report = (
                anchor == 'pathToReport' 
                and options['desc'] in configPars[SECTION]
            )
            if check_path_to_report:
                # Check that path to report in ini file is a relative path 
                # -->  we don't overwrite it with the absolute path
                path_to_report_ini = configPars[SECTION][options['desc']]
                if io.is_part_of_path(value, path_to_report_ini):
                    continue
                
            configPars[SECTION][options['desc']] = value

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
            parser_value = parser_args[options['parser_arg']]
            option = configPars.get(SECTION, options['desc'], fallback=None)
            if parser_value:
                # User explicitly passed a value --> use it regardless of ini
                continue
            
            if option is None or not option:
                continue
            dtype_converter = options['dtype']
            value = dtype_converter(option)
            parser_args[options['parser_arg']] = value
            
            if anchor == 'raiseOnCritical':
                parser_args['raise_on_critical_present'] = True
        return parser_args
    
    @exception_handler_cli
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
        
        log_folder_path = parser_args['log_folderpath']
        if log_folder_path:
            self._setup_logger_file_handler(log_folder_path)
        if not log_folder_path:
            parser_args['log_folderpath'] = self.logs_path
        
        disable_final_report = parser_args['disable_final_report']
        report_folderpath = parser_args['report_folderpath']

        if not disable_final_report:
            report_filepath = self._check_report_filepath(
                report_folderpath, params_path, force_default=force_default,
                report_filename=parser_args['report_filename']
            )
            if report_filepath is None:
                self.logger.info(
                    'spotMAX execution stopped by the user. '
                    'Report filepath was not provided.'
                )
                self.quit()
                return
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

    @exception_handler_cli
    def init_params(self, params_path, metadata_csv_path=''):        
        self._params = config.analysisInputsParams(
            params_path, cast_dtypes=False
        )
        
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
        
        self.configPars = config.ConfigParser()
        self.configPars.read(self.ini_params_file_path, encoding="utf-8")
        
        self.check_metadata()
        proceed, missing_params = self.check_missing_params()
        if not proceed:
            return False, None
        self.cast_loaded_values_filepaths()
        proceed = self.check_paths_exist()
        if not proceed:
            return False, None
        self.cast_loaded_values_dtypes()
        self.set_abs_exp_paths()
        self.set_metadata()
        self.check_init_neural_network()
        proceed = self.check_contradicting_params()
        if not proceed:
            return False, None
        return True, missing_params
    
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
            self.logger.info('*'*60)
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
            new_filename = cellacdc.io.get_filename_cli(
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

    def _check_multi_or_missing_run_numbers(
            self, run_nums, exp_path, scanner_paths, user_run_number
        ):
        if len(run_nums) > 1 and user_run_number is None:
            # Multiple run numbers detected
            run_number = self._ask_user_multiple_run_nums(
                run_nums, exp_path
            )
            if run_number is None:
                self.logger.info(
                    'spotMAX stopped by the user. Run number was not provided.'
                )
                self.quit()
        elif user_run_number is None:
            # Single run number --> we still need to check if already exists
            ask_run_number = False
            for exp_path, exp_info in scanner_paths[run_nums[0]].items():
                if exp_info['numPosSpotCounted'] > 0:
                    ask_run_number = True
                    break
            else:
                run_number = 1
            
            if ask_run_number:
                run_number = self._ask_user_multiple_run_nums(
                    run_nums, exp_path
                )
                if run_number is None:
                    self.logger.info(
                        'spotMAX stopped by the user.'
                        'Run number was not provided.'
                    )
                    self.quit()
        elif user_run_number is not None:
            # Check that user run number is not already existing
            if user_run_number in run_nums:
                run_num_info = scanner_paths[user_run_number]
                ask_run_number = False
                for exp_path, exp_info in run_num_info.items():
                    if exp_info['numPosSpotCounted'] > 0:
                        ask_run_number = True
                        break
                
                if ask_run_number:
                    user_run_number = self._ask_user_run_num_exists(
                        user_run_number, run_nums, exp_path
                    )
                    if user_run_number is None:
                        self.logger.info(
                            'spotMAX stopped by the user.'
                            'Run number was not provided.'
                        )
                        self.quit()
                
            run_number = user_run_number
        return run_number        
        
    def _ask_user_run_num_exists(self, user_run_num, run_nums, exp_path):
        default_option = f'Overwrite existing run number {user_run_num}'
        options = ('Choose a different run number', default_option )
        question = 'What do you want to do'
        txt = (
            f'[WARNING]: The requested run number {user_run_num} already exists '
            'in the folder path below! '
            f'(run numbers presents are {run_nums}):\n\n'
            f'{exp_path}'
        )
        if self._force_default:
            self.logger.info('*'*60)
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

    def _ask_user_multiple_run_nums(self, run_nums, exp_path):
        new_run_num = max(run_nums)+1
        default_option = f'Save as new run number {new_run_num}'
        options = ('Choose run number to overwrite', default_option )
        question = 'What do you want to do'
        txt = (
            '[WARNING]: The following experiment was already analysed '
            f'(run numbers presents are {run_nums}):\n\n'
            f'{exp_path}'
        )
        if self._force_default:
            self.logger.info('*'*60)
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

    @exception_handler_cli
    def set_abs_exp_paths(self):
        SECTION = 'File paths and channels'
        ANCHOR = 'folderPathsToAnalyse'
        loaded_exp_paths = self._params[SECTION][ANCHOR]['loadedVal']
        user_run_number = self._params[SECTION]['runNumber'].get('loadedVal')
        self.exp_paths_list = []
        run_num_log = []
        run_num_exp_path_processed = {}
        for exp_path in loaded_exp_paths:
            acdc_myutils.addToRecentPaths(exp_path, logger=self.logger.info)
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
                if pos_foldername.startswith('Position_'):
                    exp_path = os.path.dirname(os.path.dirname(pos_path))
                else:
                    # Images folder without a Position_n folder as parent folder
                    exp_path = os.path.dirname(pos_path)
                exp_paths = (
                    {exp_path: {'pos_foldernames': [pos_foldername]}}
                )
            else:
                exp_paths = {}
            
            exp_path = exp_path.replace('\\', '/')
            
            # Scan and determine run numbers
            pathScanner = io.expFolderScanner(
                exp_path, logger_func=self.logger.info
            )
            pathScanner.getExpPaths(exp_path)
            pathScanner.infoExpPaths(pathScanner.expPaths)
            
            if exp_path not in run_num_exp_path_processed:
                run_nums = sorted([int(r) for r in pathScanner.paths.keys()])
                run_number = self._check_multi_or_missing_run_numbers(
                    run_nums, exp_path, pathScanner.paths, user_run_number
                )
            else:
                run_number = run_num_exp_path_processed[exp_path]
            
            run_num_exp_path_processed[exp_path] = run_number
            self._store_run_number(run_number, pathScanner.paths, exp_paths)
            run_num_log.append(f'  * Run number = {run_number} ("{exp_path}")')
            self.exp_paths_list.append(exp_paths)
        self.set_channel_names()
        self.logger.info('\n'.join(run_num_log))
    
    def set_channel_names(self):
        SECTION = 'File paths and channels'
        section_params = self._params[SECTION]
        spots_ch_endname = section_params['spotsEndName'].get('loadedVal')
        ref_ch_endname = section_params['refChEndName'].get('loadedVal')
        segm_endname = section_params['segmEndName'].get('loadedVal')
        ref_ch_segm_endname = section_params['refChSegmEndName'].get(
            'loadedVal'
        )
        spots_ch_segm_endname = section_params['spotChSegmEndName'].get(
            'loadedVal'
        )
        lineage_table_endname = section_params['lineageTableEndName'].get(
            'loadedVal'
        )
        text_to_append = section_params['textToAppend'].get('loadedVal', '')
        if text_to_append is None:
            text_to_append = ''
        df_spots_file_ext = section_params['dfSpotsFileExtension'].get(
            'loadedVal', '.h5'
        )
        if df_spots_file_ext is None:
            df_spots_file_ext = '.h5'
        if self.exp_paths_list:
            for i in range(len(self.exp_paths_list)):
                for exp_path in list(self.exp_paths_list[i].keys()):
                    exp_info = self.exp_paths_list[i][exp_path]
                    exp_info['spotsEndName'] = spots_ch_endname
                    exp_info['refChEndName'] = ref_ch_endname
                    exp_info['segmEndName'] = segm_endname
                    exp_info['spotChSegmEndName'] = spots_ch_segm_endname
                    exp_info['refChSegmEndName'] = ref_ch_segm_endname
                    exp_info['lineageTableEndName'] = lineage_table_endname
                    exp_info['textToAppend'] = text_to_append
                    exp_info['df_spots_file_ext'] = df_spots_file_ext
                    self.exp_paths_list[i][exp_path] = exp_info
        else:
            self.single_path_info = {
                'spots_ch_filepath': spots_ch_endname,
                'ref_ch_filepath': ref_ch_endname,
                'segm_filepath': segm_endname,
                'ref_ch_segm_filepath': ref_ch_segm_endname,
                'lineage_table_filepath': lineage_table_endname,
                'text_to_append': text_to_append,
                'df_spots_file_ext': df_spots_file_ext
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
        if metadata['SizeZ'] == 1:
            zyx_resolution_limit_pxl = (0, *zyx_resolution_limit_pxl[1:])
        metadata['zyxResolutionLimitPxl'] = zyx_resolution_limit_pxl
        metadata['zyxResolutionLimitUm'] = zyx_resolution_limit_um
        deltaTolerance = transformations.get_expand_obj_delta_tolerance(
            zyx_resolution_limit_pxl
        )
        metadata['deltaTolerance'] = deltaTolerance

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

    @exception_handler_cli
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

    @exception_handler_cli
    def _get_missing_metadata(self):
        SECTION = 'METADATA'
        missing_metadata = []
        for anchor, options in self._params[SECTION].items():
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

    def _check_timelapse_metadata(self, missing_metadata):
        SECTION = 'METADATA'
        timelapse_metadata = ('SizeT', 'stopFrameNum')
        loaded_timelapse_metadata = {}
        for anchor, options in self._params[SECTION].items():
            if anchor not in timelapse_metadata:
                continue
            desc = options['desc']
            if desc in missing_metadata:
                loaded_value = 1
            else:
                loaded_value = self._params[SECTION][anchor]['loadedVal']
            loaded_timelapse_metadata[desc] = (anchor, loaded_value)
        
        if any([val>1 for _, val in loaded_timelapse_metadata.values()]):
            # One of the timelapse metadata value suggest we are dealing with 
            # timelapse data and metadata is required.
            return
        
        # Both timelapse values are missing or 1 --> it is fine if they are missing
        for desc, (anchor, loaded_value) in loaded_timelapse_metadata.items():          
            self._params[SECTION][anchor]['loadedVal'] = 1
            try:
                missing_metadata.remove(desc)
            except Exception as e:
                pass
        return missing_metadata
    
    def _check_zstack_metadata(self, missing_metadata):
        SECTION = 'METADATA'
        zstack_metadata = ('SizeZ', 'voxelDepth', 'zResolutionLimit')
        loaded_zstack_metadata = {}
        for anchor, options in self._params[SECTION].items():
            if anchor not in zstack_metadata:
                continue
            desc = options['desc']
            if desc in missing_metadata:
                loaded_value = 1
            else:
                loaded_value = self._params[SECTION][anchor]['loadedVal']
            loaded_zstack_metadata[desc] = (anchor, loaded_value)
        
        if any([val>1 for _, val in loaded_zstack_metadata.values()]):
            # One of the zstack metadata value suggest we are dealing with 
            # zstack data and metadata is required.
            return
        
        # Both zstack values are missing or 1 --> it is fine if they are missing
        for desc, (anchor, loaded_value) in loaded_zstack_metadata.items():          
            self._params[SECTION][anchor]['loadedVal'] = 1
            try:
                missing_metadata.remove(desc)
            except Exception as e:
                pass
        return missing_metadata
    
    def check_metadata(self):
        missing_metadata = self._get_missing_metadata()
        if not missing_metadata:
            return

        missing_metadata = self._check_timelapse_metadata(missing_metadata)
        missing_metadata = self._check_zstack_metadata(missing_metadata)
        if not missing_metadata:
            return
        
        missing_metadata_str = [f'    * {v}' for v in missing_metadata]
        missing_metadata_format = '\n'.join(missing_metadata_str)
        print('*'*60)
        err_msg = (
            f'The parameters file "{self.ini_params_filename}" is missing '
            'the following REQUIRED metadata:\n\n'
            f'{missing_metadata_format}\n\n'
            'Add them to the file (see path below) '
            'at the [METADATA] section. If you do not have timelapse data and\n'
            'the "Analyse until frame number" is missing you need to\n'
            'to write "Analyse until frame number = 1".\n\n'
            f'Parameters file path: "{self.ini_params_file_path}"'
        )
        self.logger.info(err_msg)
        if self.is_cli:
            print('*'*60)
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
    
    def _save_missing_params_to_ini(self, missing_params, ini_filepath):
        configPars = config.ConfigParser()
        configPars.read(ini_filepath, encoding="utf-8")

        for param in missing_params:
            section_name, desc, default_val, anchor = param
            if anchor == 'runNumber':
                # We do not force any run number, this will be determined later.
                continue
            if section_name not in configPars.sections():
                configPars[section_name] = {}
            
            if desc in configPars[section_name]:
                continue
            
            options = self._params[section_name][anchor]
            parser_func = options.get('parser')
            if parser_func is not None:
                value = parser_func(default_val)
            else:
                value = str(default_val)
            configPars[section_name][desc] = value
        
        with open(ini_filepath, 'w', encoding="utf-8") as file:
            configPars.write(file)
    
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
            self.logger.info('*'*60)
            self.logger.info(info_txt)
            io._log_forced_default(options[0], self.logger.info)
            self._set_default_val_params(missing_params)
            return True, missing_params
        
        while True:
            answer = io.get_user_input(
                question, options=options, info_txt=info_txt, 
                logger=self.logger.info
            )
            if answer == 'No, stop process' or answer == None:
                return False, missing_params
            elif answer == 'Yes, use default values':
                self._set_default_val_params(missing_params)
                return True, missing_params
            else:
                print('')
                default_values_format = self._get_default_values_params(
                    missing_params
                )
                self.logger.info(
                    f'Default values:\n\n{default_values_format}'
                )
                print('-'*60)
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
                # At least one option suggests that ref. channel is required.
                param_requiring_ref_ch = options['desc']
                break
        else:
            return missing_ref_ch_msg

        missing_ref_ch_msg = (
            '[ERROR]: You requested to use the reference channel for the analysis '
            'but the entry "Reference channel end name or path" is missing in the '
            '.ini params file.\n\n'
            'If you do not need a reference channel, then set the following '
            'parameters in the parameters file:\n\n'
            f'   - {param_requiring_ref_ch} = False'
            '\n\n'
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

    @exception_handler_cli
    def check_missing_params(self):
        missing_params = self._get_missing_params()
        if not missing_params:
            return True, missing_params
        
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
                print('*'*60)
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
            proceed, missing_params = self._ask_user_input_missing_params(
                missing_params, info_txt=err_msg
            )
            if not proceed:
                self.logger.info(
                    'spotMAX execution stopped by the user. '
                    'Some parameters are missing'
                )
                self.quit()
                return False, False
            
        return True, missing_params

    def _ask_loaded_ref_ch_segm_and_segm_ref_ch(self):
        default_option = 'Do not segment the ref. channel'
        options = ('Do not load the ref. ch. segm. data', default_option)
        question = 'What do you want to do'
        txt = (
            f'[WARNING]: You requested to load the ref. channel segmentation data '
            f'but ALSO to segment the ref. channel.'
        )
        if self._force_default:
            self.logger.info('*'*60)
            self.logger.info(txt)
            io._log_forced_default(default_option, self.logger.info)
            return 'do_not_segment_ref_ch'
        answer = io.get_user_input(
            question, options=options, info_txt=txt, logger=self.logger.info
        )
        if answer is None:
            return
        elif answer == options[0]:
            return 'do_not_load_ref_ch_segm'
        elif answer == options[1]:
            return 'do_not_segment_ref_ch'
    
    def check_init_neural_network(self):
        self.nnet_params = None
        self.nnet_model = None
        
        SECTION = 'File paths and channels'
        ANCHOR = 'spotChSegmEndName'
        section_params = self._params[SECTION]
        spots_ch_segm_endname = section_params[ANCHOR].get('loadedVal', '')
        if spots_ch_segm_endname:
            # User provided a segm mask --> no need to use neural net
            return        
        
        SECTION = 'Spots channel'
        ANCHOR = 'spotPredictionMethod'
        section_params = self._params[SECTION]
        spots_prediction_method = section_params[ANCHOR].get('loadedVal')
        if spots_prediction_method != 'Neural network':
            # Neural network is not required
            return
        
        try:
            self.logger.info('-'*60)                
            self.logger.info('Initializing neural network...')
            from spotmax.nnet import model
            self.nnet_params = model.get_nnet_params_from_ini_params(
                self._params, use_default_for_missing=self._force_default
            )
            self.nnet_model = model.Model(**self.nnet_params['init'])
            self.nnet_params['segment']['verbose'] = False
        except Exception as err:
            self.logger.exception(traceback.format_exc())
            raise err
    
    @exception_handler_cli
    def check_contradicting_params(self):
        SECTION = 'File paths and channels'
        section_params = self._params[SECTION]
        ref_ch_segm_endname = section_params['refChSegmEndName'].get('loadedVal')
        
        SECTION = 'Reference channel'
        section_params = self._params[SECTION]
        do_segment_ref_ch = section_params['segmRefCh'].get('loadedVal')
        
        if do_segment_ref_ch and ref_ch_segm_endname:
            answer = self._ask_loaded_ref_ch_segm_and_segm_ref_ch()
            if answer is None:
                return False
            if answer == 'do_not_segment_ref_ch':
                SECTION = 'Reference channel'
                self._params[SECTION]['segmRefCh']['loadedVal'] = False
            elif answer == 'do_not_load_ref_ch_segm':
                SECTION = 'File paths and channels'
                self._params[SECTION]['refChSegmEndName']['loadedVal'] = ''
        return True
    
    def cast_loaded_values_filepaths(self):
        for section_name in list(self._params.keys()):
            if section_name != 'File paths and channels':
                continue
            anchor_names = list(self._params[section_name].keys())
            for anchor_name in anchor_names:
                to_dtype = self._params[section_name][anchor_name].get('dtype')
                if to_dtype is None:
                    continue
                option = self._params[section_name][anchor_name]
                value = option['loadedVal']
                if value is None:
                    value = option['initialVal']
                else:
                    value = to_dtype(value)
                self._params[section_name][anchor_name]['loadedVal'] = value
    
    def cast_loaded_values_dtypes(self):
        for section_name in list(self._params.keys()):
            if section_name == 'File paths and channels':
                continue
            anchor_names = list(self._params[section_name].keys())
            for anchor_name in anchor_names:
                to_dtype = self._params[section_name][anchor_name].get('dtype')
                if to_dtype is None:
                    continue
                option = self._params[section_name][anchor_name]
                value = option['loadedVal']
                if value is None:
                    value = option['initialVal']
                else:
                    try:
                        value = to_dtype(value)
                    except Exception as e:
                        import pdb; pdb.set_trace()
                self._params[section_name][anchor_name]['loadedVal'] = value
    
    def check_paths_exist(self):
        SECTION = 'File paths and channels'
        ANCHOR = 'folderPathsToAnalyse'
        loaded_exp_paths = self._params[SECTION][ANCHOR]['loadedVal']
        for exp_path in loaded_exp_paths:
            if not os.path.exists(exp_path):
                self.logger.info('='*60)
                txt = (
                    '[ERROR]: The provided experiment path does not exist: '
                    f'{exp_path}{error_up_str}'
                )
                self.logger.info(txt)
                self.logger.info('spotMAX aborted due to ERROR. See above more details.')
                return False
            if not os.path.isdir(exp_path):
                self.logger.info('='*60)
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
            # print('WARNING: error calculating chisquare')

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
            semiax_len, zyx_c, self.V_shape, return_filled_mask=True
        )
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

    def get_local_spot_mask(
            self, semiax_len, ogrid_bounds=None, return_center=False
        ):
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
        if np.count_nonzero(mask_s) <= 7:
            # Do not allow masks with less than 7 pixels since it is the 
            # the number of coeffs --> dof would be = 0
            mask_expanded = np.zeros_like(mask_s)
            for z, mask_slice in enumerate(mask_s):
                slice_expanded = skimage.morphology.dilation(mask_slice)
                mask_expanded[z] = slice_expanded
            mask_s = mask_expanded
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

        if do_sum:
            global_mask[slice_G_to_L] += cropped_mask
        else:
            global_mask[slice_G_to_L][cropped_mask] = True
        if additional_local_arr is not None:
            additional_global_arr[slice_G_to_L] = cropped_mask_2
        
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

    def insert_grown_spot_id(
            self, grow_step_i, id, zyx_vox_dim, zyx_seed_size, zyx_c, 
            spots_3D_lab
        ):
        a, c = self.calc_semiax_len(
            grow_step_i, zyx_vox_dim, zyx_seed_size
        )
        semiax_len = a, c
        local_spot_mask = self.get_local_spot_mask(semiax_len)
        Z, Y, X = self.V_shape

        slice_G_to_L, slice_crop = self.get_slice_G_to_L(
            semiax_len, zyx_c, Z, Y, X
        )
        cropped_mask = local_spot_mask[slice_crop]
        # Avoid spot overwriting existing spot
        cropped_mask[spots_3D_lab[slice_G_to_L] != 0] = False
        spots_3D_lab[slice_G_to_L][cropped_mask] = id
        return spots_3D_lab

    def get_spots_mask(
            self, i, zyx_vox_dim, zyx_resolution, zyx_centers,
            method='min_spheroid', dtype=bool, ids=[]
        ):
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
        for c, zyx_c in enumerate(zyx_centers):
            (temp_mask, _, slice_G_to_L,
            slice_crop) = self.index_local_into_global_mask(
                temp_mask, local_spot_mask, zyx_c, semiax_len, Z, Y, X,
                return_slice=True
            )
            if dtype == bool:
                spots_mask = np.logical_or(spots_mask, temp_mask)
            elif dtype == np.uint32:
                cropped_mask = local_spot_mask[slice_crop]
                spots_mask[slice_G_to_L][cropped_mask] = ids[c]
            in_pbar.update(1)
        in_pbar.close()
        return spots_mask

    def expand_spots_labels(
            self, labels, zyx_vox_size, zyx_seed_size, spots_centers,
            grow_iter=0
        ):
        _, nearest_label_coords = scipy.ndimage.distance_transform_edt(
            labels==0, return_indices=True, sampling=zyx_vox_size,
        )
        dilate_mask = self.get_spots_mask(
            grow_iter, zyx_vox_size, zyx_seed_size, spots_centers
        )
        labels_out = np.zeros_like(labels)

        # build the coordinates to find nearest labels
        masked_nearest_label_coords = [
            dimension_indices[dilate_mask]
            for dimension_indices in nearest_label_coords
        ]
        nearest_labels = labels[tuple(masked_nearest_label_coords)]
        labels_out[dilate_mask] = nearest_labels
        return labels_out

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
            self, expanded_obj, spots_img, df_spots, zyx_vox_size, 
            zyx_spot_min_vol_um, verbose=0, inspect=0, 
            ref_ch_mask_or_labels=None, use_gpu=False,
            logger=None
        ):
        self.logger = logger
        self.spots_img_local = spots_img[expanded_obj.slice]
        super().__init__(self.spots_img_local)
        ID = expanded_obj.label
        self.ID = expanded_obj.label
        self.df_spots_ID = df_spots.loc[ID].copy()
        self.zyx_vox_size = zyx_vox_size
        self.obj_bbox_lower = tuple(expanded_obj.crop_obj_start)
        self.obj_image = expanded_obj.image
        self.zyx_spot_min_vol_um = zyx_spot_min_vol_um
        if ref_ch_mask_or_labels is not None:
            self.ref_ch_mask_local = ref_ch_mask_or_labels[expanded_obj.slice] > 0
        else:
            self.ref_ch_mask_local = None
        self.verbose = verbose
        self.inspect = inspect
        # z0, y0, x0, sz, sy, sx, A = coeffs; B added as one coeff
        self.num_coeffs = 7
        self._tol = 1e-10
        self.use_gpu = use_gpu

    def fit(self):
        verbose = self.verbose
        inspect = self.inspect
        df_spots_ID = self.df_spots_ID

        self.spotSIZE()
        self.compute_neigh_intersect()
        self._fit()
        self._quality_control()

        if self.fit_again_idx:
            self._fit_again()

        cols_to_drop = ['intersecting_idx', 'neigh_idx',  's', 'neigh_ids']
        _df_spotFIT = (
            self._df_spotFIT.reset_index()
            .drop(cols_to_drop, axis=1)
            .set_index('id')
        )
        _df_spotFIT.index.names = ['spot_id']

        df_spots_ID = self.df_spots_ID

        self.df_spotFIT_ID = df_spots_ID.join(_df_spotFIT, how='outer')
        self.df_spotFIT_ID.index.names = ['spot_id']

    def spotSIZE(self):
        df_spots_ID = self.df_spots_ID
        spots_img_denoise = filters.gaussian(
            self.spots_img_local, 0.8, use_gpu=self.use_gpu,
            logger_func=self.logger.info
        )
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
        backgr_median = np.median(backgr_vals)
        backgr_std = backgr_vals.std()

        limit = backgr_median + 3*backgr_std

        # Build prev_iter_expanded_lab mask for the expansion process
        self.spot_ids = df_spots_ID.index.to_list()
        zyx_seed_size = np.array(zyx_spot_min_vol_um)/2
        prev_iter_expanded_lab = self.get_spots_mask(
            0, zyx_vox_dim, zyx_seed_size, spots_centers, dtype=np.uint32,
            ids=self.spot_ids
        )
        spots_3D_lab = np.zeros_like(prev_iter_expanded_lab)

        # Start expanding the labels
        zs, ys, xs = zyx_seed_size
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
        drop_spots_ids = set()
        for i in range(max_i+1):
            # Note that expanded_labels has id from df_spots_ID
            # expanded_labels = transformations.expand_labels(
            #     prev_iter_expanded_lab, distance=yvd, 
            #     zyx_vox_size=zyx_vox_dim
            # )
            expanded_labels = self.expand_spots_labels(
                prev_iter_expanded_lab, zyx_vox_dim, zyx_seed_size, 
                spots_centers, grow_iter=i+1
            )

            # Replace expanded labels with the ones that stopped growing
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
                
                expanded_spot_mask = expanded_labels[s_obj.slice]==id
                prev_iter_spot_mask = prev_iter_expanded_lab[s_obj.slice]==id
                local_spot_surf_mask = np.logical_xor(
                    expanded_spot_mask, prev_iter_spot_mask
                )
                surf_vals = spots_img_denoise[s_obj.slice][local_spot_surf_mask]
                if len(surf_vals) == 0:
                    drop_spots_ids.add(id)
                    continue
                    
                surf_mean = surf_vals.mean()

                if surf_mean > limit and i < max_i:
                    continue

                stop_grow_info.append((id, s_obj.image, s_obj.slice))
                stop_grow_ids.append(id)
                self.spots_yx_size_um[o] = ys+yvd*i
                self.spots_z_size_um[o] = zs+yvd*i
                self.spots_yx_size_pxl[o] = (ys+yvd*i)/yvd
                self.spots_z_size_pxl[o] = (zs+yvd*i)/zvd
                # Insert grown spot into spots lab used for fitting
                c_idx = self.spot_ids.index(id)
                zyx_c = spots_centers[c_idx]
                spots_3D_lab = self.insert_grown_spot_id(
                    i, id, zyx_vox_dim, zyx_seed_size, zyx_c, 
                    spots_3D_lab
                )
                raw_spot_surf_vals = (
                    self.spots_img_local[s_obj.slice][local_spot_surf_mask]
                )
                self.Bs_guess[o] = np.median(raw_spot_surf_vals)
                _spot_surf_5ps[o] = np.quantile(raw_spot_surf_vals, 0.05)
                _mean = raw_spot_surf_vals.mean()
                _spot_surf_means[o] = _mean
                _std = raw_spot_surf_vals.std()
                _spot_surf_stds[o] = _std
                _spot_B_mins[o] = _mean-3*_std
            
            prev_iter_expanded_lab = expanded_labels
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
        
        self.df_spots_ID = self.df_spots_ID.drop(index=drop_spots_ids)
        
        # Used as a lower bound for B parameter in spotfit
        self.B_mins = _spot_B_mins

        self.spots_3D_lab_ID = spots_3D_lab

    def _fit(self):
        verbose = self.verbose
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
        iterable = zip(
            df_intersect.index,
            df_intersect['id'],
            df_intersect['intersecting_idx'],
            df_intersect['neigh_idx']
        )
        for count, (s, s_id, intersect_idx, neigh_idx) in enumerate(iterable):
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
                fitted_coeffs[i] for i in intersect_idx
                if not all_intersect_fitted_bool[i]
            ]
            if verbose > 2:
                print(
                    'Not-all-neighbours-fitted coeffs (model initial guess): '
                    f'{not_all_intersect_fitted_coeffs}'
                )

            # Fit n intersecting spots as sum of n gaussian + model constants
            fit_idx = intersect_idx
            if verbose > 2:
                print(
                    f'Fitting spot idx: {fit_idx}, with centers {spots_centers}'
                )

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
                num_spots_s, num_coeffs, fit_ids, fit_idx, spots_centers,
                spots_3D_lab_ID, spots_rp, spots_radii_pxl, spots_img,
                spots_Bs_guess, spots_B_mins
            )
            # bar_f = '{desc:<25}{percentage:3.0f}%|{bar:40}{r_bar}'
            model.pbar = tqdm(
                desc=f'Fitting spot {s} ({count+1}/{num_spots})',
                total=100*len(z), unit=' fev', position=5, leave=False, 
                ncols=100
            )
            leastsq_result = scipy.optimize.least_squares(
                model.residuals, init_guess_s,
                args=(s_data, z, y, x, num_spots_s, num_coeffs),
                # jac=model.jac_gauss3D,
                kwargs={'const': const},
                loss='linear', f_scale=0.1,
                bounds=bounds, ftol=self._tol,
                xtol=self._tol, gtol=self._tol
            )

            # model.pbar.update(100*len(z)-model.pbar.n)
            model.pbar.close()

            if self.debug:
                from . import _debug
                _debug._spotfit_fit(
                    model._gauss3D, spots_img, leastsq_result, num_spots_s,
                    num_coeffs, z, y, x, s_data, spots_centers, self.ID, 
                    fit_ids, init_guess_s, bounds, fit_idx
                )

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
            all_intersect_fitted = all(
                [True if fitted_coeffs[i] else False for i in intersect_idx]
            )
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
        # Get intersect ids by expanding each single object by 1 pixel
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
            expanded_spot_3D = transformations.expand_labels(
                spot_3D_lab, distance=yvd, zyx_vox_size=zyx_vox_dim
            )
            spot_surf_mask = np.logical_xor(expanded_spot_3D>0, spot_3D_mask)
            intersect_ids = np.unique(spots_3D_lab_ID[spot_surf_mask])
            intersect_idx = [
                self.spot_ids.index(id) for id in intersect_ids if id!=0
            ]
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

        self.df_intersect = pd.DataFrame({
            'id': self.spot_ids,
            'obj_id': obj_ids,
            'num_intersect': num_intersect,
            'num_neigh': num_neigh,
            'intersecting_idx': all_intersect_idx,
            'neigh_idx': all_neigh_idx,
            'neigh_ids': all_neigh_ids}
        ).sort_values('num_intersect')
        self.df_intersect.index.name = 's'

    def _quality_control(self):
        """
        Calculate goodness_of_fit metrics for each spot
        and determine which peaks should be fitted again
        """
        df_spotFIT = (
            self.df_intersect
            .reset_index()
            .set_index(['obj_id', 's'])
        )
        df_spotFIT['QC_passed_fit'] = 0
        df_spotFIT['null_ks_test_fit'] = 0
        df_spotFIT['null_chisq_test_fit'] = 0
        df_spotFIT['solution_found_fit'] = 0

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

                all_gof_metrics[s] = [
                    reduced_chisq, p_chisq, RMSE,
                    ks, p_ks, NRMSE, F_NRMSE
                ]

        # Automatic outliers detection
        NRMSEs = all_gof_metrics[:,5]
        Q1, Q3 = np.quantile(NRMSEs, q=(0.25, 0.75))
        IQR = Q3-Q1
        self.QC_limit = Q3 + (1.5*IQR)

        if self.debug:
            from ._debug import _spotfit_quality_control
            _spotfit_quality_control(self.QC_limit, all_gof_metrics)

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
                num_spots_s, num_coeffs, [s_id], [s], spots_centers,
                spots_3D_lab_ID, spots_rp, spots_radii_pxl, img,
                spots_Bs_guess, spots_B_mins
            )

            # Fit with constants
            s_data = img[z_s, y_s, x_s]
            desc = f'Fitting spot {s} ({count+1}/{num_spots})'
            model.pbar = tqdm(
                desc=desc, total=100*len(z_s), unit=' fev',
                position=4, leave=False, ncols=100
            )
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

    def store_metrics_good_spots(
            self, obj_id, s, fitted_coeffs_s, I_tot, I_foregr, gof_metrics,
            solution_found, B_fit
        ):

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
        self._df_spotFIT.at[(obj_id, s), 'sigma_yx_mean_fit'] = (
            (abs(sy_fit)+abs(sx_fit))/2
        )

        _vol = 4/3*np.pi*abs(sz_fit)*abs(sy_fit)*abs(sx_fit)
        self._df_spotFIT.at[(obj_id, s), 'spheroid_vol_vox_fit'] = _vol

        self._df_spotFIT.at[(obj_id, s), 'A_fit'] = A_fit
        self._df_spotFIT.at[(obj_id, s), 'B_fit'] = B_fit

        self._df_spotFIT.at[(obj_id, s), 'total_integral_fit'] = I_tot
        self._df_spotFIT.at[(obj_id, s), 'foreground_integral_fit'] = I_foregr

        (reduced_chisq, p_chisq,
        ks, p_ks, RMSE, NRMSE, F_NRMSE) = gof_metrics

        self._df_spotFIT.at[(obj_id, s), 'reduced_chisq_fit'] = reduced_chisq
        self._df_spotFIT.at[(obj_id, s), 'p_chisq_fit'] = p_chisq

        self._df_spotFIT.at[(obj_id, s), 'KS_stat_fit'] = ks
        self._df_spotFIT.at[(obj_id, s), 'p_KS_fit'] = p_ks

        self._df_spotFIT.at[(obj_id, s), 'RMSE_fit'] = RMSE
        self._df_spotFIT.at[(obj_id, s), 'NRMSE_fit'] = NRMSE
        self._df_spotFIT.at[(obj_id, s), 'F_NRMSE_fit'] = F_NRMSE

        QC_passed = int(NRMSE<self.QC_limit)
        self._df_spotFIT.at[(obj_id, s), 'QC_passed_fit'] = QC_passed

        self._df_spotFIT.at[(obj_id, s), 'null_ks_test_fit'] = int(p_ks > 0.05)
        self._df_spotFIT.at[(obj_id, s), 'null_chisq_test_fit'] = (
            int(p_chisq > 0.05))

        self._df_spotFIT.at[(obj_id, s), 'solution_found_fit'] = (
            int(solution_found))

class Kernel(_ParamsParser):
    def __init__(self, debug=False, is_cli=True):
        self.logger, self.log_path, self.logs_path = utils.setup_cli_logger()
        super().__init__(debug=debug, is_cli=is_cli, log=self.logger.info)
        self.debug = debug
        self.is_batch_mode = False
        self.is_cli = is_cli
        self._force_close_on_critical = False
        self._SpotFit = _spotFIT(debug=debug)
        self._current_frame_i = -1
        self._current_step = 'Kernel initialization'
        self._current_pos_path = 'Not determined yet'
    
    def _preprocess(self, image_data, is_ref_ch=False, verbose=True):
        SECTION = 'Pre-processing'
        ANCHOR = 'removeHotPixels'
        options = self._params[SECTION][ANCHOR]
        do_remove_hot_pixels = options.get('loadedVal')
        if verbose and do_remove_hot_pixels:
            print('')
            self.logger.info(f'Removing hot pixels...')
        if do_remove_hot_pixels:
            image_data = filters.remove_hot_pixels(image_data, progress=False)
        
        options = self._params[SECTION].get('gaussSigma')
        if is_ref_ch:
            ref_ch_section_params = self._params['Reference channel']
            options = ref_ch_section_params.get('refChGaussSigma', options)
        
        sigma = options.get('loadedVal')
        if sigma is None:
            sigma = options.get('initialVal')
        
        if sigma == 0:
            return image_data

        if verbose:
            print('')
            self.logger.info(f'Applying gaussian filter with sigma = {sigma}...')

        # if self.debug:
        #     return np.load(
        #         r'G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git'
        #         r'\spotMAX_v2\data\test_simone_pos\2_test_missed_spots_edges_worm'
        #         r'\Position_2\Images\20909_SampleD_Gonad1_fused_gauss_filtered.npy'
        #     )

        use_gpu = self._get_use_gpu()
        filtered_data = filters.gaussian(
            image_data, sigma, use_gpu=use_gpu, logger_func=self.logger.info
        )

        return filtered_data

    def _get_use_gpu(self):
        SECTION = 'Configuration'
        ANCHOR = 'useGpu'
        options = self._params[SECTION][ANCHOR]
        use_gpu = options.get('loadedVal')
        if use_gpu == None:
            use_gpu = False
        return use_gpu 
    
    def sharpen_spots(self, raw_spots_img, metadata):
        """Difference of Gaussians (DoG) detector. The same as TrackMate DoG 
        detector. Source: https://imagej.net/plugins/trackmate/detectors/difference-of-gaussian

        Parameters
        ----------
        raw_spots_img : (Z, Y, X) ndarray
            Raw spots' signal 3D z-stack image.
        metadata : dict
            Dictionary with 'zyxResolutionLimitPxl' key.

        Returns
        -------
        (Z, Y, X) ndarray
            Filtered image.
        """   
             
        print('')
        self.logger.info(f'Applying sharpening filter...')

        # if self.debug:
        #     return np.load(
        #         r'G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git'
        #         r'\spotMAX_v2\data\test_simone_pos\2_test_missed_spots_edges_worm'
        #         r'\Position_2\Images\20909_SampleD_Gonad1_fused_sharpened.npy'
        #     )
        use_gpu = self._get_use_gpu()
        
        resolution_limit_radii = metadata['zyxResolutionLimitPxl']
        
        filtered = filters.DoG_spots(
            raw_spots_img, resolution_limit_radii, use_gpu=use_gpu, 
            logger_func=self.logger.info
        )
        return filtered
    
    def _extract_img_from_segm_obj(self, image, lab, obj, lineage_table):
        slicer = transformations.SliceImageFromSegmObject(lab, lineage_table)
        return slicer.slice(image, obj)
    
    def _add_aggregated_ref_ch_features(
            self, df_agg, frame_i, ID, ref_ch_mask_local, vox_to_um3=None
        ):
        vol_voxels = np.count_nonzero(ref_ch_mask_local)
        df_agg.at[(frame_i, ID), 'ref_ch_vol_vox'] = vol_voxels
        if vox_to_um3 is not None:
            df_agg.at[(frame_i, ID), 'ref_ch_vol_fl'] = vol_voxels*vox_to_um3

        ref_ch_lab = skimage.measure.label(ref_ch_mask_local)
        rp = skimage.measure.regionprops(ref_ch_lab)
        num_fragments = len(rp)
        df_agg.at[(frame_i, ID), 'ref_ch_num_fragments'] = num_fragments
        
        return df_agg
    
    def add_ref_ch_features(
            self, 
            df_agg,
            lab_rp,
            ref_ch_segm,
            vox_to_um3=None,
            frame_i=0,
            verbose=True
        ):
        if verbose and frame_i == 0:
            print('')
            self.logger.info('Quantifying reference channel...')
        desc = 'Quantifying reference channel'
        pbar = tqdm(
            total=len(lab_rp), ncols=100, desc=desc, position=3, 
            leave=False
        )
        for obj in lab_rp:
            ref_ch_lab_local = ref_ch_segm[obj.slice].copy()
            ref_ch_lab_local[ref_ch_lab_local!=obj.label] = 0
            ref_ch_mask_local = ref_ch_lab_local > 0

            # Add numerical features
            df_agg = self._add_aggregated_ref_ch_features(
                df_agg, frame_i, obj.label, ref_ch_mask_local, 
                vox_to_um3=vox_to_um3
            )
            pbar.update()
        pbar.close()
        return df_agg
    
    def ref_ch_to_physical_units(self, df_agg, metadata):
        vox_to_um3_factor = metadata['vox_to_um3_factor']
        df_agg['ref_ch_vol_um3'] = df_agg['ref_ch_vol_vox']*vox_to_um3_factor
        return df_agg

    def _is_lab_all_zeros(self, lab):
        if lab is None:
            return False
        return not np.any(lab)

    @exception_handler_cli
    def segment_quantify_ref_ch(
            self, ref_ch_img, threshold_method='threshold_otsu', lab_rp=None, 
            lab=None, lineage_table=None, keep_only_largest_obj=False, 
            do_aggregate=False, df_agg=None, frame_i=0, 
            vox_to_um3=None, zyx_tolerance=None, ridge_filter_sigmas=0.0, 
            verbose=True
        ):
        if self._is_lab_all_zeros(lab):
            df_agg['ref_ch_vol_vox'] = np.nan
            df_agg['ref_ch_num_fragments'] = np.nan
            ref_ch_segm = np.zeros(lab.shape, dtype=bool)
            return ref_ch_segm, df_agg

        if lab is None:
            lab = np.ones(ref_ch_img.shape, dtype=np.uint8)
            lineage_table = None
        
        if lab_rp is None:
            lab_rp = skimage.measure.regionprops(lab)
        
        if df_agg is None:
            IDs = [obj.label for obj in lab_rp]
            df_data = {'frame_i': [frame_i]*len(IDs), 'Cell_ID': IDs}
            df_agg = pd.DataFrame(df_data).set_index(['frame_i', 'Cell_ID'])
        
        if isinstance(threshold_method, str):
            threshold_func = getattr(skimage.filters, threshold_method)
        else:
            threshold_func = threshold_method
        
        ref_ch_segm = pipe.reference_channel_semantic_segm(
            ref_ch_img, 
            lab=lab,
            keep_only_largest_obj=keep_only_largest_obj,
            lineage_table=lineage_table,
            do_aggregate=do_aggregate,
            logger_func=self.logger.info,
            thresholding_method=threshold_func,
            ridge_filter_sigmas=ridge_filter_sigmas,
            keep_input_shape=True,
            do_preprocess=False,
            return_only_segm=True,
            do_try_all_thresholds=False,
        )
        
        df_agg = self.add_ref_ch_features(
            df_agg, lab_rp, ref_ch_segm, 
            vox_to_um3=vox_to_um3,
            frame_i=frame_i,
            verbose=verbose
        )
        
        return ref_ch_segm, df_agg
    
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

    @exception_handler_cli
    def spots_detection(
            self, 
            spots_img, 
            zyx_resolution_limit_pxl, 
            sharp_spots_img=None,
            raw_spots_img=None, 
            transf_spots_nnet_img=None,
            ref_ch_img=None, 
            ref_ch_mask_or_labels=None, 
            frame_i=0, 
            lab=None, 
            rp=None, 
            backgr_is_outside_ref_ch_mask=False, 
            df_agg=None, 
            do_keep_spots_in_ref_ch=False, 
            gop_filtering_thresholds=None, 
            dist_transform_spheroid=None,
            detection_method='peak_local_max',
            spots_ch_segm_mask=None, 
            prediction_method='Thresholding',
            threshold_method='threshold_otsu', 
            do_aggregate=False,
            lineage_table=None, 
            min_size_spheroid_mask=None,
            spot_footprint=None, 
            dfs_lists=None, 
            save_spots_mask=True,
            verbose=True,
        ):
        if sharp_spots_img is None:
            sharp_spots_img = spots_img

        if self._is_lab_all_zeros(lab):
            return

        if lab is None:
            lab = np.ones(spots_img.shape, dtype=np.uint8)
        
        if rp is None:
            rp = skimage.measure.regionprops(lab)
        
        if df_agg is None:
            IDs = [obj.label for obj in rp]
            df_data = {'frame_i': [frame_i]*len(IDs), 'Cell_ID': IDs}
            df_agg = pd.DataFrame(df_data).set_index(['frame_i', 'Cell_ID'])
        
        if spot_footprint is None:
            zyx_radii_pxl = [val/2 for val in zyx_resolution_limit_pxl]
            spot_footprint = transformations.get_local_spheroid_mask(
                zyx_radii_pxl
            )
        
        if gop_filtering_thresholds is None:
            gop_filtering_thresholds = {}
        
        if self.nnet_model is not None and transf_spots_nnet_img is None:
            # Use raw image for neural network if no data was explicity passed
            transf_spots_nnet_img = raw_spots_img
        
        df_spots_coords = self._spots_detection(
            sharp_spots_img, lab, detection_method,
            threshold_method, do_aggregate, spot_footprint,
            transf_spots_nnet_img=transf_spots_nnet_img,
            spots_ch_segm_mask=spots_ch_segm_mask, 
            lineage_table=lineage_table, 
            verbose=verbose, 
            save_spots_mask=save_spots_mask
        )
        
        df_spots_det, df_spots_gop = self._spots_filter(
            df_spots_coords, spots_img, 
            sharp_spots_img, 
            ref_ch_img, 
            ref_ch_mask_or_labels, 
            backgr_is_outside_ref_ch_mask, 
            lab, rp, frame_i, 
            zyx_resolution_limit_pxl,
            min_size_spheroid_mask=min_size_spheroid_mask,
            dfs_lists=dfs_lists,
            raw_spots_img=raw_spots_img,
            do_keep_spots_in_ref_ch=do_keep_spots_in_ref_ch, 
            gop_filtering_thresholds=gop_filtering_thresholds,
            lineage_table=lineage_table, 
            dist_transform_spheroid=dist_transform_spheroid,
            verbose=verbose,
        )
        if df_spots_det is not None:
            dfs_segm_obj = self._add_aggregated_spots_features(
                df_spots_det, df_spots_gop, df_agg
            )
            return df_spots_det, df_spots_gop, *dfs_segm_obj

    def _add_local_coords_from_aggr(
            self, aggr_spots_coords, aggregated_lab, spots_objs=None
        ):
        aggr_lab_rp = skimage.measure.regionprops(aggregated_lab)
        if len(aggr_spots_coords) == 0:
            zz, yy, xx = [], [], []
        elif aggr_spots_coords.shape[1] == 2:
            # Add z=0 for spots detected in 2D images
            yy, xx = aggr_spots_coords.T
            zz = [0]*len(xx)
        else:
            zz, yy, xx = aggr_spots_coords.T
        zeros = [0]*len(zz)
        df_spots_coords = pd.DataFrame({
            'z_aggr': zz, 'y_aggr': yy, 'x_aggr': xx, 
            'z_local': zeros, 'y_local': zeros, 'x_local': zeros,
            'Cell_ID': aggregated_lab[zz, yy, xx]
        })
        if spots_objs is not None:
            df_spots_coords['spot_obj'] = spots_objs
        
        df_spots_coords = df_spots_coords.set_index('Cell_ID').sort_index()
        
        num_spots_objs_txts = []
        pbar = tqdm(
            total=len(aggr_lab_rp), ncols=100, position=3, leave=False
        )
        for obj in aggr_lab_rp:
            if obj.label not in df_spots_coords.index:
                continue
            min_z, min_y, min_x = obj.bbox[:3]
            zz_local = df_spots_coords.loc[[obj.label], 'z_aggr'] - min_z
            df_spots_coords.loc[[obj.label], 'z_local'] = zz_local

            yy_local = df_spots_coords.loc[[obj.label], 'y_aggr'] - min_y
            df_spots_coords.loc[[obj.label], 'y_local'] = yy_local

            xx_local = df_spots_coords.loc[[obj.label], 'x_aggr'] - min_x
            df_spots_coords.loc[[obj.label], 'x_local'] = xx_local

            s = f'  * Object ID {obj.label} = {len(zz_local)}'
            num_spots_objs_txts.append(s)
            pbar.update()
        pbar.close()

        return df_spots_coords, num_spots_objs_txts
    
    def _spots_detection(
            self, sharp_spots_img, lab, 
            detection_method, 
            threshold_method, 
            do_aggregate, footprint, 
            transf_spots_nnet_img=None,
            spots_ch_segm_mask=None,
            save_spots_mask=True,
            lineage_table=None, 
            verbose=True
        ):
        if verbose:
            print('')
            self.logger.info('Detecting spots...')

        spots_objs = None
        
        # Detect peaks on aggregated image
        aggregated = transformations.aggregate_objs(
            sharp_spots_img, lab, lineage_table=lineage_table, 
            zyx_tolerance=self.metadata['deltaTolerance'],
            additional_imgs_to_aggr=[spots_ch_segm_mask, transf_spots_nnet_img],
            debug=self.debug
        )
        aggr_spots_img, aggregated_lab, aggr_imgs = aggregated
        aggr_spots_ch_segm_mask = aggr_imgs[0]
        aggr_transf_spots_nnet_img = aggr_imgs[1]
        
        if aggr_spots_ch_segm_mask is not None:
            labels = aggr_spots_ch_segm_mask.astype(np.uint8)
        else:
            labels = pipe.spots_semantic_segmentation(
                aggr_spots_img, 
                lab=aggregated_lab, 
                spots_zyx_radii=self.metadata['deltaTolerance'],
                lineage_table=lineage_table,
                do_aggregate=do_aggregate,
                logger_func=self.logger.info,
                thresholding_method=threshold_method,
                nnet_model=self.nnet_model,
                nnet_params=self.nnet_params,
                nnet_input_data=aggr_transf_spots_nnet_img,
                do_preprocess=False,
                do_try_all_thresholds=False,
                keep_input_shape=True,
                return_only_segm=True,
                pre_aggregated=True
            )
        
        if detection_method == 'peak_local_max':
            aggr_spots_coords = skimage.feature.peak_local_max(
                np.squeeze(aggr_spots_img), 
                footprint=np.squeeze(footprint), 
                labels=np.squeeze(labels)
            )
        elif detection_method == 'label_prediction_mask':
            prediction_lab = skimage.measure.label(labels>0)
            prediction_lab_rp = skimage.measure.regionprops(prediction_lab)
            num_spots = len(prediction_lab_rp)
            aggr_spots_coords = np.zeros((num_spots, 3), dtype=int)
            if save_spots_mask:
                spots_objs = []
            for s, spot_obj in enumerate(prediction_lab_rp):
                aggr_zyx_coords = tuple([round(c) for c in spot_obj.centroid])
                aggr_spots_coords[s] = aggr_zyx_coords
                if not save_spots_mask:
                    continue
                zmin, ymin, xmin, _, _, _ = spot_obj.bbox
                spot_obj.zyx_local_center = (
                    aggr_zyx_coords[0] - zmin,
                    aggr_zyx_coords[1] - ymin,
                    aggr_zyx_coords[2] - xmin
                )
                spots_objs.append(spot_obj)

        df_spots_coords, num_spots_objs_txts = self._add_local_coords_from_aggr(
            aggr_spots_coords, aggregated_lab, spots_objs=spots_objs
        )

        # if self.debug:
        #     from . import _debug
        #     ID = 36
        #     _debug._spots_detection(
        #         aggregated_lab, ID, labels, aggr_spots_img, df_spots_coords
        #     )

        if verbose:
            print('')
            print('*'*60)
            num_spots_objs_txt = '\n'.join(num_spots_objs_txts)
            self.logger.info(
                f'Number of spots per segmented object:\n{num_spots_objs_txt}'
            )
            print('-'*60)
        return df_spots_coords
        
    def _spots_filter(
            self, df_spots_coords, spots_img, sharp_spots_img, 
            ref_ch_img, ref_ch_mask_or_labels,  backgr_is_outside_ref_ch_mask, 
            lab, rp, frame_i, zyx_resolution_limit_pxl, 
            dfs_lists=None,
            threshold_val=None, 
            min_size_spheroid_mask=None,
            raw_spots_img=None, 
            gop_filtering_thresholds=None, 
            do_keep_spots_in_ref_ch=False, 
            prediction_mask=None,
            lineage_table=None, 
            dist_transform_spheroid=None,
            verbose=True,
        ):
        if verbose:
            print('')
            self.logger.info('Filtering valid spots...')
        
        if dfs_lists is None:
            dfs_spots_det = []
            dfs_spots_gop = []
            keys = []
        else:
            dfs_spots_det = dfs_lists['dfs_spots_detection']
            dfs_spots_gop = dfs_lists['dfs_spots_gop_test']
            keys = dfs_lists['keys']

        # We slice the object with some added tolerance
        Z, Y, X = lab.shape
        delta_tol = self.metadata['deltaTolerance']

        num_spots_filtered_log = []
        desc = 'Filtering spots'
        pbar = tqdm(
            total=len(rp), ncols=100, desc=desc, position=3, leave=False
        )
        last_spot_id = 0
        for obj_idx, obj in enumerate(rp):
            expanded_obj = transformations.get_expanded_obj_slice_image(
                obj, delta_tol, lab
            )
            obj_slice, obj_image, crop_obj_start = expanded_obj

            local_spots_img = spots_img[obj_slice]
            local_sharp_spots_img = sharp_spots_img[obj_slice]

            result = transformations.init_df_features(
                df_spots_coords, obj, crop_obj_start, 
                self.metadata['zyxResolutionLimitPxl']
            )
            df_obj_spots_det, expanded_obj_coords = result
            if df_obj_spots_det is None:
                # 0 spots for this obj (object ID not present in index)
                s = f'  * Object ID {obj.label} = 0 --> 0 (0 iterations)'
                num_spots_filtered_log.append(s)
                continue
            
            # Increment spot_id with previous object
            df_obj_spots_det['spot_id'] += last_spot_id
            df_obj_spots_det = df_obj_spots_det.set_index('spot_id').sort_index()
            
            keys.append((frame_i, obj.label))
            num_spots_detected = len(df_obj_spots_det)
            last_spot_id += num_spots_detected
            
            dfs_spots_det.append(df_obj_spots_det)

            if ref_ch_mask_or_labels is not None:
                local_ref_ch_mask = ref_ch_mask_or_labels[obj_slice]>0
                local_ref_ch_mask = np.logical_and(local_ref_ch_mask, obj_image)
            else:
                local_ref_ch_mask = None

            if ref_ch_img is not None:
                local_ref_ch_img = ref_ch_img[obj_slice]
            else:
                local_ref_ch_img = None
            
            if raw_spots_img is not None:
                raw_spots_img_obj = raw_spots_img[obj_slice]

            df_obj_spots_gop = df_obj_spots_det.copy()
            if do_keep_spots_in_ref_ch:
                df_obj_spots_gop = self._drop_spots_not_in_ref_ch(
                    df_obj_spots_gop, local_ref_ch_mask, expanded_obj_coords
                )
            
            debug = False
            i = 0
            while True:     
                num_spots_prev = len(df_obj_spots_gop)
                if num_spots_prev == 0:
                    num_spots_filtered = 0
                    break
                
                # if obj.label == 36:
                #     debug = True
                #     import pdb; pdb.set_trace()
                
                df_obj_spots_gop = self._compute_obj_spots_metrics(
                    local_spots_img, 
                    df_obj_spots_gop, 
                    obj_image, 
                    local_sharp_spots_img, 
                    raw_spots_img_obj=raw_spots_img_obj,
                    min_size_spheroid_mask=min_size_spheroid_mask, 
                    dist_transform_spheroid=dist_transform_spheroid,
                    ref_ch_mask_obj=local_ref_ch_mask, 
                    ref_ch_img_obj=local_ref_ch_img,
                    backgr_is_outside_ref_ch_mask=backgr_is_outside_ref_ch_mask,
                    zyx_resolution_limit_pxl=zyx_resolution_limit_pxl,
                    debug=debug
                )
                if i == 0:
                    # Store metrics at first iteration
                    df_obj_spots_det = df_obj_spots_gop.copy()
                
                # if self.debug and obj.label == 79:
                # from . import _debug
                # _debug._spots_filtering(
                #     local_spots_img, df_obj_spots_gop, obj, obj_image
                # )
                
                df_obj_spots_gop = self.filter_spots(
                    df_obj_spots_gop, gop_filtering_thresholds,
                    is_spotfit=False, debug=False
                )
                num_spots_filtered = len(df_obj_spots_gop)   

                if num_spots_filtered == num_spots_prev or num_spots_filtered == 0:
                    # Number of filtered spots stopped decreasing --> stop loop
                    break

                i += 1

            nsd, nsf = num_spots_detected, num_spots_filtered
            s = f'  * Object ID {obj.label} = {nsd} --> {nsf} ({i} iterations)'
            num_spots_filtered_log.append(s)

            dfs_spots_gop.append(df_obj_spots_gop)

            pbar.update()
        pbar.close()

        if verbose:
            print('')
            print('*'*60)
            info = '\n'.join(num_spots_filtered_log)
            self.logger.info(
                f'Number of spots after filtering valid spots:\n{info}'
            )
            print('-'*60)
        
        if dfs_lists is None:
            names = ['frame_i', 'Cell_ID', 'spot_id']
            df_spots_det = pd.concat(dfs_spots_det, keys=keys, names=names)
            df_spots_gop = pd.concat(dfs_spots_gop, keys=keys, names=names)
            return df_spots_det, df_spots_gop
        else:
            return None, None
    
    def _translate_coords_segm_crop(self, *dfs, crop_to_global_coords=(0,0,0)):
        dfs_translated = []
        for i, df in enumerate(dfs):
            if df is None:
                dfs_translated.append(None)
                continue
            
            df = df.drop(columns=ZYX_LOCAL_EXPANDED_COLS)
            df[ZYX_GLOBAL_COLS] += crop_to_global_coords
            try:
                df[ZYX_FIT_COLS] += crop_to_global_coords
            except Exception as e:
                # Spotfit coordinates are not always present
                pass
            dfs_translated.append(df)
        return dfs_translated
    
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
        df_agg_det = df_agg_det.join(df_agg, how='outer')
        
        df_agg_gop = (
            df_spots_gop.reset_index().groupby(['frame_i', 'Cell_ID'])
            .agg(**func)
        )
        df_agg_gop = df_agg_gop.join(df_agg, how='outer')
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
            df_agg_spotfit = df_agg_spotfit.join(df_agg, how='outer')
        else:
            df_agg_spotfit = None

        df_agg_det = self._add_missing_cells_df_agg(df_agg, df_agg_det)
        df_agg_gop = self._add_missing_cells_df_agg(df_agg, df_agg_gop)
        if df_agg_spotfit is not None:
            df_agg_spotfit = self._add_missing_cells_df_agg(df_agg, df_agg_spotfit)

        return df_agg_det, df_agg_gop, df_agg_spotfit
    
    @exception_handler_cli
    def measure_spots_spotfit(
            self, spots_img, df_spots, zyx_voxel_size, zyx_spot_min_vol_um,
            rp=None, dfs_lists=None, lab=None, frame_i=0, 
            ref_ch_mask_or_labels=None, verbose=True
        ):
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
        
        delta_tol = self.metadata['deltaTolerance']
        desc = 'Measuring spots'
        pbar = tqdm(total=len(rp), ncols=100, desc=desc, position=3, leave=False)
        for obj in rp:
            if obj.label not in df_spots.index:
                continue
            expanded_obj = transformations.get_expanded_obj(obj, delta_tol, lab)
            self._SpotFit.set_args(
                expanded_obj, spots_img, df_spots, zyx_voxel_size, 
                zyx_spot_min_vol_um, ref_ch_mask_or_labels=ref_ch_mask_or_labels,
                use_gpu=self._get_use_gpu(), logger=self.logger
            )
            self._SpotFit.fit()
            dfs_spots_spotfit.append(self._SpotFit.df_spotFIT_ID)
            keys.append((frame_i, obj.label))
            pbar.update()
        pbar.close()
    
    def _get_spot_intensities(
            self, spots_img, zyx_center, local_spot_mask
        ):
        # Get the spot intensities
        slice_global_to_local, slice_crop_local = (
            transformations.get_slices_local_into_global_3D_arr(
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
        in_ref_ch_spots_mask = ref_ch_mask[zz, yy, xx] > 0
        return df[in_ref_ch_spots_mask]

    def _add_spot_vs_ref_location(self, ref_ch_mask, zyx_center, df, idx):
        is_spot_in_ref_ch = int(ref_ch_mask[zyx_center] > 0)
        df.at[idx, 'is_spot_inside_ref_ch'] = is_spot_in_ref_ch
        _, dist_2D_from_ref_ch = utils.nearest_nonzero(
            ref_ch_mask[zyx_center[0]], zyx_center[1], zyx_center[2]
        )
        df.at[idx, 'spot_dist_2D_from_ref_ch'] = dist_2D_from_ref_ch

    def _get_normalised_spot_ref_ch_intensities(
            self, normalised_spots_img_obj, normalised_ref_ch_img_obj,
            spheroid_mask, slice_global_to_local
        ):
        norm_spot_slice = (normalised_spots_img_obj[slice_global_to_local])
        norm_spot_slice_dt = norm_spot_slice
        norm_spot_intensities = norm_spot_slice_dt[spheroid_mask]

        norm_ref_ch_slice = (normalised_ref_ch_img_obj[slice_global_to_local])
        norm_ref_ch_slice_dt = norm_ref_ch_slice
        norm_ref_ch_intensities = norm_ref_ch_slice_dt[spheroid_mask]

        return norm_spot_intensities, norm_ref_ch_intensities

    # @acdc_myutils.exec_time
    def _compute_obj_spots_metrics(
            self, spots_img_obj, df_obj_spots, obj_mask, 
            sharp_spots_img_obj, raw_spots_img_obj=None, 
            min_size_spheroid_mask=None, dist_transform_spheroid=None,
            ref_ch_mask_obj=None, ref_ch_img_obj=None, 
            zyx_resolution_limit_pxl=None, 
            backgr_is_outside_ref_ch_mask=False,
            debug=False
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
            Boolean mask of the segmentation object.
        sharp_spots_img_obj : (Z, Y, X) ndarray
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
            It must have the same shape of `min_size_spheroid_mask`.
            If None, this will be initialized with 1s, meaning that normalisation 
            will not be performed.
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
        backgr_is_outside_ref_ch_mask : bool, optional by default False
            If True, the background mask are made of the pixels that are inside 
            the segmented object but outside of the reference channel mask.
        zyx_resolution_limit_pxl : (z, y, x) tuple or None, optional
            Resolution limit in (z, y, x) direction in pixels, by default None. 
            If `min_size_spheroid_mask` is None, this will be used to computed 
            the boolean mask of the smallest spot expected.
        """ 

        local_peaks_coords = df_obj_spots[ZYX_LOCAL_EXPANDED_COLS].to_numpy()
        result = transformations.get_spheroids_maks(
            local_peaks_coords, obj_mask.shape, 
            min_size_spheroid_mask=min_size_spheroid_mask, 
            zyx_radii_pxl=zyx_resolution_limit_pxl,
            debug=debug
        )
        spheroids_mask, min_size_spheroid_mask = result
        # if debug:
        #     from cellacdc.plot import imshow
        #     imshow(
        #         spheroids_mask, spots_img_obj, 
        #         points_coords=local_peaks_coords
        #     )
        #     import pdb; pdb.set_trace()

        # Check if spots_img needs to be normalised
        if backgr_is_outside_ref_ch_mask:
            backgr_mask = np.logical_and(ref_ch_mask_obj, ~spheroids_mask)
            normalised_ref_ch_img_obj, ref_ch_norm_value = self._normalise_img(
                ref_ch_img_obj, backgr_mask, raise_if_norm_zero=False
            )
            df_obj_spots.loc[:, 'ref_ch_normalising_value'] = ref_ch_norm_value
            normalised_spots_img_obj, spots_norm_value = self._normalise_img(
                spots_img_obj, backgr_mask, raise_if_norm_zero=True
            )
            df_obj_spots.loc[:, 'spots_normalising_value'] = spots_norm_value
        else:
            backgr_mask = np.logical_and(obj_mask, ~spheroids_mask)
            normalised_spots_img_obj = spots_img_obj
            normalised_ref_ch_img_obj = ref_ch_img_obj

        # Calculate background metrics
        backgr_vals = sharp_spots_img_obj[backgr_mask]
        for name, func in distribution_metrics_func.items():
            df_obj_spots.loc[:, f'background_{name}'] = func(backgr_vals)
        
        if raw_spots_img_obj is None:
            raw_spots_img_obj = spots_img_obj

        pbar_desc = 'Computing spots features'
        pbar = tqdm(
            total=len(df_obj_spots), ncols=100, desc=pbar_desc, position=3, 
            leave=False
        )
        spot_ids_to_drop = []
        for row in df_obj_spots.itertuples():
            spot_id = row.Index
            zyx_center = tuple(
                [getattr(row, col) for col in ZYX_LOCAL_EXPANDED_COLS]
            )

            slices = transformations.get_slices_local_into_global_3D_arr(
                zyx_center, spots_img_obj.shape, min_size_spheroid_mask.shape
            )
            slice_global_to_local, slice_crop_local = slices
            
            # Background values at spot z-slice
            backgr_mask_z_spot = backgr_mask[zyx_center[0]]
            sharp_spot_obj_z = sharp_spots_img_obj[zyx_center[0]]
            backgr_vals_z_spot = sharp_spot_obj_z[backgr_mask_z_spot]
            
            if len(backgr_vals_z_spot) == 0:
                # This is most likely because the reference channel mask at 
                # center z-slice is smaller than the spot resulting 
                # in no background mask (since the background is outside of 
                # the spot but inside the ref. ch. mask) --> there is not 
                # enough ref. channel to consider this a valid spot.
                spot_ids_to_drop.append(spot_id)
                continue
            
            if debug:
                print('')
                zyx_local = tuple(
                    [getattr(row, col) for col in ZYX_LOCAL_COLS]
                )
                zyx_global = tuple(
                    [getattr(row, col) for col in ZYX_GLOBAL_COLS]
                )
                print(f'Local coordinates = {zyx_local}')
                print(f'Global coordinates = {zyx_global}')
                print(f'Spot raw intensity at center = {raw_spots_img_obj[zyx_center]}')
                from ._debug import _compute_obj_spots_metrics
                win = _compute_obj_spots_metrics(
                    sharp_spot_obj_z, backgr_mask_z_spot, 
                    spheroids_mask[zyx_center[0]], 
                    zyx_center[1:], block=False
                )

            # Crop masks
            spheroid_mask = min_size_spheroid_mask[slice_crop_local]
            spot_slice = spots_img_obj[slice_global_to_local]

            # Get the sharp spot sliced
            sharp_spot_slice_z = sharp_spot_obj_z[slice_global_to_local[-2:]]
            
            if dist_transform_spheroid is None:
                # Do not optimise for high spot density
                sharp_spot_slice_z_transf = sharp_spot_slice_z
            else:
                dist_transf = dist_transform_spheroid[slice_crop_local]
                sharp_spot_slice_z_transf = (
                    transformations.normalise_spot_by_dist_transf(
                        sharp_spot_slice_z, dist_transf.max(axis=0),
                        backgr_vals_z_spot, how='range'
                ))

            # Get spot intensities
            spot_intensities = spot_slice[spheroid_mask]
            spheroid_mask_proj = spheroid_mask.max(axis=0)
            sharp_spot_intensities_z_edt = (
                sharp_spot_slice_z_transf[spheroid_mask_proj]
            )

            value = spots_img_obj[zyx_center]
            df_obj_spots.at[spot_id, 'spot_preproc_intensity_at_center'] = value
            features.add_distribution_metrics(
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

                features.add_distribution_metrics(
                    raw_spot_intensities, df_obj_spots, spot_id, 
                    col_name='spot_raw_*name_in_spot_minimumsize_vol'
                )

            # When comparing to the background we use the sharpened image 
            # at the center z-slice of the spot
            features.add_ttest_values(
                sharp_spot_intensities_z_edt, backgr_vals_z_spot, 
                df_obj_spots, spot_id, name='spot_vs_backgr',
                logger_func=self.logger.info
            )
            
            features.add_effect_sizes(
                sharp_spot_intensities_z_edt, backgr_vals_z_spot, 
                df_obj_spots, spot_id, name='spot_vs_backgr',
                debug=debug
            )
            
            if ref_ch_img_obj is None:
                # Raw reference channel not present --> continue
                pbar.update()
                continue

            normalised_spot_intensities, normalised_ref_ch_intensities = (
                self._get_normalised_spot_ref_ch_intensities(
                    normalised_spots_img_obj, normalised_ref_ch_img_obj,
                    spheroid_mask, slice_global_to_local
                )
            )
            features.add_ttest_values(
                normalised_spot_intensities, normalised_ref_ch_intensities, 
                df_obj_spots, spot_id, name='spot_vs_ref_ch',
                logger_func=self.logger.info
            )
            features.add_effect_sizes(
                normalised_spot_intensities, normalised_ref_ch_intensities, 
                df_obj_spots, spot_id, name='spot_vs_ref_ch'
            )
            self._add_spot_vs_ref_location(
                ref_ch_mask_obj, zyx_center, df_obj_spots, spot_id
            )                
            
            value = ref_ch_img_obj[zyx_center]
            df_obj_spots.at[spot_id, 'ref_ch_raw_intensity_at_center'] = value

            ref_ch_intensities = (
                ref_ch_img_obj[slice_global_to_local][spheroid_mask]
            )
            features.add_distribution_metrics(
                ref_ch_intensities, df_obj_spots, spot_id, 
                col_name='ref_ch_raw_*name_in_spot_minimumsize_vol'
            )
            pbar.update()
        pbar.close()
        if spot_ids_to_drop:
            df_obj_spots = df_obj_spots.drop(index=spot_ids_to_drop)
        return df_obj_spots
    
    @exception_handler_cli
    def filter_spots(
            self, df: pd.DataFrame, features_thresholds: dict, is_spotfit=False,
            debug=False
        ):
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
            if not is_spotfit and feature_name.endswith('_fit'):
                # Ignore _fit features if not spotfit
                continue
            if is_spotfit and not feature_name.endswith('_fit'):
                # Ignore non _fit features if spotfit
                continue
            if feature_name not in df.columns:
                # Warn and ignore missing features
                self._warn_feature_is_missing(feature_name, df)
                continue
            _min, _max = thresholds
            if _min is not None:
                queries.append(f'({feature_name} > {_min})')
            if _max is not None:
                queries.append(f'({feature_name} < {_max})')

        if not queries:
            return df
        
        query = ' & '.join(queries)

        return df.query(query)

    def _warn_feature_is_missing(self, missing_feature, df):
        self.logger.info(f"\n{'='*60}")
        txt = (
            f'[WARNING]: The feature name "{missing_feature}" is not present '
            'in the table. It cannot be used for filtering spots at '
            f'this stage.{error_up_str}'
        )
        self.logger.info(txt)
    
    def _critical_feature_is_missing(self, missing_feature, df):
        format_colums = [f'    * {col}' for col in df.columns]
        format_colums = '\n'.join(format_colums)
        self.logger.info(f"\n{'='*60}")
        txt = (
            f'[ERROR]: The feature name "{missing_feature}" is not present in the table.\n\n'
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
        df_agg[ZYX_RESOL_COLS] = metadata['zyxResolutionLimitPxl']
        for obj in rp:
            idx = (frame_i, obj.label)
            cell_area_pxl = obj.area
            cell_area_um2 = cell_area_pxl*pxl_to_um2
            cell_vol_vox, cell_vol_fl = cellacdc.measure.rotational_volume(
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

    def _add_spotfit_features_to_df_spots_gop(self, df_spots_fit, df_spots_gop):
        idx = df_spots_fit.index
        for col in df_spots_fit.columns:
            if col in df_spots_gop.columns:
                continue
            
            df_spots_gop[col] = np.nan
            df_spots_gop.loc[idx, col] = df_spots_fit.loc[idx, col]
    
    def _filter_spots_by_size(
            self, df_spots_fit: pd.DataFrame, spotfit_minsize, spotfit_maxsize
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
        save_datetime = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
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
            f'Parameters file: "{self._report["params_path"]}"\n\n'
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
        self.logger.info('#'*60)
        self.logger.info(
            f'Final report saved to "{report_filepath}"'
        )
        self.logger.info('#'*60)

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
            print('')
            print('-'*60)
            self.logger.info(
                f'  - Error at frame index {self._current_frame_i}\n'
                f'  - Analysis step "{self._current_step}"\n'
                f'  - Folder path "{self._current_pos_path}"'
            )
            print('-'*60)
            self.quit(error)
        else:
            self.logger.exception(traceback_str)
            if not hasattr(self, '_report'):
                return
            if self._current_pos_path not in self._report['pos_info']:
                self._report['pos_info'][self._current_pos_path] = {
                    'errors': [], 'warnings': []
                }
            self._report['pos_info'][self._current_pos_path]['errors'].append(
                error
            )

    @handle_log_exception_cli
    def check_preprocess_data_nnet_across_exp(
            self, exp_path, pos_foldernames, spots_ch_endname
        ):
        if self.nnet_params is None:
            return {pos:None for pos in pos_foldernames}
        
        preprocess_across_experiment = (
            self.nnet_params['init']['preprocess_across_experiment']
        )
        if not preprocess_across_experiment:
            return {pos:None for pos in pos_foldernames}
        
        self.logger.info(f'Pre-processing "{spots_ch_endname}" channel data across experiment...')
        transformed_data = transformations.load_preprocess_nnet_data_across_exp(
            exp_path, pos_foldernames, spots_ch_endname, self.nnet_model, 
            callback_channel_not_found=self._critical_channel_not_found
        )
        
        return transformed_data
    
    @handle_log_exception_cli
    def check_preprocess_data_nnet_across_time(
            self, spots_data, 
            transformed_spots_ch_nnet=None
        ):
        if self.nnet_params is None:
            # Neural network not required
            return transformed_spots_ch_nnet
        
        SizeT = self.metadata['SizeT']
        if SizeT == 1:
            # Do not preprocess across timepoints for static data
            return transformed_spots_ch_nnet
        
        preprocess_across_timepoints = (
            self.nnet_params['init']['preprocess_across_timepoints']
        )
        if not preprocess_across_timepoints:
            # Processing across timepoints disabled by the user
            return transformed_spots_ch_nnet
        
        if transformed_spots_ch_nnet is None:
            input_data = spots_data
        else:
            input_data = transformed_spots_ch_nnet
        
        transformed_data = self.nnet_model.preprocess(input_data)
        return transformed_data
        
    
    @handle_log_exception_cli
    def _run_from_images_path(
            self, images_path, 
            spots_ch_endname: str='', 
            ref_ch_endname: str='', 
            segm_endname: str='', 
            spots_ch_segm_endname: str='', 
            ref_ch_segm_endname: str='', 
            lineage_table_endname: str='', 
            text_to_append: str='',
            transformed_spots_ch_nnet: dict=None,
        ):
        self._current_step = 'Loading data from images path'
        data = self.get_data_from_images_path(
            images_path, spots_ch_endname, ref_ch_endname, segm_endname, 
            spots_ch_segm_endname, ref_ch_segm_endname, lineage_table_endname,
            transformed_spots_ch_nnet=transformed_spots_ch_nnet
        )
        do_segment_ref_ch = (
            self._params['Reference channel']['segmRefCh']['loadedVal']
        )
        do_aggregate = (
            self._params['Pre-processing']['aggregate']['loadedVal']
        )
        ref_ch_data = data.get('ref_ch')
        segm_rp = data.get('segm_rp')
        segm_data = data.get('segm')
        df_agg = data.get('df_agg')
        ref_ch_segm_data = data.get('ref_ch_segm')
        acdc_df = data.get('lineage_table')
        
        zyx_resolution_limit_pxl = self.metadata['zyxResolutionLimitPxl']

        verbose = not self._params['Configuration']['reduceVerbosity']['loadedVal']

        stopFrameNum = self.metadata['stopFrameNum']
        if stopFrameNum > len(segm_data):
            stopFrameNum = len(segm_data)
            self.metadata['stopFrameNum'] = stopFrameNum

        desc = 'Adding segmentation objects features'
        self._current_step = desc
        pbar = tqdm(
            total=stopFrameNum, ncols=100, desc=desc, position=2, leave=False
        )
        for frame_i in range(stopFrameNum):
            self._current_frame_i = frame_i
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
        
        segment_ref_ch = (
            ref_ch_data is not None and do_segment_ref_ch
            and ref_ch_segm_data is None
        )
        if segment_ref_ch:
            print('')
            self.logger.info('Segmenting reference channel...')
            self._current_step = 'Segmenting reference channel'
            SECTION = 'Reference channel'
            ref_ch_threshold_method = (
                self._params[SECTION]['refChThresholdFunc']['loadedVal']
            )
            is_ref_ch_single_obj = (
                self._params[SECTION]['refChSingleObj']['loadedVal']
            )
            ridge_filter_sigmas = (
                self._params[SECTION]['refChRidgeFilterSigmas']['loadedVal']
            )
            save_ref_ch_segm = (
                self._params[SECTION]['saveRefChMask']['loadedVal']
            )
            vox_to_um3 = self.metadata.get('vox_to_um3_factor', 1)
            ref_ch_segm_data = np.zeros(ref_ch_data.shape, dtype=np.uint32)
            desc = 'Frames completed (segm. ref. ch.)'
            pbar = tqdm(
                total=stopFrameNum, ncols=100, desc=desc, position=2, 
                leave=stopFrameNum>1
            )
            for frame_i in range(stopFrameNum):
                if acdc_df is not None:
                    lineage_table = acdc_df.loc[frame_i]
                else:
                    lineage_table = None
                lab_rp = segm_rp[frame_i]
                ref_ch_img = ref_ch_data[frame_i]
                ref_ch_img = self._preprocess(
                    ref_ch_img, is_ref_ch=True, verbose=frame_i==0
                )
                lab = segm_data[frame_i]
                ref_ch_lab, df_agg = self.segment_quantify_ref_ch(
                    ref_ch_img, lab_rp=lab_rp, lab=lab, 
                    threshold_method=ref_ch_threshold_method, 
                    keep_only_largest_obj=is_ref_ch_single_obj,
                    df_agg=df_agg, 
                    frame_i=frame_i, 
                    do_aggregate=do_aggregate,
                    lineage_table=lineage_table, 
                    vox_to_um3=vox_to_um3,
                    zyx_tolerance=self.metadata['deltaTolerance'],
                    ridge_filter_sigmas=ridge_filter_sigmas,
                    verbose=verbose,                    
                )
                ref_ch_segm_data[frame_i] = ref_ch_lab
                pbar.update()
            pbar.close()

            if save_ref_ch_segm:
                basename = data.get('basename', '')
                io.save_ref_ch_mask(
                    ref_ch_segm_data, images_path, ref_ch_endname, basename,
                    text_to_append=text_to_append, pad_width=data['pad_width']
                )

            df_agg = self.ref_ch_to_physical_units(df_agg, self.metadata)

            data['df_agg'] = df_agg
            data['ref_ch_segm'] = ref_ch_segm_data
        
        if 'spots_ch' not in data:
            dfs = {'agg_detection': data['df_agg']}
            return dfs, data
        
        self._current_step = 'Detecting spots'
        spots_data = data.get('spots_ch')
        spots_ch_segm_data = data.get('spots_ch_segm')
        min_size_spheroid_mask = transformations.get_local_spheroid_mask(
            zyx_resolution_limit_pxl
        )
        optimise_with_edt = (
            self._params['Spots channel']['optimiseWithEdt']['loadedVal']
        ) 
        if optimise_with_edt:
            edt_spheroid = transformations.norm_distance_transform_edt(
                min_size_spheroid_mask
            )
        else:
            edt_spheroid = None
        
        # Get footprint passed to peak_local_max --> use half the radius
        # since spots can overlap by the radius according to resol limit
        spot_footprint = transformations.get_local_spheroid_mask(
            [val/2 for val in zyx_resolution_limit_pxl]
        )
        
        do_sharpen_spots = (
            self._params['Pre-processing']['sharpenSpots']['loadedVal']
        )
        SECTION = 'Reference channel'
        backgr_is_outside_ref_ch_mask = (
            self._params[SECTION]['bkgrMaskOutsideRef']['loadedVal']
        )
        do_keep_spots_in_ref_ch = (
            self._params[SECTION]['keepPeaksInsideRef']['loadedVal']
        )

        SECTION = 'Spots channel'
        gop_filtering_thresholds = (
            self._params[SECTION]['gopThresholds']['loadedVal']
        )
        if gop_filtering_thresholds is None:
            gop_filtering_thresholds = {}
        
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
        save_spots_mask = (
            self._params[SECTION]['saveSpotsMask']['loadedVal']
        )
        dfs_lists = {
            'dfs_spots_detection': [], 'dfs_spots_gop_test': [], 'keys': []
        }
        if do_spotfit:
            dfs_lists['dfs_spots_spotfit'] = []
            dfs_lists['spotfit_keys'] = []
        
        transformed_spots_ch_nnet = data.get('transformed_spots_ch')
        
        transformed_spots_ch_nnet = self.check_preprocess_data_nnet_across_time(
            spots_data, transformed_spots_ch_nnet=transformed_spots_ch_nnet
        )
        
        desc = 'Frames completed (spot detection)'
        pbar = tqdm(
            total=stopFrameNum, ncols=100, desc=desc, position=2, 
            leave=stopFrameNum>1
        )
        for frame_i in range(stopFrameNum):
            self._current_frame_i = frame_i
            raw_spots_img = spots_data[frame_i]
            if spots_ch_segm_data is not None:
                spots_ch_segm_mask = spots_ch_segm_data[frame_i] > 0
            else:
                spots_ch_segm_mask = None
            if transformed_spots_ch_nnet is not None:
                transf_spots_nnet_img = transformed_spots_ch_nnet[frame_i]
            else:
                transf_spots_nnet_img = None
            preproc_spots_img = self._preprocess(raw_spots_img)
            if do_sharpen_spots:
                sharp_spots_img = self.sharpen_spots(
                    raw_spots_img, self.metadata
                )
            else:
                sharp_spots_img = None
            lab = segm_data[frame_i]
            rp = segm_rp[frame_i]
            if ref_ch_data is not None:
                ref_ch_img = ref_ch_data[frame_i]
                filtered_ref_ch_img = self._preprocess(ref_ch_img)
            else:
                ref_ch_img = None
                filtered_ref_ch_img = None
            if ref_ch_segm_data is not None:
                ref_ch_mask_or_labels = ref_ch_segm_data[frame_i]
            else:
                ref_ch_mask_or_labels = None
            if acdc_df is not None:
                lineage_table = acdc_df.loc[frame_i]
            else:
                lineage_table = None
            
            self.spots_detection(
                preproc_spots_img, zyx_resolution_limit_pxl, 
                sharp_spots_img=sharp_spots_img,
                ref_ch_img=filtered_ref_ch_img, 
                frame_i=frame_i, lab=lab, rp=rp,
                ref_ch_mask_or_labels=ref_ch_mask_or_labels, 
                df_agg=df_agg,
                raw_spots_img=raw_spots_img, 
                transf_spots_nnet_img=transf_spots_nnet_img,
                dfs_lists=dfs_lists,
                min_size_spheroid_mask=min_size_spheroid_mask,
                dist_transform_spheroid=edt_spheroid,
                spot_footprint=spot_footprint,
                backgr_is_outside_ref_ch_mask=backgr_is_outside_ref_ch_mask,
                do_keep_spots_in_ref_ch=do_keep_spots_in_ref_ch,
                gop_filtering_thresholds=gop_filtering_thresholds,
                spots_ch_segm_mask=spots_ch_segm_mask,
                prediction_method=prediction_method,
                threshold_method=threshold_method,
                detection_method=detection_method,
                do_aggregate=do_aggregate,
                lineage_table=lineage_table,
                save_spots_mask=save_spots_mask,
                verbose=verbose
            )
            pbar.update()
        pbar.close()

        if not dfs_lists['dfs_spots_detection']:
            missing_cols = list(aggregate_spots_feature_func.keys())
            if not do_spotfit: 
                missing_cols = missing_cols[:2]
            df_agg_det = df_agg.copy()
            df_agg_det[missing_cols] = np.nan
            df_agg_gop = df_agg_det
            # Lab was all 0s
            if do_spotfit:
                df_agg_spotfit = df_agg_gop
            else:
                df_agg_spotfit = None
            dfs = {
                'spots_detection': None,
                'spots_gop': None,
                'spots_spotfit': None,
                'agg_detection': df_agg_det,
                'agg_gop': df_agg_gop,
                'agg_spotfit': df_agg_spotfit
            }
            return dfs, data

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
                    zyx_spot_min_vol_um, rp=rp, dfs_lists=dfs_lists, lab=lab,
                    ref_ch_mask_or_labels=ref_ch_mask_or_labels,
                    frame_i=frame_i, verbose=verbose
                )
                pbar.update()
            pbar.close()
            
            keys = dfs_lists['spotfit_keys']
            df_spots_fit = pd.concat(
                dfs_lists['dfs_spots_spotfit'], keys=keys, names=names
            )
            self._add_spotfit_features_to_df_spots_gop(
                df_spots_fit, df_spots_gop
            )
            df_spots_fit = self.filter_spots(
                df_spots_fit, gop_filtering_thresholds,
                is_spotfit=True, debug=False
            )
            # df_spots_fit = self._filter_spots_by_size(
            #     df_spots_fit, spotfit_minsize, spotfit_maxsize
            # )
        else:
            df_spots_fit = None
        
        dfs_translated = self._translate_coords_segm_crop(
            df_spots_det, df_spots_gop, df_spots_fit, 
            crop_to_global_coords=data['crop_to_global_coords']
        )
        df_spots_det, df_spots_gop, df_spots_fit = dfs_translated
        
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

        return dfs, data

    @exception_handler_cli
    def _run_exp_paths(self, exp_paths):
        """Run spotMAX analysis from a dictionary of Cell-ACDC style experiment 
        paths

        Parameters
        ----------
        exp_paths : dict
            Dictionary where the keys are the experiment paths containing the 
            Position folders with the following values: `run_number`, 
            `pos_foldernames`, `spotsEndName`, `refChEndName`, `segmEndName`, 
            `refChSegmEndName`, and `lineageTableEndName`.

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
            spots_ch_segm_endname = exp_info['spotChSegmEndName']
            ref_ch_segm_endname = exp_info['refChSegmEndName']
            lineage_table_endname = exp_info['lineageTableEndName']
            text_to_append = exp_info['textToAppend']
            df_spots_file_ext = exp_info['df_spots_file_ext']
            desc = 'Experiments completed'
            pbar_pos = tqdm(
                total=len(exp_paths), ncols=100, desc=desc, position=1
            ) 
            transformed_data_nnet = self.check_preprocess_data_nnet_across_exp(
                exp_path, pos_foldernames, spots_ch_endname
            )
            for pos in pos_foldernames:
                print('')
                pos_path = os.path.join(exp_path, pos)
                rel_path = os.path.join(
                    exp_parent_foldername, exp_foldername, pos
                )
                self.logger.info(f'Analysing "...{os.sep}{rel_path}"...')
                images_path = os.path.join(pos_path, 'Images')
                self._current_pos_path = pos_path
                dfs, data = self._run_from_images_path(
                    images_path, 
                    spots_ch_endname=spots_ch_endname, 
                    ref_ch_endname=ref_ch_endname, 
                    segm_endname=segm_endname,
                    spots_ch_segm_endname=spots_ch_segm_endname,
                    ref_ch_segm_endname=ref_ch_segm_endname, 
                    lineage_table_endname=lineage_table_endname,
                    text_to_append=text_to_append,
                    transformed_spots_ch_nnet=transformed_data_nnet[pos]
                )      
                if dfs is None:
                    # Error raised, logged while dfs is None
                    continue
                self.add_post_analysis_features(dfs)
                dfs = self.filter_requested_features(dfs)
                dfs = self.filter_requested_features(dfs, on_aggr=True)
                self.save_dfs_and_spots_masks(
                    pos_path, dfs, 
                    images_path=images_path,
                    basename=data.get('basename', ''),
                    spots_ch_endname=spots_ch_endname,
                    uncropped_shape=data.get('spots_ch.shape'),
                    run_number=run_number, 
                    text_to_append=text_to_append, 
                    df_spots_file_ext=df_spots_file_ext
                )
                pbar_pos.update()
            pbar_pos.close()
            pbar_exp.update()
        pbar_exp.close()
        self.logger.info('spotMAX analysis completed.')
    
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

    def add_post_analysis_features(self, dfs):
        zyx_voxel_size = self.metadata['zyxVoxelSize']
        for key, df in dfs.items():
            if df is None:
                continue
            if 'z' not in df.columns:
                continue
            features.add_consecutive_spots_distance(df, zyx_voxel_size)
            
            if 'z_fit' not in df.columns:
                continue
            features.add_consecutive_spots_distance(
                df, zyx_voxel_size, suffix='_fit'
            )
    
    def filter_requested_features(self, dfs, on_aggr=False):
        if on_aggr:
            df_key_startswith = 'agg_'
            section = 'Aggregated measurements to save'
        else:
            df_key_startswith = 'spots_'
            section = 'Single-spot measurements to save'
        
        if not self.configPars.has_section(section):
            return dfs
        
        columns_regexes = self.configPars.options(section)
        filtered_dfs = {}
        for key, df in dfs.items():
            if df is None:
                filtered_dfs[key] = None
                continue
            
            if not key.startswith(df_key_startswith):
                filtered_dfs[key] = df
                continue
            
            columns_to_filter = BASE_COLUMNS.copy()
            for regex in columns_regexes:
                pattern = rf'^{regex}'
                filtered_df = df.filter(regex=pattern)
                columns_to_filter.extend(filtered_df.columns)
            filtered_dfs[key] = df.filter(columns_to_filter)
        return filtered_dfs

    def _copy_ini_params_to_spotmax_out(
            self, spotmax_out_path, run_number, text_to_append
        ):
        analysis_inputs_filepath = os.path.join(
            spotmax_out_path, 
            f'{run_number}_analysis_parameters{text_to_append}.ini'
        )
        
        # Add the run number
        configPars = config.ConfigParser()
        configPars.read(self.ini_params_file_path, encoding="utf-8")
        SECTION = 'File paths and channels'
        if SECTION not in configPars.sections():
            configPars[SECTION] = {}
        ANCHOR = 'runNumber'
        option = self._params[SECTION][ANCHOR]['desc']
        configPars[SECTION][option] = str(run_number)
        
        with open(analysis_inputs_filepath, 'w', encoding="utf-8") as file:
            configPars.write(file)
    
    def save_dfs_and_spots_masks(
            self, 
            folder_path, dfs, 
            images_path='',
            basename='',
            spots_ch_endname='',
            uncropped_shape=None,
            run_number=1, 
            text_to_append='', 
            df_spots_file_ext='.h5'
        ):
        if not df_spots_file_ext.startswith('.'):
            df_spots_file_ext = f'.{df_spots_file_ext}'
        
        spotmax_out_path = os.path.join(folder_path, 'spotMAX_output')
        if not os.path.exists(spotmax_out_path):
            os.mkdir(spotmax_out_path)
        
        if text_to_append and not text_to_append.startswith('_'):
            text_to_append = f'_{text_to_append}'
        
        # Remove existing run numbers (they might have a different text appended)
        for file in utils.listdir(spotmax_out_path):
            file_path = os.path.join(spotmax_out_path, file)
            if not os.path.isfile(file_path):
                continue
            if not file.startswith(f'{run_number}_'):
                continue
            if file.find('analysis_parameters') != -1 or file.find('spot') != -1:
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.logger.info(
                        f'[WARNING]: File "{file_path}" could not be deleted.'
                    )

        self._copy_ini_params_to_spotmax_out(
            spotmax_out_path, run_number, text_to_append
        )
        
        for key, filename in DFs_FILENAMES.items():
            df_spots_filename = filename.replace('*rn*', str(run_number))
            df_spots_filename = df_spots_filename.replace(
                '*desc*', text_to_append
            )
            df_spots = dfs.get(key, None)

            if df_spots is not None:
                if 'spot_obj' in df_spots.columns:
                    df_spots = io.save_spots_masks(
                        df_spots, images_path, basename, filename, 
                        spots_ch_endname, run_number, 
                        text_to_append=text_to_append,
                        mask_shape=uncropped_shape
                    )
                    
                io.save_df_spots(
                    df_spots, spotmax_out_path, df_spots_filename,
                    extension=df_spots_file_ext
                )
            
            agg_filename = f'{df_spots_filename}_aggregated.csv'
            agg_key = key.replace('spots', 'agg')
            df_agg = dfs.get(agg_key, None)

            if df_agg is not None:
                df_agg.to_csv(os.path.join(spotmax_out_path, agg_filename))

    @exception_handler_cli
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
        
        io.save_last_used_ini_filepath(params_path)
        
        proceed, missing_params = self.init_params(
            params_path, metadata_csv_path=metadata_csv_path
        )
        if not proceed:
            self.quit()
            return

        if parser_args is not None:
            params_path = self.ini_params_file_path
            self.add_parser_args_to_params_ini_file(parser_args, params_path)
        self._save_missing_params_to_ini(
            missing_params, self.ini_params_file_path
        )
        
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

        self.logger.info('='*60)
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
                'spotMAX command-line interface closed. '
                f'{utils.get_salute_string()}'
            )
        self.logger.info('='*60)
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
    # NOTE: The Abbe diffraction limit formula often uses the letter `d` because
    # it is the "diameter of the aperture in meters" or simply the "minumum 
    # resolvable distance". This should not be confused with the diameter of 
    # the spot. The diameter of the spot can be determined using the Rayleigh 
    # criterion, which says "two point sources are regarded as just resolved 
    # when the principal diffraction maximum (center) of the Airy disk of one 
    # image coincides with the first minimum of the Airy disk of the other". 
    # In other words, `d` from the Abbe formula is the distance from the center
    # of the Airy disk to the first minimum, hence the RADIUS of the spot.
    # However, when computing the peaks location with `peak_local_max` we 
    # provide a spot footprint with diameter = `d` since we cannot allow 
    # overlapping between spots in contrast to the Rayleigh criterion.
    
    # Sources:
    # - https://en.wikipedia.org/wiki/Angular_resolution
    # - https://en.wikipedia.org/wiki/Airy_disk
    # - https://en.wikipedia.org/wiki/Diffraction-limited_system
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
        dataToCont2D = dataToCont.max