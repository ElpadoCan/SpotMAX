import os
import sys
import re
import difflib
import pathlib
import time
import copy
import logging

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pprint import pprint

from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
from natsort import natsorted

import skimage
import skimage.io
import skimage.color

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMessageBox

from . import dialogs, utils, core, html_func

class channelName:
    def __init__(self, which_channel=None, QtParent=None, load=True):
        self.parent = QtParent
        self.is_first_call = True
        self.which_channel = which_channel
        if load:
            self.last_sel_channel = self._load_last_selection()
        else:
            self.last_sel_channel = None
        self.was_aborted = False

    def reloadLastSelectedChannel(self, which):
        self.which_channel = which
        self.last_sel_channel = self._load_last_selection()

    def checkDataIntegrity(self, filenames):
        char = filenames[0][:2]
        startWithSameChar = all([f.startswith(char) for f in filenames])
        if not startWithSameChar:
            txt = html_func.paragraph("""
                The system detected files inside the folder
                that <b>do not start with the same, common basename</b>
                (see which filenames in the box below).<br><br>
                To ensure correct loading of the data, the folder where
                the file(s) is/are should either contain a single image file or
                only files that <b>start with the same, common basename.</b><br><br>
                For example the following filenames:<br><br>
                F014_s01_phase_contr.tif<br>
                F014_s01_mCitrine.tif<br><br>
                are named correctly since they all start with the
                the common basename "F014_s01_". After the common basename you
                can write whatever text you want. In the example above,
                "phase_contr"  and "mCitrine" are the channel names.<br><br>
                We recommend using the module 0. or the provided Fiji/ImageJ
                macro to create the right data structure.<br><br>
                Data loading may still be successfull, so the system will
                still try to load data now.
            """)
            msg = widgets.myMessageBox()
            details = "\n".join(files)
            details = f'Files detected:\n\n{details}'
            msg.warning(
                self.parent, 'Data structure compromised', txt,
                detailedText=details
            )
            return False
        return True

    def getChannels(self, filenames, images_path, useExt=('.tif', '.h5')):
        # First check if metadata.csv already has the channel names
        metadata_csv_path = None
        for file in utils.listdir(images_path):
            if file.endswith('metadata.csv'):
                metadata_csv_path = os.path.join(images_path, file)
                break

        chNames_found = False
        if metadata_csv_path is not None:
            df = pd.read_csv(metadata_csv_path)
            if 'Description' in df.columns:
                channelNamesMask = df.Description.str.contains('channel_\d+_name')
                channelNames = df[channelNamesMask]['values'].to_list()
                if channelNames:
                    channel_names = channelNames.copy()
                    basename = None
                    for chName in channelNames:
                        chSaved = []
                        for file in filenames:
                            if file.find(chName) != -1:
                                chSaved.append(True)
                                chName_idx = file.find(chName)
                                basename = file[:chName_idx]
                        if not any(chSaved):
                            channel_names.remove(chName)

                    if basename is not None:
                        self.basenameNotFound = False
                        self.basename = basename
                        return channel_names, False

        # Find basename as intersection of filenames
        channel_names = []
        self.basenameNotFound = False
        isBasenamePresent = self.checkDataIntegrity(filenames)
        basename = filenames[0]
        for file in filenames:
            # Determine the basename based on intersection of all .tif
            _, ext = os.path.splitext(file)
            if useExt is None:
                sm = difflib.SequenceMatcher(None, file, basename)
                i, j, k = sm.find_longest_match(0, len(file),
                                                0, len(basename))
                basename = file[i:i+k]
            elif ext in useExt:
                sm = difflib.SequenceMatcher(None, file, basename)
                i, j, k = sm.find_longest_match(0, len(file),
                                                0, len(basename))
                basename = file[i:i+k]
        self.basename = basename
        basenameNotFound = [False]
        for file in filenames:
            filename, ext = os.path.splitext(file)
            if useExt is None:
                channel_name = filename.split(basename)[-1]
                channel_names.append(channel_name)
                if channel_name == filename:
                    # Warn that an intersection could not be found
                    basenameNotFound.append(True)
            elif ext in useExt:
                channel_name = filename.split(basename)[-1]
                channel_names.append(channel_name)
                if channel_name == filename:
                    # Warn that an intersection could not be found
                    basenameNotFound.append(True)
        if any(basenameNotFound):
            self.basenameNotFound = True
            filenameNOext, _ = os.path.splitext(basename)
            self.basename = f'{filenameNOext}_'
        if self.which_channel is not None:
            # Search for "phase" and put that channel first on the list
            if self.which_channel == 'segm':
                is_phase_contr_li = [c.lower().find('phase')!=-1
                                     for c in channel_names]
                if any(is_phase_contr_li):
                    idx = is_phase_contr_li.index(True)
                    channel_names[0], channel_names[idx] = (
                                      channel_names[idx], channel_names[0])
        return channel_names, any(basenameNotFound)

    def _load_last_selection(self):
        last_sel_channel = None
        ch = self.which_channel
        if self.which_channel is not None:
            _path = os.path.dirname(os.path.realpath(__file__))
            settings_path = os.path.join(_path, 'settings')
            txt_path = os.path.join(settings_path, f'{ch}_last_sel.txt')
            if os.path.exists(txt_path):
                with open(txt_path) as txt:
                    last_sel_channel = txt.read()
        return last_sel_channel

    def _save_last_selection(self, selection):
        ch = self.which_channel
        if self.which_channel is not None:
            _path = os.path.dirname(os.path.realpath(__file__))
            settings_path = os.path.join(_path, 'settings')
            if not os.path.exists(settings_path):
                os.mkdir(settings_path)
            txt_path = os.path.join(settings_path, f'{ch}_last_sel.txt')
            with open(txt_path, 'w') as txt:
                txt.write(selection)

    def askSelectChannel(self, parent, channel_names, informativeText='',
                 CbLabel='Select channel name to load:  '):
        font = QFont()
        font.setPixelSize(13)
        win = dialogs.QDialogCombobox(
            'Select channel name', channel_names,
            informativeText, CbLabel=CbLabel,
            parent=parent, defaultChannelName=self.last_sel_channel
        )
        win.setFont(font)
        win.exec_()
        if win.cancel:
            self.was_aborted = True
        self.channel_name = win.selectedItemText
        if not win.cancel:
            self._save_last_selection(self.channel_name)
        self.is_first_call = False

    def setUserChannelName(self):
        if self.basenameNotFound:
            reverse_ch_name = self.channel_name[::-1]
            idx = reverse_ch_name.find('_')
            if idx != -1:
                self.user_ch_name = self.channel_name[-idx:]
            else:
                self.user_ch_name = self.channel_name[-4:]
        else:
            self.user_ch_name = self.channel_name

class expFolderScanner:
    def __init__(self, homePath=''):
        self.is_first_call = True
        self.expPaths = []
        self.homePath = homePath

    def getExpPaths(self, path, signals=None):
        """Recursively scan the directory tree to search for folders that
        contain Position folders. When found, the path will be appended to
        self.expPaths attribute

        Parameters
        ----------
        path : str or Path
            Path to check if it contains Position folders.
        signals : attribute of QObject subclass or None.
            If not None it is used to emit signals and communicate with
            main GUI thread (e.g., to update progress bar).

        Returns
        -------
        None

        """
        if not os.path.isdir(path):
            return

        if self.is_first_call:
            self.is_first_call = False
            if signals is not None:
                signals.progress.emit(
                    'Searching valid experiment folders...',
                    'INFO'
                )
                signals.initProgressBar.emit(0)

        ls = natsorted(utils.listdir(path))
        isExpPath = any([
            f.find('Position_')!=-1 and os.path.isdir(os.path.join(path, f))
            for f in ls
        ])

        if isExpPath:
            self.expPaths.append(path)
        else:
            with ThreadPoolExecutor(4) as ex:
                ex.map(self.getExpPaths, [os.path.join(path, f) for f in ls])

    def _setInfoExpPath(self, exp_path):
        """
        See infoExpPaths for more details
        """
        ls = natsorted(utils.listdir(exp_path))

        posFoldernames = natsorted([
            f for f in ls
            if f.find('Position_')!=-1
            and os.path.isdir(os.path.join(exp_path, f))
        ])

        self.paths[1][exp_path] = {
            'numPosSpotCounted': 0,
            'numPosSpotSized': 0,
            'posFoldernames': posFoldernames,
        }
        for p, pos in enumerate(posFoldernames):
            posPath = os.path.join(exp_path, pos)
            spotmaxOutPath = os.path.join(posPath, 'spotMAX_output')
            imagesPath = os.path.join(posPath, 'Images')
            isSpotmaxOutPresent = os.path.exists(spotmaxOutPath)
            if not isSpotmaxOutPresent:
                self.paths[1][exp_path][pos] = {
                    'isPosSpotCounted': False,
                    'isPosSpotSized': False
                }
            else:
                spotmaxFiles = natsorted(utils.listdir(spotmaxOutPath))
                if not spotmaxFiles:
                    continue
                run_nums = self.runNumbers(spotmaxOutPath)


                for run in run_nums:
                    if p==0:
                        analysisInputs_df = self.loadAnalysisInputs(
                            spotmaxOutPath, run
                        )
                        self.paths[run][exp_path] = {
                            'numPosSpotCounted': 0,
                            'numPosSpotSized': 0,
                            'posFoldernames': posFoldernames,
                            'analysisInputs': analysisInputs_df
                        }

                    isSpotCounted, isSpotSized = self.analyseRunNumber(
                        spotmaxOutPath, run
                    )
                    self.paths[run][exp_path][pos] = {
                        'isPosSpotCounted': isSpotCounted,
                        'isPosSpotSized': isSpotSized
                    }
                    if isSpotCounted:
                        self.paths[run][exp_path]['numPosSpotCounted'] += 1
                    if isSpotSized:
                        self.paths[run][exp_path]['numPosSpotSized'] += 1

    def addMissingRunsInfo(self):
        paths = copy.deepcopy(self.paths)
        for run, runInfo in self.paths.items():
            for exp_path, expInfo in runInfo.items():
                posFoldernames = expInfo['posFoldernames']
                for pos in posFoldernames:
                    try:
                        posInfo = expInfo[pos]
                    except KeyError as e:
                        paths[run][exp_path][pos] = {
                            'isPosSpotCounted': False,
                            'isPosSpotSized': False
                        }
        self.paths = paths

    def infoExpPaths(self, expPaths, signals=None):
        """Method used to determine how each experiment was analysed.

        Parameters
        ----------
        expPaths : type
            Description of parameter `expPaths`.

        Returns
        -------
        dict
            A nested dictionary with the following keys:
                expInfo = paths[run_number][exp_path] --> dict
                numPosSpotCounted = expInfo['numPosSpotCounted'] --> int
                numPosSpotSized = expInfo['numPosSpotSized'] --> int
                posFoldernames = expInfo['posFoldernames'] --> list of strings
                analysisInputs_df = expInfo['analysisInputs'] --> pd.DataFrame
                pos1_info = expInfo['Position_1'] --> dict
                    isPos1_spotCounted = pos1_info['isPosSpotCounted'] --> bool
                    isPos1_spotSized = pos1_info['isPosSpotSized'] --> bool

        """
        self.paths = defaultdict(lambda: defaultdict(dict))


        if signals is not None:
            signals.progress.emit(
                'Scanning experiment folder(s)...',
                'INFO'
            )
        else:
            print('Scanning experiment folders...')
        for exp_path in tqdm(expPaths, unit=' folder', ncols=100):
            self._setInfoExpPath(exp_path)
            if signals is not None:
                signals.progressBar.emit(1)

        self.addMissingRunsInfo()

    def loadAnalysisInputs(self, spotmaxOutPath, run):
        df = None
        for file in utils.listdir(spotmaxOutPath):
            m = re.match(f'{run}_(\w*)analysis_inputs.csv', file)
            if m is not None:
                csvPath = os.path.join(spotmaxOutPath, file)
                df = pd.read_csv(csvPath, index_col='Description')
                df1 = utils.pdDataFrame_boolTo0s1s(df, labelsToCast='allRows')
                if not df.equals(df1):
                    df1.to_csv(csvPath)
                    df = df1
        return df

    def runNumbers(self, spotmaxOutPath):
        run_nums = set()
        spotmaxFiles = natsorted(utils.listdir(spotmaxOutPath))
        if not spotmaxFiles:
            return run_nums
        run_nums = [
            re.findall('(\d+)_(\d)_', f) for f in spotmaxFiles
        ]
        run_nums = [int(m[0][0]) for m in run_nums if m]
        run_nums = set(run_nums)
        return run_nums

    def analyseRunNumber(self, spotmaxOutPath, run):
        numSpotCountFilesPresent = 0
        numSpotSizeFilesPresent = 0

        p_ellip_test_csv_filename = f'{run}_3_p-_ellip_test_data_Summary'
        p_ellip_test_h5_filename = f'{run}_3_p-_ellip_test_data'

        spotSize_csv_filename = f'{run}_4_spotfit_data_Summary'
        spotSize_h5_filename = f'{run}_4_spotFIT_data'
        for file in utils.listdir(spotmaxOutPath):
            isSpotCountCsvPresent = (
                file.find(p_ellip_test_csv_filename)!=-1
                and file.endswith('.csv')
            )
            if isSpotCountCsvPresent:
                numSpotCountFilesPresent += 1

            isSpotCount_h5_present = (
                file.find(p_ellip_test_h5_filename)!=-1
                and file.endswith('.h5')
            )
            if isSpotCount_h5_present:
                numSpotCountFilesPresent += 1

            isSpotSizeCsvPresent = (
                file.find(spotSize_csv_filename)!=-1
                and file.endswith('.csv')
            )
            if isSpotSizeCsvPresent:
                numSpotSizeFilesPresent += 1
            isSpotSize_h5_present = (
                file.find(spotSize_h5_filename)!=-1
                and file.endswith('.h5')
            )
            if isSpotSize_h5_present:
                numSpotSizeFilesPresent += 1

        isPosSpotCounted = numSpotCountFilesPresent == 2
        isPosSpotSized = numSpotSizeFilesPresent == 2

        return isPosSpotCounted, isPosSpotSized

    def input(self, app=None):
        win = dialogs.selectPathsSpotmax(self.paths, self.homePath, app=app)
        win.show()
        win.exec_()
        self.selectedPaths = win.selectedPaths

    def validPosPaths(self):
        pass

class loadData:
    def __init__(self, channelDataPath, user_ch_name, QParent=None):
        # Additional loaded data
        self.loadedRelativeFilenamesData = {}

        # Dictionary of keys to keep track which channels are merged
        self.loadedMergeRelativeFilenames = {}

        # Dictionary of keys to keep track which channel is skeletonized
        self.skeletonizedRelativeFilename = ''

        # Dictionary of keys to keep track which channel is contoured
        self.contouredRelativeFilename = ''

        # Gradient levels for each channel name (layer 0, layer 1, etc)
        self.gradientLevels = {}

        # Skeleton coords as calulcated in self.skeletonize()
        self.skelCoords = {}

        # Contour coords as calulcated in self.contours()
        self.contCoords = {}

        # For .h5 files we can load a subset of the entire file.
        # loadSizeT and loadSizeZ are asked at askInputMetadata method
        self.loadSizeT, self.loadSizeZ = None, None

        self.bkgrROIs = []
        self.parent = QParent
        self.channelDataPath = str(channelDataPath)
        self.user_ch_name = user_ch_name
        self.images_path = os.path.dirname(channelDataPath)
        self.pos_path = os.path.dirname(self.images_path)
        self.h5_path = ''
        self.spotmaxOutPath = os.path.join(self.pos_path, 'spotMAX_output')
        self.exp_path = os.path.dirname(self.pos_path)
        self.pos_foldername = os.path.basename(self.pos_path)
        self.cropROI = None
        path_li = os.path.normpath(channelDataPath).split(os.sep)
        self.relPath = os.path.join('', *path_li[-4:])
        filename_ext = os.path.basename(channelDataPath)
        self.filename, self.ext = os.path.splitext(filename_ext)
        self.cca_df_colnames = [
            'cell_cycle_stage',
            'generation_num',
            'relative_ID',
            'relationship',
            'emerg_frame_i',
            'division_frame_i',
            'is_history_known',
            'corrected_assignment'
        ]
        self.loadLastEntriesMetadata()

    def loadLastEntriesMetadata(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        settings_path = os.path.join(src_path, 'settings')
        if not os.path.exists(settings_path):
            self.last_md_df = None
            return
        csv_path = os.path.join(settings_path, 'last_entries_metadata.csv')
        if not os.path.exists(csv_path):
            self.last_md_df = None
        else:
            self.last_md_df = pd.read_csv(csv_path).set_index('Description')

    def saveLastEntriesMetadata(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        settings_path = os.path.join(src_path, 'settings')
        if not os.path.exists:
            return
        csv_path = os.path.join(settings_path, 'last_entries_metadata.csv')
        self.metadata_df.to_csv(csv_path)

    def getBasenameAndChNames(self, load=True):
        ls = utils.listdir(self.images_path)
        channelNameUtil = channelName(load=load)
        self.chNames, _ = channelNameUtil.getChannels(ls, self.images_path)
        self.basename = channelNameUtil.basename
        self.allRelFilenames = [
            file[len(self.basename):] for file in ls
            if os.path.splitext(file)[1] == '.tif'
            or os.path.splitext(file)[1] == '.npy'
            or os.path.splitext(file)[1] == '.npz'
            or os.path.splitext(file)[1] == '.h5'
        ]

    def checkH5memoryFootprint(self):
        if self.ext != '.h5':
            return 0
        else:
            Y, X = self.h5_dset.shape[-2:]
            size = self.loadSizeT*self.loadSizeZ*Y*X
            itemsize = self.h5_dset.dtype.itemsize
            required_memory = size*itemsize
            return required_memory

    def shouldLoadTchunk(self, current_t):
        if self.ext != '.h5':
            return False

        coord1_window = self.t0_window + self.loadSizeT - 1
        halfWindowSize = int(self.loadSizeT/2)

        coord0_chunk = coord1_window + 1
        chunkSize = current_t + halfWindowSize - coord0_chunk + 1

        rightBoundary = self.SizeT-halfWindowSize
        leftBoundary = halfWindowSize

        if current_t <= halfWindowSize and leftBoundary >= self.t0_window:
            return False
        elif current_t >= rightBoundary and rightBoundary <= coord1_window:
            return False

        return True

    def shouldLoadZchunk(self, current_idx):
        if self.ext != '.h5':
            return False

        coord1_window = self.z0_window + self.loadSizeZ - 1
        halfWindowSize = int(self.loadSizeZ/2)

        coord0_chunk = coord1_window + 1
        chunkSize = current_idx + halfWindowSize - coord0_chunk + 1

        rightBoundary = self.SizeZ-halfWindowSize
        leftBoundary = halfWindowSize
        if current_idx <= halfWindowSize and leftBoundary >= self.z0_window:
            return False
        elif current_idx >= rightBoundary and rightBoundary <= coord1_window:
            return False

        return True

    def loadChannelDataChunk(self, current_idx, axis=0, worker=None):
        is4D = self.SizeZ > 1 and self.SizeT > 1
        is3Dz = self.SizeZ > 1 and self.SizeT == 1
        is3Dt = self.SizeZ == 1 and self.SizeT > 1
        is2D = self.SizeZ == 1 and self.SizeT == 1
        if is4D:
            if axis==0:
                axis1_range = (self.z0_window, self.z0_window+self.loadSizeZ)
                chData, t0_window, z0_window = utils.shiftWindow_axis0(
                    self.h5_dset, self.chData, self.loadSizeT, self.t0_window,
                    current_idx, axis1_interval=axis1_range, worker=worker
                )
            elif axis==1:
                axis0_range = (self.t0_window, self.t0_window+self.loadSizeT)
                chData, t0_window, z0_window = utils.shiftWindow_axis1(
                    self.h5_dset, self.chData, self.loadSizeZ, self.z0_window,
                    current_idx, axis0_interval=axis0_range, worker=worker
                )
        elif is3Dz:
            chData, t0_window, z0_window = utils.shiftWindow_axis0(
                self.h5_dset, self.chData, self.loadSizeZ, self.z0_window,
                current_idx, axis1_interval=None, worker=worker
            )
        elif is3Dt:
            chData, t0_window, z0_window = utils.shiftWindow_axis0(
                self.h5_dset, self.chData, self.loadSizeT, self.t0_window,
                current_idx, axis1_interval=None, worker=worker
            )
        self.chData = chData
        self.t0_window = t0_window
        self.z0_window = z0_window

    def loadChannelData(self):
        self.z0_window = 0
        self.t0_window = 0
        if self.ext == '.h5':
            self.h5f = h5py.File(self.channelDataPath, 'r')
            self.h5_dset = self.h5f['data']
            self.chData_shape = self.h5_dset.shape
            readH5 = self.loadSizeT is not None or self.loadSizeZ is not None
            if not readH5:
                return

            is4D = self.SizeZ > 1 and self.SizeT > 1
            is3Dz = self.SizeZ > 1 and self.SizeT == 1
            is3Dt = self.SizeZ == 1 and self.SizeT > 1
            is2D = self.SizeZ == 1 and self.SizeT == 1
            if is4D:
                midZ = int(self.SizeZ/2)
                halfZLeft = int(self.loadSizeZ/2)
                halfZRight = self.loadSizeZ-halfZLeft
                z0 = midZ-halfZLeft
                z1 = midZ+halfZRight
                self.z0_window = z0
                self.t0_window = 0
                self.chData = self.h5_dset[:self.loadSizeT, z0:z1]
            elif is3Dz:
                midZ = int(self.SizeZ/2)
                halfZLeft = int(self.loadSizeZ/2)
                halfZRight = self.loadSizeZ-halfZLeft
                z0 = midZ-halfZLeft
                z1 = midZ+halfZRight
                self.z0_window = z0
                self.chData = np.squeeze(self.h5_dset[z0:z1])
            elif is3Dt:
                self.t0_window = 0
                self.chData = np.squeeze(self.h5_dset[:self.loadSizeT])
            elif is2D:
                self.chData = self.h5_dset[:]
        elif self.ext == '.npz':
            self.chData = np.load(self.channelDataPath)['arr_0']
            self.chData_shape = self.chData.shape
        elif self.ext == '.npy':
            self.chData = np.load(self.channelDataPath)
            self.chData_shape = self.chData.shape
        else:
            try:
                self.chData = skimage.io.imread(self.channelDataPath)
                self.chData_shape = self.chData.shape
            except ValueError:
                self.chData = self._loadVideo(self.channelDataPath)
                self.chData_shape = self.chData.shape
            except Exception as e:
                traceback.print_exc()
                self.criticalExtNotValid()

    def _loadVideo(self, path):
        video = cv2.VideoCapture(path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            _, frame = video.read()
            if frame.shape[-1] == 3:
                frame = skimage.color.rgb2gray(frame)
            if i == 0:
                chData = np.zeros((num_frames, *frame.shape), frame.dtype)
            chData[i] = frame
        return chData

    def absoluteFilename(self, relFilename):
        absoluteFilename = f'{self.basename}{relFilename}'
        return absoluteFilename

    def absoluteFilepath(self, relFilename):
        absoluteFilename = f'{self.basename}{relFilename}'
        return os.path.join(self.images_path, absoluteFilename)

    def detectMultiSegmNpz(self):
        ls = utils.listdir(self.images_path)
        segm_files = [file for file in ls if file.endswith('segm.npz')]
        is_multi_npz = len(segm_files)>1
        if is_multi_npz:
            font = QFont()
            font.setPixelSize(13)
            win = apps.QDialogMultiSegmNpz(
                segm_files, self.pos_path, parent=self.parent
            )
            win.setFont(font)
            win.exec_()
            if win.removeOthers:
                for file in segm_files:
                    if file == win.selectedItemText:
                        continue
                    os.remove(os.path.join(self.images_path, file))
            return win.selectedItemText, win.cancel
        else:
            return '', False

    def loadOtherFiles(
            self,
            load_segm_data=False,
            load_acdc_df=False,
            load_shifts=False,
            loadSegmInfo=False,
            load_delROIsInfo=False,
            loadBkgrData=False,
            loadBkgrROIs=False,
            load_last_tracked_i=False,
            load_metadata=False,
            load_dataPrep_ROIcoords=False,
            getTifPath=False,
            load_ref_ch_mask=False,
            selectedSegmNpz=''
        ):
        self.segmFound = False if load_segm_data else None
        self.acd_df_found = False if load_acdc_df else None
        self.shiftsFound = False if load_shifts else None
        self.segmInfoFound = False if loadSegmInfo else None
        self.delROIsInfoFound = False if load_delROIsInfo else None
        self.bkgrDataFound = False if loadBkgrData else None
        self.bkgrROisFound = False if loadBkgrROIs else None
        self.last_tracked_i_found = False if load_last_tracked_i else None
        self.metadataFound = False if load_metadata else None
        self.dataPrep_ROIcoordsFound = False if load_dataPrep_ROIcoords else None
        self.TifPathFound = False if getTifPath else None
        self.refChMaskFound = False if load_ref_ch_mask else None
        ls = utils.listdir(self.images_path)

        for file in ls:
            filePath = os.path.join(self.images_path, file)
            if selectedSegmNpz:
                is_segm_file = file == selectedSegmNpz
            else:
                is_segm_file = file.endswith('segm.npz')

            if load_segm_data and is_segm_file:
                self.segmFound = True
                self.segm_npz_path = filePath
                self.segm_data = np.load(filePath)['arr_0']
                squeezed_arr = np.squeeze(self.segm_data)
                if squeezed_arr.shape != self.segm_data.shape:
                    self.segm_data = squeezed_arr
                    np.savez_compressed(filePath, squeezed_arr)
            elif getTifPath and file.find(f'{self.user_ch_name}.tif')!=-1:
                self.tif_path = filePath
                self.TifPathFound = True
            elif load_acdc_df and file.endswith('acdc_output.csv'):
                self.acd_df_found = True
                acdc_df = pd.read_csv(
                      filePath, index_col=['frame_i', 'Cell_ID']
                )
                acdc_df = self.BooleansTo0s1s(acdc_df, inplace=True)
                acdc_df = self.intToBoolean(acdc_df)
                self.acdc_df = acdc_df
            elif load_shifts and file.endswith('align_shift.npy'):
                self.shiftsFound = True
                self.loaded_shifts = np.load(filePath)
            elif loadSegmInfo and file.endswith('segmInfo.csv'):
                self.segmInfoFound = True
                df = pd.read_csv(filePath)
                if 'filename' not in df.columns:
                    df['filename'] = self.filename
                self.segmInfo_df = df.set_index(['filename', 'frame_i'])
            elif load_delROIsInfo and file.endswith('delROIsInfo.npz'):
                self.delROIsInfoFound = True
                self.delROIsInfo_npz = np.load(filePath)
            elif loadBkgrData and file.endswith(f'{self.filename}_bkgrRoiData.npz'):
                self.bkgrDataFound = True
                self.bkgrData = np.load(filePath)
            elif loadBkgrROIs and file.endswith('dataPrep_bkgrROIs.json'):
                self.bkgrROisFound = True
                with open(filePath) as json_fp:
                    bkgROIs_states = json.load(json_fp)

                for roi_state in bkgROIs_states:
                    Y, X = self.chData_shape[-2:]
                    roi = pg.ROI(
                        [0, 0], [1, 1],
                        rotatable=False,
                        removable=False,
                        pen=pg.mkPen(color=(150,150,150)),
                        maxBounds=QRectF(QRect(0,0,X,Y))
                    )
                    roi.setState(roi_state)
                    self.bkgrROIs.append(roi)
            elif load_dataPrep_ROIcoords and file.endswith('dataPrepROIs_coords.csv'):
                df = pd.read_csv(filePath)
                if 'description' in df.columns:
                    df = df.set_index('description')
                    if 'value' in df.columns:
                        self.dataPrep_ROIcoordsFound = True
                        self.dataPrep_ROIcoords = df
            elif (load_metadata and file.endswith('metadata.csv')
                and not file.endswith('segm_metadata.csv')
                ):
                self.metadataFound = True
                self.metadata_df = pd.read_csv(filePath).set_index('Description')
                self.extractMetadata()
            elif file.endswith('mask.npy') or file.endswith('mask.npz'):
                self.refChMaskFound = True
                self.refChMask = np.load(filePath)

        if load_last_tracked_i:
            self.last_tracked_i_found = True
            try:
                self.last_tracked_i = max(self.acdc_df.index.get_level_values(0))
            except AttributeError as e:
                # traceback.print_exc()
                self.last_tracked_i = None

        else:
            is_segm_file = file.endswith('segm.npz')

        if load_segm_data and not self.segmFound:
            # Try to check if there is npy segm data
            for file in ls:
                if file.endswith('segm.npy'):
                    filePath = os.path.join(self.images_path, file)
                    self.segmFound = True
                    self.segm_npz_path = filePath
                    self.segm_data = np.load(filePath)
                    break

        self.setNotFoundData()

    def segmLabels(self, frame_i):
        if self.segm_data is None:
            return None

        if self.SizeT > 1:
            lab = self.segm_data[frame_i]
        else:
            lab = self.segm_data
        return lab

    def computeSegmRegionprops(self):
        if self.segm_data is None:
            self.rp = None
            return

        if self.SizeT > 1:
            self.rp = [
                skimage.measure.regionprops(lab) for lab in self.segm_data
            ]
            self.newIDs = []
            self.IDs = []
            self.rpDict = []
            for frame_i, rp in enumerate(self.rp):
                if frame_i == 0:
                    self.newIDs.append([])
                    continue
                prevIDs = [obj.label for obj in self.regionprops(frame_i-1)]
                currentIDs = [obj.label for obj in rp]
                newIDs = [ID for ID in currentIDs if ID not in prevIDs]
                self.IDs.append(currentIDs)
                self.newIDs.append(newIDs)
                self.computeRotationalCellVolume(rp)
                rpDict = {obj.label:obj for obj in rp}
                self.rpDict.append(rpDict)
        else:
            self.rp = skimage.measure.regionprops(self.segm_data)
            self.IDs = [obj.label for obj in self.rp]
            self.rpDict = {obj.label:obj for obj in self.rp}
            self.computeRotationalCellVolume(self.rp)

    def computeRotationalCellVolume(self, rp):
        for obj in rp:
            vol_vox, vol_fl = core.rotationalVolume(
                obj,
                PhysicalSizeY=self.PhysicalSizeY,
                PhysicalSizeX=self.PhysicalSizeX
            )
            obj.vol_vox, obj.vol_fl = vol_vox, vol_fl
        return vol_vox, vol_fl


    def getNewIDs(self, frame_i):
        if frame_i == 0:
            return []

        if self.SizeT > 1:
            return self.newIDs[frame_i]
        else:
            return []

    def IDs(self, frame_i):
        if self.SizeT > 1:
            return self.IDs[frame_i]
        else:
            return self.IDs

    def regionprops(self, frame_i, returnDict=False):
        if self.SizeT > 1:
            if returnDict:
                return self.rpDict[frame_i]
            else:
                return self.rp[frame_i]
        else:
            if returnDict:
                return self.rpDict
            else:
                return self.rp

    def cca_df(self, frame_i):
        if self.acdc_df is None:
            return None

        cca_df = self.acdc_df.loc[frame_i][self.cca_df_colnames]
        return cca_df

    def extractMetadata(self):
        if 'SizeT' in self.metadata_df.index:
            self.SizeT = int(self.metadata_df.at['SizeT', 'values'])
        elif self.last_md_df is not None and 'SizeT' in self.last_md_df.index:
            self.SizeT = int(self.last_md_df.at['SizeT', 'values'])
        else:
            self.SizeT = 1

        if 'SizeZ' in self.metadata_df.index:
            self.SizeZ = int(self.metadata_df.at['SizeZ', 'values'])
        elif self.last_md_df is not None and 'SizeZ' in self.last_md_df.index:
            self.SizeZ = int(self.last_md_df.at['SizeZ', 'values'])
        else:
            self.SizeZ = 1

        if 'TimeIncrement' in self.metadata_df.index:
            self.TimeIncrement = float(
                self.metadata_df.at['TimeIncrement', 'values']
            )
        elif self.last_md_df is not None and 'TimeIncrement' in self.last_md_df.index:
            self.TimeIncrement = float(self.last_md_df.at['TimeIncrement', 'values'])
        else:
            self.TimeIncrement = 1

        if 'PhysicalSizeX' in self.metadata_df.index:
            self.PhysicalSizeX = float(
                self.metadata_df.at['PhysicalSizeX', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(self.last_md_df.at['PhysicalSizeX', 'values'])
        else:
            self.PhysicalSizeX = 1

        if 'PhysicalSizeY' in self.metadata_df.index:
            self.PhysicalSizeY = float(
                self.metadata_df.at['PhysicalSizeY', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(self.last_md_df.at['PhysicalSizeY', 'values'])
        else:
            self.PhysicalSizeY = 1

        if 'PhysicalSizeZ' in self.metadata_df.index:
            self.PhysicalSizeZ = float(
                self.metadata_df.at['PhysicalSizeZ', 'values']
            )
        elif self.last_md_df is not None and 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(self.last_md_df.at['PhysicalSizeZ', 'values'])
        else:
            self.PhysicalSizeZ = 1

        load_last_segmSizeT = (
            self.last_md_df is not None
            and 'segmSizeT' in self.last_md_df.index
            and self.SizeT > 1
        )
        if 'segmSizeT' in self.metadata_df.index:
             self.segmSizeT = int(
                 self.metadata_df.at['segmSizeT', 'values']
             )
        elif load_last_segmSizeT:
            self.segmSizeT = int(self.last_md_df.at['segmSizeT', 'values'])
        else:
            self.segmSizeT = self.SizeT

    def setNotFoundData(self):
        if self.segmFound is not None and not self.segmFound:
            self.segm_data = None
        if self.acd_df_found is not None and not self.acd_df_found:
            self.acdc_df = None
        if self.shiftsFound is not None and not self.shiftsFound:
            self.loaded_shifts = None
        if self.segmInfoFound is not None and not self.segmInfoFound:
            self.segmInfo_df = None
        if self.delROIsInfoFound is not None and not self.delROIsInfoFound:
            self.delROIsInfo_npz = None
        if self.bkgrDataFound is not None and not self.bkgrDataFound:
            self.bkgrData = None
        if self.dataPrep_ROIcoordsFound is not None and not self.dataPrep_ROIcoordsFound:
            self.dataPrep_ROIcoords = None
        if self.last_tracked_i_found is not None and not self.last_tracked_i_found:
            self.last_tracked_i = None
        if self.TifPathFound is not None and not self.TifPathFound:
            self.tif_path = None
        if self.refChMaskFound is not None and not self.refChMaskFound:
            self.refChMask = None

        if self.metadataFound is None:
            # Loading metadata was not requested
            return

        if self.metadataFound:
            return

        if self.chData.ndim == 3:
            if len(self.chData) > 49:
                self.SizeT, self.SizeZ = len(self.chData), 1
            else:
                self.SizeT, self.SizeZ = 1, len(self.chData)
        elif self.chData.ndim == 4:
            self.SizeT, self.SizeZ = self.chData_shape[:2]
        else:
            self.SizeT, self.SizeZ = 1, 1

        self.TimeIncrement = 1.0
        self.PhysicalSizeX = 1.0
        self.PhysicalSizeY = 1.0
        self.PhysicalSizeZ = 1.0
        self.segmSizeT = self.SizeT
        self.metadata_df = None

        if self.last_md_df is None:
            # Last entered values do not exists
            return

        # Since metadata was not found use the last entries saved in temp folder
        if 'TimeIncrement' in self.last_md_df.index:
            self.TimeIncrement = float(self.last_md_df.at['TimeIncrement', 'values'])
        if 'PhysicalSizeX' in self.last_md_df.index:
            self.PhysicalSizeX = float(self.last_md_df.at['PhysicalSizeX', 'values'])
        if 'PhysicalSizeY' in self.last_md_df.index:
            self.PhysicalSizeY = float(self.last_md_df.at['PhysicalSizeY', 'values'])
        if 'PhysicalSizeZ' in self.last_md_df.index:
            self.PhysicalSizeZ = float(self.last_md_df.at['PhysicalSizeZ', 'values'])
        if 'segmSizeT' in self.last_md_df.index:
            self.segmSizeT = int(self.last_md_df.at['segmSizeT', 'values'])

    def checkMetadata_vs_shape(self):
        pass

    def buildPaths(self):
        if self.basename.endswith('_'):
            basename = self.basename
        else:
            basename = f'{self.basename}_'
        base_path = f'{self.images_path}/{basename}'
        self.slice_used_align_path = f'{base_path}slice_used_alignment.csv'
        self.slice_used_segm_path = f'{base_path}slice_segm.csv'
        self.align_npz_path = f'{base_path}{self.user_ch_name}_aligned.npz'
        self.align_old_path = f'{base_path}phc_aligned.npy'
        self.align_shifts_path = f'{base_path}align_shift.npy'
        self.segm_npz_path = f'{base_path}segm.npz'
        self.last_tracked_i_path = f'{base_path}last_tracked_i.txt'
        self.acdc_output_csv_path = f'{base_path}acdc_output.csv'
        self.segmInfo_df_csv_path = f'{base_path}segmInfo.csv'
        self.delROIs_info_path = f'{base_path}delROIsInfo.npz'
        self.dataPrepROI_coords_path = f'{base_path}dataPrepROIs_coords.csv'
        # self.dataPrepBkgrValues_path = f'{base_path}dataPrep_bkgrValues.csv'
        self.dataPrepBkgrROis_path = f'{base_path}dataPrep_bkgrROIs.json'
        self.metadata_csv_path = f'{base_path}metadata.csv'
        self.analysis_inputs_path = f'{base_path}analysis_inputs.ini'

    def setBlankSegmData(self, SizeT, SizeZ, SizeY, SizeX):
        Y, X = self.chData_shape[-2:]
        if self.segmFound is not None and not self.segmFound:
            if SizeT > 1:
                self.segm_data = np.zeros((SizeT, Y, X), int)
            else:
                self.segm_data = np.zeros((Y, X), int)

    def loadAllChannelsPaths(self):
        tif_paths = []
        npy_paths = []
        npz_paths = []
        basename = self.basename[0:-1]
        for filename in utils.listdir(self.images_path):
            file_path = os.path.join(self.images_path, filename)
            f, ext = os.path.splitext(filename)
            m = re.match(f'{basename}.*\.tif', filename)
            if m is not None:
                tif_paths.append(file_path)
                # Search for npy fluo data
                npy = f'{f}_aligned.npy'
                npz = f'{f}_aligned.npz'
                npy_found = False
                npz_found = False
                for name in utils.listdir(self.images_path):
                    _path = os.path.join(self.images_path, name)
                    if name == npy:
                        npy_paths.append(_path)
                        npy_found = True
                    if name == npz:
                        npz_paths.append(_path)
                        npz_found = True
                if not npy_found:
                    npy_paths.append(None)
                if not npz_found:
                    npz_paths.append(None)
        self.tif_paths = tif_paths
        self.npy_paths = npy_paths
        self.npz_paths = npz_paths

    def askInputMetadata(
            self, numPos,
            ask_SizeT=False,
            ask_TimeIncrement=False,
            ask_PhysicalSizes=False,
            save=False
        ):
        font = QFont()
        font.setPixelSize(13)
        metadataWin = dialogs.QDialogMetadata(
            self.SizeT, self.SizeZ, self.TimeIncrement,
            self.PhysicalSizeZ, self.PhysicalSizeY, self.PhysicalSizeX,
            ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes, numPos,
            parent=self.parent, font=font, imgDataShape=self.chData_shape,
            PosData=self, fileExt=self.ext
        )
        metadataWin.setFont(font)
        metadataWin.exec_()
        if metadataWin.cancel:
            return False

        self.SizeT = metadataWin.SizeT
        self.SizeZ = metadataWin.SizeZ
        self.loadSizeS = metadataWin.loadSizeS
        self.loadSizeT = metadataWin.loadSizeT
        self.loadSizeZ = metadataWin.loadSizeZ

        source = metadataWin if ask_TimeIncrement else self
        self.TimeIncrement = source.TimeIncrement

        source = metadataWin if ask_PhysicalSizes else self
        self.PhysicalSizeZ = source.PhysicalSizeZ
        self.PhysicalSizeY = source.PhysicalSizeY
        self.PhysicalSizeX = source.PhysicalSizeX
        if save:
            self.saveMetadata()
        return True

    def transferMetadata(self, from_PosData):
        self.SizeT = from_PosData.SizeT
        self.SizeZ = from_PosData.SizeZ
        self.PhysicalSizeZ = from_PosData.PhysicalSizeZ
        self.PhysicalSizeY = from_PosData.PhysicalSizeY
        self.PhysicalSizeX = from_PosData.PhysicalSizeX

    def saveMetadata(self):
        if self.metadata_df is None:
            self.metadata_df = pd.DataFrame({
                'SizeT': self.SizeT,
                'SizeZ': self.SizeZ,
                'TimeIncrement': self.TimeIncrement,
                'PhysicalSizeZ': self.PhysicalSizeZ,
                'PhysicalSizeY': self.PhysicalSizeY,
                'PhysicalSizeX': self.PhysicalSizeX,
                'segmSizeT': self.segmSizeT
            }, index=['values']).T
            self.metadata_df.index.name = 'Description'
        else:
            self.metadata_df.at['SizeT', 'values'] = self.SizeT
            self.metadata_df.at['SizeZ', 'values'] = self.SizeZ
            self.metadata_df.at['TimeIncrement', 'values'] = self.TimeIncrement
            self.metadata_df.at['PhysicalSizeZ', 'values'] = self.PhysicalSizeZ
            self.metadata_df.at['PhysicalSizeY', 'values'] = self.PhysicalSizeY
            self.metadata_df.at['PhysicalSizeX', 'values'] = self.PhysicalSizeX
            self.metadata_df.at['segmSizeT', 'values'] = self.segmSizeT
        try:
            self.metadata_df.to_csv(self.metadata_csv_path)
        except PermissionError:
            msg = QMessageBox()
            warn_cca = msg.critical(
                self.parent, 'Permission denied',
                f'The below file is open in another app (Excel maybe?).\n\n'
                f'{self.metadata_csv_path}\n\n'
                'Close file and then press "Ok".',
                msg.Ok
            )
            self.metadata_df.to_csv(self.metadata_csv_path)

        self.saveLastEntriesMetadata()

    def validRuns(self):
        if not os.path.exists(self.spotmaxOutPath):
            return []

        scanner = expFolderScanner()
        run_nums = scanner.runNumbers(self.spotmaxOutPath)
        valid_run_nums = [
            run for run in run_nums
            if scanner.analyseRunNumber(self.spotmaxOutPath, run)[0]
        ]
        return valid_run_nums

    def h5_files(self, run):
        if not os.path.exists(self.spotmaxOutPath):
            return []

        orig_data_h5_filename = f'{run}_0_Orig_data'
        ellip_test_h5_filename = f'{run}_1_ellip_test_data'
        p_test_h5_filename = f'{run}_2_p-_test_data'
        p_ellip_test_h5_filename = f'{run}_3_p-_ellip_test_data'
        spotSize_h5_filename = f'{run}_4_spotFIT_data'
        h5_files = []
        for file in utils.listdir(self.spotmaxOutPath):
            _, ext = os.path.splitext(file)
            if ext != '.h5':
                continue
            if file.find(orig_data_h5_filename) != -1:
                h5_files.append(file)
            elif file.find(ellip_test_h5_filename) != -1:
                h5_files.append(file)
            elif file.find(p_test_h5_filename) != -1:
                h5_files.append(file)
            elif file.find(p_ellip_test_h5_filename) != -1:
                h5_files.append(file)
            elif file.find(spotSize_h5_filename) != -1:
                h5_files.append(file)
        return natsorted(h5_files)

    def skeletonize(self, dataToSkel):
        if self.SizeT > 1:
            self.skelCoords = []
            for data in dataToSkel:
                skelCoords = core.skeletonize(data, is_zstack=self.SizeZ>1)
                self.skelCoords.append(skelCoords)
        else:
            skelCoords = core.skeletonize(dataToSkel, is_zstack=self.SizeZ>1)
            self.skelCoords = skelCoords

    def contours(self, dataToCont):
        if self.SizeT > 1:
            self.contCoords = []
            self.contScatterCoords = []
            for data in dataToCont:
                contCoords = core.findContours(data, is_zstack=self.SizeZ>1)
                self.contCoords.append(contCoords)
                contScatterCoords = self.scatterContCoors(contCoords)
                self.contScatterCoords.append(contScatterCoords)
        else:
            contCoords = core.findContours(dataToCont, is_zstack=self.SizeZ>1)
            self.contCoords = contCoords
            self.contScatterCoords = self.scatterContCoors(contCoords)

    def scatterContCoors(self, contCoords):
        contScatterCoords = {}
        for z, allObjContours in contCoords.items():
            xx_cont = []
            yy_cont = []
            for objID, contours_li in allObjContours.items():
                for cont in contours_li:
                    xx_cont.extend(cont[:,0])
                    yy_cont.extend(cont[:,1])
            contScatterCoords[z] = (np.array(xx_cont), np.array(yy_cont))
        return contScatterCoords

    @staticmethod
    def BooleansTo0s1s(acdc_df, csv_path=None, inplace=True):
        """
        Function used to convert "FALSE" strings and booleans to 0s and 1s
        to avoid pandas interpreting as strings or numbers
        """
        if not inplace:
            acdc_df = acdc_df.copy()
        colsToCast = ['is_cell_dead', 'is_cell_excluded']
        for col in colsToCast:
            isInt = pd.api.types.is_integer_dtype(acdc_df[col])
            isFloat = pd.api.types.is_float_dtype(acdc_df[col])
            isObject = pd.api.types.is_object_dtype(acdc_df[col])
            isString = pd.api.types.is_string_dtype(acdc_df[col])
            isBool = pd.api.types.is_bool_dtype(acdc_df[col])
            if isFloat or isBool:
                acdc_df[col] = acdc_df[col].astype(int)
            elif isString or isObject:
                acdc_df[col] = (acdc_df[col].str.lower() == 'true').astype(int)
        if csv_path is not None:
            acdc_df.to_csv(csv_path)
        return acdc_df

    def intToBoolean(self, acdc_df):
        colsToCast = ['is_cell_dead', 'is_cell_excluded']
        for col in colsToCast:
            acdc_df[col] = acdc_df[col] > 0
        return acdc_df


    def criticalExtNotValid(self):
        err_title = f'File extension {self.ext} not valid.'
        err_msg = (
            f'The requested file {self.relPath}\n'
            'has an invalid extension.\n\n'
            'Valid extensions are .tif, .tiff, .npy or .npz'
        )
        if self.parent is None:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            raise FileNotFoundError(err_title)
        else:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            msg = QMessageBox()
            msg.critical(self.parent, err_title, err_msg, msg.Ok)
            return None

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QStyleFactory

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    print('Searching experiment folders...')
    homePath = r'G:\My Drive\1_MIA_Data\Anika\Mutants\Petite'
    # homePath = r'G:\My Drive\1_MIA_Data\Anika\WTs\SCD'
    pathScanner = expFolderScanner(
        homePath = homePath
    )


    t0 = time.time()
    pathScanner.getExpPaths(pathScanner.homePath)
    t1 = time.time()

    print((t1-t0)*1000)

    t0 = time.time()
    pathScanner.infoExpPaths(pathScanner.expPaths)
    t1 = time.time()

    print((t1-t0)*1000)

    print(pathScanner.paths.keys())

    # selectedRunPaths = pathScanner.paths[1]
    # for exp_path, expInfo in selectedRunPaths.items():
    #     print('------------------------------------------')
    #     print(exp_path)
    #     pprint(expInfo)

    pathScanner.input()
