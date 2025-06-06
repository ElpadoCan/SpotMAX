import sys
import os
import re
import subprocess

from importlib import import_module
from functools import wraps

import numpy as np
import pandas as pd
import h5py

import skimage.io

from qtpy.QtCore import Signal, QObject, QRunnable, QMutex, QWaitCondition

from cellacdc.workers import worker_exception_handler, workerLogger
from cellacdc import load as acdc_load

from . import io, utils
from . import transformations
from . import printl

"""
QRunnables or QObjects that run in QThreadPool or QThread in a PyQT app
example of usage:

    self.progressWin = acdc_apps.QDialogWorkerProgress(
        title='Loading data...', parent=self,
        pbarDesc=f'Loading "{channelDataPath}"...'
    )
    self.progressWin.show(self.app)
    self.startLoadDataWorker()

def startLoadDataWorker(self):
    worker = qtworkers.loadDataWorker(self)
    worker.signals.finished.connect(self.loadDataWorkerFinished)
    worker.signals.progress.connect(self.workerProgress)
    worker.signals.initProgressBar.connect(self.workerInitProgressbar)
    worker.signals.progressBar.connect(self.workerUpdateProgressbar)
    worker.signals.critical.connect(self.workerCritical)
    self.threadPool.start(worker)

def loadDataWorkerFinished(self):
    self.progressWin.workerFinished = True
    self.progressWin.close()
    ... more code
"""

class signals(QObject):
    finished = Signal(object)
    finishedNextStep = Signal(object, str, str)
    progress = Signal(str, object)
    sigLoadedData = Signal(object, object, str, str)
    initProgressBar = Signal(int)
    progressBar = Signal(int)
    critical = Signal(object)
    debug = Signal(object)
    sigLoadingNewChunk = Signal(object)

class AnalysisWorker(QRunnable):
    def __init__(self, ini_filepath, is_tempfile, log_filepath: os.PathLike=''):
        QRunnable.__init__(self)
        self.signals = signals()
        self._ini_filepath = ini_filepath
        self._is_tempfile = is_tempfile
        self._log_filepath = log_filepath
        self.logger = workerLogger(self.signals.progress)

    def getCommandForClipboard(self):
        command = f'{self.cli_command()},'
        command_cp = re.sub(r'-p, (.*?),', r'-p, "\1"', command)
        command_cp = re.sub(r'-l, (.*?),', '', command_cp)
        command_cp = command_cp.replace(',', '')
        return command_cp
    
    def cli_command(self):
        command = f'spotmax, -p, {self._ini_filepath}'
        if self._log_filepath:
            command = f'{command}, -l, {self._log_filepath}'
        return command
    
    @worker_exception_handler
    def run(self):
        from . import _process
        command = self.cli_command()
        
        self.logger.log(f'Full command: {command}')
        
        # command = r'python, spotmax\test.py'
        command_format = self.getCommandForClipboard()
        self.logger.log(f'SpotMAX analysis started with command `{command_format}`')
        args = [sys.executable, _process.__file__, '-c', command]
        subprocess.run(args)
        run_number = io.get_run_number_from_ini_filepath(self._ini_filepath)
        self.signals.finished.emit(
            (self._ini_filepath, self._is_tempfile, run_number)
        )

class ComputeAnalysisStepWorker(QRunnable):
    def __init__(self, module_func, anchor, **kwargs):
        QRunnable.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.kwargs = kwargs
        self.anchor = anchor
        self.module_func = module_func
    
    @worker_exception_handler
    def run(self):
        self.logger.log('')
        self.logger.log(f'Computing analysis step...')
        module_name, func_name = self.module_func.rsplit('.', 1)
        module = import_module(f'spotmax.{module_name}')
        func = getattr(module, func_name)
        self.kwargs['logger_func'] = self.logger.log
        
        result = func(**self.kwargs)
        output = (
            result, 
            self.kwargs.get('image'),
            self.kwargs.get('lab'),
            self.anchor
        )
        self.signals.finished.emit(output)

class LoadImageWorker(QRunnable):
    def __init__(
            self, filepath='', channel='', images_path='', 
            loop_to_exist_on_finished=None
        ):
        QRunnable.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self._filepath = filepath
        self._channel = channel
        self._images_path = images_path
        self._loop = loop_to_exist_on_finished
    
    @worker_exception_handler
    def run(self):
        if not self._filepath and not self._channel:
            raise FileNotFoundError(
                'Neither a file path or a channel name was provided '
                'to the worker.'
            )
        
        if self._filepath:
            filepath = self._filepath
            channel = ''
        else:
            images_path = self._images_path
            channel = self._channel
            filepath = acdc_load.get_filename_from_channel(images_path, channel)
        
        self.logger.log(f'Loading image data from {filepath}...')
        image_data = acdc_load.load_image_file(filepath)
        self.signals.finished.emit(
            (self, filepath, channel, image_data, self._loop)
        )

class Runnable(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
    
    def emitDebugSignal(self, to_debug):
        self.mutex.lock()
        self.signals.debug.emit((*to_debug, self))
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

class DummyThread(QObject):
    def __init__(self):
        super().__init__()
    
    def quit(self):
        pass
    
    def deleteLater(self):
        return super().deleteLater()

class TuneKernelWorker(Runnable):
    def __init__(self, kernel):
        super().__init__()
        self._kernel = kernel
    
    @worker_exception_handler
    def run(self):
        print('\n')
        self.logger.log('Running auto-tuning process...')
        result = self._kernel.run(
            logger_func=self.logger.log, emitDebug=self.emitDebugSignal
        )
        
        self.signals.finished.emit(result)
    
    def thread(self):
        return DummyThread()
    
    def deleteLater(self):
        pass

class CropImageBasedOnSegmDataWorker(QRunnable):
    def __init__(
            self, image_data, segm_data, delta_tolerance, SizeZ, 
            on_finished_callback, nnet_input_data=None, 
            extend_segm_3D_range=None
        ):
        QRunnable.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.image_data = image_data
        self.segm_data = segm_data
        self.nnet_input_data = nnet_input_data
        self.delta_tolerance = delta_tolerance
        self.SizeZ = SizeZ
        self.on_finished_callback = on_finished_callback
        self.extend_segm_3D_range = extend_segm_3D_range
    
    def _add_missing_axis(self):
        # Add axis for Z if missing (2D image data or 2D segm data)
        self.logger.log('Cropping based on segm data...')
        segm_data = self.segm_data
        image_data = self.image_data
        nnet_input_data = self.nnet_input_data
        if image_data.ndim == 3:
            image_data = image_data[:, np.newaxis]
        
        if nnet_input_data is not None:
            if nnet_input_data.ndim == 3:
                nnet_input_data = nnet_input_data[:, np.newaxis]
        
        if segm_data.ndim == 3:
            if self.SizeZ == 1:
                segm_data = segm_data[:, np.newaxis]
            else:
                T, Y, X = self.segm_data.shape
                new_shape = (T, self.SizeZ, Y, X)
                tiled_segm_data = np.zeros(new_shape, dtype=segm_data.dtype)
                for frame_i, lab in enumerate(segm_data):
                    tiled_segm_data[frame_i, :] = lab
                segm_data = tiled_segm_data
        
        if self.extend_segm_3D_range is not None:
            segm_data = transformations.extend_3D_segm_in_z(
                segm_data, self.extend_segm_3D_range, errors='ignore'
            )
        
        return segm_data, image_data, nnet_input_data
    
    @worker_exception_handler
    def run(self):
        segm_data, image_data, nnet_input_data = self._add_missing_axis()
        crop_info = transformations.crop_from_segm_data_info(
            segm_data, self.delta_tolerance
        )
        segm_slice, pad_widths, crop_to_global_coords = crop_info
        
        if not np.any(self.segm_data):
            segm_data = np.ones_like(segm_data)
            result = (
                image_data, segm_data, crop_to_global_coords, 
                self.on_finished_callback, nnet_input_data
            )
            self.signals.finished.emit(result)
            return
        
        image_cropped = image_data[segm_slice]
        segm_data_cropped = segm_data[segm_slice]
        nnet_input_data_cropped = None
        if nnet_input_data is not None:
            nnet_input_data_cropped = nnet_input_data[segm_slice]
        result = (
            image_cropped, segm_data_cropped, crop_to_global_coords, 
            self.on_finished_callback, nnet_input_data_cropped
        )
        self.signals.finished.emit(result)

class pathScannerWorker(QRunnable):
    def __init__(self, selectedPath):
        QRunnable.__init__(self)
        self.signals = signals()
        self.selectedPath = selectedPath

    @worker_exception_handler
    def run(self):
        selectedPath = self.selectedPath
        areDirsPosFolders = [
            f.find('Position_')!=-1 and os.path.isdir(os.path.join(selectedPath, f))
            for f in utils.listdir(selectedPath)
        ]
        is_selectedPath = any(areDirsPosFolders)

        pathScanner = io.expFolderScanner(selectedPath)
        if is_selectedPath:
            pathScanner.expPaths = [selectedPath]
        else:
            pathScanner.getExpPaths(
                pathScanner.homePath, signals=self.signals
            )
            numExps = len(pathScanner.expPaths)
            self.signals.progress.emit(
                f'Number of valid experiments found = {numExps}',
                'INFO'
            )

        self.signals.initProgressBar.emit(len(pathScanner.expPaths))
        pathScanner.infoExpPaths(pathScanner.expPaths, signals=self.signals)
        self.signals.finished.emit(pathScanner)

class loadDataWorker(QRunnable):
    def __init__(self, mainWin, selectedPos, selectedExpName):
        QRunnable.__init__(self)
        self.signals = signals()
        self.selectedPos = selectedPos
        self.selectedExpName = selectedExpName
        self.mainWin = mainWin

    @worker_exception_handler
    def run(self):
        expInfo = self.mainWin.expPaths[self.selectedExpName]

        posDataRef = self.mainWin.posDataRef
        channelDataPaths = expInfo['channelDataPaths'][:posDataRef.loadSizeS]

        user_ch_name = self.mainWin.user_ch_name
        logger = self.mainWin.logger
        dataSide = self.mainWin.expData[self.mainWin.lastLoadedSide]
        self.signals.initProgressBar.emit(len(channelDataPaths))
        for channelDataPath in channelDataPaths:
            posFoldername = channelDataPath.parents[1].name
            skipPos = (
                self.selectedPos is not None
                and not posFoldername == self.selectedPos
            )
            if skipPos:
                # To avoid memory issues we load single pos for time-lapse and
                # all pos for static data
                continue

            posData = io.loadData(channelDataPath, user_ch_name)
            self.signals.progress.emit(
                f'Loading {posData.relPath}...',
                'INFO'
            )
            posData.loadSizeS = posDataRef.loadSizeS
            posData.loadSizeT = posDataRef.loadSizeT
            posData.loadSizeZ = posDataRef.loadSizeZ
            posData.SizeT = posDataRef.SizeT
            posData.SizeZ = posDataRef.SizeZ
            posData.getBasenameAndChNames(load=False)
            posData.buildPaths()
            posData.loadChannelData()
            posData.loadOtherFiles(
                load_segm_data=True,
                load_acdc_df=True,
                loadSegmInfo=True,
                load_last_tracked_i=True,
                load_metadata=True,
                load_ref_ch_mask=True,
                endNameSegm=self.mainWin.selectedSegmEndame
            )
            if posDataRef.SizeZ > 1:
                SizeZ = posData.chData_shape[-3]
                posData.SizeZ = SizeZ
            else:
                posData.SizeZ = 1

            posData.TimeIncrement = posDataRef.TimeIncrement
            posData.PhysicalSizeZ = posDataRef.PhysicalSizeZ
            posData.PhysicalSizeY = posDataRef.PhysicalSizeY
            posData.PhysicalSizeX = posDataRef.PhysicalSizeX
            posData.saveMetadata()

            posData.computeSegmRegionprops()

            logger.info(f'Channel data shape = {posData.chData_shape}')
            logger.info(f'Loaded data shape = {posData.chData.shape}')
            # logger.info(f'Metadata:')
            # logger.info(posData.metadata_df)

            dataSide.append(posData)

            self.signals.progressBar.emit(1)

        self.signals.finished.emit(None)

class LazyLoaderWorker(QObject):
    sigLoadingFinished = Signal(object)

    def __init__(self, mutex, waitCond, readH5mutex, waitReadH5cond):
        QObject.__init__(self)
        self.signals = signals()
        self.mutex = mutex
        self.waitCond = waitCond
        self.exit = False
        self.sender = None
        self.H5readWait = False
        self.waitReadH5cond = waitReadH5cond
        self.readH5mutex = readH5mutex
        self.isFinished = False

    def setArgs(self, posData, current_idx, axis, updateImgOnFinished):
        self.wait = False
        self.updateImgOnFinished = updateImgOnFinished
        self.posData = posData
        self.current_idx = current_idx
        self.axis = axis

    def pauseH5read(self):
        self.readH5mutex.lock()
        self.waitReadH5cond.wait(self.mutex)
        self.readH5mutex.unlock()

    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.signals.progress.emit(
                    'Closing lazy loader...', 'INFO'
                )
                break
            elif self.wait:
                self.signals.progress.emit(
                    'Lazy loader paused.', 'INFO'
                )
                self.pause()
            else:
                self.signals.progress.emit(
                    'Lazy loader resumed.', 'INFO'
                )
                self.posData.loadChannelDataChunk(
                    self.current_idx, axis=self.axis, worker=self
                )
                self.sigLoadingFinished.emit(self.side)
                self.wait = True

        self.signals.finished.emit(None)
        self.isFinished = True

class LoadH5StoreWorker(QRunnable):
    def __init__(self, expData, h5_filename, side):
        QRunnable.__init__(self)
        self.signals = signals()
        self.expData = expData
        self.h5_filename = h5_filename
        self.side = side

    @worker_exception_handler
    def run(self):
        for posData in self.expData[self.side]:
            h5_path = os.path.join(
                posData.spotmaxOutPath, self.h5_filename
            )
            if not os.path.exists(h5_path):
                posData.hdf_store = None
                self.progress.emit(
                    f'WARNING: {self.h5_filename} not found '
                    f'in {posData.spotmaxOutPath}',
                    'WARNING'
                )
                continue

            posData.h5_path = h5_path
            posData.hdf_store = pd.HDFStore(posData.h5_path, mode='r')
            self.signals.progressBar.emit(1)
        self.signals.finished.emit(self.side)

class LoadRelFilenameDataWorker(QRunnable):
    """
    Load data given a list of relative filenames
    (filename without the common basename)
    """
    def __init__(self, expData, relFilenames, side, nextStep):
        QRunnable.__init__(self)
        self.signals = signals()
        self.expData = expData
        self.relFilenames = relFilenames
        self.side = side
        self.nextStep = nextStep

    @worker_exception_handler
    def run(self):
        for posData in self.expData[self.side]:
            for relFilename in self.relFilenames:
                if relFilename in posData.loadedRelativeFilenamesData:
                    continue
                filepath = posData.absoluteFilepath(relFilename)
                filename = os.path.basename(filepath)
                self.signals.progress.emit(f'Loading {filepath}...', 'INFO')
                ext = os.path.splitext(filename)[1]
                if ext == '.tif':
                    data = skimage.io.imread(filepath)
                elif ext == '.npy':
                    data = np.load(filepath)
                elif ext == '.npz':
                    data = np.load(filepath)['arr_0']
                elif ext == '.h5':
                    h5f = h5py.File(filepath, 'r')
                    data = h5f['data']
                self.signals.sigLoadedData.emit(
                    posData, data, relFilename, self.nextStep
                )
            self.signals.progressBar.emit(1)
        self.signals.finishedNextStep.emit(
            self.side, self.nextStep, self.relFilenames[0]
        )

class skeletonizeWorker(QRunnable):
    def __init__(self, expData, side, initFilename=False):
        QRunnable.__init__(self)
        self.signals = signals()
        self.expData = expData
        self.side = side
        self.initFilename = initFilename

    @worker_exception_handler
    def run(self):
        for posData in self.expData[self.side]:
            if self.initFilename:
                relFilename = list(posData.loadedRelativeFilenamesData)[0]
                posData.skeletonizedRelativeFilename = relFilename
            relFilename = posData.skeletonizedRelativeFilename
            filepath = posData.absoluteFilepath(relFilename)
            filename = os.path.basename(filepath)
            dataToSkel = posData.loadedRelativeFilenamesData[relFilename]

            self.signals.progress.emit(f'Skeletonizing {filepath}...', 'INFO')
            posData.skeletonize(dataToSkel)

            self.signals.progressBar.emit(1)
        self.signals.finished.emit(self.side)


class findContoursWorker(QRunnable):
    def __init__(self, expData, side, initFilename=False):
        QRunnable.__init__(self)
        self.signals = signals()
        self.expData = expData
        self.side = side
        self.initFilename = initFilename

    @worker_exception_handler
    def run(self):
        for posData in self.expData[self.side]:
            if self.initFilename:
                relFilename = list(posData.loadedRelativeFilenamesData)[0]
                posData.contouredRelativeFilename = relFilename
            relFilename = posData.contouredRelativeFilename
            filepath = posData.absoluteFilepath(relFilename)
            filename = os.path.basename(filepath)
            dataToCont = posData.loadedRelativeFilenamesData[relFilename]

            self.signals.progress.emit(
                f'Computing contour of {filepath}...', 'INFO'
            )
            posData.contours(dataToCont)

            self.signals.progressBar.emit(1)
        self.signals.finished.emit(self.side)

class PreprocessNnetDataAcrossExpWorker(QRunnable):
    def __init__(
            self, exp_path, pos_foldernames, spots_ch_endname, nnet_model, 
            loop_to_exist_on_finished=None
        ):
        QRunnable.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self._loop = loop_to_exist_on_finished
        self.nnet_model = nnet_model
        self.exp_path = exp_path
        self.pos_foldernames = pos_foldernames
        self.spots_ch_endname = spots_ch_endname
    
    @worker_exception_handler
    def run(self):
        transformed_data = transformations.load_preprocess_nnet_data_across_exp(
            self.exp_path, self.pos_foldernames, self.spots_ch_endname, 
            self.nnet_model
        )
        self.signals.finished.emit(
            (self, transformed_data, self._loop)
        )

class PreprocessNnetDataAcrossTimeWorker(QRunnable):
    def __init__(
            self, input_data, nnet_model, loop_to_exist_on_finished=None
        ):
        QRunnable.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self._loop = loop_to_exist_on_finished
        self.nnet_model = nnet_model
        self.input_data = input_data
    
    @worker_exception_handler
    def run(self):
        transformed_data = self.nnet_model.preprocess(self.input_data)
        self.signals.finished.emit(
            (self, transformed_data, self._loop)
        )
    
class ComputeFeaturesWorker(QRunnable):
    def __init__(self, **features_kwargs):
        QRunnable.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.features_kwargs = features_kwargs
    
    @worker_exception_handler
    def run(self):
        self.logger.log('')
        self.logger.log(f'Computing features...')
        self.signals.finished.emit(None)