from functools import partial
import os
import shutil
import datetime
import traceback
import re
from queue import Queue

import numpy as np
import pandas as pd

from qtpy.QtCore import (
    Qt, QTimer, QThreadPool, QMutex, QWaitCondition, QEvent
)
from qtpy.QtGui import QIcon, QGuiApplication
from qtpy.QtWidgets import QDockWidget, QToolBar, QAction, QAbstractSlider

# Interpret image data as row-major instead of col-major
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
try:
    import numba
    pg.setConfigOption("useNumba", True)
except Exception as e:
    pass

try:
    import cupy as cp
    pg.setConfigOption("useCupy", True)
except Exception as e:
    pass

import cellacdc
cellacdc.GUI_INSTALLED = True

from cellacdc import gui as acdc_gui
from cellacdc import apps as acdc_apps
from cellacdc import widgets as acdc_widgets
from cellacdc import exception_handler
from cellacdc import load as acdc_load

from . import qtworkers, io, printl, dialogs
from . import logs_path, html_path, html_func
from . import widgets, config
from . import transformations

from . import qrc_resources_spotmax

ANALYSIS_STEP_RESULT_SLOTS = {
    'gaussSigma': '_displayGaussSigmaResult',
    'removeHotPixels': '_displayRemoveHotPixelsResult',
    'sharpenSpots': '_displaySharpenSpotsResult',
    'spotPredictionMethod': '_displayspotPredictionResult',
    'refChThresholdFunc': '_displayspotSegmRefChannelResult'
}

PARAMS_SLOTS = {
    'gaussSigma': ('sigComputeButtonClicked', '_computeGaussSigma'),
    'removeHotPixels': ('sigComputeButtonClicked', '_computeRemoveHotPixels'),
    'sharpenSpots': ('sigComputeButtonClicked', '_computeSharpenSpots'),
    'spotPredictionMethod': ('sigComputeButtonClicked', '_computeSpotPrediction'),
    'refChThresholdFunc': ('sigComputeButtonClicked', '_computeSegmentRefChannel')
}

class spotMAX_Win(acdc_gui.guiWin):
    def __init__(
            self, app, debug=False, parent=None, buttonToRestore=None, 
            mainWin=None, executed=False, version=None
        ):
        super().__init__(
            app, parent=parent, buttonToRestore=buttonToRestore, 
            mainWin=mainWin, version=version
        )

        self._version = version
        self._appName = 'spotMAX'
    
    def run(self, module='spotmax_gui', logs_path=logs_path):
        super().run(module=module, logs_path=logs_path)

        self.setWindowTitle("spotMAX - GUI")
        self.setWindowIcon(QIcon(":icon_spotmax.ico"))

        self.initGui()
        self.createThreadPool()
    
    def createThreadPool(self):
        self.maxThreads = QThreadPool.globalInstance().maxThreadCount()
        self.threadCount = 0
        self.threadQueue = Queue()
        self.threadPool = QThreadPool.globalInstance()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            posData = self.data[self.pos_i]
            return
        super().keyPressEvent(event)
    
    def gui_setCursor(self, modifiers, event):
        cursorsInfo = super().gui_setCursor(modifiers, event)
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        
        setAutoTuneCursor = (
            self.isAutoTuneRunning and not event.isExit()
            and noModifier
        )
        cursorsInfo['setAutoTuneCursor'] = setAutoTuneCursor
        overrideCursor = self.app.overrideCursor()
        if setAutoTuneCursor and overrideCursor is None:
            self.app.setOverrideCursor(Qt.CrossCursor)
        return cursorsInfo
        
    def gui_hoverEventImg1(self, event, isHoverImg1=True):
        cursorsInfo = super().gui_hoverEventImg1(event, isHoverImg1=isHoverImg1)
        if cursorsInfo is None:
            return
        
        if event.isExit():
            return
        
        x, y = event.pos()
        if cursorsInfo['setAutoTuneCursor']:
            self.setHoverCircleAutoTune(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
    
    def getIDfromXYPos(self, x, y):
        posData = self.data[self.pos_i]
        xdata, ydata = int(x), int(y)
        Y, X = self.get_2Dlab(posData.lab).shape
        if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            return ID
        else:
            return
    
    @exception_handler
    def gui_mousePressEventImg1(self, event):
        super().gui_mousePressEventImg1(event)
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        posData = self.data[self.pos_i]
        left_click = event.button() == Qt.MouseButton.LeftButton and not alt
        
        canAddPointAutoTune = self.isAutoTuneRunning and left_click
        
        x, y = event.pos().x(), event.pos().y()
        ID = self.getIDfromXYPos(x, y)
        if ID is None:
            return
        
        if canAddPointAutoTune:
            self.addAutoTunePoint(x, y)
        
        
    def gui_createRegionPropsDockWidget(self):
        super().gui_createRegionPropsDockWidget(side=Qt.RightDockWidgetArea)
        self.gui_createParamsDockWidget()
    
    def gui_createParamsDockWidget(self):
        self.computeDockWidget = QDockWidget('spotMAX Tab Control', self)
        computeTabControl = dialogs.guiTabControl(
            parent=self.computeDockWidget, logging_func=self.logger.info
        )
        computeTabControl.addAutoTuneTab()
        computeTabControl.initState(False)
        computeTabControl.currentChanged.connect(self.tabControlPageChanged)

        self.computeDockWidget.setWidget(computeTabControl)
        self.computeDockWidget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable 
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.computeDockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )

        self.addDockWidget(Qt.LeftDockWidgetArea, self.computeDockWidget)
        
        self.connectAutoTuneSlots()
        self.initAutoTuneColors()
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        self.LeftClickButtons.append(autoTuneTabWidget.autoTuningButton)
    
    def gui_createShowPropsButton(self):
        super().gui_createShowPropsButton(side='right') 
        self.gui_createShowParamsDockButton()

    def gui_createShowParamsDockButton(self):
        self.showParamsDockButton = acdc_widgets.expandCollapseButton()
        self.showParamsDockButton.setToolTip('Analysis parameters')
        self.showParamsDockButton.setFocusPolicy(Qt.NoFocus)
        self.leftSideDocksLayout.addWidget(self.showParamsDockButton)
        
    def gui_connectActions(self):
        super().gui_connectActions()
        self.showParamsDockButton.sigClicked.connect(self.showComputeDockWidget)
        self.computeDockWidget.widget().sigRunAnalysis.connect(
            self.runAnalysis
        )
        self.addSpotsCoordinatesAction.triggered.connect(
            self.addSpotsCoordinatesTriggered
        )
    
    def addSpotsCoordinatesTriggered(self):
        posData = self.data[self.pos_i]
        h5files = posData.getSpotmaxH5files()
        self.spotsItems.setPosition(posData.spotmax_out_path)
        toolbutton = self.spotsItems.addLayer(h5files)
        if toolbutton is None:
            self.logger.info(
                'Add spots layer process cancelled.'
            )
            return
        self.spotmaxToolbar.addWidget(toolbutton)
        self.ax1.addItem(toolbutton.item)

        self.spotsItems.setData(
            posData.frame_i, toolbutton=toolbutton,
            z=self.currentZ(checkIfProj=True)
        )
    
    def currentZ(self, checkIfProj=True):
        if checkIfProj and self.zProjComboBox.currentText() != 'single z-slice':
            return
        
        return self.zSliceScrollBar.sliderPosition()
        
    def _setWelcomeText(self):
        html_filepath = os.path.join(html_path, 'gui_welcome.html')
        with open(html_filepath) as html_file:
            htmlText = html_file.read()
        self.ax1.infoTextItem.setHtml(htmlText)
    
    def _disableAcdcActions(self, *actions):
        for action in actions:
            action.setVisible(False)
            action.setDisabled(True)
    
    def reInitGui(self):
        super().reInitGui()
        try:
            self.disconnectParamsGroupBoxSignals()
        except Exception as e:
            # printl(traceback.format_exc())
            pass
        self.showParamsDockButton.setDisabled(False)
        self.computeDockWidget.widget().initState(False)
        

    def initGui(self):
        self._setWelcomeText()
        self._disableAcdcActions(
            self.newAction, self.manageVersionsAction, self.openFileAction
        )
        self.ax2.hide()
    
    def showComputeDockWidget(self, checked=False):
        if self.showParamsDockButton.isExpand:
            self.computeDockWidget.setVisible(False)
        else:
            self.computeDockWidget.setVisible(True)
            self.computeDockWidget.setEnabled(True)

    def _loadFromExperimentFolder(self, selectedPath):
        # Replaces cellacdc.gui._loadFromExperimentFolder
        self.funcDescription = 'scanning experiment paths'
        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title='Path scanner progress', parent=self,
            pbarDesc='Scanning experiment folder...'
        )
        self.progressWin.show(self.app)
        self.pathScanner = io.PathScanner(self, self.progressWin)
        self.pathScanner.start(selectedPath)
        return self.pathScanner.images_paths

    @exception_handler
    def runAnalysis(self, ini_filepath, is_tempfile):
        self.logger.info('Starting spotMAX analysis...')
        self._analysis_started_datetime = datetime.datetime.now()
        self.funcDescription = 'starting analysis process'
        worker = qtworkers.analysisWorker(ini_filepath, is_tempfile)

        worker.signals.finished.connect(self.analysisWorkerFinished)
        worker.signals.progress.connect(self.workerProgress)
        # worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        # worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)
        
    def analysisWorkerFinished(self, args):
        ini_filepath, is_tempfile = args
        self.logger.info('Analysis finished')
        if is_tempfile:
            tempdir = os.path.dirname(ini_filepath)
            self.logger.info(f'Deleting temp folder "{tempdir}"')
            shutil.rmtree(tempdir)
        self._analysis_finished_datetime = datetime.datetime.now()
        delta = self._analysis_finished_datetime-self._analysis_started_datetime
        delta_sec = str(delta).split('.')[0]
        ff = r'%d %b %Y, %H:%M:%S'
        txt = (
            'spotMAX analysis finished!\n\n'
            f'    * Started on: {self._analysis_finished_datetime.strftime(ff)}\n'
            f'    * Ended on: {self._analysis_finished_datetime.strftime(ff)}\n'
            f'    * Total execution time = {delta_sec} H:mm:ss\n'
        )
        line_str = '-'*60
        close_str = '*'*60
        self.logger.info(f'{line_str}\n{txt}\n{close_str}')
        txt = html_func.paragraph(txt.replace('\n', '<br>'))
        msg = acdc_widgets.myMessageBox()
        msg.information(self, 'spotMAX analysis finished!', txt)
    
    def gui_createActions(self):
        super().gui_createActions()

        self.addSpotsCoordinatesAction = QAction(self)
        self.addSpotsCoordinatesAction.setIcon(QIcon(":addPlotSpots.svg"))
        self.addSpotsCoordinatesAction.setToolTip('Add plot for spots coordinates')
    
    def gui_createToolBars(self):
        super().gui_createToolBars()

        self.addToolBarBreak(Qt.LeftToolBarArea)
        self.spotmaxToolbar = QToolBar("spotMAX toolbar", self)
        self.spotmaxToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(Qt.LeftToolBarArea, self.spotmaxToolbar)
        self.spotmaxToolbar.addAction(self.addSpotsCoordinatesAction)
        self.spotmaxToolbar.setVisible(False)
        self.spotsItems = widgets.SpotsItems()
    
    def gui_addTopLayerItems(self):
        super().gui_addTopLayerItems()

    def loadingDataCompleted(self):
        super().loadingDataCompleted()
        posData = self.data[self.pos_i]
        self.setWindowTitle(f'spotMAX - GUI - "{posData.exp_path}"')
        self.spotmaxToolbar.setVisible(True)
        self.computeDockWidget.widget().initState(True)
        
        self.setRunNumbers()
        
        self.setAnalysisParameters()
        self.connectParamsGroupBoxSignals()
        self.autoTuningAddItems()
    
    def disconnectParamsGroupBoxSignals(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        for section, params in ParamsGroupBox.params.items():
            for anchor, param in params.items():
                formWidget = param['formWidget']
                signal_slot = PARAMS_SLOTS.get(anchor)
                if signal_slot is None:
                    continue
                formWidget.setComputeButtonConnected(False)
                signal, slot = signal_slot
                signal = getattr(formWidget, signal)
                signal.disconnect()
    
    @exception_handler
    def _computeGaussSigma(self, formWidget):
        self.funcDescription = 'Initial gaussian filter'
        module_func = 'filters.gaussian'
        anchor = 'gaussSigma'
        
        posData = self.data[self.pos_i]
        image = posData.img_data[posData.frame_i]
        
        sigma = formWidget.widget.value()
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        configParams = ParamsGroupBox.params['Configuration']
        use_gpu = configParams['useGpu']['widget'].isChecked()
        
        kwargs = {
            'image': image, 'sigma': sigma, 'use_gpu': use_gpu
        }
        
        self.startComputeAnalysisStepWorker(module_func, anchor, **kwargs)
    
    def startCropImageBasedOnSegmDataWorkder(
            self, image_data, segm_data, on_finished_callback
        ):
        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title=self.funcDescription, parent=self,
            pbarDesc=self.funcDescription
        )
        self.progressWin.mainPbar.setMaximum(0)
        self.progressWin.show(self.app)
            
        posData = self.data[self.pos_i]
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        deltaTolerance = np.array(spots_zyx_radii)
        delta_tolerance = np.ceil(deltaTolerance).astype(int)
        
        worker = qtworkers.CropImageBasedOnSegmDataWorker(
            image_data, segm_data, delta_tolerance, posData.SizeZ,
            on_finished_callback
        )
        worker.signals.finished.connect(self.cropImageWorkerFinished)
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)
    
    def cropImageWorkerFinished(self, result):
        image_cropped, segm_data_cropped, on_finished_callback = result
        
        if on_finished_callback is None:
            return
        
        posData = self.data[self.pos_i]
        image = image_cropped[posData.frame_i]
        func, args, kwargs = on_finished_callback
        kwargs['image'] = image
        if 'lab' in kwargs:
            lab = segm_data_cropped[posData.frame_i]
            if not np.any(lab):
                # Without segm data we evaluate the entire image
                lab = None
            kwargs['lab'] = lab
        func(*args, **kwargs)
    
    @exception_handler
    def _computeRemoveHotPixels(self, formWidget):
        self.funcDescription = 'Remove hot pixels'
        module_func = 'filters.remove_hot_pixels'
        anchor = 'removeHotPixels'
        
        posData = self.data[self.pos_i]
        args = [module_func, anchor]
        kwargs = {}
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            posData.img_data, posData.segm_data, 
            on_finished_callback=on_finished_callback
        )
    
    def setHoverCircleAutoTune(self, x, y):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        size = round(spots_zyx_radii[-1])
        self.setHoverToolSymbolData(
            [x], [y], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            size=size
        )
    
    def tabControlPageChanged(self, index):
        if not self.dataIsLoaded:
            return
        if index == 1:
            # Autotune tab toggled
            self.setAutoTunePointSize()
    
    
    def setAutoTunePointSize(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        size = round(spots_zyx_radii[-1])
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.setAutoTunePointSize(size)
    
    @exception_handler
    def _computeSharpenSpots(self, formWidget):
        self.funcDescription = 'Sharpen spots (DoG filter)'
        module_func = 'filters.DoG_spots'
        anchor = 'sharpenSpots'
        
        posData = self.data[self.pos_i]
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        
        configParams = ParamsGroupBox.params['Configuration']
        use_gpu = configParams['useGpu']['widget'].isChecked()
        args = [module_func, anchor]
        kwargs = {
            'spots_zyx_radii': spots_zyx_radii, 'use_gpu': use_gpu
        }
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            posData.img_data, posData.segm_data, 
            on_finished_callback=on_finished_callback
        )
    
    @exception_handler
    def _computeSpotPrediction(self, formWidget):
        self.funcDescription = 'Spots location semantic segmentation'
        module_func = 'pipe.spots_semantic_segmentation'
        anchor = 'spotPredictionMethod'
        
        posData = self.data[self.pos_i]
        lineage_table = None
        if posData.acdc_df is not None:
            lineage_table = posData.acdc_df.loc[posData.frame_i]
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        initial_sigma = preprocessParams['gaussSigma']['widget'].value()
        
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        do_sharpen = preprocessParams['sharpenSpots']['widget'].isChecked()
        do_remove_hot_pixels = (
            preprocessParams['removeHotPixels']['widget'].isChecked()
        )
        do_aggregate = preprocessParams['aggregate']['widget'].isChecked()
        
        configParams = ParamsGroupBox.params['Configuration']
        use_gpu = configParams['useGpu']['widget'].isChecked()
        
        args = [module_func, anchor]
        kwargs = {
            'lab': None, 'initial_sigma': initial_sigma, 
            'spots_zyx_radii': spots_zyx_radii, 'do_sharpen': do_sharpen, 
            'do_remove_hot_pixels': do_remove_hot_pixels,
            'lineage_table': lineage_table, 'do_aggregate': do_aggregate, 
            'use_gpu': use_gpu
        }
        
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            posData.img_data, posData.segm_data, 
            on_finished_callback=on_finished_callback
        )
    
    def askReferenceChannelEndname(self):
        posData = self.data[self.pos_i]
        selectChannelWin = acdc_widgets.QDialogListbox(
            'Select channel to load',
            'Selec <b>reference channel</b> name:\n',
            posData.chNames, multiSelection=False, parent=self
        )
        selectChannelWin.exec_()
        if selectChannelWin.cancel:
            return
        return selectChannelWin.selectedItemsText[0]
    
    def loadImageDataFromChannelName(self, channel):
        posData = self.data[self.pos_i]
        images_path = posData.images_path
        filepath = acdc_load.get_filename_from_channel(images_path, channel)
        printl(filepath)
        if not filepath:
            raise FileNotFoundError(f'{channel} channel not found in {images_path}')
        filename_ext = os.path.basename(filepath)
        filename, ext = os.path.splitext(filename_ext)
        imgData = posData.fluo_data_dict.get(filename)
        if imgData is None:
            imgData = acdc_load.load_image_file(filepath)
            if posData.SizeT == 1:
                imgData = imgData[np.newaxis]
        return imgData
    
    @exception_handler
    def _computeSegmentRefChannel(self, formWidget):
        posData = self.data[self.pos_i]
        
        self.funcDescription = 'Reference channel semantic segmentation'
        module_func = 'pipe.reference_channel_semantic_segm'
        anchor = 'refChThresholdFunc'
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        filePathParams = ParamsGroupBox.params['File paths and channels']
        refChEndName = filePathParams['refChEndName']['widget'].text()
        if not refChEndName:
            refChEndName = self.askReferenceChannelEndname()
            if refChEndName is None:
                self.logger.info('Segmenting reference channel cancelled.')
        
        self.logger.info(f'Loading "{refChEndName}" reference channel data...')
        refChannelData = self.loadImageDataFromChannelName(refChEndName)        
        
        lineage_table = None
        if posData.acdc_df is not None:
            lineage_table = posData.acdc_df.loc[posData.frame_i]
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        initial_sigma = preprocessParams['gaussSigma']['widget'].value()
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        do_remove_hot_pixels = (
            preprocessParams['removeHotPixels']['widget'].isChecked()
        )
        do_aggregate = preprocessParams['aggregate']['widget'].isChecked()
        
        configParams = ParamsGroupBox.params['Configuration']
        use_gpu = configParams['useGpu']['widget'].isChecked()
        
        args = [module_func, anchor]
        kwargs = {
            'lab': None, 'initial_sigma': initial_sigma, 
            'do_remove_hot_pixels': do_remove_hot_pixels,
            'lineage_table': lineage_table, 'do_aggregate': do_aggregate, 
            'use_gpu': use_gpu
        }
        
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            refChannelData, posData.segm_data, 
            on_finished_callback=on_finished_callback
        )
    
    def _displayGaussSigmaResult(self, result, image):
        from cellacdc.plot import imshow
        posData = self.data[self.pos_i]
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        anchor = 'gaussSigma'
        sigma = preprocessParams[anchor]['widget'].value()
        titles = ['Raw image', f'Filtered image (sigma = {sigma})']
        imshow(
            image, result, axis_titles=titles, parent=self, 
            window_title='Pre-processing - Gaussian filter',
            color_scheme=self._colorScheme
        )
    
    def _displayRemoveHotPixelsResult(self, result, image):
        from cellacdc.plot import imshow
        
        titles = ['Raw image', f'Hot pixels removed']
        window_title = 'Pre-processing - Remove hot pixels'
        imshow(
            image, result, axis_titles=titles, parent=self, 
            window_title=window_title, color_scheme=self._colorScheme
        )
    
    def _displaySharpenSpotsResult(self, result, image):
        from cellacdc.plot import imshow
        
        titles = ['Raw image', f'Sharpened (DoG filter)']
        window_title = 'Pre-processing - Sharpening (DoG filter)'
        imshow(
            image, result, axis_titles=titles, parent=self, 
            window_title=window_title, color_scheme=self._colorScheme
        )
    
    def _displayspotPredictionResult(self, result, image):
        from cellacdc.plot import imshow
        posData = self.data[self.pos_i]
        
        titles = list(result.keys())
        titles[0] = 'Input image'
        prediction_images = list(result.values())
        
        window_title = 'Spots channel - Spots segmentation method'
        
        imshow(
            *prediction_images, axis_titles=titles, parent=self, 
            window_title=window_title, color_scheme=self._colorScheme
        )
    
    def _displayspotSegmRefChannelResult(self, result, image):
        from cellacdc.plot import imshow
        posData = self.data[self.pos_i]
        
        titles = list(result.keys())
        titles[0] = 'Input image'
        prediction_images = list(result.values())
        
        window_title = 'Reference channel - Semantic segmentation'
        
        imshow(
            *prediction_images, axis_titles=titles, parent=self, 
            window_title=window_title, color_scheme=self._colorScheme
        )
    
    def startComputeAnalysisStepWorker(self, module_func, anchor, **kwargs):
        if self.progressWin is None:
            self.progressWin = acdc_apps.QDialogWorkerProgress(
                title=self.funcDescription, parent=self,
                pbarDesc=self.funcDescription
            )
            self.progressWin.mainPbar.setMaximum(0)
            self.progressWin.show(self.app)
        
        worker = qtworkers.ComputeAnalysisStepWorker(module_func, anchor, **kwargs)
        worker.signals.finished.connect(self.computeAnalysisStepWorkerFinished)
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        worker.signals.debug.connect(self.workerDebug)
        self.threadPool.start(worker)
    
    def workerDebug(self, to_debug):
        pass
    
    def computeAnalysisStepWorkerFinished(self, output: tuple):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        result, image, anchor = output
        self.logger.info(f'{self.funcDescription} process ended.')
        displayFunc = ANALYSIS_STEP_RESULT_SLOTS[anchor]
        displayFunc = getattr(self, displayFunc)
        displayFunc(result, image)
    
    def connectParamsGroupBoxSignals(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        for section, params in ParamsGroupBox.params.items():
            for anchor, param in params.items():
                formWidget = param['formWidget']
                signal_slot = PARAMS_SLOTS.get(anchor)
                if signal_slot is None:
                    continue
                formWidget.setComputeButtonConnected(True)
                signal, slot = signal_slot
                signal = getattr(formWidget, signal)
                slot = getattr(self, slot)
                signal.connect(slot)
    
    def connectAutoTuneSlots(self):
        self.isAutoTuneRunning = False
        self.isAutoTuningForegr = True
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.sigStartAutoTune.connect(self.startAutoTuning)
        autoTuneTabWidget.sigStopAutoTune.connect(self.stopAutoTuning)
        
        autoTuneTabWidget.sigTrueFalseToggled.connect(
            self.autoTuningTrueFalseToggled
        )
        autoTuneTabWidget.sigColorChanged.connect(
            self.autoTuningColorChanged
        )
    
    def addAutoTunePoint(self, x, y):
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.addAutoTunePoint(x, y)
    
    def initAutoTuneColors(self):
        setting_name = 'autoTuningTrueSpotsColor'
        default_color = '255-0-0-255'
        try:
            rgba = self.df_settings.at[setting_name, 'value']
        except Exception as e:
            rgba = default_color 
        trueColor = [float(val) for val in rgba.split('-')][:3]
        
        setting_name = 'autoTuningFalseSpotsColor'
        default_color = '0-255-255-255'
        try:
            rgba = self.df_settings.at[setting_name, 'value']
        except Exception as e:
            rgba = default_color 
        falseColor = [float(val) for val in rgba.split('-')][:3]        
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget    
        autoTuneTabWidget.initAutoTuneColors(trueColor, falseColor)
    
    def autoTuningTrueFalseToggled(self, checked):
        if not self.dataIsLoaded:
            return
        self.isAutoTuningForegr = checked
        self.autoTuningSetItemsColor(checked)
    
    def setScatterItemsBrushPen(self, items, rgba):
        if isinstance(items, pg.ScatterPlotItem):
            items = [items]
        
        r, g, b = rgba[:3]
        for item in items:
            item.setPen(r,g,b, width=2)
            item.setBrush(r,g,b, 50)
    
    def autoTuningSetItemsColor(self, true_spots: bool):
        if true_spots:
            setting_name = 'autoTuningTrueSpotsColor'
            default_color = '255-0-0-255'
        else:
            setting_name = 'autoTuningFalseSpotsColor'
            default_color = '0-255-255-255'
        try:
            rgba = self.df_settings.at[setting_name, 'value']
        except Exception as e:
            rgba = default_color 
        
        items = [
            self.ax2_BrushCircle, 
            self.ax1_BrushCircle
        ]
        
        r, g, b, a = [int(val) for val in rgba.split('-')]
        self.setScatterItemsBrushPen(items, (r,g,b,a))
    
    def autoTuningColorChanged(self, rgba, true_spots: bool):
        if true_spots:
            setting_name = 'autoTuningTrueSpotsColor'
        else:
            setting_name = 'autoTuningFalseSpotsColor'
        value = '-'.join([str(v) for v in rgba])
        self.df_settings.at[setting_name, 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)
        self.autoTuningSetItemsColor(true_spots)
    
    def autoTuningAddItems(self):
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        self.ax1.addItem(autoTuneGroupbox.trueItem)
        self.ax1.addItem(autoTuneGroupbox.falseItem)
        self.autoTuningSetItemsColor(True)
        
    def connectLeftClickButtons(self):
        super().connectLeftClickButtons()
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        button = autoTuneTabWidget.autoTuningButton
        button.toggled.connect(button.onToggled)
    
    def startAutoTuning(self):
        if not self.dataIsLoaded:
            return
        self.isAutoTuneRunning = True
        self.setAutoTunePointSize()
    
    def stopAutoTuning(self):
        if not self.dataIsLoaded:
            return
        self.isAutoTuneRunning = False
    
    def setRunNumbers(self):
        posData = self.data[self.pos_i]
        # Scan and determine run numbers
        pathScanner = io.expFolderScanner(
            posData.exp_path, logger_func=self.logger.info
        )
        pathScanner.getExpPaths(posData.exp_path)
        pathScanner.infoExpPaths(pathScanner.expPaths)
        run_nums = sorted([int(r) for r in pathScanner.paths.keys()])
        self.loaded_exp_run_nums = run_nums
        
    def setAnalysisParameters(self):
        paramsGroupbox = self.computeDockWidget.widget().parametersQGBox
        posData = self.data[self.pos_i]
        segmFilename = os.path.basename(posData.segm_npz_path)
        segmEndName = segmFilename[len(posData.basename):]
        runNum = max(self.loaded_exp_run_nums) + 1
        try:
            emWavelen = posData.emWavelens[self.user_ch_name]
        except Exception as e:
            emWavelen = 500.0
        loadedValues = {
            'File paths and channels': [
                {'anchor': 'folderPathsToAnalyse', 'value': posData.pos_path},
                {'anchor': 'spotsEndName', 'value': self.user_ch_name},
                {'anchor': 'segmEndName', 'value': segmEndName},
                {'anchor': 'runNumber', 'value': runNum}
            ],
            'METADATA': [
                {'anchor': 'SizeT', 'value': posData.SizeT},
                {'anchor': 'stopFrameNum', 'value': posData.SizeT},
                {'anchor': 'SizeZ', 'value': posData.SizeZ},
                {'anchor': 'pixelWidth', 'value': posData.PhysicalSizeX},
                {'anchor': 'pixelHeight', 'value': posData.PhysicalSizeY},
                {'anchor': 'voxelDepth', 'value': posData.PhysicalSizeZ},
                {'anchor': 'numAperture', 'value': posData.numAperture},
                {'anchor': 'emWavelen', 'value': emWavelen}
            ]
        }
        analysisParams = config.analysisInputsParams(params_path=None)
        for section, params in loadedValues.items():
            for paramValue in params:
                anchor = paramValue['anchor']
                widget = paramsGroupbox.params[section][anchor]['widget']
                valueSetter = analysisParams[section][anchor]['valueSetter']
                setterFunc = getattr(widget, valueSetter)
                value = paramValue['value']
                setterFunc(value)
    
    def resizeComputeDockWidget(self):
        guiTabControl = self.computeDockWidget.widget()
        paramsGroupbox = guiTabControl.parametersQGBox
        paramsScrollArea = guiTabControl.parametersTab
        autotuneScrollArea = guiTabControl.autoTuneTabWidget
        verticalScrollbar = paramsScrollArea.verticalScrollBar()
        groupboxWidth = autotuneScrollArea.size().width()
        scrollbarWidth = verticalScrollbar.size().width()
        minWidth = groupboxWidth + scrollbarWidth + 30
        self.resizeDocks([self.computeDockWidget], [minWidth], Qt.Horizontal)
        self.showParamsDockButton.click()
    
    def zSliceScrollBarActionTriggered(self, action):
        super().zSliceScrollBarActionTriggered(action)
        if action != QAbstractSlider.SliderAction.SliderMove:
            return
        posData = self.data[self.pos_i]
        self.spotsItems.setData(
            posData.frame_i, z=self.currentZ(checkIfProj=True)
        )
    
    def updateAllImages(self, *args, **kwargs):
        posData = self.data[self.pos_i]
        super().updateAllImages(*args, **kwargs)
        self.spotsItems.setData(
            posData.frame_i, z=self.currentZ(checkIfProj=True)
        )
    
    def updatePos(self):
        self.setSaturBarLabel()
        self.checkManageVersions()
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        self.initContoursImage()
        self.initTextAnnot()
        self.postProcessing()
        posData = self.data[self.pos_i]
        self.spotsItems.setPosition(posData.spotmax_out_path)
        self.spotsItems.loadSpotsTables()
        self.updateAllImages(updateFilters=True)
        self.zoomToCells()
        self.updateScrollbars()
        self.computeSegm()

    def show(self):
        super().show()
        self.showParamsDockButton.setMaximumWidth(15)
        self.showParamsDockButton.setMaximumHeight(60)
        self.realTimeTrackingToggle.setChecked(True)
        self.realTimeTrackingToggle.setDisabled(True)
        self.realTimeTrackingToggle.label.hide()
        self.realTimeTrackingToggle.hide()
        self.computeDockWidget.hide()
        QTimer.singleShot(50, self.resizeComputeDockWidget)