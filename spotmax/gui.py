import sys
import os
import shutil
import datetime
import traceback
import re
import pprint
from queue import Queue

import numpy as np
import pandas as pd

from qtpy.QtCore import (
    Qt, QTimer, QThreadPool, QMutex, QWaitCondition, QEventLoop
)
from qtpy.QtGui import QIcon, QGuiApplication
from qtpy.QtWidgets import QDockWidget, QToolBar, QAction, QAbstractSlider

# Interpret image data as row-major instead of col-major
import pyqtgraph as pg

from spotmax.filters import remove_hot_pixels
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
from cellacdc import io as acdc_io
from cellacdc.myutils import get_salute_string, determine_folder_type

from . import qtworkers, io, printl, dialogs
from . import logs_path, html_path, html_func
from . import widgets, config
from . import tune, utils
from . import core
from . import base_lineage_table_values

from . import qrc_resources_spotmax

LINEAGE_COLUMNS = list(base_lineage_table_values.keys())

ANALYSIS_STEP_RESULT_SLOTS = {
    'gaussSigma': '_displayGaussSigmaResult',
    'refChGaussSigma': '_displayGaussSigmaResult',
    'refChRidgeFilterSigmas': '_displayRidgeFilterResult',
    'removeHotPixels': '_displayRemoveHotPixelsResult',
    'sharpenSpots': '_displaySharpenSpotsResult',
    'spotPredictionMethod': '_displaySpotPredictionResult',
    'refChSegmentationMethod': '_displaySegmRefChannelResult'
}

PARAMS_SLOTS = {
    'gaussSigma': ('sigComputeButtonClicked', '_computeGaussFilter'),
    'refChGaussSigma': ('sigComputeButtonClicked', '_computeRefChGaussSigma'),
    'refChRidgeFilterSigmas': ('sigComputeButtonClicked', '_computeRefChRidgeFilter'),
    'removeHotPixels': ('sigComputeButtonClicked', '_computeRemoveHotPixels'),
    'sharpenSpots': ('sigComputeButtonClicked', '_computeSharpenSpots'),
    'spotPredictionMethod': ('sigComputeButtonClicked', '_computeSpotPrediction'),
    'refChSegmentationMethod': ('sigComputeButtonClicked', '_computeSegmentRefChannel')
}

SliderSingleStepAdd = acdc_gui.SliderSingleStepAdd
SliderSingleStepSub = acdc_gui.SliderSingleStepSub
SliderPageStepAdd = acdc_gui.SliderPageStepAdd
SliderPageStepSub = acdc_gui.SliderPageStepSub
SliderMove = acdc_gui.SliderMove

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
        self._executed = executed
    
    def run(self, module='spotmax_gui', logs_path=logs_path):
        super().run(module=module, logs_path=logs_path)

        self.setWindowTitle("spotMAX - GUI")
        self.setWindowIcon(QIcon(":icon_spotmax.ico"))

        self.initGui()
        self.createThreadPool()
        self.setMaxNumThreadsNumbaParam()
    
    def setMaxNumThreadsNumbaParam(self):
        SECTION = 'Configuration'
        ANCHOR = 'numbaNumThreads'
        paramsGroupbox = self.computeDockWidget.widget().parametersQGBox
        widget = paramsGroupbox.params[SECTION][ANCHOR]['widget']
        if not core.NUMBA_INSTALLED:
            widget.setDisabled(True)
        else:
            import numba
            widget.setMaximum(numba.config.NUMBA_NUM_THREADS)
    
    def createThreadPool(self):
        self.maxThreads = QThreadPool.globalInstance().maxThreadCount()
        self.threadCount = 0
        self.threadQueue = Queue()
        self.threadPool = QThreadPool.globalInstance()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            pass
        super().keyPressEvent(event)
    
    def gui_setCursor(self, modifiers, event):
        cursorsInfo = super().gui_setCursor(modifiers, event)
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        
        setAutoTuneCursor = (
            self.isAddAutoTunePoints and not event.isExit()
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
        
        self.onHoverAutoTunePoints(x, y)
        self.onHoverInspectPoints(x, y)
    
    def onHoverAutoTunePoints(self, x, y):
        if not self.isAutoTuneTabActive:
            return
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        z = self.currentZ()
        frame_i = self.data[self.pos_i].frame_i
        hoveredPoints = autoTuneTabWidget.getHoveredPoints(frame_i, z, y, x)
        if not hoveredPoints:
            return
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.setInspectFeatures(hoveredPoints)
    
    def onHoverInspectPoints(self, x, y):
        z = self.currentZ()
        frame_i = self.data[self.pos_i].frame_i
        point_features = self.spotsItems.getHoveredPointData(frame_i, z, y, x)
        inspectResultsTab = self.computeDockWidget.widget().inspectResultsTab
        inspectResultsTab.setInspectFeatures(point_features)
        
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
        
        canAddPointAutoTune = self.isAddAutoTunePoints and left_click
        
        x, y = event.pos().x(), event.pos().y()
        ID = self.getIDfromXYPos(x, y)
        if ID is None:
            return
        
        if canAddPointAutoTune:
            z = self.currentZ()
            self.addAutoTunePoint(posData.frame_i, z, y, x)
        
    def gui_createRegionPropsDockWidget(self):
        super().gui_createRegionPropsDockWidget(side=Qt.RightDockWidgetArea)
        self.gui_createParamsDockWidget()
    
    def gui_createParamsDockWidget(self):
        self.computeDockWidget = QDockWidget('spotMAX Tab Control', self)
        guiTabControl = dialogs.guiTabControl(
            parent=self.computeDockWidget, logging_func=self.logger.info
        )
        guiTabControl.addAutoTuneTab()
        guiTabControl.addInspectResultsTab()
        guiTabControl.initState(False)
        guiTabControl.currentChanged.connect(self.tabControlPageChanged)

        self.computeDockWidget.setWidget(guiTabControl)
        self.computeDockWidget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable 
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.computeDockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )

        self.addDockWidget(Qt.LeftDockWidgetArea, self.computeDockWidget)
        
        self.connectParamsBaseSignals()
        self.connectAutoTuneSlots()
        self.initAutoTuneColors()
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        self.LeftClickButtons.append(autoTuneTabWidget.addAutoTunePointsButton)
    
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
        
        inspectTabWidget = self.computeDockWidget.widget().inspectResultsTab
        inspectTabWidget.loadAnalysisButton.clicked.connect(
            self.loadAnalysisPathSelected
        )
    
    def checkDataLoaded(self):
        if self.dataIsLoaded:
            return True
        
        txt = html_func.paragraph("""
            Before visualizing results from a previous analysis you need to <b>load some 
            image data</b>.<br><br>
            To do so, click on the <code>Open folder</code> button on the left of 
            the top toolbar (Ctrl+O) and choose an experiment folder to load. 
        """)
        msg = acdc_widgets.myMessageBox()
        msg.warning(self, 'Data not loaded', txt)
        
        return False
        
    
    def loadAnalysisPathSelected(self):
        proceed = self.checkDataLoaded()
        if not proceed:
            return
        
        self.addSpotsCoordinatesAction.trigger()        
    
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
        toolbutton.action = self.spotmaxToolbar.addWidget(toolbutton)
        self.ax1.addItem(toolbutton.item)

        self.spotsItems.setData(
            posData.frame_i, toolbutton=toolbutton,
            z=self.currentZ(checkIfProj=True)
        )
    
    def currentZ(self, checkIfProj=True):
        posData = self.data[self.pos_i]
        if posData.SizeZ == 1:
            return 0
        
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
    
    def isNeuralNetworkRequested(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params['Spots channel']
        anchor = 'spotPredictionMethod'
        return spotsParams[anchor]['widget'].currentText() == 'spotMAX AI'
    
    def isBioImageIOModelRequested(self, section, anchor):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params[section]
        return spotsParams[anchor]['widget'].currentText() == 'BioImage.IO model'
    
    def isPreprocessAcrossExpRequired(self):
        if not self.isNeuralNetworkRequested():
            return False
        
        nnetParams = self.getNeuralNetParams()
        if not nnetParams['init']['preprocess_across_experiment']:
            # Pre-processing not requested
            return False
        
        return True

    def isPreprocessAcrossTimeRequired(self):
        if not self.isNeuralNetworkRequested():
            return False
        
        posData = self.data[self.pos_i]
        if posData.SizeT == 1:
            return False
        
        nnetParams = self.getNeuralNetParams()
        if not nnetParams['init']['preprocess_across_timepoints']:
            # Pre-processing not requested
            return False
        
        return True
    
    def reInitGui(self):
        super().reInitGui()
        
        self.annotateToolbar.setDisabled(True)
        self.annotateToolbar.setVisible(False)
        
        for toolButton in self.spotsItems.buttons:
            self.spotmaxToolbar.removeAction(toolButton.action)
            
        self.spotsItems = widgets.SpotsItems(self)
        
        try:
            self.disconnectParamsGroupBoxSignals()
        except Exception as e:
            # printl(traceback.format_exc())
            pass
        self.showParamsDockButton.setDisabled(False)
        self.computeDockWidget.widget().initState(False)
        
        self.transformedDataNnetExp = None
        self.transformedDataTime = None
        
    def initGui(self):
        self.isAnalysisRunning = False
        
        self._setWelcomeText()
        self._disableAcdcActions(
            self.newAction, self.manageVersionsAction, self.openFileAction
        )
        self.ax2.hide()
        
        self.measurementsMenu.setDisabled(False)
        self.setMeasurementsAction.setText('Set Cell-ACDC measurements...')
    
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
        self.isAnalysisRunning = True
        self.logger.info('Starting spotMAX analysis...')
        self._analysis_started_datetime = datetime.datetime.now()
        self.funcDescription = 'starting analysis process'
        worker = qtworkers.AnalysisWorker(ini_filepath, is_tempfile)

        worker.signals.finished.connect(self.analysisWorkerFinished)
        worker.signals.progress.connect(self.workerProgress)
        # worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        # worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)
        
    def analysisWorkerFinished(self, args):
        self.isAnalysisRunning = False
        ini_filepath, is_tempfile = args
        self.logger.info('Analysis finished')
        if is_tempfile:
            tempdir = os.path.dirname(ini_filepath)
            self.logger.info(f'Deleting temp folder "{tempdir}"')
            shutil.rmtree(tempdir)
        log_path, errors = utils.parse_log_file()
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
        msg_kwargs = {
            'path_to_browse': os.path.dirname(log_path),
            'browse_button_text': 'Show log file'
        }
        if errors:
            details = '\n\n'.join(errors)
            msg_kwargs['detailsText'] = details
            txt.replace(
                'spotMAX analysis finished!', 
                'spotMAX analysis ended with ERRORS'
            )
            txt = (
                f'{txt}\n'
                'WARNING: Analysis ended with errors. '
                'See summary of errors below and more details in the '
                'log file:\n\n'
                f'`{log_path}`\n'
            )
            msg_func = 'critical'
        else:
            msg_func = 'information'
        self.logger.info(f'{line_str}\n{txt}\n{close_str}')
        txt = html_func.paragraph(txt.replace('\n', '<br>'))
        txt = re.sub('`(.+)`', r'<code>\1</code>', txt)
        msg = acdc_widgets.myMessageBox()
        
        msg_args = (self, 'spotMAX analysis finished', txt)
        getattr(msg, msg_func)(*msg_args, **msg_kwargs)
        
        if msg_func == 'information':
            self.askVisualizeResults()
    
    def askVisualizeResults(self):        
        txt = html_func.paragraph(
            'Do you want to visualize the results in the GUI?'
        )
        msg = acdc_widgets.myMessageBox(wrapText=False)
        _, yesButton = msg.question(
            self, 'Visualize results?', txt, buttonsTexts=('No', 'Yes')
        )
        if msg.clickedButton == yesButton:
            if not self.dataIsLoaded:
                txt = html_func.paragraph("""
            In order to visualize the results you need to <b>load some 
            image data first</b>.<br><br>
            
            To do so, click on the <code>Open folder</code> button on the left of 
            the top toolbar (Ctrl+O) and choose an experiment folder to load.<br><br>
            
            After loading the image data you can visualize the results by clicking 
            on the <code>Visualize detected spots from a previous analysis</code> 
            button on the left-side toolbar. 
        """)
                msg = acdc_widgets.myMessageBox(wrapText=True)
                msg.warning(self, 'Data not loaded', txt)
                return
            self.addSpotsCoordinatesAction.trigger()
        
    
    def gui_createActions(self):
        super().gui_createActions()

        self.addSpotsCoordinatesAction = QAction(self)
        self.addSpotsCoordinatesAction.setIcon(QIcon(":addPlotSpots.svg"))
        self.addSpotsCoordinatesAction.setToolTip(
            'Visualize detected spots from a previous analysis'
        )
    
    def gui_createToolBars(self):
        super().gui_createToolBars()

        self.addToolBarBreak(Qt.LeftToolBarArea)
        self.spotmaxToolbar = QToolBar("spotMAX toolbar", self)
        self.spotmaxToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(Qt.LeftToolBarArea, self.spotmaxToolbar)
        self.spotmaxToolbar.addAction(self.addSpotsCoordinatesAction)
        self.spotmaxToolbar.setVisible(False)
        self.spotsItems = widgets.SpotsItems(self)
    
    def gui_addTopLayerItems(self):
        super().gui_addTopLayerItems()

    def gui_connectEditActions(self):
        super().gui_connectEditActions()
    
    def loadingDataCompleted(self):
        super().loadingDataCompleted()
        posData = self.data[self.pos_i]
        self.setWindowTitle(f'spotMAX - GUI - "{posData.exp_path}"')
        self.spotmaxToolbar.setVisible(True)
        self.computeDockWidget.widget().initState(True)
        
        self.isAutoTuneTabActive = False
        
        self.setRunNumbers()
        
        self.setAnalysisParameters()
        self.connectParamsGroupBoxSignals()
        self.autoTuningAddItems()
        self.initTuneKernel()
        self.hideAcdcToolbars()
        
        self.setFocusGraphics()
    
    def hideAcdcToolbars(self):
        self.editToolBar.setVisible(False)
        self.editToolBar.setDisabled(True)
        self.placeHolderToolbar.setVisible(False)
        self.placeHolderToolbar.setDisabled(True)
        self.ccaToolBar.setVisible(False)
        self.ccaToolBar.setDisabled(True)

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
    def _computeGaussFilter(self, formWidget):
        self.funcDescription = 'Initial gaussian filter'
        module_func = 'pipe.preprocess_image'
        anchor = 'gaussSigma'
        
        posData = self.data[self.pos_i]
        
        args = [module_func, anchor]
        all_kwargs = self.paramsToKwargs()
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return
        keys = ['do_remove_hot_pixels', 'gauss_sigma', 'use_gpu']
        kwargs = {key:all_kwargs[key] for key in keys}
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            posData.img_data, posData.segm_data, 
            on_finished_callback=on_finished_callback
        )
    
    @exception_handler
    def _computeRefChGaussSigma(self, formWidget):
        self.funcDescription = 'Initial gaussian filter'
        module_func = 'pipe.preprocess_image'
        anchor = 'refChGaussSigma'
        
        posData = self.data[self.pos_i]
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        filePathParams = ParamsGroupBox.params['File paths and channels']
        refChEndName = filePathParams['refChEndName']['widget'].text()
        if not refChEndName:
            refChEndName = self.askReferenceChannelEndname()
            if refChEndName is None:
                self.logger.info('Segmenting reference channel cancelled.')
                return
            filePathParams['refChEndName']['widget'].setText(refChEndName)
        
        self.logger.info(f'Loading "{refChEndName}" reference channel data...')
        refChannelData = self.loadImageDataFromChannelName(refChEndName) 
        
        args = [module_func, anchor]
        all_kwargs = self.paramsToKwargs(is_spots_ch_required=False)
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return
        keys = ['do_remove_hot_pixels', 'ref_ch_gauss_sigma', 'use_gpu']
        kwargs = {key:all_kwargs[key] for key in keys}
        kwargs['gauss_sigma'] = kwargs.pop('ref_ch_gauss_sigma')
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            refChannelData, posData.segm_data, 
            on_finished_callback=on_finished_callback
        )

    @exception_handler
    def _computeRefChRidgeFilter(self, formWidget):
        self.funcDescription = 'Ridge filter (enhances networks)'
        module_func = 'pipe.ridge_filter'
        anchor = 'refChRidgeFilterSigmas'
        
        posData = self.data[self.pos_i]
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        filePathParams = ParamsGroupBox.params['File paths and channels']
        refChEndName = filePathParams['refChEndName']['widget'].text()
        if not refChEndName:
            refChEndName = self.askReferenceChannelEndname()
            if refChEndName is None:
                self.logger.info('Segmenting reference channel cancelled.')
                return
            filePathParams['refChEndName']['widget'].setText(refChEndName)
        
        self.logger.info(f'Loading "{refChEndName}" reference channel data...')
        refChannelData = self.loadImageDataFromChannelName(refChEndName) 
        
        args = [module_func, anchor]
        all_kwargs = self.paramsToKwargs(is_spots_ch_required=False)
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return
        keys = ['do_remove_hot_pixels', 'ref_ch_ridge_sigmas']
        kwargs = {key:all_kwargs[key] for key in keys}
        kwargs['ridge_sigmas'] = kwargs.pop('ref_ch_ridge_sigmas')
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            refChannelData, posData.segm_data, 
            on_finished_callback=on_finished_callback
        )

    def warnTrueSpotsAutoTuneNotAdded(self):
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph("""
            You did not add any points for true spots!<br><br>
            To perform auto-tuning, you need to add points that will be used 
            as true positives.<br><br>
            Press the <code>Start adding points</code> button and click on 
            valid spots before starting autotuning. Thanks! 
        """)
        msg.critical(self, 'True spots not added!', txt)
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.autoTuningButton.setChecked(False)
    
    def storeCroppedDataAndStartTuneKernel(self, *args, **kwargs):
        kernel = args[0]
        image_data_cropped = kwargs['image_data_cropped']
        segm_data_cropped = kwargs['segm_data_cropped']
        crop_to_global_coords = kwargs['crop_to_global_coords']
        
        kernel.set_image_data(image_data_cropped)
        kernel.set_segm_data(segm_data_cropped)
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        
        trueItem = autoTuneGroupbox.trueItem
        coords = trueItem.coordsToNumpy(includeData=True)
        kernel.set_tzyx_true_spots_coords(coords, crop_to_global_coords)
        
        falseItem = autoTuneGroupbox.falseItem
        coords = falseItem.coordsToNumpy(includeData=True)
        kernel.set_tzyx_false_spots_coords(coords, crop_to_global_coords)
        
        self.startTuneKernelWorker(kernel)
        
    def storeCroppedRefChDataAndStartTuneKernel(self, *args, **kwargs):
        kernel = args[0]
        ref_ch_data_cropped = kwargs['image_data_cropped']
        segm_data_cropped = kwargs['segm_data_cropped']
        
        kernel.set_ref_ch_data(ref_ch_data_cropped)
        kernel.set_segm_data(segm_data_cropped)
        
        if kernel.image_data() is None:
            posData = self.data[self.pos_i]
            on_finished_callback = (
                self.storeCroppedDataAndStartTuneKernel, args, kwargs
            )
            self.startCropImageBasedOnSegmDataWorkder(
                posData.img_data, posData.segm_data, 
                on_finished_callback=on_finished_callback
            )
    
    def startCropImageBasedOnSegmDataWorkder(
            self, image_data, segm_data, on_finished_callback,
            nnet_input_data=None
        ):
        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title='Cropping based on segm data', parent=self,
            pbarDesc='Cropping based on segm data'
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
            on_finished_callback, nnet_input_data=nnet_input_data
        )
        worker = self.connectDefaultWorkerSlots(worker)
        worker.signals.finished.connect(self.cropImageWorkerFinished)
        self.threadPool.start(worker)
    
    def cropImageWorkerFinished(self, result):
        (image_data_cropped, segm_data_cropped, crop_to_global_coords, 
         on_finished_callback, nnet_input_data_cropped) = result
        
        if on_finished_callback is None:
            return
        
        func, args, kwargs = on_finished_callback
        
        if 'image_data_cropped' in kwargs:
            kwargs['image_data_cropped'] = image_data_cropped
        
        if 'image_data_cropped' in kwargs:
            kwargs['segm_data_cropped'] = segm_data_cropped
        
        if 'crop_to_global_coords' in kwargs:
            kwargs['crop_to_global_coords'] = crop_to_global_coords
        
        if 'nnet_input_data_cropped' in kwargs:
            kwargs['nnet_input_data_cropped'] = nnet_input_data_cropped
        
        posData = self.data[self.pos_i]
        image = image_data_cropped[posData.frame_i]
        kwargs['image'] = image
        if nnet_input_data_cropped is not None:
            kwargs['nnet_input_data'] = nnet_input_data_cropped[posData.frame_i]
            
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
        size = spots_zyx_radii[-1]*2
        self.setHoverToolSymbolData(
            [x], [y], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            size=size
        )
    
    def copyParamsToAutoTuneWidget(self):
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        section = 'METADATA'
        anchor = 'yxResolLimitMultiplier'
        autoTuneGroupbox.params[section][anchor]['widget'].setValue(
            ParamsGroupBox.params[section][anchor]['widget'].value()
        )
    
    def tabControlPageChanged(self, index):
        if not self.dataIsLoaded:
            return
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget        
        self.isAutoTuneTabActive = False
        if index == 1:
            # AutoTune tab toggled
            autoTuneTabWidget.setAutoTuneItemsVisible(True)
            self.copyParamsToAutoTuneWidget()
            self.setAutoTunePointSize()
            self.initTuneKernel()
            self.isAutoTuneTabActive = True
        elif index == 2:
            autoTuneTabWidget.setAutoTuneItemsVisible(False)
    
    def setAutoTunePointSize(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        size = spots_zyx_radii[-1]*2
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.setAutoTunePointSize(size)
    
    @exception_handler
    def _computeSharpenSpots(self, formWidget):
        self.funcDescription = 'Sharpen spots (DoG filter)'
        module_func = 'filters.DoG_spots'
        anchor = 'sharpenSpots'
        
        posData = self.data[self.pos_i]
        
        keys = ['spots_zyx_radii', 'use_gpu']
        all_kwargs = self.paramsToKwargs()
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return
        kwargs = {key:all_kwargs[key] for key in keys}
        
        args = [module_func, anchor]

        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            posData.img_data, posData.segm_data, 
            on_finished_callback=on_finished_callback
        )
    
    def getLineageTable(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        filePathParams = ParamsGroupBox.params['File paths and channels']
        acdcDfEndNameWidget = filePathParams['lineageTableEndName']['widget']
        acdcDfEndName = acdcDfEndNameWidget.text()
        if not acdcDfEndName:
            return None, True
        
        acdcDfEndName, _ = os.path.splitext(acdcDfEndName)
        
        posData = self.data[self.pos_i]
        loadedAcdcDfEndname = posData.getAcdcDfEndname()
        
        if acdcDfEndName == loadedAcdcDfEndname:
            return posData.acdc_df[LINEAGE_COLUMNS].copy(), True
        
        df, proceed = self.warnLoadedAcdcDfDifferentFromRequested(
            loadedAcdcDfEndname, acdcDfEndName
        )
        return df, proceed
    
    def checkRequestedSegmEndname(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        filePathParams = ParamsGroupBox.params['File paths and channels']
        segmEndNameWidget = filePathParams['segmEndName']['widget']
        segmEndName = segmEndNameWidget.text()
        if not segmEndName:
            return True
        
        segmEndName, _ = os.path.splitext(segmEndName)
        
        posData = self.data[self.pos_i]
        loadedSegmEndname = posData.getSegmEndname()
        
        if loadedSegmEndname == segmEndName:
            return True
        
        proceed = self.warnLoadedSegmDifferentFromRequested(
            loadedSegmEndname, segmEndName
        )    
        return proceed
    
    def checkRequestedSpotsChEndname(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        filePathParams = ParamsGroupBox.params['File paths and channels']
        spotChEndnameWidget = filePathParams['spotsEndName']['widget']
        spotChEndname = spotChEndnameWidget.text()
        if not spotChEndname:
            return self.warnSpotsChNotProvided()

        posData = self.data[self.pos_i]
        spotChEndname, _ = os.path.splitext(spotChEndname)
        
        if posData.user_ch_name == spotChEndname:
            return True
        
        return self.warnSpotsChWillBeIgnored(posData.user_ch_name, spotChEndname)
        
    def warnSpotsChWillBeIgnored(self, loaded, requested):
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph(f"""
            You requested <code>{requested}</code> channel for the spots 
            image data (parameter `Spots channel end name or path`),<br>
            but you loaded the channel <code>{loaded}</code>.<br><br>
            How do you want to proceed?
        """)
        continueWithLoadedButton = acdc_widgets.okPushButton(
            f'Continue with `{loaded}` data'
        )
        msg.warning(
            self, 'Spots channel name not provided', txt,
            buttonsTexts=('Cancel', continueWithLoadedButton)
        )
        return not msg.cancel    
        
    def warnSpotsChNotProvided(self):
        posData = self.data[self.pos_i]
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph(f"""
            You did not provide <b>any channel name for the spots image data</b>, 
            (parameter `Spots channel end name or path`).<br><br>
            How do you want to proceed?
        """)
        continueWithLoadedButton = acdc_widgets.okPushButton(
            f'Continue with `{posData.user_ch_name}` data'
        )
        msg.warning(
            self, 'Spots channel name not provided', txt,
            buttonsTexts=('Cancel', continueWithLoadedButton)
        )
        return not msg.cancel
    
    def warnLoadedSegmDifferentFromRequested(self, loaded, requested):
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph(f"""
            You loaded the segmentation file ending with <code>{loaded},</code> 
            but in the parameter `Cells segmentation end name or path`<br>
            you requested the file <code>{requested}</code>.<br><br>
            How do you want to proceed?
        """)
        keepLoadedButton = acdc_widgets.okPushButton(
            f'Continue with `{loaded}`'
        )
        msg.warning(
            self, 'Mismatch between loaded and requested file', txt,
            buttonsTexts=('Cancel', keepLoadedButton)
        )
        return not msg.cancel
    
    def warnLoadedAcdcDfDifferentFromRequested(self, loaded, requested):
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph(f"""
            You loaded the lineage table ending with <code>{loaded},</code> 
            but in the parameter `Table with lineage info end name or path`<br>
            you requested the table name <code>{requested}</code>.<br><br>
            How do you want to proceed?
        """)
        loadRequestedButton = acdc_widgets.OpenFilePushButton(
            f'Load table `{requested}`'
        )
        keepLoadedButton = acdc_widgets.okPushButton(
            f'Keep table `{loaded}`'
        )
        msg.warning(
            self, 'Mismatch between loaded and requested file', txt,
            buttonsTexts=('Cancel', loadRequestedButton, keepLoadedButton)
        )
        if msg.cancel:
            return None, False
        
        posData = self.data[self.pos_i]
        if msg.clickedButton == loadRequestedButton:
            filepath = acdc_io.get_filepath_from_channel_name(
                posData.images_path, posData.basename
            )
            self.logger.info(f'Loading table from "{filepath}"...')
            df = acdc_load._load_acdc_df_file(filepath)
            return df, True
        
        return posData.acdc_df[LINEAGE_COLUMNS].copy(), True
    
    def paramsToKwargs(self, is_spots_ch_required=True):
        posData = self.data[self.pos_i]
        
        if is_spots_ch_required:
            proceed = self.checkRequestedSpotsChEndname()
            if not proceed:
                return
        
        lineage_table = None
        if posData.acdc_df is not None:
            acdc_df, proceed = self.getLineageTable()
            if not proceed:
                return 
            if acdc_df is not None:
                lineage_table = acdc_df.loc[posData.frame_i]
        
        proceed = self.checkRequestedSegmEndname()
        if not proceed:
            return
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        filePathParams = ParamsGroupBox.params['File paths and channels']
        refChEndName = filePathParams['refChEndName']['widget'].text()
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        gauss_sigma = preprocessParams['gaussSigma']['widget'].value()
        
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        do_sharpen = preprocessParams['sharpenSpots']['widget'].isChecked()
        do_remove_hot_pixels = (
            preprocessParams['removeHotPixels']['widget'].isChecked()
        )
        do_aggregate = preprocessParams['aggregate']['widget'].isChecked()
        
        refChParams = ParamsGroupBox.params['Reference channel']
        ref_ch_gauss_sigma = refChParams['refChGaussSigma']['widget'].value()
        
        refChParams = ParamsGroupBox.params['Reference channel']
        refChRidgeSigmasWidget = refChParams['refChRidgeFilterSigmas']
        ref_ch_ridge_sigmas = refChRidgeSigmasWidget['widget'].value()
        if isinstance(ref_ch_ridge_sigmas, float) and ref_ch_ridge_sigmas>0:
            ref_ch_ridge_sigmas = [ref_ch_ridge_sigmas]
        
        spotsParams = ParamsGroupBox.params['Spots channel']
        optimise_with_edt = (
            spotsParams['optimiseWithEdt']['widget'].isChecked()
        )
        
        configParams = ParamsGroupBox.params['Configuration']
        use_gpu = configParams['useGpu']['widget'].isChecked()
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        tune_features_range = autoTuneTabWidget.selectedFeatures()
        
        kwargs = {
            'lab': None, 
            'gauss_sigma': gauss_sigma, 
            'ref_ch_gauss_sigma': ref_ch_gauss_sigma, 
            'ref_ch_ridge_sigmas': ref_ch_ridge_sigmas,
            'spots_zyx_radii': spots_zyx_radii, 
            'do_sharpen': do_sharpen, 
            'do_remove_hot_pixels': do_remove_hot_pixels,
            'lineage_table': lineage_table, 
            'do_aggregate': do_aggregate, 
            'optimise_with_edt': optimise_with_edt,
            'use_gpu': use_gpu, 'sigma': gauss_sigma, 
            'ref_ch_endname': refChEndName,
            'tune_features_range': tune_features_range
        }
        
        return kwargs
    
    def checkPreprocessAcrossExp(self):
        if not self.isPreprocessAcrossExpRequired():
            return True

        if self.transformedDataNnetExp is not None:
            # Data already pre-processed
            return True
        
        proceed = self.startAndWaitPreprocessAcrossExpWorker()
        return proceed
        
    def checkPreprocessAcrossTime(self):
        if not self.isPreprocessAcrossTimeRequired():
            return True

        if self.transformedDataTime is not None:
            # Data already pre-processed
            return True

        if self.transformedDataNnetExp is not None:
            posData = self.data[self.pos_i]
            input_data = self.transformedDataNnetExp[posData.pos_foldername]
        else:
            input_data = posData.img_data
        
        proceed = self.startAndWaitPreprocessAcrossTimeWorker(input_data)
        return proceed
        
        
    @exception_handler
    def _computeSpotPrediction(self, formWidget):
        proceed = self.checkPreprocessAcrossExp()
        if not proceed:
            self.logger.info('Computing spots segmentation cancelled.')
            return
        
        self.checkPreprocessAcrossTime()
        self.funcDescription = 'Spots location semantic segmentation'
        module_func = 'pipe.spots_semantic_segmentation'
        anchor = 'spotPredictionMethod'
        
        posData = self.data[self.pos_i]        
        args = [module_func, anchor]
        keys = [
            'lab', 'gauss_sigma', 'spots_zyx_radii', 'do_sharpen',
            'do_remove_hot_pixels', 'lineage_table', 'do_aggregate', 
            'use_gpu'
        ]
        all_kwargs = self.paramsToKwargs()
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return
        kwargs = {key:all_kwargs[key] for key in keys}
        
        kwargs = self.addNnetKwargs(kwargs)
        
        section = 'Spots channel'
        kwargs = self.addBioImageIOModelKwargs(kwargs, section, anchor)
        
        self.logNnetParams(kwargs.get('nnet_params'))
        self.logNnetParams(
            kwargs.get('bioimageio_params'), model_name='BioImage.IO model'
        )
        
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            posData.img_data, posData.segm_data, 
            on_finished_callback=on_finished_callback,
            nnet_input_data=kwargs.get('nnet_input_data')
        )
    
    def logNnetParams(self, nnet_params, model_name='spotMAX AI model'):
        if nnet_params is None:
            return
        
        text = '-'*60
        text = (
            f'{text}\nRunning {model_name} with the following parameters:\n'
        )
        text = f'{text}  1. Initialization:\n'
        for param, value in nnet_params['init'].items():
            text = f'{text}    - {param}: {value}\n'
        
        text = f'{text}  2. Segmentation:\n'
        for param, value in nnet_params['segment'].items():
            text = f'{text}    - {param}: {value}\n'
            
        closing = '*'*60
        text = f'{text}{closing}'
        self.logger.info(text)        
        
    def addNnetKwargs(self, kwargs):
        if not self.isNeuralNetworkRequested():
            return kwargs
        
        kwargs['nnet_model'] = self.getNeuralNetworkModel()
        kwargs['nnet_params'] = self.getNeuralNetParams()
        kwargs['nnet_input_data'] = self.getNeuralNetInputData()
        
        threshold_func = self.getSpotsThresholdMethod()
        kwargs['thresholding_method'] = threshold_func
        kwargs['do_try_all_thresholds'] = False
        
        return kwargs
    
    def addBioImageIOModelKwargs(self, kwargs, section, anchor):
        if not self.isBioImageIOModelRequested(section, anchor):
            return kwargs
        
        kwargs['bioimageio_model'] = self.getBioImageIOModel(section, anchor)
        kwargs['bioimageio_params'] = self.getBioImageIOParams(section, anchor)
        
        threshold_func = self.getRefChThresholdMethod()
        kwargs['thresholding_method'] = threshold_func
        kwargs['do_try_all_thresholds'] = False
        
        return kwargs
    
    def getNeuralNetInputData(self):
        useTranformedDataTime = (
            self.transformedDataTime is not None
            and self.isPreprocessAcrossTimeRequired()
        )
        if useTranformedDataTime:
            return self.transformedDataTime
        
        useTranformedDataExp = (
            self.transformedDataNnetExp is not None
            and self.isPreprocessAcrossExpRequired()
        )
        
        if useTranformedDataExp:
            posData = self.data[self.pos_i]
            nnet_input_data = self.transformedDataNnetExp[posData.pos_foldername]
            if posData.SizeT == 1:
                nnet_input_data = nnet_input_data[np.newaxis]
            return nnet_input_data
        
        # We return None so that the network will use the raw image
        return
    
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
    
    @exception_handler
    def startAndWaitPreprocessAcrossExpWorker(self):        
        selectedPath = self.getSelectExpPath()
        if selectedPath is None:
            self.logger.info('Experiment path not selected')
            return False
        
        folder_type = determine_folder_type(selectedPath)
        is_pos_folder, is_images_folder, _ = folder_type
        if is_pos_folder:
            images_paths = [os.path.join(selectedPath, 'Images')]
        elif is_images_folder:
            images_paths = [selectedPath]
        else:
            images_paths = self._loadFromExperimentFolder(selectedPath)
        
        if not images_paths:
            self.logger.info(
                'Selected experiment path does not contain valid '
                'Position folders.'
            )
            return False
        
        pos_foldernames = [
            os.path.basename(os.path.dirname(images_path))
            for images_path in images_paths
        ]
        exp_path = os.path.dirname(os.path.dirname(images_paths[0]))
        
        nnet_model = self.getNeuralNetworkModel()
        
        spots_ch_endname = self.getSpotsChannelEndname()
        if not spots_ch_endname:
            raise ValueError(
                '"Spots channel end name or path" parameter not provided.'
            )
        
        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title='Preprocessing data', parent=self,
            pbarDesc='Preprocessing data across experiment'
        )
        self.progressWin.mainPbar.setMaximum(0)
        self.progressWin.show(self.app)
        
        loop = QEventLoop()
        worker = qtworkers.PreprocessNnetDataAcrossExpWorker(
            exp_path, pos_foldernames, spots_ch_endname, nnet_model, 
            loop_to_exist_on_finished=loop
        )
        self.connectDefaultWorkerSlots(worker)
        worker.signals.finished.connect(self.preprocesAcrossExpFinished)
        self.threadPool.start(worker)
        loop.exec_()
        return True
    
    @exception_handler
    def startAndWaitPreprocessAcrossTimeWorker(self, input_data):
        nnet_model = self.getNeuralNetworkModel()
        
        pbarDesc = 'Preprocessing data across time-points'
        if self.progressWin is None:
            self.progressWin = acdc_apps.QDialogWorkerProgress(
                title='Preprocessing data across time-points', parent=self,
                pbarDesc=pbarDesc
            )
            self.progressWin.mainPbar.setMaximum(0)
            self.progressWin.show(self.app)
        else:
            self.progressWin.progressLabel.setText(pbarDesc)
        
        loop = QEventLoop()
        worker = qtworkers.PreprocessNnetDataAcrossTimeWorker(
            input_data, nnet_model, loop_to_exist_on_finished=loop
        )
        self.connectDefaultWorkerSlots(worker)
        worker.signals.finished.connect(self.preprocesAcrossTimeFinished)
        self.threadPool.start(worker)
        loop.exec_()
        return True
    
    def preprocesAcrossExpFinished(self, result):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        worker, transformed_data, loop = result
        self.transformedDataNnetExp = transformed_data
        if loop is not None:
            loop.exit()
    
    def preprocesAcrossTimeFinished(self, result):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        worker, transformed_data, loop = result
        self.transformedDataTime = transformed_data
        if loop is not None:
            loop.exit()
    
    def startLoadImageDataWorker(
            self, filepath='', channel='', images_path='', 
            loop_to_exist_on_finished=None
        ):
        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title='Loading image data', parent=self,
            pbarDesc='Loading image data'
        )
        self.progressWin.mainPbar.setMaximum(0)
        self.progressWin.show(self.app)
        
        worker = qtworkers.LoadImageWorker(
            filepath=filepath, channel=channel, images_path=images_path,
            loop_to_exist_on_finished=loop_to_exist_on_finished
        )
        self.connectDefaultWorkerSlots(worker)
        worker.signals.finished.connect(self.loadImageDataWorkerFinished)
        self.threadPool.start(worker)
        return worker

    def loadImageDataWorkerFinished(self, output):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        worker, filepath, channel, image_data, loop = output
        worker.image_data = image_data
        if loop is not None:
            loop.exit()
    
    def loadImageDataFromChannelName(self, channel, get_image_data=True):
        posData = self.data[self.pos_i]
        images_path = posData.images_path
        filepath = acdc_load.get_filename_from_channel(images_path, channel)
        if not filepath:
            raise FileNotFoundError(f'{channel} channel not found in {images_path}')
        filename_ext = os.path.basename(filepath)
        filename, ext = os.path.splitext(filename_ext)
        imgData = posData.fluo_data_dict.get(filename)
        if imgData is None:
            if get_image_data:
                loop = QEventLoop()
            worker = self.startLoadImageDataWorker(
                filepath=filepath, loop_to_exist_on_finished=loop
            )
            if get_image_data:
                loop.exec_()
            
            imgData = worker.image_data
            if posData.SizeT == 1:
                imgData = imgData[np.newaxis]
        return imgData
    
    @exception_handler
    def _computeSegmentRefChannel(self, formWidget):
        posData = self.data[self.pos_i]
        
        self.funcDescription = 'Reference channel semantic segmentation'
        module_func = 'pipe.reference_channel_semantic_segm'
        anchor = 'refChSegmentationMethod'
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        filePathParams = ParamsGroupBox.params['File paths and channels']
        refChEndName = filePathParams['refChEndName']['widget'].text()
        if not refChEndName:
            refChEndName = self.askReferenceChannelEndname()
            if refChEndName is None:
                self.logger.info('Segmenting reference channel cancelled.')
                return
            filePathParams['refChEndName']['widget'].setText(refChEndName)
        
        self.logger.info(f'Loading "{refChEndName}" reference channel data...')
        refChannelData = self.loadImageDataFromChannelName(refChEndName)        
        
        keys = [
            'lab', 'ref_ch_gauss_sigma', 'do_remove_hot_pixels', 'lineage_table',
            'do_aggregate', 'use_gpu', 'ref_ch_ridge_sigmas'
        ]
        all_kwargs = self.paramsToKwargs(is_spots_ch_required=False)
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return
        kwargs = {key:all_kwargs[key] for key in keys}
        kwargs['gauss_sigma'] = kwargs.pop('ref_ch_gauss_sigma')
        kwargs['ridge_filter_sigmas'] = kwargs.pop('ref_ch_ridge_sigmas')
        
        section = 'Reference channel'
        kwargs = self.addBioImageIOModelKwargs(kwargs, section, anchor)
        self.logNnetParams(
            kwargs.get('bioimageio_params'), model_name='BioImage.IO model'
        )
        
        args = [module_func, anchor]
        
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
    
    def _displayRefChGaussSigmaResult(self, result, image):
        from cellacdc.plot import imshow
        posData = self.data[self.pos_i]
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        refChParams = ParamsGroupBox.params['Reference channel']
        anchor = 'refChGaussSigma'
        sigma = refChParams[anchor]['widget'].value()
        titles = ['Raw image', f'Filtered image (sigma = {sigma})']
        imshow(
            image, result, axis_titles=titles, parent=self, 
            window_title='Reference channel - Gaussian filter',
            color_scheme=self._colorScheme
        )
    
    def _displayRidgeFilterResult(self, result, image):
        from cellacdc.plot import imshow
        posData = self.data[self.pos_i]
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        refChParams = ParamsGroupBox.params['Reference channel']
        anchor = 'refChRidgeFilterSigmas'
        sigmas = refChParams[anchor]['widget'].value()
        titles = ['Raw image', f'Filtered image (sigmas = {sigmas})']
        imshow(
            image, result, axis_titles=titles, parent=self, 
            window_title='Reference channel - Ridge filter (enhances networks)',
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
    
    def _displaySpotPredictionResult(self, result, image):
        from cellacdc.plot import imshow
        posData = self.data[self.pos_i]
        
        if 'neural_network' in result:
            selected_threshold_method = self.getSpotsThresholdMethod()
            titles = [
                'Input image', f'{selected_threshold_method}', 'spotMAX AI'
            ]
            prediction_images = [
                result['input_image'], 
                result['custom'], 
                result['neural_network'],
            ]
        elif 'bioimageio_model' in result:
            selected_threshold_method = self.getSpotsThresholdMethod()
            titles = [
                'Input image', f'{selected_threshold_method}', 
                'BioImage.IO model'
            ]
            prediction_images = [
                result['input_image'], 
                result['custom'], 
                result['bioimageio_model'],
            ] 
        else:
            titles = list(result.keys())
            titles[0] = 'Input image'
            prediction_images = list(result.values())
        
        window_title = 'Spots channel - Spots segmentation method'
        
        imshow(
            *prediction_images, axis_titles=titles, parent=self, 
            window_title=window_title, color_scheme=self._colorScheme
        )
    
    def _displaySegmRefChannelResult(self, result, image):
        from cellacdc.plot import imshow
        
        if 'bioimageio_model' in result:
            selected_threshold_method = self.getRefChThresholdMethod()
            titles = [
                'Input image', f'{selected_threshold_method}', 
                'BioImage.IO model'
            ]
            prediction_images = [
                result['input_image'], 
                result['custom'], 
                result['bioimageio_model'],
            ] 
        else:
            titles = list(result.keys())
            titles[0] = 'Input image'
            prediction_images = list(result.values())
        
        window_title = 'Reference channel - Semantic segmentation'
        
        imshow(
            *prediction_images, axis_titles=titles, parent=self, 
            window_title=window_title, color_scheme=self._colorScheme
        )
    
    def connectDefaultWorkerSlots(self, worker):
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        worker.signals.debug.connect(self.workerDebug)
        return worker
    
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
    
    @exception_handler
    def workerDebug(self, to_debug):
        try:
            from . import _debug
            worker = to_debug[-1]
            _debug._gui_autotune_compute_features(to_debug)
            # _debug._gui_autotune_f1_score(to_debug)
        except Exception as error:
            raise error
        finally:
            worker.waitCond.wakeAll()
    
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
    
    def connectParamsBaseSignals(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        removeHotPixelsToggle = preprocessParams['removeHotPixels']['widget']
        removeHotPixelsToggle.toggled.connect(self.onRemoveHotPixelsToggled)
        
        metadataParams = ParamsGroupBox.params['METADATA']
        pixelWidthWidget = metadataParams['pixelWidth']['widget']
        pixelWidthWidget.valueChanged.connect(self.onPixelWidthValueChanged)
        
        configParams = ParamsGroupBox.params['Configuration']
        useGpuToggle = configParams['useGpu']['widget']
        useGpuToggle.toggled.connect(self.onUseGpuToggled)
        
        filePathParams = ParamsGroupBox.params['File paths and channels']
        expPathsWidget = filePathParams['folderPathsToAnalyse']['widget']
        expPathsWidget.textChanged.connect(self.onExpPathsTextChanged)
        
        spotsChNameWidget = filePathParams['spotsEndName']['widget']
        spotsChNameWidget.textChanged.connect(self.onSpotsChannelTextChanged)
    
    def onExpPathsTextChanged(self):
        self.transformedDataNnetExp = None
        self.transformedDataTime = None
    
    def onSpotsChannelTextChanged(self):
        self.transformedDataNnetExp = None
        self.transformedDataTime = None
    
    @exception_handler
    def getNeuralNetworkModel(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params['Spots channel']
        anchor = 'spotPredictionMethod'
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        if spotPredictionMethodWidget.nnetModel is None:
            raise ValueError(
                'Neural network parameters were not initialized. Before trying '
                'to use it, you need to initialize the model\'s parameters by '
                'clicking on the settings button on the right of the selection '
                'box at the "Spots segmentation method" parameter.'
            )
        
        return spotPredictionMethodWidget.nnetModel    

    @exception_handler
    def getBioImageIOModel(self, section, anchor):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params[section]
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        if spotPredictionMethodWidget.bioImageIOModel is None:
            raise ValueError(
                'BioImage.IO model parameters were not initialized. Before trying '
                'to use it, you need to initialize the model\'s parameters by '
                'clicking on the settings button on the right of the selection '
                'box at the "Spots segmentation method" parameter.'
            )
        
        return spotPredictionMethodWidget.bioImageIOModel   
    
    def getSpotsThresholdMethod(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params['Spots channel']
        anchor = 'spotThresholdFunc'
        spotThresholdFuncWidget = spotsParams[anchor]['widget']
        return spotThresholdFuncWidget.currentText()

    def getRefChThresholdMethod(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        refChParams = ParamsGroupBox.params['Reference channel']
        anchor = 'refChThresholdFunc'
        thresholdFuncWidget = refChParams[anchor]['widget']
        return thresholdFuncWidget.currentText()
    
    @exception_handler
    def getSelectExpPath(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        filePathParams = ParamsGroupBox.params['File paths and channels']
        pathsToAnalyse = filePathParams['folderPathsToAnalyse']['widget'].text()
        caster = filePathParams['folderPathsToAnalyse']['dtype']
        pathsToAnalyse = caster(pathsToAnalyse)  
        if len(pathsToAnalyse) == 0:
            return 
        
        if len(pathsToAnalyse) == 1:
            return pathsToAnalyse[0]

        selectWin = acdc_widgets.QDialogListbox(
            'Select experiment to process',
            'You provided multiple experiment folders, but you can visualize '
            'only one at the time.\n\n'
            'Select which experiment folder to pre-process\n',
            pathsToAnalyse, multiSelection=False, parent=self
        )
        selectWin.exec_()
        if selectWin.cancel:
            return
        return selectWin.selectedItemsText[0]
    
    @exception_handler
    def getSpotsChannelEndname(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        filePathParams = ParamsGroupBox.params['File paths and channels']
        return filePathParams['spotsEndName']['widget'].text()
    
    @exception_handler
    def getNeuralNetParams(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params['Spots channel']
        anchor = 'spotPredictionMethod'
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        if spotPredictionMethodWidget.nnetModel is None:
            raise ValueError(
                'Neural network parameters were not initialized. Before trying '
                'to use it, you need to initialize the model\'s parameters by '
                'clicking on the settings button on the right of the selection '
                'box at the "Spots segmentation method" parameter.'
            )
        
        return spotPredictionMethodWidget.nnetParams  
    
    @exception_handler
    def getBioImageIOParams(self, section, anchor):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params[section]
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        if spotPredictionMethodWidget.bioImageIOModel is None:
            raise ValueError(
                'BioImage.IO model parameters were not initialized. Before trying '
                'to use it, you need to initialize the model\'s parameters by '
                'clicking on the settings button on the right of the selection '
                'box at the "Spots segmentation method" parameter.'
            )
        
        return spotPredictionMethodWidget.bioImageIOParams  
    
    def onRemoveHotPixelsToggled(self, checked):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        anchor = 'spotPredictionMethod'
        spotsParams = ParamsGroupBox.params['Spots channel']
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        spotPredictionMethodWidget.setDefaultRemoveHotPixels(checked)
    
    def onUseGpuToggled(self, checked):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        anchor = 'spotPredictionMethod'
        spotsParams = ParamsGroupBox.params['Spots channel']
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        spotPredictionMethodWidget.setDefaultUseGpu(checked)
    
    def onPixelWidthValueChanged(self, value):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        anchor = 'spotPredictionMethod'
        spotsParams = ParamsGroupBox.params['Spots channel']
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        spotPredictionMethodWidget.setDefaultPixelWidth(value)
    
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
        self.isAddAutoTunePoints = False
        self.isAutoTuningForegr = True
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.sigStartAutoTune.connect(self.startAutoTuning)
        autoTuneTabWidget.sigStopAutoTune.connect(self.stopAutoTuning)
        autoTuneTabWidget.sigAddAutoTunePointsToggle.connect(
            self.addAutoTunePointsToggled
        )
        
        autoTuneTabWidget.sigTrueFalseToggled.connect(
            self.autoTuningTrueFalseToggled
        )
        autoTuneTabWidget.sigColorChanged.connect(
            self.autoTuningColorChanged
        )
        autoTuneTabWidget.sigFeatureSelected.connect(
            self.autoTuningFeatureSelected
        )
        
        autoTuneTabWidget.sigYXresolMultiplChanged.connect(
            self.autoTuningYXresolMultiplChanged
        )
    
    def autoTuningYXresolMultiplChanged(self, value):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        metadataParams['yxResolLimitMultiplier']['widget'].setValue(value)
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        size = spots_zyx_radii[-1]*2
        self.ax2_BrushCircle.setSize(size)
        self.ax1_BrushCircle.setSize(size)
    
    def addAutoTunePoint(self, frame_i, z, y, x):
        self.setAutoTunePointSize()
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.addAutoTunePoint(frame_i, z, x, y)
    
    def doAutoTune(self):
        posData = self.data[self.pos_i]
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.autoTuneGroupbox.setDisabled(True)
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        
        trueItem = autoTuneGroupbox.trueItem
        coords = trueItem.coordsToNumpy(includeData=True)
        
        if len(coords) == 0:
            self.warnTrueSpotsAutoTuneNotAdded()
            return
        
        all_kwargs = self.paramsToKwargs()
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return
        kernel = posData.tuneKernel
        kernel.set_kwargs(all_kwargs)
        
        args = [kernel]
        kwargs = {
            'lab': None, 'image_data_cropped': None, 
            'segm_data_cropped': None, 'crop_to_global_coords': None
        }
        
        if kernel.ref_ch_endname() and kernel.ref_ch_data() is None:
            ref_ch_data = self.loadImageDataFromChannelName(
                kernel.ref_ch_endname()
            )
            
            on_finished_callback = (
                self.storeCroppedRefChDataAndStartTuneKernel, args, kwargs
            )
            
            self.startCropImageBasedOnSegmDataWorkder(
                ref_ch_data, posData.segm_data, 
                on_finished_callback=on_finished_callback
            )
        
        elif kernel.image_data() is None:
            on_finished_callback = (
                self.storeCroppedDataAndStartTuneKernel, args, kwargs
            )
            self.startCropImageBasedOnSegmDataWorkder(
                posData.img_data, posData.segm_data, 
                on_finished_callback=on_finished_callback
            )
        else:
            self.startTuneKernelWorker(kernel)
        
    def startTuneKernelWorker(self, kernel):
        worker = qtworkers.TuneKernelWorker(kernel)
        self.connectDefaultWorkerSlots(worker)
        worker.signals.finished.connect(self.tuneKernelWorkerFinished)
        self.threadPool.start(worker)
        return worker
    
    def startInspectHoveredSpotWorker(self):
        pass

    def tuneKernelWorkerFinished(self, result: tune.TuneResult):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.autoTuningButton.setChecked(False)
        autoTuneTabWidget.setTuneResult(result)
        
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph("""
            Auto-tuning process finished. Results will be displayed on the 
            `Tune parameters` tab.<br>
        """)
        msg.information(self, 'Auto-tuning finished', txt)
        
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
    
    def autoTuningFeatureSelected(self, editFeatureButton, featureText, colName):
        if colName.find('vs_ref_ch') != -1:
            ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
            filePathParams = ParamsGroupBox.params['File paths and channels']
            refChEndName = filePathParams['refChEndName']['widget'].text()
            if refChEndName:
                return
            refChEndName = self.askReferenceChannelEndname()
            if refChEndName is None:
                self.logger.info('Loading reference channel cancelled.')
                editFeatureButton.clearSelectedFeature()
                return
            filePathParams['refChEndName']['widget'].setText(refChEndName)
            
            self.logger.info(f'Loading "{refChEndName}" reference channel data...')
    
    def autoTuningAddItems(self):
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        self.ax1.addItem(autoTuneGroupbox.trueItem)
        self.ax1.addItem(autoTuneGroupbox.falseItem)
        self.autoTuningSetItemsColor(True)
    
    def updatePos(self):
        super().updatePos()
        self.initTuneKernel()
        
    def initTuneKernel(self):
        posData = self.data[self.pos_i]
        posData.tuneKernel = tune.TuneKernel()
        
    def connectLeftClickButtons(self):
        super().connectLeftClickButtons()
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        button = autoTuneTabWidget.addAutoTunePointsButton
        button.toggled.connect(button.onToggled)
    
    def addAutoTunePointsToggled(self, checked):
        self.isAddAutoTunePoints = checked
        self.zProjComboBox.setDisabled(checked)
        if checked:
            self.setAutoTunePointSize()
    
    def startAutoTuning(self):
        if not self.dataIsLoaded:
            return
        self.isAutoTuneRunning = True
        self.doAutoTune()
        
    def stopAutoTuning(self):
        if not self.dataIsLoaded:
            return
        self.isAutoTuneRunning = False
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.autoTuneGroupbox.setDisabled(False)
    
    def setRunNumbers(self):
        posData = self.data[self.pos_i]
        # Scan and determine run numbers
        pathScanner = io.expFolderScanner(
            posData.exp_path, logger_func=self.logger.info
        )
        pathScanner.getExpPaths(posData.exp_path)
        pathScanner.infoExpPaths(pathScanner.expPaths)
        run_nums = set()
        for run_num, expsInfo in pathScanner.paths.items():
            for expPath, expInfo in expsInfo.items():
                numPosSpotCounted = expInfo.get('numPosSpotCounted', 0)
                if numPosSpotCounted > 0:
                    run_nums.add(run_num)
        run_nums = sorted(list(run_nums))
        
        self.loaded_exp_run_nums = run_nums
    
    def initDefaultParamsNnet(self, posData):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        anchor = 'spotPredictionMethod'
        spotsParams = ParamsGroupBox.params['Spots channel']
        if posData is not None:
            spotsParams[anchor]['widget'].setPosData(self.data[self.pos_i])
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        do_remove_hot_pixels = (
            preprocessParams['removeHotPixels']['widget'].isChecked()
        )
        
        configParams = ParamsGroupBox.params['Configuration']
        use_gpu = configParams['useGpu']['widget'].isChecked()
        
        PhysicalSizeX = posData.PhysicalSizeX
        
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        spotPredictionMethodWidget.setDefaultPixelWidth(PhysicalSizeX)
        spotPredictionMethodWidget.setDefaultRemoveHotPixels(
            do_remove_hot_pixels
        )
        spotPredictionMethodWidget.setDefaultUseGpu(use_gpu)
    
    def setAnalysisParameters(self):
        paramsGroupbox = self.computeDockWidget.widget().parametersQGBox
        posData = self.data[self.pos_i]
        self.computeDockWidget.widget().loadPreviousParamsButton.setStartPath(
            posData.pos_path
        )
        segmFilename = os.path.basename(posData.segm_npz_path)
        segmEndName = segmFilename[len(posData.basename):]
        runNum = max(self.loaded_exp_run_nums, default=0) + 1
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
        self.initDefaultParamsNnet(posData)
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
        autoTuneScrollArea = guiTabControl.autoTuneTabWidget
        verticalScrollbar = paramsScrollArea.verticalScrollBar()
        groupboxWidth = autoTuneScrollArea.size().width()
        scrollbarWidth = verticalScrollbar.size().width()
        minWidth = groupboxWidth + scrollbarWidth + 30
        self.resizeDocks([self.computeDockWidget], [minWidth], Qt.Horizontal)
        self.showParamsDockButton.click()
    
    def zSliceScrollBarActionTriggered(self, action):
        super().zSliceScrollBarActionTriggered(action)
        if action != SliderMove:
            return
        posData = self.data[self.pos_i]
        self.spotsItems.setData(
            posData.frame_i, z=self.currentZ(checkIfProj=True)
        )
        self.setVisibleAutoTunePoints()
    
    def framesScrollBarMoved(self, frame_n):
        super().framesScrollBarMoved(frame_n)
        self.setVisibleAutoTunePoints()
    
    def setVisibleAutoTunePoints(self):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        z = self.currentZ()
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.setVisibleAutoTunePoints(frame_i, z)
    
    def updateZproj(self, how):
        super().updateZproj(how)
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        addAutoTunePointsButton = autoTuneTabWidget.addAutoTunePointsButton
        addAutoTunePointsButton.setDisabled(how != 'single z-slice')            
    
    def updateAllImages(self, *args, **kwargs):
        posData = self.data[self.pos_i]
        super().updateAllImages(*args, **kwargs)
        self.spotsItems.setData(
            posData.frame_i, z=self.currentZ(checkIfProj=True)
        )
        self.setVisibleAutoTunePoints()
    
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
    
    def warnClosingWhileAnalysisIsRunning(self):
        txt = html_func.paragraph("""
            The analysis is still running (see progress in the terminal).<br><br>
            Are you sure you want to close and abort the analysis process?<br>
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        noButton, yesButton = msg.warning(
            self, 'Analysis still running!', txt,
            buttonsTexts=(
                'No, do not close', 
                'Yes, stop analysis and close spotMAX'
            )
        )
        return msg.clickedButton == yesButton
    
    def closeEvent(self, event):
        if self.isAnalysisRunning:
            proceed = self.warnClosingWhileAnalysisIsRunning()
            if not proceed:
                event.ignore()
                return
        super().closeEvent(event)
        if not sys.stdout == self.logger.default_stdout:
            return
        if not self._executed:
            return
        print('**********************************************')
        print(f'SpotMAX closed. {get_salute_string()}')
        print('**********************************************')