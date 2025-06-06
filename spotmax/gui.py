import warnings
import os
import shutil
import datetime
import traceback
import re
from queue import Queue

from typing import Tuple

import numpy as np
import pandas as pd

import skimage.measure

from qtpy.QtCore import (
    Qt, QTimer, QThreadPool, QMutex, QWaitCondition, QEventLoop, 
    QObject
)
from qtpy.QtGui import QIcon, QGuiApplication, QMouseEvent
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
from cellacdc import io as acdc_io
from cellacdc.myutils import get_salute_string, determine_folder_type
from cellacdc import qrc_resources
from cellacdc import base_cca_dict
from cellacdc import myutils as acdc_myutils

from . import qtworkers, io, printl, dialogs
from . import logs_path, html_path, html_func
from . import widgets, config
from . import tune, utils
from . import core
from . import transformations
from . import icon_path
from . import issues_url
from . import features
from . import prompts
from . import spotmax_path
from . import _warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

LINEAGE_COLUMNS = list(base_cca_dict.keys())

ANALYSIS_STEP_RESULT_SLOTS = {
    'gaussSigma': '_displayGaussSigmaResult',
    'refChGaussSigma': '_displayGaussSigmaResult',
    'refChRidgeFilterSigmas': '_displayRidgeFilterResult',
    'removeHotPixels': '_displayRemoveHotPixelsResult',
    'sharpenSpots': '_displaySharpenSpotsResult',
    'spotPredictionMethod': '_displaySpotPredictionResult',
    'spotDetectionMethod': '_displaySpotDetectionResult',
    'refChSegmentationMethod': '_displaySegmRefChannelResult',
    'spotMinSizeLabels': '_displaySpotFootprint',
    'extend3DsegmRange': '_displayExtend3DsegmRange',
}

PARAMS_SLOTS = {
    'gaussSigma': ('sigComputeButtonClicked', '_computeGaussFilter'),
    'refChGaussSigma': ('sigComputeButtonClicked', '_computeRefChGaussSigma'),
    'refChRidgeFilterSigmas': (
        'sigComputeButtonClicked', '_computeRefChRidgeFilter'
    ),
    'removeHotPixels': ('sigComputeButtonClicked', '_computeRemoveHotPixels'),
    'sharpenSpots': ('sigComputeButtonClicked', '_computeSharpenSpots'),
    'spotPredictionMethod': ('sigComputeButtonClicked', '_computeSpotPrediction'),
    'spotDetectionMethod': ('sigComputeButtonClicked', '_computeSpotDetection'),
    'refChSegmentationMethod': (
        'sigComputeButtonClicked', '_computeSegmentRefChannel'
    ),
    'spotMinSizeLabels': (
        'sigComputeButtonClicked', '_computeSpotFootprint'
    ),
    'extend3DsegmRange' : (
        'sigComputeButtonClicked', '_computeExtend3DsegmRange'
    ),
}

SliderSingleStepAdd = acdc_gui.SliderSingleStepAdd
SliderSingleStepSub = acdc_gui.SliderSingleStepSub
SliderPageStepAdd = acdc_gui.SliderPageStepAdd
SliderPageStepSub = acdc_gui.SliderPageStepSub
SliderMove = acdc_gui.SliderMove

class spotMAX_Win(acdc_gui.guiWin):
    def __init__(self, app, executed=False, debug=False, **kwargs):
        super().__init__(app, **kwargs)

        self._version = kwargs.get('version')
        self._appName = 'SpotMAX'
        self._executed = executed
    
    def run(self, module='spotmax_gui', logs_path=logs_path):
        super().run(module=module, logs_path=logs_path)

        self.initGui()
        self.createThreadPool()
        self.setMaxNumThreadsNumbaParam()
        self.hideCellACDCtools()
    
    def setWindowIcon(self, icon=None):
        if icon is None:
            icon = QIcon(icon_path)
        super().setWindowIcon(icon)
    
    def setWindowTitle(self, title="SpotMAX - GUI"):
        super().setWindowTitle(title)
    
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
    
    def dragEnterEvent(self, event):
        dragged_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(dragged_path):
            exp_path = dragged_path
            basename = os.path.basename(dragged_path)
            if basename.find('Position_')!=-1 or basename=='Images':
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.acceptProposedAction()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        dropped_path = event.mimeData().urls()[0].toLocalFile()
        self.logger.info(f'Dragged and dropped path "{dropped_path}"')
        basename = os.path.basename(dropped_path)
        if os.path.isdir(dropped_path):
            exp_path = dropped_path
            self.openFolder(exp_path=exp_path)
        elif dropped_path.endswith('.ini'):
            guiTabControl = self.computeDockWidget.widget()
            guiTabControl.loadPreviousParams(dropped_path)
        else:
            self.openFile(file_path=dropped_path)
    
    def gui_setCursor(self, modifiers, event):
        cursorsInfo = super().gui_setCursor(modifiers, event)
        noModifier = modifiers == Qt.NoModifier
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        
        setAutoTuneCursor = (
            self.isAddAutoTunePoints 
            and not event.isExit()
            and noModifier
            and autoTuneTabWidget.isYXresolMultiplActive
        )
        cursorsInfo['setAutoTuneCursor'] = setAutoTuneCursor
        
        setAutoTuneZplaneCursor = (
            self.isAddAutoTunePoints 
            and not event.isExit()
            and noModifier
            and autoTuneTabWidget.isZresolLimitActive
        )
        cursorsInfo['setAutoTuneZplaneCursor'] = setAutoTuneZplaneCursor
        
        setEditResultsCursor = (
            self.isEditingResults
            and not event.isExit()
            and noModifier
        )
        cursorsInfo['setEditResultsCursor'] = setEditResultsCursor
        overrideCursor = self.app.overrideCursor()
        
        isCrossCursor = (
            overrideCursor is None
            and (setAutoTuneCursor or setEditResultsCursor)
        )
        if isCrossCursor:
            self.app.setOverrideCursor(Qt.CrossCursor)
        return cursorsInfo
    
    def gui_hoverEventImg1(self, event, isHoverImg1=True):
        cursorsInfo = super().gui_hoverEventImg1(event, isHoverImg1=isHoverImg1)
        if cursorsInfo is None:
            return
        
        if event.isExit():
            self.LinePlotItem.clearData()
            return

        x, y = event.pos()
        if cursorsInfo['setAutoTuneCursor']:
            self.setHoverCircleAutoTune(x, y)
        elif cursorsInfo['setEditResultsCursor']:
            self.setHoverCircleEditResults(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
        
        if cursorsInfo['setAutoTuneZplaneCursor']:
            self.setAutoTuneZplaneCursorData(x, y)
        else:
            self.LinePlotItem.clearData()
        
        self.onHoverAutoTunePoints(x, y)
        self.onHoverInspectPoints(x, y)
        self.onHoverRefChMasks(x, y)
    
    def onHoverRefChMasks(self, x, y):
        if self.dfs_ref_ch_features is None:
            return
        
        if not self.dfs_ref_ch_features:
            return
        
        ID = self.currentLab2D[int(y), int(x)]
        if ID == 0:
            if self.highlightedRefChObjItem.image is not None:
                self.highlightedRefChObjLab[:] = 0
                self.highlightedRefChObjItem.clear()
            return
        
        inspectResultsTab = self.computeDockWidget.widget().inspectResultsTab
        if not inspectResultsTab.areFeaturesSelected():
            return
        
        if self.highlightedRefChObjLab[int(y), int(x)] == ID:
            return
        
        if self.isSegm3D:
            subObjID = self.refChSubObjsLab[self.z_lab(), int(y), int(x)]
        else:
            subObjID = self.refChSubObjsLab[int(y), int(x)]
            
        self.highlightRefChObject(ID, subObjID)
        
        posData = self.data[self.pos_i]
        inspectResultsTab = self.computeDockWidget.widget().inspectResultsTab
        inspectResultsTab.setInspectedRefChFeatures(
            self.dfs_ref_ch_features[self.pos_i], posData.frame_i, ID, subObjID
        )
    
    def highlightRefChObject(self, ID, subObjID):
        inspectResultsTab = self.computeDockWidget.widget().inspectResultsTab
        isWholeObjRequested = inspectResultsTab.isWholeObjRequested()
        
        lut = np.zeros((2, 4), dtype=np.uint8)
        rgb = self.lut[ID].copy() 
        lut[1, :-1] = rgb
        lut[1, -1] = 178  
        self.highlightedRefChObjItem.setLookupTable(lut)
        
        posData = self.data[self.pos_i]
        obj_idx = posData.IDs_idxs[ID]
        obj = posData.rp[obj_idx]
        sub_obj = None
        if not isWholeObjRequested:
            sub_obj = self.localRefChSubObjsLabRp[ID][subObjID]
        
        obj_slice = self.getObjSlice(obj.slice)
        obj_image = self.getObjImage(obj.image, obj.bbox)
        self.highlightedRefChObjLab[:] = 0
        
        if sub_obj is None:
            self.highlightedRefChObjLab[obj_slice][obj_image] = ID
        else:
            sub_obj_slice = self.getObjSlice(sub_obj.slice)
            sub_obj_image = self.getSubObjImage(obj, sub_obj)
            localLab = self.highlightedRefChObjLab[obj_slice]
            localLab[sub_obj_slice][sub_obj_image] = ID
        
        self.highlightedRefChObjItem.setImage(self.highlightedRefChObjLab)
    
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
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        
        canAddPointAutoTune = (
            self.isAddAutoTunePoints and left_click
            and autoTuneTabWidget.isYXresolMultiplActive
        )
        
        canEditResults = (
            self.isEditingResults and left_click
        )
        
        x, y = event.pos().x(), event.pos().y()
        ID = self.getIDfromXYPos(x, y)
        if ID is None:
            return
        
        if canAddPointAutoTune:
            z = self.currentZ()
            self.addAutoTunePoint(posData.frame_i, z, y, x)
        
        if canEditResults:
            z = self.currentZ()
            self.spotsItems.editPoint(
                posData.frame_i, z, y, x, self.img1.image
            )
    
    def gui_createPlotItems(self):
        super().gui_createPlotItems()
        
        self.LinePlotItem = widgets.ParentLinePlotItem(
            pen=pg.mkPen('r', width=2)
        )
        self.topLayerItems.append(self.LinePlotItem)
        childrenLineItem = self.LinePlotItem.addChildrenItem()
        self.topLayerItemsRight.append(childrenLineItem)
    
    def gui_createRegionPropsDockWidget(self):
        super().gui_createRegionPropsDockWidget(side=Qt.RightDockWidgetArea)
        self.gui_createParamsDockWidget()
    
    def gui_createParamsDockWidget(self):
        self.computeDockWidget = widgets.DockWidget('SpotMAX Tab Control', self)
        guiTabControl = dialogs.guiTabControl(
            parent=self.computeDockWidget, logging_func=self.logger.info
        )
        guiTabControl.addAutoTuneTab()
        guiTabControl.addInspectResultsTab()
        guiTabControl.addLeftClickButtons(self.checkableQButtonsGroup)
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
        self.computeDockWidget.widget().sigParametersLoaded.connect(
            self.parametersLoaded
        )
        self.addSpotsCoordinatesAction.triggered.connect(
            self.addSpotsCoordinatesTriggered
        )
        
        inspectTabWidget = self.computeDockWidget.widget().inspectResultsTab
        inspectTabWidget.loadAnalysisButton.clicked.connect(
            self.loadAnalysisPathSelected
        )
        inspectTabWidget.loadRefChDfButton.clicked.connect(
            self.loadReferenceChannelFeaturesTable
        )
    
    def checkDataLoaded(self):
        if self.isDataLoaded:
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
    
    def warnWrongRefChOrMasksLoaded(self, loaded_segm_filepath):
        loaded_segm_filename = os.path.basename(loaded_segm_filepath)
        path_to_browse = os.path.dirname(loaded_segm_filepath)
        html_pattern = (
            '_run_num&lt;run_number&gt;_&lt;ref_ch_name&gt;'
            '_ref_ch_segm_mask_&lt;appended_text&gt;.npz'
        )
        txt = html_func.paragraph(f"""
            The loaded segmentation file does <b>not have valid reference 
            channel masks</b>.<br><br>
            In order to inspect reference channel features make sure you 
            <b>load the correct channel and the correct masks</b>.<br><br>
            The masks filename ends with the following pattern:<br><br>
            <code>{html_pattern}</code>
            <br><br>
            Loaded segmentation file:
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Loaded segmentation data is not valid ref. ch. masks', txt, 
            commands=(loaded_segm_filepath,), path_to_browse=path_to_browse
        )
    
    def warnRefChFeaturesTableNotFound(
            self, file_not_found, spotmax_out_folder
        ):
        files = utils.listdir(spotmax_out_folder)
        files = utils.sort_strings_by_template(files, file_not_found)
        
        txt = html_func.paragraph(f"""
            The requested file <code>{file_not_found}</code> was 
            <b>not found</b>.<br><br>
            
            Did you load the <b>correct reference channel masks</b>?<br><br>
            
            See below the present files sorted by similarity to requested file:
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Reference channel features table not found', txt, 
            path_to_browse=spotmax_out_folder, 
            detailsText='\n'.join(files)
        )
        
    def loadReferenceChannelFeaturesTable(self):
        posData = self.data[self.pos_i]  
        
        run_num_info = io.run_num_and_appended_text_from_segm_path(
            posData.segm_npz_path, posData.user_ch_name, 
            channel_type='ref_ch'
        )
        
        if run_num_info is None:
            self.warnWrongRefChOrMasksLoaded(posData.segm_npz_path)
            return
        
        run_num, appended_text = run_num_info
        df_ref_ch_filename, found = io.get_ref_channel_dfs_features_filename(
            run_num, appended_text, posData.spotmax_out_path
        )
        
        if not found:
            self.warnRefChFeaturesTableNotFound(
                df_ref_ch_filename, posData.spotmax_out_path
            )
            return
        
        self.logger.info(
            f'Loading reference channel features tables `{df_ref_ch_filename}`...'
        )
        self.dfs_ref_ch_features = []
        for _posData in self.data:
            spotmax_files = utils.listdir(_posData.spotmax_out_path)
            df_files = [
                file for file in spotmax_files if file==df_ref_ch_filename
            ]
            if not df_files:
                self.dfs_ref_ch_features.append(None)
                continue
            
            df_file = df_files[0]
            df_filepath = os.path.join(
                _posData.spotmax_out_path, df_file
            )
            df = io.load_df_ref_ch_features(df_filepath)
            self.dfs_ref_ch_features.append(df)
        
        inspectResultsTab = self.computeDockWidget.widget().inspectResultsTab
        inspectResultsTab.setLoadedRefChannelFeaturesFile(df_ref_ch_filename)
        
        self.initHighlightRefChannelObjImage()
    
    def getSubObjImage(self, obj, sub_obj):
        if not self.isSegm3D:
            return sub_obj.image
        
        zProjHow = self.zProjComboBox.currentText()
        isZslice = zProjHow == 'single z-slice'
        if not isZslice:
            return sub_obj.image.max(axis=0)
        
        min_z = obj.bbox[0]
        z = self.z_lab()
        local_z = z - min_z
        sub_obj_local_z = local_z - sub_obj.bbox[0]
        return sub_obj.image[sub_obj_local_z]
    
    def initHighlightRefChannelObjImage(self):
        if self.dfs_ref_ch_features is None:
            return
        
        if not self.dfs_ref_ch_features:
            return
        
        self.highlightedRefChObjItem = pg.ImageItem()
        self.ax1.addItem(self.highlightedRefChObjItem)
        
        posData = self.data[self.pos_i]
        self.highlightedRefChObjLab = np.zeros(self.currentLab2D.shape, np.uint8)   
        self.refChSubObjsLab = np.zeros_like(posData.lab)
        self.localRefChSubObjsLabRp = {}
        for obj in posData.rp:
            ref_ch_lab = skimage.measure.label(obj.image)
            self.refChSubObjsLab[obj.slice][obj.image] = ref_ch_lab[obj.image]
            ref_ch_rp = {
                sub_obj.label:sub_obj
                for sub_obj in skimage.measure.regionprops(ref_ch_lab)
            }
            self.localRefChSubObjsLabRp[obj.label] = ref_ch_rp
    
    def warnNoSpotsFilesFound(self, spotmax_out_path):
        txt = html_func.paragraph(f"""
            There are no valid files in the following folder:<br><br>
            <code>{spotmax_out_path}</code><br><br>
            This could be because the number of detected spots was 0,  
            you did not run any analysis yet, or the analysis ended 
            with errors.<br><br>
            If you need help with this feel free to reach out on or 
            {html_func.href('GitHub page', issues_url)}.
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'No valid files found', txt)
    
    def addSpotsCoordinatesTriggered(self, checked=True, selected_file=None):
        posData = self.data[self.pos_i]
        if not os.path.exists(posData.spotmax_out_path):
            _warnings.warnSpotmaxOutFolderDoesNotExist(
                posData.spotmax_out_path, qparent=self
            )
        df_spots_files = {}
        for _posData in self.data:
            df_spots_files[_posData.spotmax_out_path] = (
                _posData.getSpotmaxSingleSpotsfiles()
            )
        if not df_spots_files:
            self.warnNoSpotsFilesFound(posData.spotmax_out_path)
            return
        
        self.spotsItems.setPosition(posData)
        toolbutton = self.spotsItems.addLayer(
            df_spots_files, selected_file=selected_file
        )
        for _posData in self.data:
            self.spotsItems.setPosition(_posData)
            self.spotsItems.loadSpotsTables()
            
        self.spotsItems.setPosition(posData)
        self.spotsItems.loadSpotsTables()
            
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
        guiTabControl = self.computeDockWidget.widget()
        guiTabControl.setCurrentIndex(2)
        
        inspectResultsTab = self.computeDockWidget.widget().inspectResultsTab
        inspectResultsTab.setLoadedData(
            self.spotsItems, posData.img_data, posData.segm_data
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
        QTimer.singleShot(100, super().resizeRangeWelcomeText)
    
    def _disableAcdcActions(self, *actions):
        for action in actions:
            action.setVisible(False)
            action.setDisabled(True)
    
    def isNeuralNetworkRequested(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params['Spots channel']
        anchor = 'spotPredictionMethod'
        return spotsParams[anchor]['widget'].currentText() == 'spotMAX AI'
    
    def isBioImageIOModelRequested(
            self, section='Spots channel', anchor='spotPredictionMethod'
        ):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params[section]
        return spotsParams[anchor]['widget'].currentText() == 'BioImage.IO model'
    
    def isSpotiflowRequested(
            self, section='Spots channel', anchor='spotPredictionMethod'
        ):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params[section]
        return spotsParams[anchor]['widget'].currentText() == 'Spotiflow'
    
    def isPreprocessAcrossExpRequired(self):
        if not self.isNeuralNetworkRequested():
            return False
        
        nnetParams = self.getNeuralNetParams()
        if nnetParams is None:
            return False
        
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
        if nnetParams is None:
            return False
        
        if not nnetParams['init']['preprocess_across_timepoints']:
            # Pre-processing not requested
            return False
        
        return True
    
    def warnRemovingPointCellIDzero(self):
        txt = html_func.paragraph("""
            It looks like you want to remove a spot that was detected 
            outside of the segmented objects.<br><br>
            While you can of course do that, these spots were already removed 
            by SpotMAX in the tables called <code>1_valid_spots</code> and 
            <code>0_detected_spots</code>.<br><br>
            You probably <b>want to edit the results in those tables</b>.<br><br>
            Anyway, do you want to continue editing the loaded spots?
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        _, yesButton = msg.question(
            self, 'Editing detected spots', txt, 
            buttonsTexts=(
                'No, I will load another table, thanks', 
                'Yes, let me edit this table'
            )
        )
        return msg.clickedButton == yesButton
    
    def reInitGui(self):
        super().reInitGui()
        
        self.loadCustomAnnotationsAction.setDisabled(True)
        self.addCustomAnnotationAction.setDisabled(True)
        
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
        self.computeDockWidget.widget().inspectResultsTab.reinitState()
        
        self.transformedDataNnetExp = None
        self.transformedDataTime = None
        self.isEditingResults = False
        self.dfs_ref_ch_features = None
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.addAutoTunePointsButton.setChecked(False)
        
        self.checkReinitTuneKernel()
        
        inspectResultsTab = self.computeDockWidget.widget().inspectResultsTab
        inspectResultsTab.resetLoadedRefChannelFeaturesFile()
        
        try:
            self.ax1.removeItem(self.highlightedRefChObjItem)
        except Exception as err:
            pass
    
    def hideCellACDCtools(self):
        # self.measurementsMenu.setDisabled(False)
        # self.setMeasurementsAction.setText('Set Cell-ACDC measurements...')
        
        self.trackingMenu.menuAction().setVisible(False)
        self.measurementsMenu.menuAction().setVisible(False)
        self.segmentMenu.menuAction().setVisible(False)
        
        removeActions = False
        for action in self.settingsMenu.actions():
            if action.isSeparator():
                removeActions = True
                continue
            if not removeActions:
                continue
            
            if removeActions:
                self.settingsMenu.removeAction(action)
    
    def initGui(self):
        self.isAnalysisRunning = False
        
        self._disableAcdcActions(
            self.newAction, self.manageVersionsAction, self.openFileAction
        )
        self.ax2.hide()
        
        self.lastLoadedIniFilepath = None
        
        self.aboutSmaxAction = QAction("About SpotMAX", self)
        self.helpMenu.addAction(self.aboutSmaxAction)
        self.aboutSmaxAction.triggered.connect(self.showAboutSmax)
        
        self._setWelcomeText()
    
    def showAboutSmax(self):
        win = dialogs.AboutSpotMAXDialog(parent=self)
        win.exec_()
    
    def parametersLoaded(self, ini_filepath):
        self.lastLoadedIniFilepath = ini_filepath
    
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
    def runAnalysis(self, ini_filepath, is_tempfile, start=True):
        self.isAnalysisRunning = True
        self.stateBeforeStartingAnalysis = self.windowState()
        self.setWindowState(Qt.WindowMinimized)
        self.setDisabled(True)
        self.logger.info('Starting SpotMAX analysis...')
        self._analysis_started_datetime = datetime.datetime.now()
        self.funcDescription = 'starting analysis process'
        
        # Close logger to pass the file to cli logger
        log_handler = self.logger.handlers[0]
        self.logger.removeHandler(log_handler)
        log_handler.close()
        
        self.logger.info(
            f'Starting SpotMAX analysis worker with log file: {self.log_path}'
        )
        
        worker = qtworkers.AnalysisWorker(
            ini_filepath, is_tempfile, log_filepath=self.log_path
        )

        command = worker.getCommandForClipboard()
        widgets.toClipboard(command)
        
        worker.signals.finished.connect(self.analysisWorkerFinished)
        worker.signals.progress.connect(self.workerProgress)
        # worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        # worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        if start:
            self.threadPool.start(worker)
        return worker
    
    def promptAnalysisWorkerFinished(self, args):
        self.isAnalysisRunning = False
        self.setWindowState(self.stateBeforeStartingAnalysis)
        self.setDisabled(False)
        ini_filepath, is_tempfile = args[:2]
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
            'SpotMAX analysis finished!\n\n'
            f'    * Started on: {self._analysis_started_datetime.strftime(ff)}\n'
            f'    * Ended on: {self._analysis_finished_datetime.strftime(ff)}\n'
            f'    * Total execution time = {delta_sec} H:mm:ss\n'
        )
        line_str = '-'*100
        close_str = '*'*100
        msg_kwargs = {
            'path_to_browse': os.path.dirname(log_path),
            'browse_button_text': 'Show log file'
        }
        if errors:
            details = '\n\n'.join(errors)
            msg_kwargs['detailsText'] = details
            txt = txt.replace(
                'SpotMAX analysis finished!', 
                'SpotMAX analysis ended with ERRORS'
            )
            txt = (
                f'{txt}\n'
                'WARNING: Analysis ended with errors. '
                'See summary of errors below and more details in the '
                'log file:'
            )
            msg_func = 'critical'
            msg_kwargs['commands'] = (log_path, )
        else:
            msg_func = 'information'
        self.logger.info(f'{line_str}\n{txt}\n{close_str}')
        txt = html_func.paragraph(txt.replace('\n', '<br>'))
        txt = re.sub('`(.+)`', r'<code>\1</code>', txt)
        msg = acdc_widgets.myMessageBox()
        
        msg_args = (self, 'SpotMAX analysis finished', txt)
        getattr(msg, msg_func)(*msg_args, **msg_kwargs)
        return msg_func == 'information'
    
    def analysisWorkerFinished(self, args):
        self.reOpenLogger()
        success = self.promptAnalysisWorkerFinished(args)
        
        if not success:
            return
        
        if self.isDataLoaded:
            runNumberLastAnalysis = args[2]
            self.askVisualizeResults(runNumberLastAnalysis)
        else:
            self.instructHowToVisualizeResults()
    
    def instructHowToVisualizeResults(self):
        txt = html_func.paragraph("""
            To visualize results load the image data, then go to the 
            <code>Inspect and/or edit results</code> tab (top-left),<br>
            and click on <code>Load results from previous analysis...</code>.
            <br><br>
            Have fun!
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.question(self, 'How to visualize results', txt)
    
    def reOpenLogger(self):
        file_handler = utils.logger_file_handler(self.log_path, mode='a')
        self.logger.addHandler(file_handler)
    
    def askVisualizeResults(self, runNumberLastAnalysis):        
        txt = html_func.paragraph(
            'Do you want to visualize the results in the GUI?'
        )
        msg = acdc_widgets.myMessageBox(wrapText=False)
        _, yesButton = msg.question(
            self, 'Visualize results?', txt, buttonsTexts=('No', 'Yes')
        )
        if msg.clickedButton != yesButton:
            return 
        
        if not self.isDataLoaded:
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
        
        posData = self.data[self.pos_i]
        sm_files = posData.getSpotmaxSingleSpotsfiles()
        selected_file = None
        for file in sm_files:
            if file.startswith(f'{selected_file}_0_detect'):
                selected_file = file
                break
            
        self.addSpotsCoordinatesTriggered(selected_file=selected_file)
        
    
    def gui_createActions(self):
        super().gui_createActions()

        self.addSpotsCoordinatesAction = QAction(self)
        self.addSpotsCoordinatesAction.setIcon(QIcon(":addPlotSpots.svg"))
        self.addSpotsCoordinatesAction.setToolTip(
            'Visualize detected spots from a previous analysis'
        )
        
        # Disable save actions from acdc gui
        self.saveAsAction.setDisabled(True)
        self.saveAction.setDisabled(True)
        self.quickSaveAction.setDisabled(True)
        self.newAction.setDisabled(True)
    
    def gui_createToolBars(self):
        super().gui_createToolBars()

        # self.addToolBarBreak(Qt.LeftToolBarArea)
        self.spotmaxToolbar = QToolBar("SpotMAX toolbar", self)
        self.spotmaxToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(Qt.LeftToolBarArea, self.spotmaxToolbar)
        self.spotmaxToolbar.addAction(self.addSpotsCoordinatesAction)
        self.spotmaxToolbar.setVisible(False)
        self.spotsItems = widgets.SpotsItems(self)
    
    def gui_addTopLayerItems(self):
        super().gui_addTopLayerItems()

    def gui_connectEditActions(self):
        super().gui_connectEditActions()
    
    def logFilesInSpotmaxOutPath(self, spotmax_out_path):
        if not os.path.exists(spotmax_out_path):
            return
        
        files_format = '\n'.join([
            f'  - {file}' for file in utils.listdir(spotmax_out_path)
        ])
        if not files_format:
            self.logger.info('SpotMAX files are not present')
            return
        
        sep = '-'*100
        self.logger.info(
            f'{sep}\nFiles present in the first spotMAX_output folder loaded:\n\n'
            f'{files_format}\n{sep}'
        )
    
    def loadingDataCompleted(self):
        super().loadingDataCompleted()
        posData = self.data[self.pos_i]
        
        self.logFilesInSpotmaxOutPath(posData.spotmax_out_path)
        
        self.setWindowTitle(f'SpotMAX - GUI - "{posData.exp_path}"')
        self.spotmaxToolbar.setVisible(True)
        self.computeDockWidget.widget().initState(True)
        
        self.isAutoTuneTabActive = False
        
        self.setRunNumbers()
        
        self.computeDockWidget.widget().setLoadedPosData(posData)
        
        self.setAnalysisParameters()
        self.connectParamsGroupBoxSignals()
        self.autoTuningAddItems()
        self.initTuneKernel()
        self.hideAcdcToolbars()
        self.connectInspectResultsTab()
        
        self.setFocusGraphics()
        
        self.modeToolBar.setVisible(False)
        
        QTimer.singleShot(300, self.autoRange)
    
    def connectInspectResultsTab(self):
        inspectResultsTab = self.computeDockWidget.widget().inspectResultsTab
        inspectResultsTab.sigEditResultsToggled.connect(
            self.editResultsToggled
        )
        inspectResultsTab.sigSaveEditedResults.connect(
            self.saveEditedResultsClicked
        )
        inspectResultsTab.sigComputeFeatures.connect(
            self.computeFeaturesEditedResultsClicked
        )
    
    def warnEditedTableFileExists(self, existing_filepath):
        folderpath = os.path.dirname(existing_filepath)
        txt = html_func.paragraph(f"""
            The following <b>file already exists</b>:<br><br>
            <code>{existing_filepath}</code><br><br>
            What should I do?                        
        """)
        buttonsTexts = ('Cancel', 'Overwrite existing file')
        msg = acdc_widgets.myMessageBox(wrapText=False)
        _, overwriteButton = msg.warning(
            self, 'File exists', txt, 
            buttonsTexts=buttonsTexts, 
            path_to_browse=folderpath
        )
        return msg.clickedButton == overwriteButton
    
    def saveEditedResultsClicked(
            self, src_df_filename, text_to_add, prompt_info=True
        ):
        self.logger.info('Saving edited tables...')
        
        parts = io.df_spots_filename_parts(src_df_filename)
        run_num, df_id, df_text, desc, ext = parts
        if not text_to_add.startswith('_'):
            text_to_add = f'_{text_to_add}'
        
        dst_filename = f'{run_num}_4_{df_id}_{df_text}{desc}{text_to_add}'
        toolbutton = self.spotsItems.getActiveButton()
        
        saved_filepaths = []
        for posData in self.data:
            spotmax_output_folderpath = posData.spotmax_out_path
            self.spotsItems.setPosition(posData)
            self.spotsItems.loadSpotsTables()
            
            df = toolbutton.df
            df = (
                df.reset_index().set_index(['frame_i', 'Cell_ID', 'spot_id'])
                .sort_index()
            )
            aggr_dst_filename = f'{dst_filename}_aggregated.csv'
            aggr_dst_filepath = os.path.join(
                spotmax_output_folderpath, aggr_dst_filename
            )
            
            if os.path.exists(aggr_dst_filepath):
                proceed = self.warnEditedTableFileExists(aggr_dst_filepath)
                if not proceed:
                    self.logger.info('Saving edited tables cancelled.')
                    continue
            
            dst_filepath = io.save_df_spots(
                df, spotmax_output_folderpath, dst_filename, 
                extension=f'.{ext}'
            )
            saved_filepaths.append(dst_filepath)
            
            images_path_filename = f'{posData.basename}{dst_filename}'
            images_dst_filepath = io.save_df_spots(
                df, posData.images_path, images_path_filename, 
                extension=f'.{ext}'
            )
            saved_filepaths.append(images_dst_filepath)
            
            posData = self.data[self.pos_i]
            segm_data = posData.segm_data
            df_agg = features.df_spots_to_aggregated(df)
            df_agg = features.add_missing_cells_to_df_agg_from_segm(
                df_agg, segm_data
            )
            src_df_agg = io.load_df_agg_from_df_spots_filename(
                spotmax_output_folderpath, src_df_filename
            )
            df_agg = features.add_missing_cols_from_src_df_agg(
                df_agg, src_df_agg
            )
            
            # Add columns from acdc_df in case the segm file changed and for 
            # example it has a new cell that was not in previous spotmax analysis
            # e.g., spotmax analysis has IDs 1, 2 but segm file (hence acdc_output)
            # has also ID 3 because it was added after analysis --> this cell 
            # does not have the columns from acdc_df yet
            df_agg = features.add_columns_from_acdc_output_file(
                df_agg, posData.acdc_df
            )
            df_agg.to_csv(aggr_dst_filepath)
            
            saved_filepaths.append(aggr_dst_filepath)
        
        # Back to current pos
        posData = self.data[self.pos_i]
        self.spotsItems.setPosition(posData)
        self.spotsItems.loadSpotsTables()
        
        saved_filepaths_format = '\n'.join(saved_filepaths)
        self.logger.info(
            f'Edited tables saved to:\n\n{saved_filepaths_format}\n'
        )
        
        if prompt_info:
            txt = html_func.paragraph("""
                Edited tables saved!<br><br>
                See below the list of new files created.
            """)
            msg = acdc_widgets.myMessageBox(wrapText=False)
            msg.information(
                self, 'Edited tables saved', txt, 
                detailsText=saved_filepaths_format, 
                path_to_browse=spotmax_output_folderpath, 
                wrapDetails=False
            )
        return f'{dst_filename}.{ext}', f'{desc}{text_to_add}'
    
    def editResultsToggled(self, checked):
        self.isEditingResults = checked
        self.spotmaxToolbar.setDisabled(checked)
    
    def computeFeaturesEditedResultsClicked(
            self, text_to_add, src_df_filename, ini_filepath
        ):
        self.funcDescription = 'Computing features of edited results'
        
        df_spots_endname, text_to_append = self.saveEditedResultsClicked(
            src_df_filename, text_to_add, prompt_info=False
        )
        self.spotsItems.edited_df_out_filename = df_spots_endname
        pos_folders_to_reanalyse = []
        for posData in self.data:
            spotmax_output_folderpath = posData.spotmax_out_path
            self.spotsItems.setPosition(posData)
            self.spotsItems.loadSpotsTables()
            toolbutton = self.spotsItems.getActiveButton()
            if 'edited' not in toolbutton.df.columns:
                continue
            pos_folders_to_reanalyse.append(posData.pos_path.replace('\\', '/'))
        
        # Back to current pos
        posData = self.data[self.pos_i]
        self.spotsItems.setPosition(posData)
        self.spotsItems.loadSpotsTables()
        
        if not pos_folders_to_reanalyse:
            prompts.warnNoneOfLoadedPosResultsEdited(qparent=self)
            return
        
        cp = config.ConfigParser()
        cp.read(ini_filepath)
        cp = io.add_folders_to_analyse_to_configparser(
            cp, pos_folders_to_reanalyse
        )
        cp = io.add_spots_coordinates_endname_to_configparser(
            cp, df_spots_endname
        )
        cp = io.add_use_default_values_to_configparser(cp)
        cp = io.add_text_to_append_to_configparser(
            cp, text_to_append
        )
        
        section = 'File paths and channels'
        option = 'Reference channel end name'
        ref_ch_name = cp[section][option]
        refChSegmEndName = self.spotsItems.getRefChannelSegmEndname(ref_ch_name)
        if refChSegmEndName:
            useSavedRefChMask = prompts.askUseSavedRefChMask(
                refChSegmEndName, qparent=self
            )
            if useSavedRefChMask:
                cp = io.add_ref_ch_segm_endname_to_configparser(
                    cp, refChSegmEndName
                )
        
        temp_ini_filepath = io.get_ini_filepath_appdata(
            text_to_add=df_spots_endname
        )
        with open(temp_ini_filepath, 'w', encoding="utf-8") as ini:
            cp.write(ini)
        
        cancel, ini_filepath = prompts.informationSpotmaxAnalysisStart(
            temp_ini_filepath
        )
        self.logger.info(
            f'Analysis parameters files saved to:\n\n{ini_filepath}\n'
        )
        if cancel:
            self.logger.info('Computing features of the edited results cancelled.')
            return
        
        worker = self.runAnalysis(ini_filepath, False, start=False)
        worker.signals.finished.disconnect()
        worker.signals.finished.connect(self.computeFeaturesWorkerFinished)
        self.threadPool.start(worker)
    
    def enableZstackWidgets(self, enabled):
        super().enableZstackWidgets(enabled)
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        section = 'METADATA'
        anchor = 'zResolutionLimit'
        ZresolMultiplWidget = autoTuneGroupbox.params[section][anchor]['widget']
        ZresolMultiplWidget.setDisabled(not enabled)
    
    def hideAcdcToolbars(self):
        self.editToolBar.setVisible(False)
        self.editToolBar.setDisabled(True)
        self.secondLevelToolbar.setVisible(False)
        self.secondLevelToolbar.setDisabled(True)
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
    def _computeSpotFootprint(self, formWidget):
        self.funcDescription = 'Get spot footprint'
        module_func = 'transformations.get_local_spheroid_mask'
        anchor = 'spotMinSizeLabels'
        
        args = [module_func, anchor]
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii_pxl = spotMinSizeLabels.pixelValues()
        zyx_radii_pxl = [val/2 for val in spots_zyx_radii_pxl]
        kwargs = {'spots_zyx_radii_pxl': zyx_radii_pxl}
        self.startComputeAnalysisStepWorker(*args, **kwargs)
    
    @exception_handler
    def _computeExtend3DsegmRange(self, formWidget):
        self.funcDescription = 'Extend 3D segmentation in z'
        module_func = 'transformations.extend_3D_segm_in_z'
        anchor = 'extend3DsegmRange'
        
        args = [module_func, anchor]
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        preProcessParams = ParamsGroupBox.params['Pre-processing']
        extend3DsegmRangeWidget = preProcessParams['extend3DsegmRange']['widget']
        extend3DsegmRange = extend3DsegmRangeWidget.value()
        
        posData = self.data[self.pos_i]
        lab = posData.segm_data[posData.frame_i]
        
        kwargs = {'segm_data': lab, 'low_high_range': extend3DsegmRange}
        self.startComputeAnalysisStepWorker(*args, **kwargs)
    
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
    
    # @exception_handler
    def storeCroppedDataToTuneKernel(self, *args, **kwargs):
        kernel = args[0]
        image_data_cropped = kwargs['image_data_cropped']
        segm_data_cropped = kwargs['segm_data_cropped']
        crop_to_global_coords = kwargs['crop_to_global_coords']
        
        pos_foldername = self.data[self.pos_i].pos_foldername
        
        kernel.set_crop_to_global_coords(pos_foldername, crop_to_global_coords)
        kernel.set_image_data(pos_foldername, image_data_cropped)
        kernel.set_segm_data(pos_foldername, segm_data_cropped)  
        
        if self.pos_i == len(self.data)-1:
            # Back to current pos
            self.pos_i = self.current_pos_i
            # self.get_data()
            self.startTuneKernelWorker(kernel)
        else:
            self.pos_i += 1
            posData = self.data[self.pos_i]
            pos_foldername = posData.pos_foldername
            kernel.set_images_path(
                pos_foldername, posData.images_path, posData.basename
            )
            on_finished_callback = (
                self.storeCroppedDataToTuneKernel, args, kwargs
            )
            self.startCropImageBasedOnSegmDataWorkder(
                posData.img_data, posData.segm_data, 
                on_finished_callback=on_finished_callback
            )
    
    @exception_handler
    def startCropImageBasedOnSegmDataWorkder(
            self, image_data, segm_data, on_finished_callback,
            nnet_input_data=None
        ):
        if self.progressWin is None:
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
        
        preProcessParams = ParamsGroupBox.params['Pre-processing']
        extend3DsegmRangeWidget = preProcessParams['extend3DsegmRange']['widget']
        extend3DsegmRange = extend3DsegmRangeWidget.value()
        
        worker = qtworkers.CropImageBasedOnSegmDataWorker(
            image_data, segm_data, delta_tolerance, posData.SizeZ,
            on_finished_callback, nnet_input_data=nnet_input_data, 
            extend_segm_3D_range=extend3DsegmRange
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
        size = spots_zyx_radii[-1]
        self.setHoverToolSymbolData(
            [x], [y], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            size=size
        )
    
    def setHoverCircleEditResults(self, x, y):
        size = self.spotsItems.getPointSize()
        self.setHoverToolSymbolData(
            [x], [y], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            size=size
        )
    
    def setAutoTuneZplaneCursorData(self, x, y):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        length = spots_zyx_radii[0]
        halfLength = length/2
        x0 = x
        x1 = x
        y0 = y - halfLength
        y1 = y + halfLength
        self.LinePlotItem.setData([x0, x1], [y0, y1])
    
    def setAutoTuneZplaneCursorLength(self, length):
        halfLength = length/2
        xx, yy = self.LinePlotItem.getData()
        if xx is None:
            return
        (x0, x1), (y0, y1) = xx, yy
        yc = y0 + abs(y0-y1)/2
        y0 = yc - halfLength
        y1 = yc + halfLength
        self.LinePlotItem.setData([x0, x1], [y0, y1])
    
    def setZstackView(self, isZstackView):
        pass
    
    def copyParamsToAutoTuneWidget(self):
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        section = 'METADATA'
        anchor = 'yxResolLimitMultiplier'
        autoTuneGroupbox.params[section][anchor]['widget'].setValue(
            ParamsGroupBox.params[section][anchor]['widget'].value()
        )
        
        anchor = 'zResolutionLimit'
        autoTuneGroupbox.params[section][anchor]['widget'].setValue(
            ParamsGroupBox.params[section][anchor]['widget'].value()
        )
        voxelDepth = (
            ParamsGroupBox.params[section]['voxelDepth']['widget'].value()
        )
        autoTuneGroupbox.params[section][anchor]['widget'].setStep(
            core.ceil(voxelDepth, precision=3)
        )
        
        plane = self.switchPlaneCombobox.currentText()
        if plane == 'xy':
            anchor = 'yxResolLimitMultiplier'
            widget = autoTuneGroupbox.params[section][anchor]['widget']
            widget.activateCheckbox.setChecked(False)
            widget.activateCheckbox.click()
        else:
            anchor = 'zResolutionLimit'
            widget = autoTuneGroupbox.params[section][anchor]['widget']
            widget.activateCheckbox.setChecked(False)
            widget.activateCheckbox.click()
    
    def checkReinitTuneKernel(self):
        if not hasattr(self, 'data'):
            return 
        
        posData = self.data[self.pos_i]
        if not hasattr(posData, 'tuneKernel'):
            return
        
        if not posData.tuneKernel.image_data():
            return
        
        posData.tuneKernel.init_input_data()
        
    
    def tabControlPageChanged(self, index):
        if not self.isDataLoaded:
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
        
        if index == 0:
            # To be safe, we reinit tune kernel since changin pre-processing 
            # parameters renders current kernel's image data obsolete
            self.checkReinitTuneKernel()
        
        if index != 1:
            autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
            autoTuneTabWidget.addAutoTunePointsButton.setChecked(False)
    
    def setAutoTunePointSize(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        
        # Note that here we visualize the spot with a diameter equal to the 
        # entered radius because we want to show the user the smaller volume
        # where a single spot can be detected. Basically this is the 
        # spot footprint passed to peak_local_max
        yx_diameter = spots_zyx_radii[-1]
        z_diameter = spots_zyx_radii[0]
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.setAutoTunePointSize(yx_diameter, z_diameter)
    
    @exception_handler
    def _computeSharpenSpots(self, formWidget):
        self.funcDescription = 'Sharpen spots (DoG filter)'
        module_func = 'filters.DoG_spots'
        anchor = 'sharpenSpots'
        
        posData = self.data[self.pos_i]
        
        keys = ['spots_zyx_radii_pxl', 'use_gpu', 'lab']
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
            columns = posData.acdc_df.columns
            cca_df = posData.acdc_df[columns.intersection(LINEAGE_COLUMNS)].copy()
            return cca_df, True
        
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
            image data (parameter `Spots channel end name`),<br>
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
            (parameter `Spots channel end name`).<br><br>
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
            but in the parameter `Cells segmentation end name`<br>
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
            but in the parameter `Table with lineage info end name`<br>
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
        
        columns = posData.acdc_df.columns
        cca_df = posData.acdc_df[columns.intersection(LINEAGE_COLUMNS)].copy()
        return cca_df, True
    
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
        optimise_for_high_spot_density = (
            spotsParams['optimiseWithEdt']['widget'].isChecked()
        )
        detection_method = (
            spotsParams['spotDetectionMethod']['widget'].currentText()
        )
        
        configParams = ParamsGroupBox.params['Configuration']
        use_gpu = configParams['useGpu']['widget'].isChecked()
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        tune_features_range = autoTuneTabWidget.selectedFeatures()
        
        zyx_voxel_size = ParamsGroupBox.zyxVoxelSize()
        
        use_spots_segm_masks = detection_method != 'peak_local_max'
        
        kwargs = {
            'lab': None, 
            'gauss_sigma': gauss_sigma, 
            'ref_ch_gauss_sigma': ref_ch_gauss_sigma, 
            'ref_ch_ridge_sigmas': ref_ch_ridge_sigmas,
            'spots_zyx_radii_pxl': spots_zyx_radii, 
            'do_sharpen': do_sharpen, 
            'do_remove_hot_pixels': do_remove_hot_pixels,
            'lineage_table': lineage_table, 
            'do_aggregate': do_aggregate, 
            'optimise_for_high_spot_density': optimise_for_high_spot_density,
            'use_gpu': use_gpu, 'sigma': gauss_sigma, 
            'ref_ch_endname': refChEndName,
            'tune_features_range': tune_features_range,
            'detection_method': detection_method,
            'zyx_voxel_size': zyx_voxel_size,
            'use_spots_segm_masks': use_spots_segm_masks
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
    def _computeSpotPrediction(self, formWidget, run=True, **kwargsToAdd):
        if not self.isSpotPredSetupCorrectly():
            return 
        
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
            'lab', 'gauss_sigma', 'spots_zyx_radii_pxl', 'do_sharpen',
            'do_remove_hot_pixels', 'lineage_table', 'do_aggregate', 
            'use_gpu'
        ]
        all_kwargs = self.paramsToKwargs()
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return None, None
        
        kwargs = {key:all_kwargs[key] for key in keys}
        
        kwargs = self.addNnetKwargsAndThresholdMethodIfNeeded(kwargs)
        
        section = 'Spots channel'
        kwargs = self.addBioImageIOModelKwargs(kwargs, section, anchor)
        
        kwargs = self.addSpotiflowModelKwargs(kwargs)
        
        kwargs = {**kwargs, **kwargsToAdd}
        
        self.logNnetParams(kwargs.get('nnet_params'))
        self.logNnetParams(
            kwargs.get('bioimageio_params'), model_name='BioImage.IO model'
        )
        
        if not run:
            return args, kwargs
        
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, args, kwargs
        )
        self.startCropImageBasedOnSegmDataWorkder(
            posData.img_data, posData.segm_data, 
            on_finished_callback=on_finished_callback,
            nnet_input_data=kwargs.get('nnet_input_data')
        )
    
    @exception_handler
    def _computeSpotDetection(self, formWidget):
        threshold_func = self.getSpotsThresholdMethod()
        kwargsToAdd = {
            'do_try_all_thresholds': False,
            'thresholding_method': threshold_func
        }
        spots_pred_args, spots_pred_kwargs = self._computeSpotPrediction(
            None, run=False, **kwargsToAdd
        )
        if spots_pred_args is None:
            return

        posData = self.data[self.pos_i]
        on_finished_callback = (
            self.startComputeAnalysisStepWorker, 
            spots_pred_args, 
            spots_pred_kwargs
        )
        
        # Add the next step to the prediction step 
        # (finished slot for ComputeAnalysisStepWorker)
        spots_pred_kwargs['onFinishedSlot'] = (
            self._computeSpotDetectionFromPredictionResult
        )
        
        self.startCropImageBasedOnSegmDataWorkder(
            posData.img_data, posData.segm_data, 
            on_finished_callback=on_finished_callback,
            nnet_input_data=spots_pred_kwargs.get('nnet_input_data')
        )
    
    def logNnetParams(self, nnet_params, model_name='spotMAX AI model'):
        if nnet_params is None:
            return
        
        text = '-'*100
        text = (
            f'{text}\nRunning {model_name} with the following parameters:\n'
        )
        text = f'{text}  1. Initialization:\n'
        for param, value in nnet_params['init'].items():
            text = f'{text}    - {param}: {value}\n'
        
        text = f'{text}  2. Segmentation:\n'
        for param, value in nnet_params['segment'].items():
            text = f'{text}    - {param}: {value}\n'
            
        closing = '*'*100
        text = f'{text}{closing}'
        self.logger.info(text)        
    
    def isSpotPredSetupCorrectly(self):
        if self.isNeuralNetworkRequested() and self.getNeuralNetParams() is None:
            return False
        
        if self.isBioImageIOModelRequested() and self.getBioImageIOParams() is None:
            return False

        return True
    
    def addNnetKwargsAndThresholdMethodIfNeeded(self, kwargs):
        if not self.isNeuralNetworkRequested():
            return kwargs
        
        kwargs['nnet_model'] = self.getNeuralNetworkModel()
        kwargs['nnet_params'] = self.getNeuralNetParams()
        kwargs['nnet_input_data'] = self.getNeuralNetInputData()
        kwargs['return_nnet_prediction'] = True
        
        threshold_func = self.getSpotsThresholdMethod()
        kwargs['thresholding_method'] = threshold_func
        kwargs['do_try_all_thresholds'] = False
        
        return kwargs
    
    def addBioImageIOModelKwargs(self, kwargs, section, anchor):
        if not self.isBioImageIOModelRequested(section, anchor):
            return kwargs
        
        kwargs['bioimageio_model'] = self.getBioImageIOModel(section, anchor)
        kwargs['bioimageio_params'] = self.getBioImageIOParams(section, anchor)
        
        if section == 'Reference channel':
            threshold_func = self.getRefChThresholdMethod()
        else:
            threshold_func = self.getSpotsThresholdMethod()
            
        kwargs['thresholding_method'] = threshold_func
        kwargs['do_try_all_thresholds'] = False
        
        return kwargs

    def addSpotiflowModelKwargs(self, kwargs):
        if not self.isSpotiflowRequested():
            return kwargs
        
        kwargs['spotiflow_model'] = self.getSpotiflowModel()
        kwargs['spotiflow_params'] = self.getSpotiflowParams()
        
        threshold_func = self.getSpotsThresholdMethod()
        kwargs['thresholding_method'] = threshold_func
        kwargs['do_try_all_thresholds'] = False
        
        return kwargs
    
    def getSpotFootprint(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii_pxl = spotMinSizeLabels.pixelValues()
        zyx_radii_pxl = [val/2 for val in spots_zyx_radii_pxl]
        spot_footprint = transformations.get_local_spheroid_mask(
            zyx_radii_pxl
        )
        return spot_footprint
    
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
                '"Spots channel end name" parameter not provided.'
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
        if kwargs is None:
            return 
        
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
    
    def _displayGaussSigmaResult(self, filtered, image):
        from cellacdc.plot import imshow
        posData = self.data[self.pos_i]
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        anchor = 'gaussSigma'
        sigma = preprocessParams[anchor]['widget'].value()
        titles = ['Raw image', f'Filtered image (sigma = {sigma})']
        imshow(
            image, filtered, axis_titles=titles, parent=self, 
            window_title='Pre-processing - Gaussian filter',
            color_scheme=self._colorScheme
        )
    
    def _displayRefChGaussSigmaResult(self, result, image):
        from cellacdc.plot import imshow
        
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
                'Input image', 
                'SpotMAX AI prediction map',
                f'{selected_threshold_method}', 
                'SpotMAX AI'
            ]
            prediction_images = [
                result['input_image'], 
                result['neural_network_prediciton'],
                result['custom'], 
                result['neural_network'],
            ]
            max_ncols = 2
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
            max_ncols = 3
        elif 'spotiflow' in result:
            selected_threshold_method = self.getSpotsThresholdMethod()
            titles = [
                'Input image', f'{selected_threshold_method}', 
                'Spotiflow'
            ]
            prediction_images = [
                result['input_image'], 
                result['custom'], 
                result['spotiflow'],
            ] 
            max_ncols = 3
        else:
            titles = list(result.keys())
            titles[0] = 'Input image'
            prediction_images = list(result.values())
            max_ncols = 4
        
        window_title = 'Spots channel - Spots segmentation method'
        
        imshow(
            *prediction_images, axis_titles=titles, parent=self, 
            window_title=window_title, color_scheme=self._colorScheme, 
            max_ncols=max_ncols
        )
    
    @exception_handler
    def _computeSpotDetectionFromPredictionResult(self, output):
        # This method is called as a slot of the finished signal in 
        # startComputeAnalysisStepWorker that is performed after 
        # _computeSpotDetection.
        result, image, lab, spotPredAnchor = output
        
        inputImage = result['input_image']
        
        if 'neural_network' in result:
            segmSpotsMask = result['neural_network']
        elif 'bioimageio_model' in result:
            segmSpotsMask = result['bioimageio_model']
        else:
            segmSpotsMask = result['custom']
        
        self.funcDescription = 'Spots detection'
        module_func = 'pipe.spot_detection'
        anchor = 'spotDetectionMethod'
        args = [module_func, anchor]
        
        keys = [
            'detection_method',
            'spots_zyx_radii_pxl',
            'lab',
        ]
        all_kwargs = self.paramsToKwargs()
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return None, None
        
        kwargs = {key:all_kwargs[key] for key in keys}
        kwargs['image'] = inputImage
        kwargs['lab'] = lab
        kwargs['return_df'] = True
        kwargs['return_spots_mask'] = True
        kwargs['spots_segmantic_segm'] = segmSpotsMask
        kwargs['spot_footprint'] = self.getSpotFootprint()
        kwargs['validate'] = True
        
        self.startComputeAnalysisStepWorker(*args, **kwargs)
    
    def _displaySpotDetectionResult(self, result, image):
        df_coords, spots_masks, valid = result
        df_spots_objs = df_coords.copy()
        df_spots_objs['spot_mask'] = spots_masks
        spots_lab = transformations.from_df_spots_objs_to_spots_lab(
            df_spots_objs, image.shape, show_pbar=True
        )
        self.logger.info(
            f'Total number of detected spots = {len(df_coords)}'
        )
        df_spots_count = df_coords[['z']].groupby(level=0).count()
        df_spots_count = df_spots_count.rename(columns={'z': 'Number of spots'})
        self.logger.info(
            f'Number of detected spots per objects:\n'
            f'{df_spots_count}'
        )
        if image.ndim == 2:
            image = image[np.newaxis]
        
        if spots_lab.ndim == 2:
            spots_lab = spots_lab[np.newaxis]
        
        from cellacdc.plot import imshow
        self.spotDetectPreviewWin = imshow(
            image, 
            spots_lab, 
            axis_titles=['Detect image', 'Spots masks'],
            # annotate_labels_idxs=[1],
            points_coords_df=df_coords,
            block=False
        )
        
        if valid:
            return
        
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        filePathParams = ParamsGroupBox.params['File paths and channels']
        segmEndNameWidget = filePathParams['segmEndName']['widget']
        segmEndName = segmEndNameWidget.text()
        _warnings.warnSpotsDetectedOutsideCells(
            segmEndName, qparent=self.spotDetectPreviewWin
        )
    
    def _displaySpotFootprint(self, spot_footprint, image):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        SizeZ = metadataParams['SizeZ']['widget'].value()
        
        if SizeZ == 1:
            spot_footprint = spot_footprint.max(axis=0)
            
        from cellacdc.plot import imshow
        imshow(
            spot_footprint, 
            window_title='Spot footprint',
            axis_titles=['Spot footprint'], 
            infer_rgb=False
        )
    
    def _displayExtend3DsegmRange(self, extended_lab, image):
        posData = self.data[self.pos_i]
        lab = posData.segm_data[posData.frame_i]            
    
        from cellacdc.plot import imshow
        imshow(
            lab, extended_lab,
            window_title='Extended 3D segm',
            axis_titles=['Input masks', 'Extended masks'],
            annotate_labels_idxs=[0, 1]
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
    
    def computeFeaturesWorkerFinished(self, args):
        success = self.promptAnalysisWorkerFinished(args)
        if not success:
            return
        self.logger.info('Loading computed features (edited results)...')
        for posData in self.data:
            df_spots = io.load_spots_table(
                posData.spotmax_out_path, self.spotsItems.edited_df_out_filename
            )
            self.spotsItems.setActiveButtonDf(df_spots)
        self.logger.info(
            'Done (features loaded from file '
            f'`{self.spotsItems.edited_df_out_filename}`)'
        )
        prompts.informationComputeFeaturesFinished(
            self.spotsItems.edited_df_out_filename, qparent=self
        )            
            
    def startComputeAnalysisStepWorker(self, module_func, anchor, **kwargs):
        if self.progressWin is None:
            self.progressWin = acdc_apps.QDialogWorkerProgress(
                title=self.funcDescription, parent=self,
                pbarDesc=self.funcDescription
            )
            self.progressWin.mainPbar.setMaximum(0)
            self.progressWin.show(self.app)
        
        onFinishedSlot = kwargs.pop(
            'onFinishedSlot', self.computeAnalysisStepWorkerFinished
        )
        
        worker = qtworkers.ComputeAnalysisStepWorker(
            module_func, anchor, **kwargs
        )
        worker.signals.finished.connect(onFinishedSlot)
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
            # _debug._gui_autotune_compute_features(to_debug)
            _debug._gui_autotune_f1_score(to_debug)
        except Exception as error:
            raise error
        finally:
            worker.waitCond.wakeAll()
    
    def computeAnalysisStepWorkerFinished(self, output: tuple):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        result, image, lab, anchor = output
        self.logger.info(f'{self.funcDescription} process ended.')
        displayFunc = ANALYSIS_STEP_RESULT_SLOTS[anchor]
        displayFunc = getattr(self, displayFunc)
        displayFunc(result, image)
    
    def connectParamsBaseSignals(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        
        preprocessParams = ParamsGroupBox.params['Pre-processing']
        removeHotPixelsToggle = preprocessParams['removeHotPixels']['widget']
        removeHotPixelsToggle.toggled.connect(self.onRemoveHotPixelsToggled)
        gaussSigmaWidget = preprocessParams['gaussSigma']['widget']
        gaussSigmaWidget.valueChanged.connect(
            self.onPreprocessGaussSigmaValueChanged
        )
        
        metadataParams = ParamsGroupBox.params['METADATA']
        pixelWidthWidget = metadataParams['pixelWidth']['widget']
        pixelWidthWidget.valueChanged.connect(self.onPixelWidthValueChanged)
        ParamsGroupBox.sigResolMultiplValueChanged.connect(
            self.onResolMultiplValueChanged
        )
        
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
    
    @exception_handler
    def getSpotiflowModel(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params['Spots channel']
        anchor = 'spotPredictionMethod'
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        if spotPredictionMethodWidget.SpotiflowModel is None:
            raise ValueError(
                'Neural network parameters were not initialized. Before trying '
                'to use it, you need to initialize the model\'s parameters by '
                'clicking on the settings button on the right of the selection '
                'box at the "Spots segmentation method" parameter.'
            )
        
        return spotPredictionMethodWidget.SpotiflowModel
    
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
            _warnings.warnNeuralNetNotInitialized(
                qparent=self, model_type='SpotMAX AI'
            )
            return 
        
        return spotPredictionMethodWidget.nnetParams  
    
    @exception_handler
    def getSpotiflowParams(self):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params['Spots channel']
        anchor = 'spotPredictionMethod'
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        if spotPredictionMethodWidget.SpotiflowModel is None:
            _warnings.warnNeuralNetNotInitialized(
                qparent=self, model_type='Spotiflow'
            )
            return 
        
        return spotPredictionMethodWidget.SpotiflowParams 
    
    @exception_handler
    def getBioImageIOParams(
            self, section='Spots channel', anchor='spotPredictionMethod'
        ):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        spotsParams = ParamsGroupBox.params[section]
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        if spotPredictionMethodWidget.bioImageIOModel is None:
            _warnings.warnNeuralNetNotInitialized(
                qparent=self, model_type='BioImage.IO model'
            )
            return
        
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
    
    def onPreprocessGaussSigmaValueChanged(self, value):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        anchor = 'spotPredictionMethod'
        spotsParams = ParamsGroupBox.params['Spots channel']
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        spotPredictionMethodWidget.setDefaultGaussianSigma(value)
    
    def onResolMultiplValueChanged(self, value):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        anchor = 'spotPredictionMethod'
        spotsParams = ParamsGroupBox.params['Spots channel']
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        spotPredictionMethodWidget.setDefaultResolutionMultiplier(value)
    
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
        autoTuneTabWidget.sigYXresolMultiplActivated.connect(
            self.autoTuningYXresolMultiplActivated
        )
        
        autoTuneTabWidget.sigZresolLimitChanged.connect(
            self.autoTuningZresolLimitChanged
        )
        autoTuneTabWidget.sigZresolLimitActivated.connect(
            self.autoTuningZresolLimitActivated
        )
    
    def autoTuningYXresolMultiplChanged(self, value):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        metadataParams['yxResolLimitMultiplier']['widget'].setValue(value)
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        size = spots_zyx_radii[-1]
        self.ax2_BrushCircle.setSize(size)
        self.ax1_BrushCircle.setSize(size)
    
    def autoTuningYXresolMultiplActivated(self, checked):
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        section = 'METADATA'
        anchor = 'zResolutionLimit'
        ZresolMultiplWidget = autoTuneGroupbox.params[section][anchor]['widget']
        ZresolMultiplWidget.activateCheckbox.setChecked(not checked)
        ZresolMultiplWidget.setDisabled(checked)
        if checked:
            plane = 'xy'
            self.switchPlaneCombobox.setCurrentText(plane)
    
    def autoTuningZresolLimitChanged(self, value):
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        metadataParams = ParamsGroupBox.params['METADATA']
        metadataParams['zResolutionLimit']['widget'].setValue(value)
        spotMinSizeLabels = metadataParams['spotMinSizeLabels']['widget']
        spots_zyx_radii = spotMinSizeLabels.pixelValues()
        self.setAutoTuneZplaneCursorLength(spots_zyx_radii[0])
    
    def autoTuningZresolLimitActivated(self, checked):
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        section = 'METADATA'
        anchor = 'yxResolLimitMultiplier'
        YXresolMultiplWidget = autoTuneGroupbox.params[section][anchor]['widget']
        YXresolMultiplWidget.activateCheckbox.setChecked(not checked)
        YXresolMultiplWidget.setDisabled(checked)
        if checked:
            plane = 'zy'
            self.switchPlaneCombobox.setCurrentText(plane)
    
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
        df_coords = trueItem.coordsToDf(includeData=True)
        
        if len(df_coords) == 0:
            self.warnTrueSpotsAutoTuneNotAdded()
            return
        
        all_kwargs = self.paramsToKwargs()
        if all_kwargs is None:
            self.logger.info('Process cancelled.')
            return
        
        # kwargs = self.addNnetKwargsAndThresholdMethodIfNeeded(kwargs)
        kernel = posData.tuneKernel
        kernel.set_kwargs(all_kwargs)
        
        args = [kernel]
        kwargs = {
            'lab': None, 
            'image_data_cropped': None, 
            'segm_data_cropped': None,
            'crop_to_global_coords': None
        }
        self.current_pos_i = self.pos_i
        if not kernel.image_data():
            self.pos_i = 0
            posData = self.data[self.pos_i]
            kernel.set_images_path(
                posData.pos_foldername, posData.images_path, posData.basename
            )
            on_finished_callback = (
                self.storeCroppedDataToTuneKernel, args, kwargs
            )
            self.startCropImageBasedOnSegmDataWorkder(
                posData.img_data, posData.segm_data, 
                on_finished_callback=on_finished_callback
            )
        else:
            self.startTuneKernelWorker(kernel)
        
    def startTuneKernelWorker(self, kernel):
        if self.progressWin is None:
            self.progressWin = acdc_apps.QDialogWorkerProgress(
                title='Tuning parameters', parent=self,
                pbarDesc='Tuning parameters'
            )
            self.progressWin.mainPbar.setMaximum(0)
            self.progressWin.show(self.app)
            
        ini_filepath, temp_dirpath = io.get_temp_ini_filepath()
        ParamsGroupBox = self.computeDockWidget.widget().parametersQGBox
        ParamsGroupBox.saveToIniFile(ini_filepath)
        kernel.set_ini_filepath(ini_filepath)
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneGroupbox = autoTuneTabWidget.autoTuneGroupbox
        
        trueItem = autoTuneGroupbox.trueItem
        df_coords = trueItem.coordsToDf(includeData=True)
        kernel.set_tzyx_true_spots_df_coords(df_coords)
        
        falseItem = autoTuneGroupbox.falseItem
        df_coords = falseItem.coordsToDf(includeData=True)
        kernel.set_tzyx_false_spots_df_coords(df_coords)
        
        worker = qtworkers.TuneKernelWorker(kernel)
        self.connectDefaultWorkerSlots(worker)
        worker.signals.finished.connect(self.tuneKernelWorkerFinished)
        self.threadPool.start(worker)
        return worker
    
    @exception_handler
    def workerCritical(self, out: Tuple[QObject, Exception]):
        worker, error = out
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info(error)
        try:
            worker.thread().quit()
            worker.deleteLater()
            worker.thread().deleteLater()
        except Exception as err:
            pass
        raise error
    
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
        
        tip_text = ("""
            Hover onto data points with mouse cursor to view any of the 
            features. Features can be selected and viewed at the bottom of 
            the <code>Tune parameters<code> tab.
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph(f"""
            Auto-tuning process finished. Results will be displayed on the 
            `Tune parameters` tab.<br>
            {html_func.to_admonition(tip_text, admonition_type='tip')}
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
        if not self.isDataLoaded:
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
        
    def initTuneKernel(self):
        posData = self.data[self.pos_i]
        posData.tuneKernel = tune.TuneKernel()
    
    def PosScrollBarMoved(self, pos_n):
        self.pos_i = pos_n-1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        posData = self.data[self.pos_i]
        self.spotsItems.hideAllItems()
        super().updateAllImages()
        self.setStatusBarLabel()
    
    def PosScrollBarReleased(self):
        super().PosScrollBarReleased()
        self.spotsItems.showItems()
        
    def connectLeftClickButtons(self):
        super().connectLeftClickButtons()
        
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        button = autoTuneTabWidget.addAutoTunePointsButton
        button.toggled.connect(button.onToggled)
    
    def addAutoTunePointsToggled(self, checked):
        self.isAddAutoTunePoints = checked
        self.zProjComboBox.setDisabled(checked)
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.setPosData(self.data[self.pos_i])
        if checked:
            self.setAutoTunePointSize()
    
    def startAutoTuning(self):
        if not self.isDataLoaded:
            return
        self.isAutoTuneRunning = True
        self.doAutoTune()
        
    def stopAutoTuning(self):
        if not self.isDataLoaded:
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
    
    def checkLoadLoadedIniFilepath(self):
        if self.lastLoadedIniFilepath is None:
            return True
        
        txt = html_func.paragraph(f"""
            You previously loaded parameters from this file:<br><br>
            <code>{self.lastLoadedIniFilepath}</code><br><br>
            Do you want to <b>update them</b> based on the loaded Position folder?
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        _, _, yesButton = msg.warning(
            self, 'Update parameters?', txt, 
            buttonsTexts=(
                'Cancel', 
                'No, keep the parameters as they are', 
                'Yes, update them'
            ), 
            path_to_browse=os.path.dirname(self.lastLoadedIniFilepath)
        )
        return msg.clickedButton == yesButton
    
    def setAnalysisParameters(self):
        proceed = self.checkLoadLoadedIniFilepath()
        if not proceed:
            self.logger.info(
                'Initializing parameters from Position folder cancelled.'
            )
            return
        paramsGroupbox = self.computeDockWidget.widget().parametersQGBox
        posData = self.data[self.pos_i]
        self.computeDockWidget.widget().loadPreviousParamsButton.setStartPath(
            posData.pos_path
        )
        if self.isNewFile:
            segmEndName = ''
        else:
            segmFilename = os.path.basename(posData.segm_npz_path)
            segmEndName = segmFilename[len(posData.basename):]
        runNum = max(self.loaded_exp_run_nums, default=0) + 1
        try:
            emWavelen = posData.emWavelens[self.user_ch_name]
        except Exception as e:
            emWavelen = 500.0
        if emWavelen == 0:
            emWavelen = 500
        
        if self.user_ch_name:
            spotsEndName = self.user_ch_name
        else:
            spotsEndName = posData.basename.split('_')[-1]
        
        folderPathsToAnalyse = [_posData.pos_path for _posData in self.data]
        folderPathsToAnalyse = '\n'.join(folderPathsToAnalyse)
        loadedValues = {
            'File paths and channels': [
                {'anchor': 'folderPathsToAnalyse', 'value': folderPathsToAnalyse},
                {'anchor': 'spotsEndName', 'value': spotsEndName},
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
        paramsScrollArea = guiTabControl.parametersTab
        # autoTuneScrollArea = guiTabControl.autoTuneTabWidget
        verticalScrollbar = paramsScrollArea.verticalScrollBar()
        # groupboxWidth = autoTuneScrollArea.size().width()
        scrollbarWidth = verticalScrollbar.size().width()
        # minWidth = groupboxWidth + scrollbarWidth + 30        
        w = guiTabControl.paramsTabButtonsContainerWidget.sizeHint().width()
        w += scrollbarWidth
        self.resizeDocks([self.computeDockWidget], [w+10], Qt.Horizontal)
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
        args = (posData.pos_foldername, frame_i, z)
        autoTuneTabWidget.setVisibleAutoTunePoints(*args)
    
    def updateZproj(self, how):
        super().updateZproj(how)
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        addAutoTunePointsButton = autoTuneTabWidget.addAutoTunePointsButton
        addAutoTunePointsButton.setDisabled(how != 'single z-slice')  
        addAutoTunePointsButton.setToolTip(
            'Functionality disabled in projection mode.\n'
            'Switch to "single z-slice" to activate it.'
        )          
    
    def updateAllImages(self, *args, **kwargs):
        posData = self.data[self.pos_i]
        super().updateAllImages(*args, **kwargs)
        self.spotsItems.setData(
            posData.frame_i, z=self.currentZ(checkIfProj=True)
        )
        self.setVisibleAutoTunePoints()
        self.initHighlightRefChannelObjImage()
    
    def updatePos(self):
        self.setStatusBarLabel()
        self.checkManageVersions()
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        self.initContoursImage()
        self.initTextAnnot()
        self.postProcessing()
        posData = self.data[self.pos_i]
        autoTuneTabWidget = self.computeDockWidget.widget().autoTuneTabWidget
        autoTuneTabWidget.updatePos(posData, self.currentZ())
        self.spotsItems.setPosition(posData)
        self.spotsItems.loadSpotsTables()
        self.updateAllImages()
        self.zoomToCells()
        self.updateScrollbars()
        self.computeSegm()
        self.initTuneKernel()
        self.computeDockWidget.widget().setLoadedPosData(posData)

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
                'Yes, stop analysis and close SpotMAX'
            )
        )
        return msg.clickedButton == yesButton
    
    def askSaveOnClosing(self, event):
        return True
    
    def closeEvent(self, event):
        self.stopAutoTuning()
        if self.isAnalysisRunning:
            self.setDisabled(False)
            proceed = self.warnClosingWhileAnalysisIsRunning()
            if not proceed:
                event.ignore()
                self.setDisabled(True)
                return
        super().closeEvent(event)
        # if not sys.stdout == self.logger.default_stdout:
        #     return
        if not self._executed:
            return
        print('**********************************************')
        print(f'SpotMAX closed. {get_salute_string()}')
        print('**********************************************')