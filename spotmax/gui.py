# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPyTop HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# TODO:
# - If checkboxes checked when left are moved move also right controls
# - Method warnLoadedDifferentPaths
# - Style dock for size and opacity of skeleton scatter
# - IndexError when skeletonizing AND overlaying
# - Check Antoine data
# - Check Nada's data
# - Add distance from reference channel metric
# - Load chunk other layers (overlay, skel and contour)
# - Load chunk positions
# - tif files should have number of frames to load disabled

"""spotMAX GUI"""
import sys
import os
import pathlib
import re
import traceback
import json
import time
import datetime
import logging
import uuid
import psutil
from pprint import pprint
from functools import partial, wraps
from natsort import natsorted
from queue import Queue
import time

import cv2
import math
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.interpolate
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from skimage.color import gray2rgb, gray2rgba

import matplotlib.pyplot as plt

from PyQt5.QtCore import (
    Qt, QFile, QTextStream, QSize, QRect, QRectF,
    QEventLoop, QTimer, QEvent, QThreadPool,
    QRunnable, pyqtSignal, QObject, QThread,
    QMutex, QWaitCondition, QSettings
)
from PyQt5.QtGui import (
    QIcon, QKeySequence, QCursor, QKeyEvent, QFont,
    QPixmap, QColor, QPainter, QPen
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QPushButton,
    QMainWindow, QMenu, QToolBar, QGroupBox,
    QScrollBar, QCheckBox, QToolButton, QSpinBox,
    QComboBox, QButtonGroup, QActionGroup,
    QShortcut, QFileDialog, QDoubleSpinBox,
    QAbstractSlider, QMessageBox, QGraphicsProxyWidget,
    QGridLayout, QStyleFactory, QWidget, QFrame,
    QHBoxLayout, QDockWidget, QAbstractSpinBox,
    QColorDialog, QVBoxLayout
)

import pyqtgraph as pg

from cellacdc import apps as acdc_apps
from cellacdc import widgets as acdc_widgets
from cellacdc import qutils as acdc_qutils
from cellacdc import exception_handler

from . import io, dialogs, utils, widgets, qtworkers, html_func
from . import spotmax_path, settings_path, colorItems_path

# NOTE: Enable icons
from . import qrc_resources, config

if os.name == 'nt':
    try:
        # Set taskbar icon in windows
        import ctypes
        myappid = 'schmollerlab.spotmax.pyqt' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        pass

bottomWidgetsKeys = (
    'howDrawSegmCombobox',
    'navigateScrollbar_label',
    'navigateScrollbar',
    'zSliceSbL0_label',
    'zSliceScrollbarLayer0',
    'zProjComboboxLayer0',
    'zSliceSbL1_label',
    'zSliceScrollbarLayer1',
    'zProjComboboxLayer1',
    'alphaScrollbar_label',
    'alphaScrollbar'
)

sideToolbarWidgetsKeys = {
    'fileToolbar': (
        'openFolderAction',
        'openFileAction',
        'reloadAction',
        'openFolderAction',
        'openFileAction',
        'reloadAction'
    ),
    'viewToolbar': (
        'overlayButton',
        'colorButton',
        'plotSpotsCoordsButton',
        'plotSkeletonButton',
        'overlayAction',
        'colorAction',
        'plotSpotsCoordsAction'
    )
}

# Interpret image data as row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
np.random.seed(1568)

def qt_debug_trace():
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    import pdb; pdb.set_trace()


class spotMAX_Win(QMainWindow):
    """Main Window."""
    sigClosed = pyqtSignal(object)

    def __init__(
            self, app, debug=False,
            parent=None, buttonToRestore=None, mainWin=None,
            executed=False
        ):
        """Initializer."""
        super().__init__(parent)

        self.debug = debug
        self.executed = executed

        logger, self.log_path, self.logs_path = utils.setupLogger()
        self.logger = logger
        self.loadLastSessionSettings()

        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin

        self.app = app
        self.num_screens = len(app.screens())

        self.funcDescription = 'Initializer'

        # Center main window and determine location of slideshow window
        # depending on number of screens available
        if self.num_screens > 1:
            screen1 = app.screens()[0]
            screen2 = app.screens()[1]
            screen2Center = screen2.size().width()/2
            screen2Left = screen1.size().width()
            self.slideshowWinLeft = int(screen2Left+screen2Center-850/2)
            self.slideshowWinTop = int(screen1.size().height()/2 - 800/2)
        else:
            screen1 = app.screens()[0]
            self.slideshowWinLeft = int(screen1.size().width()-850)
            self.slideshowWinTop = int(screen1.size().height()/2 - 800/2)

        self.setWindowTitle("spotMAX - GUI")

        self.setWindowIcon(QIcon(":logo.svg"))
        self.setAcceptDrops(True)

        self.rightClickButtons = []
        self.leftClickButtons = []
        self.initiallyHiddenItems = {'left': [], 'right': [], 'top': []}
        self.checkableToolButtons = []

        self.lastSelectedColorItem = None

        self.countKeyPress = 0
        self.xHoverImg, self.yHoverImg = None, None

        self.areActionsConnected = {'left': False, 'right': False}
        self.dataLoaded = {'left': False, 'right': False}

        # List of io.loadData class for each position of the experiment
        self.expData = {'left': [], 'right': []}

        self.bottomWidgets = {'left': {}, 'right': {}}
        self.sideToolbar = {'left': {}, 'right': {}}
        self.axes = {'left': None, 'right': None}
        self.imgItems = {'left': None, 'right': None}
        self.histItems = {'left': {}, 'right': {}}
        self.isFirstOpen = True

        # Buttons added to QButtonGroup will be mutually exclusive
        self.checkableQButtonsGroup = QButtonGroup(self)
        self.checkableQButtonsGroup.setExclusive(False)

        self.gui_createFonts()
        self.gui_setListItems()
        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createTopToolBars()
        self.gui_createComputeDockWidget()
        self.gui_createLeftToolBars()
        self.gui_createRightToolBars()
        self.gui_createSpotsClickedContextMenuActions()
        self.gui_createSkeletonClickedContextMenuActions()

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.gui_createBottomLayout()

        self.gui_addBottomLeftWidgets()
        self.gui_addBottomRightWidgets()

        self.graphLayout = pg.GraphicsLayoutWidget(parent=self)
        self.gui_addGraphicsItems()

        self.gui_createSideLayout()
        self.gui_addSideLayoutWidgets()

        self.gui_createThreadPool()

        mainContainer = QWidget()

        mainLayout = QGridLayout()
        mainLayout.addLayout(self.sideLayout, 0, 0, 2, 1)
        mainLayout.addWidget(self.graphLayout, 0, 1)
        mainLayout.addLayout(self.bottomLayout, 1, 1)

        mainContainer.setLayout(mainLayout)
        self.setCentralWidget(mainContainer)

        self.gui_init(first_call=True)

    def setVersion(self, version):
        self._version = version

    def loadLastSessionSettings(self):
        colorItems_path = os.path.join(settings_path, 'colorItems.json')
        csv_path = os.path.join(settings_path, 'gui_settings.csv')
        self.settings_csv_path = csv_path
        styleItems = ['spots', 'skel']
        if os.path.exists(csv_path):
            self.df_settings = pd.read_csv(csv_path, index_col='setting')
            if 'is_bw_inverted' not in self.df_settings.index:
                self.df_settings.at['is_bw_inverted', 'value'] = 'No'
            else:
                self.df_settings.loc['is_bw_inverted'] = (
                    self.df_settings.loc['is_bw_inverted'].astype(str)
                )
            if 'fontSize' not in self.df_settings.index:
                self.df_settings.at['fontSize', 'value'] = '10pt'
            if 'overlayColor' not in self.df_settings.index:
                self.df_settings.at['overlayColor', 'value'] = '255-255-0'
            if 'how_normIntensities' not in self.df_settings.index:
                raw = 'Do not normalize. Display raw image'
                self.df_settings.at['how_normIntensities', 'value'] = raw

            for what in styleItems:
                if f'{what}_opacity' not in self.df_settings.index:
                    self.df_settings.at[f'{what}_opacity', 'value'] = '0.3'
                if f'{what}_pen_width' not in self.df_settings.index:
                    self.df_settings.at[f'{what}_pen_width', 'value'] = '2.0'
                if f'{what}_size' not in self.df_settings.index:
                    self.df_settings.at[f'{what}_size', 'value'] = '3'
        else:
            idx = [
                'is_bw_inverted',
                'fontSize',
                'overlayColor',
                'how_normIntensities',
            ]
            values = ['No', '10pt', '255-255-0', 'raw']
            for what in styleItems:
                idx.append(f'{what}_opacity')
                values.append(f'0.3')
                idx.append(f'{what}_pen_width')
                values.append(f'2.0')
                idx.append(f'{what}_size')
                values.append(f'3')

            self.df_settings = pd.DataFrame({'setting': idx,
                                             'value': values}
                                           ).set_index('setting')

    def dragEnterEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(file_path):
            selectedPath = file_path
            basename = os.path.basename(file_path)
            if basename.find('Position_')!=-1 or basename=='Images':
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def dropEvent(self, event):
        if event.pos().x() > self.geometry().width()/2:
            side = 'right'
        else:
            side = 'left'
        event.setDropAction(Qt.CopyAction)
        file_path = event.mimeData().urls()[0].toLocalFile()
        basename = os.path.basename(file_path)
        if os.path.isdir(file_path):
            selectedPath = file_path
            self.openFolder(side, selectedPath=selectedPath)
        else:
            self.openFile(side, file_path=file_path)

    def leaveEvent(self, event):
        if self.slideshowWin is not None:
            mainWinGeometry = self.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft+mainWinWidth
            mainWinBottom = mainWinTop+mainWinHeight

            slideshowWinGeometry = self.slideshowWin.geometry()
            slideshowWinLeft = slideshowWinGeometry.left()
            slideshowWinTop = slideshowWinGeometry.top()
            slideshowWinWidth = slideshowWinGeometry.width()
            slideshowWinHeight = slideshowWinGeometry.height()

            # Determine if overlap
            overlap = (
                (slideshowWinTop < mainWinBottom) and
                (slideshowWinLeft < mainWinRight)
            )

            autoActivate = (
                self.expData_loaded and not
                overlap and not
                self.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.slideshowWin.setFocus(True)
                self.slideshowWin.activateWindow()
        else:
            event.ignore()

    def enterEvent(self, event):
        if self.slideshowWin is not None:
            mainWinGeometry = self.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft+mainWinWidth
            mainWinBottom = mainWinTop+mainWinHeight

            slideshowWinGeometry = self.slideshowWin.geometry()
            slideshowWinLeft = slideshowWinGeometry.left()
            slideshowWinTop = slideshowWinGeometry.top()
            slideshowWinWidth = slideshowWinGeometry.width()
            slideshowWinHeight = slideshowWinGeometry.height()

            # Determine if overlap
            overlap = (
                (slideshowWinTop < mainWinBottom) and
                (slideshowWinLeft < mainWinRight)
            )

            autoActivate = (
                self.expData_loaded and not
                overlap and not
                self.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.setFocus(True)
                self.activateWindow()
        else:
            event.ignore()

    @exception_handler
    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_P:
            print(self.computeDockWidget.frameSize())
            pass
            # posData = self.currentPosData('left')
            # print(posData.chData.shape)
            # print(posData.chData_shape)
            # print(sys.getsizeof(posData.chData)*1E-6)
            # print(f'Start t0 = {posData.t0_window}, start z0 = {posData.z0_window}')
            # pass
        elif ev.key() == Qt.Key_Left:
            if not self.dataLoaded['left'] and not self.dataLoaded['right']:
                ev.ignore()
                return

            areBothPlotsVisible = (
                self.dataLoaded['left'] and self.dataLoaded['right']
            )
            if areBothPlotsVisible:
                side = self.sideFromCursor()
            else:
                side = 'left' if self.dataLoaded['left'] else 'right'
            bottomWidgets = self.bottomWidgets[side]
            navigateScrollbar = bottomWidgets['navigateScrollbar']
            navigateScrollbar.triggerAction(QAbstractSlider.SliderSingleStepSub)

        elif ev.key() == Qt.Key_Right:
            if not self.dataLoaded['left'] and not self.dataLoaded['right']:
                ev.ignore()
                return

            areBothPlotsVisible = (
                self.dataLoaded['left'] and self.dataLoaded['right']
            )
            if areBothPlotsVisible:
                side = self.sideFromCursor()
            else:
                side = 'left' if self.dataLoaded['left'] else 'right'

            bottomWidgets = self.bottomWidgets[side]
            navigateScrollbar = bottomWidgets['navigateScrollbar']
            navigateScrollbar.triggerAction(QAbstractSlider.SliderSingleStepAdd)

    def sideFromCursor(self):
        xCursor = QCursor.pos().x()
        windowWidth = self.geometry().width()
        windowCenter = windowWidth/2
        if xCursor<windowWidth/2:
            side = 'left'
        else:
            side = 'right'
        return side

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        # Open Recent submenu
        self.openRecentMenu = fileMenu.addMenu("Open Recent")
        fileMenu.addAction(self.saveAction)
        # Separator
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Edit menu
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addSeparator()
        # Font size
        self.fontSize = self.df_settings.at['fontSize', 'value']
        self.fontSizeMenu = editMenu.addMenu("Font size")
        fontActionGroup = QActionGroup(self)
        fs = int(re.findall('(\d+)pt', self.fontSize)[0])
        for i in range(2,25):
            action = QAction(self)
            action.setText(f'{i}')
            action.setCheckable(True)
            if i == fs:
                action.setChecked(True)
                self.fontSizeAction = action
            fontActionGroup.addAction(action)
            action = self.fontSizeMenu.addAction(action)
        editMenu.addAction(self.editTextIDsColorAction)
        editMenu.addAction(self.enableAutoZoomToCellsAction)


        # Image menu
        ImageMenu = menuBar.addMenu("&Image")
        ImageMenu.addSeparator()
        ImageMenu.addAction(self.imgPropertiesAction)
        filtersMenu = ImageMenu.addMenu("Filters")
        filtersMenu.addAction(self.gaussBlurAction)
        filtersMenu.addAction(self.edgeDetectorAction)
        filtersMenu.addAction(self.entropyFilterAction)
        normalizeIntensitiesMenu = ImageMenu.addMenu("Normalize intensities")
        normalizeIntensitiesMenu.addAction(self.normalizeRawAction)
        normalizeIntensitiesMenu.addAction(self.normalizeToFloatAction)
        # normalizeIntensitiesMenu.addAction(self.normalizeToUbyteAction)
        normalizeIntensitiesMenu.addAction(self.normalizeRescale0to1Action)
        normalizeIntensitiesMenu.addAction(self.normalizeByMaxAction)

        # Analyse menu
        analyseMenu = menuBar.addMenu("&Analyse")
        analyseMenu.addAction(self.setMeasurementsAction)

        # Help menu
        helpMenu = menuBar.addMenu("&Help")
        # helpMenu.addAction(self.tipsAction)
        # helpMenu.addAction(self.UserManualAction)
        # helpMenu.addAction(self.aboutAction)

    def gui_createTopToolBars(self):
        toolbarSize = 34

        # File toolbar
        fileToolBar = self.addToolBar("File")
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        fileToolBar.setMovable(False)
        fileToolBar.addAction(self.saveAction)
        fileToolBar.addAction(self.showInFileManagerAction)
        self.initiallyHiddenItems['top'].append(self.showInFileManagerAction)
        fileToolBar.addAction(self.undoAction)
        fileToolBar.addAction(self.redoAction)
        self.expNameText = '   Relative path: '
        expNameLabel = QLabel(self.expNameText)
        self.expNameLabelAction = fileToolBar.addWidget(expNameLabel)
        self.expNameCombobox = QComboBox(fileToolBar)
        self.expNameCombobox.SizeAdjustPolicy(QComboBox.AdjustToContents)
        self.expNameAction = fileToolBar.addWidget(self.expNameCombobox)
        self.initiallyHiddenItems['top'].append(self.expNameAction)
        self.initiallyHiddenItems['top'].append(self.expNameLabelAction)

        self.topFileToolBar = fileToolBar

        # Navigation toolbar
        viewerToolbar = QToolBar("Navigation", self)
        # viewerToolbar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(viewerToolbar)

        self.slideshowButton = QToolButton(self)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        self.slideshowButton.setToolTip('Open slideshow (Ctrl+W)')
        action = viewerToolbar.addWidget(self.slideshowButton)

        self.rulerButton = QToolButton(self)
        self.rulerButton.setIcon(QIcon(":ruler.svg"))
        self.rulerButton.setCheckable(True)
        self.rulerButton.setToolTip(
            'Measure straight line. '
            'Length is displayed on the bottom-right corner.'
        )
        action = viewerToolbar.addWidget(self.rulerButton)
        self.leftClickButtons.append(self.rulerButton)

        self.initiallyHiddenItems['top'].append(viewerToolbar)
        self.topViewerToolbar = viewerToolbar

    def gui_createSideFileToolbar(self, side):
        self.sideToolbar[side]['fileToolbar'] = {}
        toolbarActions = self.sideToolbar[side]['fileToolbar']

        openFolderAction = QAction(
            QIcon(":folder-open.svg"),
            f"Load data from folder onto the {side} image...", self
        )
        toolbarActions['openFolderAction'] = openFolderAction

        openFileAction = QAction(
            QIcon(":image.svg"),
            f"Load single image/video file onto the {side} image...", self
        )
        toolbarActions['openFileAction'] = openFileAction

        # reloadAction = QAction(
        #     QIcon(":reload.svg"), "Reload {side} image file", self
        # )
        # toolbarActions['reloadAction'] = reloadAction

        fileToolBar = QToolBar(f"{side} image file toolbar", self)
        fileToolBar.setMovable(False)

        QtSide = Qt.LeftToolBarArea if side == 'left' else Qt.RightToolBarArea
        self.addToolBar(QtSide, fileToolBar)

        fileToolBar.addAction(openFolderAction)
        fileToolBar.addAction(openFileAction)
        # fileToolBar.addAction(reloadAction)
        # self.initiallyHiddenItems[side].append(reloadAction)

    def gui_createSideViewToolbar(self, side):
        self.sideToolbar[side]['viewToolbar'] = {}
        viewToolbarDict = self.sideToolbar[side]['viewToolbar']

        overlayButton = widgets.DblClickQToolButton(self)
        overlayButton.setIcon(QIcon(":overlay.svg"))
        overlayButton.setCheckable(True)
        overlayButton.setToolTip('Overlay fluorescent image\n'
        'NOTE: Button has a green background if you successfully loaded fluorescent data\n\n'
        'If you need to overlay a different channel load (or load it again)\n'
        'from "File --> Load fluorescent images" menu.')
        viewToolbarDict['overlayButton'] = overlayButton
        self.checkableToolButtons.append(overlayButton)

        colorButton = pg.ColorButton(
            self, color=(0,255,255)
        )
        colorButton.clicked.disconnect()
        colorButton.hide()
        viewToolbarDict['colorButton'] = colorButton

        plotSpotsCoordsButton = widgets.DblClickQToolButton(self)
        plotSpotsCoordsButton.setIcon(QIcon(":plotSpots.svg"))
        plotSpotsCoordsButton.setCheckable(True)
        plotSpotsCoordsButton.setToolTip('Plot spots coordinates')
        viewToolbarDict['plotSpotsCoordsButton'] = plotSpotsCoordsButton
        self.checkableToolButtons.append(plotSpotsCoordsButton)

        plotSkeletonButton = widgets.DblClickQToolButton(self)
        plotSkeletonButton.setIcon(QIcon(":skeletonize.svg"))
        plotSkeletonButton.setCheckable(True)
        plotSkeletonButton.setToolTip('Plot skeleton of selected data')
        viewToolbarDict['plotSkeletonButton'] = plotSkeletonButton
        self.checkableToolButtons.append(plotSkeletonButton)

        plotContourButton = widgets.DblClickQToolButton(self)
        plotContourButton.setIcon(QIcon(":contour.svg"))
        plotContourButton.setCheckable(True)
        plotContourButton.setToolTip('Plot contour of selected data')
        viewToolbarDict['plotContourButton'] = plotContourButton
        self.checkableToolButtons.append(plotContourButton)

        viewToolbar = QToolBar(f"{side} image controls", self)
        viewToolbar.setMovable(False)
        QtSide = Qt.LeftToolBarArea if side == 'left' else Qt.RightToolBarArea
        self.addToolBar(QtSide, viewToolbar)

        overlayAction = viewToolbar.addWidget(overlayButton)
        viewToolbarDict['overlayAction'] = overlayAction
        self.initiallyHiddenItems[side].append(overlayAction)

        toolColorButton = widgets.colorToolButton()
        toolColorButton.setToolTip('Edit colors')
        viewToolbarDict['toolColorButton'] = toolColorButton
        colorAction = viewToolbar.addWidget(toolColorButton)
        viewToolbarDict['colorAction'] = colorAction
        self.initiallyHiddenItems[side].append(colorAction)

        plotSpotsCoordsAction = viewToolbar.addWidget(plotSpotsCoordsButton)
        viewToolbarDict['plotSpotsCoordsAction'] = plotSpotsCoordsAction
        self.initiallyHiddenItems[side].append(plotSpotsCoordsAction)

        plotSkeletonAction = viewToolbar.addWidget(plotSkeletonButton)
        viewToolbarDict['plotSkeletonAction'] = plotSkeletonAction
        self.initiallyHiddenItems[side].append(plotSkeletonAction)

        plotContourAction = viewToolbar.addWidget(plotContourButton)
        viewToolbarDict['plotContourAction'] = plotContourAction
        self.initiallyHiddenItems[side].append(plotContourAction)

    def gui_createLeftToolBars(self):
        self.gui_createSideFileToolbar('left')
        self.gui_createSideViewToolbar('left')

    def gui_createRightToolBars(self):
        self.gui_createSideFileToolbar('right')
        self.gui_createSideViewToolbar('right')

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_createActions(self):
        # File actions
        self.saveAction = QAction(
            QIcon(":file-save.svg"), "&Save (Ctrl+S)", self
        )

        self.showInFileManagerAction = QAction(
            QIcon(":drawer.svg"), "&Show in Explorer/Finder", self
        )
        self.exitAction = QAction("&Exit", self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo (Ctrl+Z)", self)
        self.redoAction = QAction(QIcon(":redo.svg"), "Redo (Ctrl+Y)", self)
        self.undoAction.setDisabled(True)
        self.redoAction.setDisabled(True)
        # String-based key sequences
        self.saveAction.setShortcut("Ctrl+S")
        self.undoAction.setShortcut("Ctrl+Z")
        self.redoAction.setShortcut("Ctrl+Y")

        self.editTextIDsColorAction = QAction('Edit text on IDs color...', self)

        self.enableAutoZoomToCellsAction = QAction(
            'Automatic zoom to all cells when pressing "Next/Previous"', self
        )
        self.enableAutoZoomToCellsAction.setCheckable(True)

        self.imgPropertiesAction = QAction('Properties...', self)

        self.gaussBlurAction = QAction('Gaussian blur...', self)
        self.gaussBlurAction.setCheckable(True)

        self.edgeDetectorAction = QAction('Edge detection...', self)
        self.edgeDetectorAction.setCheckable(True)

        self.entropyFilterAction = QAction('Object detection...', self)
        self.entropyFilterAction.setCheckable(True)

        self.normalizeRawAction = QAction(
            'Do not normalize. Display raw image', self
        )
        self.normalizeToFloatAction = QAction(
            'Convert to floating point format with values [0, 1]', self
        )
        # self.normalizeToUbyteAction = QAction(
        #     'Rescale to 8-bit unsigned integer format with values [0, 255]', self)
        self.normalizeRescale0to1Action = QAction(
            'Rescale to [0, 1]', self
        )
        self.normalizeByMaxAction = QAction(
            'Normalize by max value', self
        )
        self.normalizeRawAction.setCheckable(True)
        self.normalizeToFloatAction.setCheckable(True)
        # self.normalizeToUbyteAction.setCheckable(True)
        self.normalizeRescale0to1Action.setCheckable(True)
        self.normalizeByMaxAction.setCheckable(True)
        self.normalizeQActionGroup = QActionGroup(self)
        self.normalizeQActionGroup.addAction(self.normalizeRawAction)
        self.normalizeQActionGroup.addAction(self.normalizeToFloatAction)
        # self.normalizeQActionGroup.addAction(self.normalizeToUbyteAction)
        self.normalizeQActionGroup.addAction(self.normalizeRescale0to1Action)
        self.normalizeQActionGroup.addAction(self.normalizeByMaxAction)

        self.setLastUserNormAction()

        self.setMeasurementsAction = QAction('Set measurements...', self)

    def gui_createComputeDockWidget(self):
        self.computeDockWidget = QDockWidget('spotMAX Tab Control', self)
        computeTabControl = dialogs.guiTabControl(parent=self.computeDockWidget)

        self.computeDockWidget.setWidget(computeTabControl)
        self.computeDockWidget.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )
        self.computeDockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )

        self.addDockWidget(Qt.LeftDockWidgetArea, self.computeDockWidget)

    def gui_createSpotsClickedContextMenuActions(self):
        showGroup = QActionGroup(self)

        self.showOnlyInsideRefAction = QAction(
            'Show only spots inside ref. channel', self
        )
        self.showOnlyInsideRefAction.setCheckable(True)

        self.showOnlyOutsideRefAction = QAction(
            'Show only spots outside ref. channel', self
        )
        self.showOnlyOutsideRefAction.setCheckable(True)

        self.showAllSpotsAction = QAction(
            'Show all spots', self
        )
        self.showAllSpotsAction.setCheckable(True)
        self.showAllSpotsAction.setChecked(True)

        showGroup.addAction(self.showOnlyInsideRefAction)
        showGroup.addAction(self.showOnlyOutsideRefAction)
        showGroup.addAction(self.showAllSpotsAction)

        self.editColorMenu = QMenu('Edit spots color', self)
        # self.editClickedSpotColor = QAction(
        #     'Clicked spot', self.editColorMenu)
        self.editInsideRefColor = QAction(
            'Spots inside ref. channel', self.editColorMenu
        )
        self.editOutsideRefColor = QAction(
            'Spots outside ref. channel', self.editColorMenu
        )
        self.editAllSpotsColor = QAction(
            'All spots', self.editColorMenu
        )
        # self.editColorMenu.addAction(self.editClickedSpotColor)
        self.editColorMenu.addAction(self.editInsideRefColor)
        self.editColorMenu.addAction(self.editOutsideRefColor)
        self.editColorMenu.addAction(self.editAllSpotsColor)

        self.spotStyleAction = QAction('Style...', self)

    def gui_createSkeletonClickedContextMenuActions(self):
        self.editSkelColorAction = QAction('Skeleton color...', self)
        self.skelStyleAction = QAction('Style...', self)

    def gui_connectActions(self):
        # Connect File actions
        fileToolbarLeft = self.sideToolbar['left']['fileToolbar']
        openFolderActionLeft = fileToolbarLeft['openFolderAction']
        openFolderActionLeft.triggered.connect(self.openFolderLeft)

        openFileActionLeft = fileToolbarLeft['openFileAction']
        openFileActionLeft.triggered.connect(self.openFileLeft)

        fileToolbarRight = self.sideToolbar['right']['fileToolbar']
        openFolderActionRight = fileToolbarRight['openFolderAction']
        openFolderActionRight.triggered.connect(self.openFolderRight)

        openFileActionRight = fileToolbarRight['openFileAction']
        openFileActionRight.triggered.connect(self.openFileRight)

        self.saveAction.triggered.connect(self.saveData)
        self.showInFileManagerAction.triggered.connect(self.showInFileManager)
        self.exitAction.triggered.connect(self.close)

        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)

        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)

        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

        self.showOnlyInsideRefAction.triggered.connect(self.updateSpots)
        self.showOnlyOutsideRefAction.triggered.connect(self.updateSpots)
        self.showAllSpotsAction.triggered.connect(self.updateSpots)

        self.expNameCombobox.currentTextChanged.connect(self.loadSelectedData)

        self.setMeasurementsAction.triggered.connect(self.setMeasurements)

    def gui_connectBottomWidgetsSignals(self, side):
        if self.areActionsConnected[side]:
            return

        bottomWidgets = self.bottomWidgets[side]

        bottomWidgets['howDrawSegmCombobox'].currentTextChanged.connect(
            self.howDrawSegmCombobox_cb
        )

        bottomWidgets['navigateScrollbar'].sigActionTriggered.connect(
            self.navigateScrollbarTriggered
        )
        bottomWidgets['navigateScrollbar'].sliderReleased.connect(
            self.navigateScrollbarReleased
        )

        bottomWidgets['zSliceScrollbarLayer0'].sigActionTriggered.connect(
            self.zSliceScrollbarLayerTriggered
        )
        bottomWidgets['zSliceScrollbarLayer0'].sliderReleased.connect(
            self.zSliceScrollbarLayerReleased
        )

        bottomWidgets['zProjComboboxLayer0'].currentTextChanged.connect(
            self.updateZprojLayer
        )

        bottomWidgets['zProjComboboxLayer1'].currentTextChanged.connect(
            self.updateZprojLayer
        )
        bottomWidgets['zSliceScrollbarLayer1'].sigActionTriggered.connect(
            self.zSliceScrollbarLayerTriggered
        )
        bottomWidgets['zSliceScrollbarLayer1'].sliderReleased.connect(
            self.zSliceScrollbarLayerReleased
        )

        bottomWidgets['alphaScrollbar'].actionTriggered.connect(
            self.updateAlphaOverlay
        )

    def gui_connectSideToolbarsSignals(self, side):
        if self.areActionsConnected[side]:
            return

        self.showComputeDockButton.clicked.connect(self.showComputeDockWidget)

        viewToolbar = self.sideToolbar[side]['viewToolbar']

        colorButton = viewToolbar['colorButton']
        toolColorButton = viewToolbar['toolColorButton']
        toolColorButton.sigClicked.connect(self.gui_selectColor)
        colorButton.sigColorChanging.connect(self.gui_setColor)
        colorButton.sigColorChanged.connect(self.gui_setColor)

        plotSpotsCoordsButton = viewToolbar['plotSpotsCoordsButton']
        plotSpotsCoordsButton.sigDoubleClickEvent.connect(
            self.plotSpotsCoordsClicked
        )
        plotSpotsCoordsButton.sigClickEvent.connect(
            self.plotSpotsCoordsClicked
        )

        overlayButton = viewToolbar['overlayButton']
        overlayButton.sigClickEvent.connect(self.overlayClicked)
        overlayButton.sigDoubleClickEvent.connect(self.overlayClicked)

        plotSkeletonButton = viewToolbar['plotSkeletonButton']
        plotSkeletonButton.sigDoubleClickEvent.connect(
            self.plotSkeletonClicked
        )
        plotSkeletonButton.sigClickEvent.connect(
            self.plotSkeletonClicked
        )

        plotContourButton = viewToolbar['plotContourButton']
        plotContourButton.sigDoubleClickEvent.connect(
            self.plotContoursClicked
        )
        plotContourButton.sigClickEvent.connect(
            self.plotContoursClicked
        )

    def gui_connectMenuBarSignals(self):
        self.fontSizeMenu.triggered.connect(self.changeFontSize)

    def gui_selectColor(self):
        viewToolbar = self.sideToolbar['left']['viewToolbar']
        side = self.side(viewToolbar['toolColorButton'])

        win = dialogs.QDialogListbox(
            'Selec color to edit', 'Select which item\'s color to edit',
            list(self.colorItems[side].keys()), multiSelection=False,
            currentItem=self.lastSelectedColorItem, parent=self
        )
        win.show()
        win.exec_()
        if win.cancel:
            return

        self.lastSelectedColorItem = win.selectedItems[0]

        viewToolbar = self.sideToolbar[side]['viewToolbar']
        colorButton = viewToolbar['colorButton']
        toolColorButton = viewToolbar['toolColorButton']
        colorButton.side = side
        colorButton.key = win.selectedItems[0].text()
        colorButton.setColor(self.colorItems[side][colorButton.key])
        toolColorButton.setColor(self.colorItems[side][colorButton.key])
        colorButton.selectColor()

    def gui_setColor(self, colorButton):
        side = colorButton.side
        key = colorButton.key
        rgb = list(colorButton.color(mode='byte'))
        viewToolbar = self.sideToolbar[side]['viewToolbar']
        toolColorButton = viewToolbar['toolColorButton']
        toolColorButton.setColor(tuple(rgb))
        if key == 'All spots':
            self.colorItems[side]["Spots inside ref. channel"] = rgb
            self.colorItems[side]["Spots outside ref. channel"] = rgb
        else:
            self.colorItems[side][key] = rgb
        try:
            colorButton.scatterItem.createBrushesPens()
        except AttributeError:
            pass
        self.updateImage(side)

    def gui_createFonts(self):
        self._font = QFont()
        self._font.setPixelSize(11)

    def gui_setListItems(self):
        self.drawSegmComboboxItems = [
            'Draw IDs and contours',
            'Draw IDs and overlay segm. masks',
            'Draw only cell cycle info',
            'Draw cell cycle info and contours',
            'Draw cell cycle info and overlay segm. masks',
            'Draw only mother-bud lines',
            'Draw only IDs',
            'Draw only contours',
            'Draw only overlay segm. masks',
            'Draw nothing'
        ]

        self.zProjItems = [
            'single z-slice',
            'max z-projection',
            'mean z-projection',
            'median z-proj.'
        ]

        with open(colorItems_path) as file:
            self.colorItems = json.load(file)

    def gui_createBottomLayout(self):
        self.bottomLayout = QHBoxLayout()
        self.bottomLayout.addSpacing(100)

    def gui_createSideLayout(self):
        self.sideLayout = QVBoxLayout()

    def gui_addSideLayoutWidgets(self):
        self.showComputeDockButton = acdc_widgets.expandCollapseButton()
        self.showComputeDockButton.setToolTip('Analysis parameters')
        self.sideLayout.addWidget(self.showComputeDockButton)
        self.sideLayout.setSpacing(0)
        self.sideLayout.setContentsMargins(0,0,0,0)
        # self.initiallyHiddenItems['left'].append(self.showComputeDockButton)

    def gui_createBottomWidgets(self, side):
        frame = dialogs.guiBottomWidgets(
            side, self.drawSegmComboboxItems, self.zProjItems,
            isCheckable=True, checked=side=='left', font=self._font
        )
        bottomWidgets = frame.bottomWidgets
        bottomWidgets['zProjComboboxLayer0'].setCurrentText('single z-slice')
        bottomWidgets['zProjComboboxLayer1'].setCurrentText('same as above')
        return frame, bottomWidgets

    def gui_addBottomLeftWidgets(self):
        frame, bottomLeftWidgets = self.gui_createBottomWidgets('left')
        self.bottomWidgets['left'] = bottomLeftWidgets
        self.bottomWidgets['left']['frame'] = frame
        self.bottomLayout.addWidget(frame)
        self.initiallyHiddenItems['left'].append(frame)

    def gui_addBottomRightWidgets(self):
        frame, bottomRightWidgets = self.gui_createBottomWidgets('right')
        self.bottomWidgets['right'] = bottomRightWidgets
        self.bottomWidgets['right']['frame'] = frame
        self.bottomLayout.addWidget(frame)
        self.bottomLayout.addSpacing(100)
        self.initiallyHiddenItems['right'].append(frame)

    def gui_createThreadPool(self):
        self.maxThreads = QThreadPool.globalInstance().maxThreadCount()
        self.threadCount = 0
        self.threadQueue = Queue()
        self.threadPool = QThreadPool.globalInstance()

        self.progressWin = None
        self.thread = QThread()
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.waitReadH5cond = QWaitCondition()
        self.readH5mutex = QMutex()
        self.worker = qtworkers.loadChunkDataWorker(
            self.mutex, self.waitCond, self.waitReadH5cond, self.readH5mutex
        )
        self.worker.moveToThread(self.thread)
        self.worker.wait = True

        self.worker.signals.finished.connect(self.thread.quit)
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.signals.progress.connect(self.workerProgress)
        self.worker.signals.sigLoadingNewChunk.connect(self.loadingNewChunk)
        self.worker.sigLoadingFinished.connect(self.loadChunkFinished)
        self.worker.signals.critical.connect(self.workerCritical)
        self.worker.signals.finished.connect(self.loadChunkWorkerClosed)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def gui_createGraphicsPlots(self):
        self.graphLayout.removeItem(self.welcomeTextLeft)
        self.graphLayout.removeItem(self.welcomeTextRight)

        # Left plot
        ax1 = pg.PlotItem()
        ax1.invertY(True)
        ax1.setAspectLocked(True)
        ax1.hideAxis('bottom')
        ax1.hideAxis('left')
        self.graphLayout.addItem(ax1, row=1, col=1)
        self.axes['left'] = ax1

        # Right plot
        ax2 = pg.PlotItem()
        ax2.setAspectLocked(True)
        ax2.invertY(True)
        ax2.hideAxis('bottom')
        ax2.hideAxis('left')
        self.graphLayout.addItem(ax2, row=1, col=2)
        self.axes['right'] = ax2

        # self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        # self.graphLayout.addItem(self.titleLabel, row=0, col=1)

    def gui_addGraphicsItems(self):
        # Welcome text
        html_path = os.path.join(spotmax_path, 'html_files', 'gui_welcome.html')
        with open(html_path) as html_file:
            htmlText = html_file.read()
        self.welcomeTextLeft = self.graphLayout.addLabel(
            text=htmlText, row=1, col=1
        )
        self.welcomeTextLeft.item.adjustSize()

        htmlText = htmlText.replace('left-side', 'right-side')
        htmlText = htmlText.replace('left image', 'right image')
        htmlText = htmlText.replace('spots channel', 'reference channel')
        self.welcomeTextRight = self.graphLayout.addLabel(
            text=htmlText, row=1, col=2
        )
        self.welcomeTextRight.item.adjustSize()

        # Auto image adjustment buttons
        proxy = QGraphicsProxyWidget()
        equalizeHistButton = QPushButton('Auto')
        equalizeHistButton.setStyleSheet(
            'QPushButton {background-color: #282828; color: #F0F0F0;}'
            'QPushButton:checked {background-color: #646464;}'
        )
        equalizeHistButton.setCheckable(True)
        proxy.setWidget(equalizeHistButton)
        self.graphLayout.addItem(proxy, row=0, col=0)
        equalizeHistButton.hide()
        self.histItems['left']['equalizeButton'] = equalizeHistButton

        proxy = QGraphicsProxyWidget()
        equalizeHistButton = QPushButton('Auto')
        equalizeHistButton.setStyleSheet(
               'QPushButton {background-color: #282828; color: #F0F0F0;}'
               'QPushButton:checked {background-color: #646464;}'
        )
        equalizeHistButton.setCheckable(True)
        proxy.setWidget(equalizeHistButton)
        self.graphLayout.addItem(proxy, row=0, col=3)
        equalizeHistButton.hide()
        self.histItems['right']['equalizeButton'] = equalizeHistButton

        # Left image histogram
        histLeft = acdc_widgets.myHistogramLUTitem()

        self.graphLayout.addItem(histLeft, row=1, col=0)
        histLeft.hide()
        self.histItems['left']['hist'] = histLeft

        # Right image histogram
        try:
            histRight = acdc_widgets.myHistogramLUTitem(gradientPosition='left')
        except TypeError:
            histRight = acdc_widgets.myHistogramLUTitem()

        self.graphLayout.addItem(histRight, row=1, col=3)
        histRight.hide()
        self.histItems['right']['hist'] = histRight

    def gui_addImageItem(self, side):
        # Blank image
        self.blank = np.zeros((256,256), np.uint8)

        imgItem = widgets.ImageItem(self.blank)
        histItem = self.histItems[side]['hist']
        # Use same left click menu as the hist item
        imgItem.menu = histItem.gradient.menu
        self.imgItems[side] = imgItem
        self.axes[side].addItem(imgItem)

    def gui_addRulerItems(self, side):
        # Ruler plotItem and scatterItem
        rulerPen = pg.mkPen(color='r', style=Qt.DashLine, width=2)
        self.axes[side].rulerPlotItem = pg.PlotDataItem(pen=rulerPen)
        self.axes[side].rulerAnchorsItem = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2)
        )
        self.axes[side].addItem(self.axes[side].rulerPlotItem)
        self.axes[side].addItem(self.axes[side].rulerAnchorsItem)

    def gui_addSpotsScatterItem(self, side):
        spotsScatterItem = widgets.ScatterPlotItem(
            self, side, 'spots', self.plotSpotsCoords,
            symbol='o', size=3, pxMode=False, hoverable=True
        )
        self.axes[side].spotsScatterItem = spotsScatterItem
        self.axes[side].addItem(spotsScatterItem)

        self.gui_connectSpotsScatterItemActions(spotsScatterItem)

    def gui_addSkelScatterItem(self, side):
        skelScatterItem = widgets.ScatterPlotItem(
            self, side, 'skel', self.plotSkeleton,
            symbol='o', size=1.5, pxMode=False, hoverable=True,
            brush=pg.mkBrush(self.colorItems[side]['Skeleton color...']),
            pen=pg.mkPen(self.colorItems[side]['Skeleton color...'])
        )
        self.axes[side].skelScatterItem = skelScatterItem
        self.axes[side].addItem(skelScatterItem)

        self.gui_connectSkelScatterItemActions(skelScatterItem)

    def gui_addContourScatterItem(self, side):
        contourScatterItem = pg.ScatterPlotItem(
            symbol='s', size=1, pxMode=False,
            brush=pg.mkBrush(self.colorItems[side]['Contour color...']),
            pen=pg.mkPen(self.colorItems[side]['Contour color...'],
            width=2)
        )
        self.axes[side].contourScatterItem = contourScatterItem
        self.axes[side].addItem(contourScatterItem)

    def gui_connectSpotsScatterItemActions(self, scatterItem):
        try:
            self.editInsideRefColor.triggered.disconnect()
            self.editOutsideRefColor.triggered.disconnect()
            self.editAllSpotsColor.triggered.disconnect()
            self.spotStyleAction.triggered.disconnect()
        except TypeError:
            pass

        self.editInsideRefColor.triggered.connect(scatterItem.selectColor)
        self.editOutsideRefColor.triggered.connect(scatterItem.selectColor)
        self.editAllSpotsColor.triggered.connect(scatterItem.selectColor)
        self.spotStyleAction.triggered.connect(scatterItem.selectStyle)

    def gui_connectSkelScatterItemActions(self, scatterItem):
        try:
            self.editSkelColorAction.triggered.disconnect()
            self.skelStyleAction.triggered.disconnect()
        except TypeError:
            pass

        self.editSkelColorAction.triggered.connect(scatterItem.selectColor)
        self.skelStyleAction.triggered.connect(scatterItem.selectStyle)

    def gui_addPlotItems(self, side):
        self.gui_addImageItem(side)
        self.gui_addSegmVisuals(side)
        self.gui_addRulerItems(side)
        self.gui_addSkelScatterItem(side)
        self.gui_addContourScatterItem(side)
        self.gui_addSpotsScatterItem(side)

    def gui_removeAllItems(self, side):
        self.axes[side].clear()

    def gui_createGraphicsItems(self):
        # Contour pens
        self.oldIDs_cpen = pg.mkPen(color=(200, 0, 0, 255*0.5), width=2)
        self.newIDs_cpen = pg.mkPen(color='r', width=3)
        self.tempNewIDs_cpen = pg.mkPen(color='g', width=3)
        self.lostIDs_cpen = pg.mkPen(color=(245, 184, 0, 100), width=4)

        # Lost ID question mark text color
        self.lostIDs_qMcolor = (245, 184, 0)

        # New bud-mother line pen
        self.NewBudMoth_Pen = pg.mkPen(color='r', width=3, style=Qt.DashLine)

        # Old bud-mother line pen
        self.OldBudMoth_Pen = pg.mkPen(color=(255,165,0), width=2,
                                       style=Qt.DashLine)


    def gui_addSegmVisuals(self, side):
        # Temporary line item connecting bud to new mother
        self.BudMothTempLine = pg.PlotDataItem(pen=self.NewBudMoth_Pen)
        self.axes['left'].addItem(self.BudMothTempLine)

        # Create enough PlotDataItems and LabelItems to draw contours and IDs.
        maxID = 0
        for posData in self.expData[side]:
            if posData.segm_data is not None:
                maxID = posData.segm_data.max()

        numItems = maxID+10
        self.axes[side].ContoursCurves = []
        self.axes[side].BudMothLines = []
        self.axes[side].LabelItemsIDs = []

        for i in range(numItems):
            # Contours
            ContCurve = pg.PlotDataItem()
            self.axes[side].ContoursCurves.append(ContCurve)
            self.axes[side].addItem(ContCurve)

            # Bud mother
            BudMothLine = pg.PlotDataItem()
            self.axes[side].BudMothLines.append(BudMothLine)
            self.axes[side].addItem(BudMothLine)

            # LabelItems
            ax1_IDlabel = pg.LabelItem()
            self.axes[side].LabelItemsIDs.append(ax1_IDlabel)
            self.axes[side].addItem(ax1_IDlabel)

    def gui_connectGraphicsEvents(self, side, disconnect=False):
        if self.areActionsConnected[side] and disconnect:
            self.imgItems[side].sigHoverEvent.disconnect()
            histItem = self.histItems[side]['hist']
            histItem.sigContextMenu.disconnect()
            histItem.sigLookupTableChanged.disconnect()
            self.axes[side].spotsScatterItem.sigClicked.disconnect()
            return

        if disconnect:
            return

        self.imgItems[side].sigHoverEvent.connect(self.gui_hoverEventImage)

        histItem = self.histItems[side]['hist']
        histItem.setImageItem(self.imgItems[side])
        # histItem.sigContextMenu.connect(self.gui_raiseContextMenuLUT)
        histItem.sigLookupTableChanged.connect(self.histLUTchanged)

        self.axes[side].spotsScatterItem.sigClicked.connect(self.spotsClicked)
        self.axes[side].skelScatterItem.sigClicked.connect(self.skelClicked)


    def gui_raiseContextMenuLUT(self, histItem, event):
        side = self.side(self.histItems['left']['hist'], sender=histItem)
        chNamesQActionGroup = self.histItems['left']['actionGroup']
        menu = QMenu(self)
        for action in chNamesQActionGroup.actions():
            menu.addAction(action)
        menu.exec(event.screenPos())

    def gui_addChannelNameQActionGroup(self, side, chName):
        # LUT histogram channel name context menu actions
        actionGroup = self.histItems[self.lastLoadedSide]['actionGroup']
        chNameAction = QAction(self)
        chNameAction.setCheckable(True)
        chNameAction.setText(chName)
        actionGroup.addAction(chNameAction)
        if len(actionGroup.actions()) == 1:
            chNameAction.setChecked(True)
        actionGroup.triggered.connect(self.channelNameLUTmenuActionTriggered)
        self.histItems[self.lastLoadedSide]['actionGroup'] = actionGroup

    def gui_addGradientLevels(self, side, filename=''):
        posData = self.currentPosData(side)
        if not filename:
            filename = posData.filename
        overlayButton = self.sideToolbar[side]['viewToolbar']['overlayButton']

        if not overlayButton.isChecked():
            filename = f'{filename}_overlayOFF'

        histItem = self.histItems[side]['hist']
        min = histItem.gradient.listTicks()[0][1]
        max = histItem.gradient.listTicks()[1][1]
        posData.gradientLevels[filename] = (min, max)

    def gui_hoverEventImage(self, imageItem, event):
        side = self.side(self.imgItems['left'], sender=imageItem)
        posData = self.currentPosData(side)
        # Update x, y, value label bottom right
        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None

        drawRulerLine = (
            self.rulerButton.isChecked()
            and not event.isExit()
        )
        if drawRulerLine:
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            xxRA, yyRA = self.axes[side].rulerAnchorsItem.getData()
            if self.isCtrlDown:
                ydata = yyRA[0]
            self.axes[side].rulerPlotItem.setData(
                [xxRA[0], xdata], [yyRA[0], ydata]
            )

        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            layer0 = self.layerImage(side)
            Y, X = layer0.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = layer0[ydata, xdata]
                _max = layer0.max()
                hoverText = (
                    f'(x={x:.2f}, y={y:.2f}, value={val:.0f}, max={_max:.3f}'
                )
                if posData.rp is not None:
                    frame_i = self.frame_i(side)
                    rp = posData.regionprops(frame_i, returnDict=True)
                    lab = posData.segmLabels(frame_i)
                    ID = lab[ydata, xdata]
                    hoverText = f'{hoverText}, ID={ID}'
                    if ID > 0:
                        obj = rp[ID]
                        hoverText = f'{hoverText}, cell_vol_fl={obj.vol_fl:.2f}'
                hoverText = f'{hoverText})'
                self.wcLabel.setText(hoverText)
            else:
                self.wcLabel.setText(f'')


    def gui_init(self, first_call=False, side='left'):
        # Clear PlotItems
        # Write the instructions again (drag and drop etc.)
        # Disable and hide widgets
        # Reinitialize attributes initialized at __init__ method
        self.slideshowWin = None
        self.segmModel = None
        self.ccaTableWin = None
        self.flag = True
        if first_call:
            return

        self.dataLoaded[side] = False
        self.gui_addTitleLabel()

        if side == 'left' and not self.expData['right']:
            # Loading on the left while right was never loaded --> hide right
            self.axes['right'].hide()
        elif side == 'right' and not self.expData['left']:
            # Loading on the right while left was never loaded --> hide left
            self.axes['left'].hide()
        elif side == 'left' and self.expData['right']:
            # Loading on the left with right already loaded --> keep right visible
            self.axes['left'].show()
        elif side == 'right' and self.expData['left']:
            # Loading on the right with left already loaded --> keep left visible
            self.axes['left'].show()

        self.gui_hideInitItems(side=side)
        self.gui_uncheckToolButtons()

    def gui_uncheckToolButtons(self):
        for button in self.checkableToolButtons:
            button.setChecked(False)

    def gui_hideInitItems(self, side='all'):
        if side == 'all':
            for side, items in self.initiallyHiddenItems.items():
                for item in items:
                    item.setVisible(False)
                    item.setEnabled(False)
                    try:
                        item.setChecked(False)
                    except AttributeError as e:
                        pass
        else:
            for item in self.initiallyHiddenItems[side]:
                item.setVisible(False)
                item.setEnabled(False)
                try:
                    if item.isCheckable():
                        item.setChecked(False)
                except AttributeError as e:
                    pass

    def gui_setItemsDataShape(self, side):
        if self.dataLoaded[side]:
            return

        bottomWidgets = self.bottomWidgets[side]
        bottomWidgets['frame'].setVisible(True)
        bottomWidgets['frame'].setDisabled(False)
        posData = self.currentPosData(side)
        numPos = posData.loadSizeS
        overlayButton = self.sideToolbar[side]['viewToolbar']['overlayButton']

        if posData.SizeZ > 1:
            midZslice = int(posData.SizeZ/2)
            bottomWidgets['zSliceScrollbarLayer0'].setDisabled(False)
            bottomWidgets['zSliceScrollbarLayer0'].setMaximum(posData.SizeZ)
            bottomWidgets['zSliceScrollbarLayer1'].setMaximum(posData.SizeZ)
            bottomWidgets['zSliceScrollbarLayer0'].setSliderPosition(midZslice)
            zSliceSbL0_label = bottomWidgets['zSliceSbL0_label']
            z_str = str(midZslice).zfill(len(str(posData.SizeZ)))
            zSliceSbL0_label.setText(
                f'First layer z-slice {z_str}/{posData.SizeZ}'
            )
            bottomWidgets['zProjComboboxLayer0'].setDisabled(False)
        else:
            bottomWidgets['zSliceScrollbarLayer0'].setDisabled(True)
            bottomWidgets['zProjComboboxLayer0'].setDisabled(True)
        if posData.SizeT > 1:
            bottomWidgets['navigateScrollbar'].setSliderPosition(0)
            t_str = str(1).zfill(len(str(posData.SizeT)))
            bottomWidgets['navigateScrollbar_label'].setText(
                f'frame n. {t_str}/{posData.SizeT}  '
            )
            bottomWidgets['navigateScrollbar'].setMaximum(posData.SizeT)
            bottomWidgets['navigateScrollbar'].setDisabled(False)
        else:
            posName = posData.pos_foldername
            bottomWidgets['navigateScrollbar'].setMaximum(numPos)
            bottomWidgets['navigateScrollbar_label'].setText(f'{posName} ')
            if numPos > 1:
                bottomWidgets['navigateScrollbar'].setDisabled(False)
            else:
                bottomWidgets['navigateScrollbar'].setDisabled(True)

        if not overlayButton.isChecked():
            bottomWidgets['zSliceScrollbarLayer1'].setDisabled(True)
            bottomWidgets['zProjComboboxLayer1'].setDisabled(True)
            bottomWidgets['alphaScrollbar'].setDisabled(True)
        elif overlayButton.isChecked() and posData.SizeZ > 1:
            bottomWidgets['zSliceScrollbarLayer1'].setDisabled(False)
            bottomWidgets['zSliceScrollbarLayer1'].setSliderPosition(midZslice)
            bottomWidgets['zSliceScrollbarLayer1'].setMaximum(posData.SizeZ)
            zSliceSbL0_label = bottomWidgets['zSliceSbL1_label']
            z_str = str(midZslice).zfill(len(str(posData.SizeZ)))
            zSliceSbL0_label.setText(
                f'First layer z-slice {z_str}/{posData.SizeZ}'
            )
            bottomWidgets['zProjComboboxLayer1'].setDisabled(False)
            bottomWidgets['alphaScrollbar'].setDisabled(False)

        if overlayButton.isChecked():
            bottomWidgets['alphaScrollbar'].setDisabled(False)

    def gui_setVisibleItems(self, side):
        if self.dataLoaded[side]:
            return

        self.topViewerToolbar.setVisible(True)
        self.topViewerToolbar.setDisabled(False)

        self.showInFileManagerAction.setVisible(True)
        self.showInFileManagerAction.setDisabled(False)

        if side == 'left' and not self.expData['right']:
            self.axes['right'].hide()
            self.histItems['right']['hist'].hide()
            self.histItems['right']['equalizeButton'].hide()
        else:
            self.axes['right'].show()
            self.histItems['right']['hist'].show()
            self.histItems['right']['equalizeButton'].show()

        if side == 'right' and not self.expData['left']:
            self.axes['left'].hide()
            self.histItems['left']['hist'].hide()
            self.histItems['left']['equalizeButton'].hide()
        else:
            self.axes['left'].show()
            self.histItems['left']['hist'].show()
            self.histItems['left']['equalizeButton'].show()

        viewToolbar = self.sideToolbar[side]['viewToolbar']
        overlayButton = viewToolbar['overlayButton']
        overlayAction = viewToolbar['overlayAction']
        overlayAction.setVisible(True)
        overlayAction.setEnabled(True)

        colorAction = viewToolbar['colorAction']
        colorAction.setVisible(True)
        colorAction.setEnabled(True)

        plotSkeletonAction = viewToolbar['plotSkeletonAction']
        plotSkeletonAction.setVisible(True)
        plotSkeletonAction.setEnabled(True)

        plotContourAction = viewToolbar['plotContourAction']
        plotContourAction.setVisible(True)
        plotContourAction.setEnabled(True)

        self.gui_setVisiblePlotSpotsCoordsAction(side)

    def gui_setVisiblePlotSpotsCoordsAction(self, side):
        posData = self.currentPosData(side)
        viewToolbar = self.sideToolbar[side]['viewToolbar']
        plotSpotsCoordsAction = viewToolbar['plotSpotsCoordsAction']
        if posData.validRuns():
            plotSpotsCoordsAction.setVisible(True)
            plotSpotsCoordsAction.setEnabled(True)
        else:
            plotSpotsCoordsAction.setVisible(False)
            plotSpotsCoordsAction.setEnabled(False)


    def gui_enableLoadButtons(self):
        try:
            isMultiChannel = len(self.ch_names) > 1
        except AttributeError:
            isMultiChannel = False

        dataLoadedOnlyRight = self.expData['right'] and not self.expData['left']
        dataLoadedOnlyLeft = self.expData['left'] and not self.expData['right']

        # Check if enable right actions
        fileToolbar = self.sideToolbar['right']['fileToolbar']
        openFolderAction = fileToolbar['openFolderAction']
        openFileAction = fileToolbar['openFileAction']
        if dataLoadedOnlyLeft and isMultiChannel:
            openFolderAction.setEnabled(True)
            openFileAction.setEnabled(False)
        else:
            openFolderAction.setEnabled(True)
            openFileAction.setEnabled(True)

        # Check if enable left actions
        fileToolbar = self.sideToolbar['left']['fileToolbar']
        openFolderAction = fileToolbar['openFolderAction']
        openFileAction = fileToolbar['openFileAction']
        if dataLoadedOnlyRight and isMultiChannel:
            openFolderAction.setEnabled(True)
            openFileAction.setEnabled(False)
        elif dataLoadedOnlyRight:
            openFileAction.setEnabled(False)
        else:
            openFolderAction.setEnabled(True)
            openFileAction.setEnabled(True)


    def gui_addTitleLabel(self, colspan=None):
        return
        # self.graphLayout.removeItem(self.titleLabel)
        # if colspan is None:
        #     colspan = 2 if self.expData['left'] and self.expData['right'] else 1
        # self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        # self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=colspan)

    def setMeasurements(self, checked=False):
        pass

    def showComputeDockWidget(self, checked=False):
        if self.showComputeDockButton.isExpand:
            self.computeDockWidget.setVisible(False)
        else:
            self.computeDockWidget.setVisible(True)
            self.computeDockWidget.setEnabled(True)
            # if self.computeDockWidgetMinWidth is not None:
            #     self.resizeDocks([self.computeDockWidget], [w+5], Qt.Horizontal)
            self.addInspectResultsTab(self.lastLoadedSide)

    def getDockWidgetWidth(self):
        if 'computeDockWidgetMinWidth' in self.df_settings.index:
            w = int(self.df_settings.at['computeDockWidgetMinWidth', 'value'])
            self.computeDockWidgetMinWidth = w
            self.resizeTimer.stop()
            self.resizeDocks([self.computeDockWidget], [w+5], Qt.Horizontal)
            self.computeDockWidget.hide()
            self.count = 0
            return

        self.count += 1
        parametersTab = self.computeDockWidget.widget().parametersTab
        horizontalScrollBar = parametersTab.horizontalScrollBar()
        w = self.computeDockWidget.frameSize().width()
        self.resizeDocks([self.computeDockWidget], [w+5], Qt.Horizontal)
        if not horizontalScrollBar.isVisible() or self.count >= 200:
            self.resizeTimer.stop()
            self.computeDockWidgetMinWidth = w+5
            self.computeDockWidget.hide()
            self.count = 0
            val = self.computeDockWidgetMinWidth
            self.df_settings.at['computeDockWidgetMinWidth', 'value'] = val
            self.df_settings.to_csv(self.settings_csv_path)

    def loadingDataAborted(self):
        self.logger.info('Loading data aborted.')
        self.gui_enableLoadButtons()
        if self.axes is not None:
            self.axes['left'].show()
            self.axes['right'].show()
    
    def cleanUpOnError(self):
        txt = 'spotMAX is in error state. Please restart.'
        self.logger.info(txt)
        self.gui_enableLoadButtons()
        if self.axes is not None:
            self.axes['left'].show()
            self.axes['right'].show()

    def setLastUserNormAction(self):
        how = self.df_settings.at['how_normIntensities', 'value']
        for action in self.normalizeQActionGroup.actions():
            if action.text() == how:
                action.setChecked(True)
                break

    def color(self, side, key, desc=''):
        r, g, b, a = self.colorItems[side][key]
        if not desc:
            return (r,g,b,a)

        if desc == 'G1':
            return (int(r*0.8), int(g*0.8), int(b*0.8), 178)
        elif desc == 'divAnnotated':
            return (int(r*0.9), int(g*0.9), int(b*0.9))
        elif desc == 'S_old':
            return (int(r*0.9), int(r*0.9), int(b*0.9))

    def setTextID(self, obj, side):
        posData = self.currentPosData(side)
        frame_i = self.frame_i(side)
        bottomWidgets = self.bottomWidgets[side]
        how = bottomWidgets['howDrawSegmCombobox'].currentText()
        LabelItemID = self.axes[side].LabelItemsIDs[obj.label-1]
        ID = obj.label
        cca_df = posData.cca_df(frame_i)
        if cca_df is None or how.find('cell cycle') == -1:
            txt = f'{ID}'
            if ID in posData.getNewIDs(frame_i):
                color = 'r'
                bold = True
            else:
                color = self.color(side, 'Text on segmented objects')
                bold = False
        else:
            cca_df_ID = cca_df.loc[ID]
            ccs = cca_df_ID['cell_cycle_stage']
            relationship = cca_df_ID['relationship']
            generation_num = cca_df_ID['generation_num']
            generation_num = 'ND' if generation_num==-1 else generation_num
            emerg_frame_i = cca_df_ID['emerg_frame_i']
            is_history_known = cca_df_ID['is_history_known']
            is_bud = relationship == 'bud'
            is_moth = relationship == 'mother'
            emerged_now = emerg_frame_i == frame_i

            # Check if the cell has already annotated division in the future
            # to use orange instead of red
            is_division_annotated = False
            if ccs == 'S' and is_bud and posData.SizeT > 1:
                for i in range(posData.frame_i+1, posData.segmSizeT):
                    cca_df = posData.cca_df(frame_i)
                    if cca_df is None:
                        break
                    _ccs = cca_df.at[ID, 'cell_cycle_stage']
                    if _ccs == 'G1':
                        is_division_annotated = True
                        break

            mothCell_S = (
                ccs == 'S'
                and is_moth
                and not emerged_now
                and not is_division_annotated
            )

            budNotEmergedNow = (
                ccs == 'S'
                and is_bud
                and not emerged_now
                and not is_division_annotated
            )

            budEmergedNow = (
                ccs == 'S'
                and is_bud
                and emerged_now
                and not is_division_annotated
            )

            txt = f'{ccs}-{generation_num}'
            if ccs == 'G1':
                color = self.color(side, 'Text on segmented objects', desc='G1')
                bold = False
            elif mothCell_S:
                color = self.color(side, 'Text on segmented objects')
                bold = False
            elif budNotEmergedNow:
                color = 'r'
                bold = False
            elif budEmergedNow:
                color = 'r'
                bold = True
            elif is_division_annotated:
                color = self.color(
                    side, 'Text on segmented objects', desc='divAnnotated'
                )
                bold = False

            if not is_history_known:
                txt = f'{txt}?'

        LabelItemID.setText(txt, color=color, bold=bold, size=self.fontSize)

        # Center LabelItem at centroid
        y, x = obj.centroid
        w, h = LabelItemID.rect().right(), LabelItemID.rect().bottom()
        LabelItemID.setPos(x-w/2, y-h/2)

    @exception_handler
    def channelNameLUTmenuActionTriggered(self, action):
        if action in self.histItems['left']['actionGroup'].actions():
            side = 'left'
        else:
            side = 'right'

        chName = action.text()
        filename = self.filename(chName, side)
        overlayButton = self.sideToolbar[side]['viewToolbar']['overlayButton']
        if overlayButton.isChecked():
            self.updateHistogramItem(side)

    def filename(self, chName, side, posData=None):
        if posData is None:
            posData = self.currentPosData(side)
        filename = f'{posData.basename}{chName}'
        return filename

    @exception_handler
    def plotSkeletonClicked(self, button, event):
        self.funcDescription = 'plotting skeleton'
        viewToolbar = self.sideToolbar['left']['viewToolbar']
        side = self.side(viewToolbar['plotSkeletonButton'])

        if not button.isChecked():
            self.checkIfDisableOverlayWidgets(side)
            self.axes[side].skelScatterItem.setData([], [])
            return

        posData = self.currentPosData(side)

        if posData.skelCoords and not button.isDoubleClick:
            self.updateImage(side)
            return

        selectChannelWin = dialogs.QDialogListbox(
            'Select data to skeletonize',
            'Select one of the following channels to load and skeletonize',
            posData.allRelFilenames, moreButtonFuncText='Cancel',
            filterItems=('.npy', '.npz'), parent=self
        )
        selectChannelWin.exec_()
        if selectChannelWin.cancel:
            button.setChecked(False)
            self.logger.info(f'Loading data to skeletonize aborted.')
            return

        selectedRelFilenames = selectChannelWin.selectedItemsText
        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title='Loading data...', parent=self,
            pbarDesc=f'Loading "{selectedRelFilenames[0]}"...'
        )
        self.progressWin.mainPbar.setMaximum(len(self.expData[side]))
        self.progressWin.show(self.app)
        self.startRelFilenameDataWorker(
            selectedRelFilenames, side, 'startSkeletonizeWorker'
        )

        self.setEnabledOverlayWidgets(side, True)

    def startSkeletonizeWorker(self, side, initFilename=False):
        self.funcDescription = 'skeletonizing'
        worker = qtworkers.skeletonizeWorker(
            self.expData, side, initFilename=initFilename
        )
        worker.signals.finished.connect(self.workerFinished)
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)

    @exception_handler
    def plotContoursClicked(self, button, event):
        self.funcDescription = 'plotting contour'
        viewToolbar = self.sideToolbar['left']['viewToolbar']
        side = self.side(viewToolbar['plotContourButton'])

        if not button.isChecked():
            self.checkIfDisableOverlayWidgets(side)
            self.axes[side].contourScatterItem.setData([], [])
            return

        posData = self.currentPosData(side)

        if posData.contCoords and not button.isDoubleClick:
            self.updateImage(side)
            return

        selectChannelWin = dialogs.QDialogListbox(
            'Select data to contour',
            'Select one of the following channels to load and display contour',
            posData.allRelFilenames, moreButtonFuncText='Cancel',
            filterItems=('.npy', '.npz'), parent=self
        )
        selectChannelWin.exec_()
        if selectChannelWin.cancel:
            button.setChecked(False)
            self.logger.info(f'Loading data to contour aborted.')
            return

        selectedRelFilenames = selectChannelWin.selectedItemsText
        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title='Loading data...', parent=self,
            pbarDesc=f'Loading "{selectedRelFilenames[0]}"...'
        )
        self.progressWin.mainPbar.setMaximum(len(self.expData[side]))
        self.progressWin.show(self.app)
        self.startRelFilenameDataWorker(
            selectedRelFilenames, side, 'startContoursWorker'
        )

        self.setEnabledOverlayWidgets(side, True)

    def startContoursWorker(self, side, initFilename=False):
        self.funcDescription = 'computing contours'
        worker = qtworkers.findContoursWorker(
            self.expData, side, initFilename=initFilename
        )
        worker.signals.finished.connect(self.workerFinished)
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)

    @exception_handler
    def plotSpotsCoordsClicked(self, button, event):
        self.funcDescription = 'plotting spots coordinates'
        viewToolbar = self.sideToolbar['left']['viewToolbar']
        side = self.side(viewToolbar['plotSpotsCoordsButton'])

        if not button.isChecked():
            self.axes[side].spotsScatterItem.setData([], [])
            return

        posData = self.currentPosData(side)

        if posData.h5_path and not button.isDoubleClick:
            self.plotSpotsCoords(side)
            return

        run_nums = posData.validRuns()
        runsInfo = {}
        for run in run_nums:
            h5_files = posData.h5_files(run)
            if not h5_files:
                continue
            runsInfo[run] = h5_files
        win = dialogs.selectSpotsH5FileDialog(runsInfo, parent=self)
        win.show()
        win.exec_()
        if win.selectedFile is None:
            self.sender().setChecked(False)
            return

        h5_path = os.path.join(posData.spotmaxOutPath, win.selectedFile)
        if h5_path == posData.h5_path:
            self.plotSpotsCoords(side)
            return

        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title='Loading spots data...', parent=self,
            pbarDesc=f'Loading "{win.selectedFile}"...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMinimum(0)
        self.progressWin.mainPbar.setMaximum(len(self.expData[side]))

        worker = qtworkers.load_H5Store_Worker(
            self.expData, win.selectedFile, side
        )
        worker.signals.finished.connect(self.load_H5Store_WorkerFinished)
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)

    def load_H5Store_WorkerFinished(self, side):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.plotSpotsCoords(side)

    def checkIfDisableOverlayWidgets(self, side):
        viewToolbar = self.sideToolbar[side]['viewToolbar']
        overlayButton = viewToolbar['overlayButton']
        plotSkeletonButton = viewToolbar['plotSkeletonButton']
        plotContourButton = viewToolbar['plotContourButton']

        disable = (
            not overlayButton.isChecked()
            and not plotSkeletonButton.isChecked()
            and not plotContourButton.isChecked()
        )
        if disable:
            self.setEnabledOverlayWidgets(side, False)

    @exception_handler
    def changeFontSize(self, action):
        self.fontSize = f'{action.text()}pt'
        self.df_settings.at['fontSize', 'value'] = self.fontSize
        self.df_settings.to_csv(self.settings_csv_path)
        LIs = zip(
            self.axes['left'].LabelItemsIDs, self.axes['right'].LabelItemsIDs
        )
        for ax1_LI, ax2_LI in LIs:
            x1, y1 = ax1_LI.pos().x(), ax1_LI.pos().y()
            if x1>0:
                w, h = ax1_LI.rect().right(), ax1_LI.rect().bottom()
                xc, yc = x1+w/2, y1+h/2
            ax1_LI.setAttr('size', self.fontSize)
            ax1_LI.setText(ax1_LI.text)
            if x1>0:
                w, h = ax1_LI.rect().right(), ax1_LI.rect().bottom()
                ax1_LI.setPos(xc-w/2, yc-h/2)
            x2, y2 = ax2_LI.pos().x(), ax2_LI.pos().y()
            if x2>0:
                w, h = ax2_LI.rect().right(), ax2_LI.rect().bottom()
                xc, yc = x2+w/2, y2+h/2
            ax2_LI.setAttr('size', self.fontSize)
            ax2_LI.setText(ax2_LI.text)
            if x2>0:
                w, h = ax2_LI.rect().right(), ax2_LI.rect().bottom()
                ax2_LI.setPos(xc-w/2, yc-h/2)

    @exception_handler
    def plotSkeleton(self, side):
        posData = self.currentPosData(side)
        if posData.SizeT > 1:
            frame_i = self.frame_i(side)
            skelCoords = posData.skelCoords[frame_i]
        else:
            skelCoords = posData.skelCoords

        if posData.SizeZ > 1:
            zz_skel, yy_skel, xx_skel = skelCoords['all']
            bottomWidgets = self.bottomWidgets[side]

            zProjHow = bottomWidgets[f'zProjComboboxLayer1'].currentText()
            z = bottomWidgets[f'zSliceScrollbarLayer1'].sliderPosition()-1
            if zProjHow == 'same as above':
                zProjHow = bottomWidgets[f'zProjComboboxLayer0'].currentText()
                z = bottomWidgets[f'zSliceScrollbarLayer0'].sliderPosition()-1

            if zProjHow == 'single z-slice':
                yy_skel, xx_skel = skelCoords[z]
        else:
            yy_skel, xx_skel = skelCoords['all']

        self.axes[side].skelScatterItem.setData(xx_skel, yy_skel)

    @exception_handler
    def plotContours(self, side):
        posData = self.currentPosData(side)
        if posData.SizeT > 1:
            frame_i = self.frame_i(side)
            contScatterCoords = posData.contScatterCoords[frame_i]
        else:
            contScatterCoords = posData.contScatterCoords

        if posData.SizeZ > 1:
            contours = contScatterCoords['proj']
            bottomWidgets = self.bottomWidgets[side]

            zProjHow = bottomWidgets[f'zProjComboboxLayer1'].currentText()
            z = bottomWidgets[f'zSliceScrollbarLayer1'].sliderPosition()-1
            if zProjHow == 'same as above':
                zProjHow = bottomWidgets[f'zProjComboboxLayer0'].currentText()
                z = bottomWidgets[f'zSliceScrollbarLayer0'].sliderPosition()-1

            if zProjHow == 'single z-slice':
                contours = contScatterCoords[z]
        else:
            contours = contScatterCoords['proj']

        xx_cont, yy_cont = contours
        xx_cont = xx_cont+0.5
        yy_cont = yy_cont+0.5
        self.axes[side].contourScatterItem.setData(xx_cont, yy_cont)

    def updateSpots(self):
        side = self.sender().parent().side
        self.plotSpotsCoords(side)

    @exception_handler
    def plotSpotsCoords(self, side):
        posData = self.currentPosData(side)
        if posData.hdf_store is None:
            self.logger.warning()

        if posData.SizeT > 1:
            frame_i = bottomWidgets['navigateScrollbar'].sliderPosition()-1
            key = f'frame_{frame_i}'
        else:
            key = 'frame_0'
        try:
            df = posData.hdf_store[key].reset_index()
        except KeyError:
            self.axes[side].spotsScatterItem.setData([], [])
            return

        bottomWidgets = self.bottomWidgets[side]
        zProjHow = bottomWidgets[f'zProjComboboxLayer0'].currentText()
        if zProjHow == 'single z-slice':
            z = bottomWidgets['zSliceScrollbarLayer0'].sliderPosition()-1
            df = df[df['z']==z]

        spotsScatterItem = self.axes[side].spotsScatterItem

        if 'is_spot_inside_ref_ch' in df.columns:
            if self.showOnlyInsideRefAction.isChecked():
                df = df[df['is_spot_inside_ref_ch'] > 0]
            elif self.showOnlyOutsideRefAction.isChecked():
                df = df[df['is_spot_inside_ref_ch'] == 0]

        yy, xx = df['y']+0.5, df['x']+0.5
        data = df['|abs|_spot'].round(4)

        # Add brushes and pens
        brushes = spotsScatterItem.brushes[side]
        pens = spotsScatterItem.pens[side]
        df['brush'] = brushes["Spots inside ref. channel"][0]
        df['pen'] = pens["Spots inside ref. channel"]

        if 'is_spot_inside_ref_ch' in df.columns:
            in_ref_mask = df['is_spot_inside_ref_ch'] > 0
            out_ref_mask = ~in_ref_mask
            in_ref_key = "Spots inside ref. channel"
            out_ref_key = "Spots outside ref. channel"
            df.loc[in_ref_mask, 'brush'] = brushes[in_ref_key][0]
            df.loc[out_ref_mask, 'brush'] = brushes[out_ref_key][0]

            df.loc[in_ref_mask, 'pen'] = pens[in_ref_key]
            df.loc[out_ref_mask, 'pen'] = pens[out_ref_key]

            # xClicked, yClicked = spotsScatterItem.clickedSpot
            # clickedMask = (df['x']==xClicked) & (df['y']==yClicked)
            # clickedKey = "Clicked spot"
            # df.loc[clickedMask, 'brush'] = brushes[clickedKey][0]
            # df.loc[clickedMask, 'pen'] = pens[clickedKey]

        spotsScatterItem.setData(
            xx, yy, data=data,
            size=int(self.df_settings.at['spots_size', 'value']),
            pen=df['pen'].to_list(),
            brush=df['brush'].to_list(),
            hoverBrush=brushes["Spots inside ref. channel"][1]
        )

    @exception_handler
    def spotsClicked(self, scatterItem, spotItems, event):
        side = self.side(self.axes['left'].spotsScatterItem, sender=scatterItem)

        xClicked = int(spotItems[0].pos().x()-0.5)
        yClicked = int(spotItems[0].pos().y()-0.5)
        scatterItem.clickedSpot = (xClicked, yClicked)
        scatterItem.clickedSpotItem = spotItems[0]

        left_click = event.button() == Qt.MouseButton.LeftButton
        right_click = event.button() == Qt.MouseButton.RightButton

        if right_click:
            menu = QMenu(self)
            menu.side = side

            self.showOnlyInsideRefAction.setParent(menu)
            menu.addAction(self.showOnlyInsideRefAction)

            self.showOnlyOutsideRefAction.setParent(menu)
            menu.addAction(self.showOnlyOutsideRefAction)

            self.showAllSpotsAction.setParent(menu)
            menu.addAction(self.showAllSpotsAction)

            menu.addSeparator()
            self.editColorMenu.side = side
            menu.addMenu(self.editColorMenu)
            self.spotStyleAction.setParent(menu)
            menu.addAction(self.spotStyleAction)

            menu.exec(event.screenPos())

    @exception_handler
    def skelClicked(self, scatterItem, spotItems, event):
        side = self.side(self.axes['left'].spotsScatterItem, sender=scatterItem)

        xClicked = int(spotItems[0].pos().x()-0.5)
        yClicked = int(spotItems[0].pos().y()-0.5)
        scatterItem.clickedSpot = (xClicked, yClicked)
        scatterItem.clickedSpotItem = spotItems[0]

        left_click = event.button() == Qt.MouseButton.LeftButton
        right_click = event.button() == Qt.MouseButton.RightButton

        if right_click:
            menu = QMenu(self)
            menu.side = side

            self.editSkelColorAction.setParent(menu)
            menu.addAction(self.editSkelColorAction)

            self.skelStyleAction.setParent(menu)
            menu.addAction(self.skelStyleAction)

            menu.exec(event.screenPos())

    def uncheckQButton(self, button):
        # Manual exclusive where we allow to uncheck all buttons
        for b in self.checkableQButtonsGroup.buttons():
            if b != button:
                b.setChecked(False)

    def undo(self):
        pass

    def redo(self):
        pass

    def newFile(self):
        pass

    def openFileLeft(self, checked=True):
        if self.expData['right']:
            self.lastLoadedSide = 'left'
            self.channelNameUtil.reloadLastSelectedChannel('left')
            self.getChannelName()
        else:
            self.openFile('left')

    def openFileRight(self, checked=True):
        if self.expData['left']:
            self.lastLoadedSide = 'right'
            self.channelNameUtil.reloadLastSelectedChannel('right')
            self.getChannelName()
        else:
            self.openFile('right')

    @exception_handler
    def openFile(self, side, file_path=''):
        self.funcDescription = 'load data'

        if not file_path:
            mostRecentPath = utils.getMostRecentPath()
            file_path = widgets.getOpenImageFileName()
            if not file_path:
                return file_path
        dirpath = os.path.dirname(file_path)
        dirname = os.path.basename(dirpath)
        if dirname != 'Images':
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            spotmax_folder = f'{timestamp}_spotmax'
            selectedPath = os.path.join(dirpath, spotmax_folder, 'Images')
            os.makedirs(selectedPath)
        else:
            selectedPath = dirpath

        filename, ext = os.path.splitext(os.path.basename(file_path))
        if ext == '.tif' or ext == '.npz' or ext == '.h5':
            self.openFolder(side, selectedPath=selectedPath, imageFilePath=file_path)
        else:
            self.logger.info('Copying file to .tif format...')
            img = skimage.io.imread(file_path)
            tif_path = utils.img_to_imageJ(img, selectedPath, filename)
            self.openFolder(
                side, selectedPath=selectedPath, imageFilePath=tif_path
            )

    def openFolderRight(self, checked=True):
        if self.expData['left']:
            self.lastLoadedSide = 'right'
            self.channelNameUtil.reloadLastSelectedChannel('right')
            self.getChannelName()
        else:
            self.openFolder('right')

    def openFolderLeft(self, checked=True):
        if self.expData['right']:
            self.lastLoadedSide = 'left'
            self.channelNameUtil.reloadLastSelectedChannel('left')
            self.getChannelName()
        else:
            self.openFolder('left')

    @exception_handler
    def openFolder(self, side, selectedPath='', imageFilePath=''):
        """Main function used to load data into GUI. Multi-step function:
            1. openFolder
            2. getChannelName
            3. getDataPaths
            4. loadSelectedData (on a different thread)
            5. updateImage

        Parameters
        ----------
        side : string
            Either 'left' or 'right'
        selectedPath : string or None
            Path selected by the user either directly, through openFile,
            or drag and drop image file.
        imageFilePath : string
            Path of the image file that was either drag and dropped or opened
            from File --> Open image/video file (openFileActionLeft).

        Returns
        -------
            None
        """
        if self.threadCount > 0:
            self.worker.exit = True
            self.waitCond.wakeAll()

        if self.isFirstOpen:
            self.gui_createGraphicsPlots()
            self.isFirstOpen = False

        self.funcDescription = 'load data'
        self.lastLoadedSide = side

        self.gui_init(side=side)

        fileToolbar = self.sideToolbar[side]['fileToolbar']
        openFolderAction = fileToolbar['openFolderAction']

        if self.slideshowWin is not None:
            self.slideshowWin.close()

        if not selectedPath:
            mostRecentPath = utils.getMostRecentPath()
            title = (
                'Select experiment folder containing Position_n folders '
                'or specific Position_n folder'
            )
            selectedPath = QFileDialog.getExistingDirectory(
                self, title, mostRecentPath
            )

        if not selectedPath:
            self.loadingDataAborted()
            self.logger.info('Open folder action aborted by the user.')
            return

        self.setWindowTitle(f'spotMAX - GUI - "{selectedPath}"')

        self.selectedPath = selectedPath
        self.addToRecentPaths(selectedPath)

        selectedPath_basename = os.path.basename(selectedPath)
        is_pos_path = selectedPath_basename.find('Position_')!=-1
        is_images_path = selectedPath_basename == 'Images'

        self.channelNameUtil = io.channelName(
            which_channel=side, QtParent=self
        )
        user_ch_name = None
        if imageFilePath:
            images_paths = [pathlib.Path(selectedPath)]
            self.images_paths = images_paths
            filenames = utils.listdir(selectedPath)
            ch_names, basenameNotFound = (
                self.channelNameUtil.getChannels(filenames, selectedPath)
            )
            filename = os.path.basename(imageFilePath)
            self.ch_names = ch_names
            user_ch_name = [
                chName for chName in ch_names if filename.find(chName)!=-1
            ][0]
            self.user_ch_name = user_ch_name
            # Skip getChannelName step since the user loaded a specific file
            self.getDataPaths()
            return

        elif is_images_path:
            images_paths = [pathlib.Path(selectedPath)]

        elif is_pos_path:
            pos_path = pathlib.Path(selectedPath)
            images_paths = [pos_path / 'Images']
        else:
            pos_foldernames = [
                f for f in utils.listdir(selectedPath)
                if f.find('Position_')!=-1
                and os.path.isdir(os.path.join(selectedPath, f))
            ]
            if len(pos_foldernames) == 1:
                pos = pos_foldernames[0]
                images_paths = [pathlib.Path(selectedPath) / pos / 'Images']
            else:
                self.progressWin = acdc_apps.QDialogWorkerProgress(
                    title='Path scanner progress', parent=self,
                    pbarDesc='Scanning experiment folder...'
                )
                self.progressWin.show(self.app)
                selectedPath = selectedPath
                self.startPathScannerWorker(selectedPath)
                # Loading process will continue after startPathScannerWorker
                # has emitted finished signal (pathScannerWorkerFinished)
                return

        self.images_paths = images_paths
        self.getChannelName()

    def startPathScannerWorker(self, selectedPath):
        self.funcDescription = 'scanning experiment paths'
        worker = qtworkers.pathScannerWorker(selectedPath)
        worker.signals.finished.connect(self.pathScannerWorkerFinished)
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)

    def pathScannerWorkerFinished(self, pathScanner):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        pathScanner.input(app=self.app)
        if pathScanner.selectedPaths:
            self.images_paths = pathScanner.selectedPaths
            self.getChannelName()
        else:
            self.loadingDataAborted()
            self.logger.info('Loading data process aborted by the user.')
            # self.titleLabel.setText('Loading data process aborted by the user.')

    def workerFinished(self, side):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.updateImage(side)

    def workerFinishedNextStep(self, side, nextStep, selectedRelFilename):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        if not nextStep:
            self.updateImage(side)
        elif nextStep == 'startSkeletonizeWorker':
            for posData in self.expData[side]:
                posData.skeletonizedRelativeFilename = selectedRelFilename

            self.progressWin = acdc_apps.QDialogWorkerProgress(
                title='Skeletonizing...', parent=self,
                pbarDesc=f'Skeletonizing {selectedRelFilename}...'
            )
            self.progressWin.mainPbar.setMaximum(len(self.expData[side]))
            self.progressWin.show(self.app)
            self.startSkeletonizeWorker(side)
        elif nextStep == 'startContoursWorker':
            for posData in self.expData[side]:
                posData.contouredRelativeFilename = selectedRelFilename

            self.progressWin = acdc_apps.QDialogWorkerProgress(
                title='Computing contours...', parent=self,
                pbarDesc=f'Computing contours {selectedRelFilename}...'
            )
            self.progressWin.mainPbar.setMaximum(len(self.expData[side]))
            self.progressWin.show(self.app)

            self.startContoursWorker(side)

    def workerProgress(self, text, loggerLevel):
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)

    @exception_handler
    def workerCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
        raise error

    def getChannelName(self):
        self.funcDescription = 'retrieving channel names'
        channelNameUtil = self.channelNameUtil
        images_path = self.images_paths[0]
        abortedByUser = False
        user_ch_name = None
        filenames = utils.listdir(images_path)
        ch_names, basenameNotFound = (
            channelNameUtil.getChannels(filenames, images_path)
        )
        if not ch_names:
            self.logger.exception(
                f'No valid channels found in the folder {images_path}.'
            )
            # self.titleLabel.setText(
            #     f'No valid channels found in the folder {images_path}.'
            # )
            self.loadingDataAborted()
            return

        if len(ch_names) > 1:
            channelNameUtil.askSelectChannel(self, ch_names)
            if channelNameUtil.was_aborted:
                self.logger.info('Channel selection aborted by the User')
                # self.titleLabel.setText('Channel selection aborted by the User')
                self.loadingDataAborted()
                return
        else:
            channelNameUtil.channel_name = ch_names[0]
        self.ch_names = ch_names
        channelNameUtil.setUserChannelName()
        user_ch_name = channelNameUtil.user_ch_name

        if user_ch_name is None:
            self.loadingDataAborted()
            self.criticalNoTifFound(images_path)
            return

        self.user_ch_name = user_ch_name
        self.getDataPaths()

    def getDataPaths(self):
        channelDataFilePaths = []
        for images_path in self.images_paths:
            h5_aligned_path = ''
            h5_path = ''
            npz_aligned_path = ''
            tif_path = ''
            files = utils.listdir(images_path)
            for file in (files):
                channelDataPath = images_path / file
                if file.endswith(f'{self.user_ch_name}_aligned.h5'):
                    h5_aligned_path = channelDataPath
                elif file.endswith(f'{self.user_ch_name}.h5'):
                    h5_path = channelDataPath
                elif file.endswith(f'{self.user_ch_name}_aligned.npz'):
                    npz_aligned_path = channelDataPath
                elif file.endswith(f'{self.user_ch_name}.tif'):
                    tif_path = channelDataPath

            if h5_aligned_path:
                self.logger.info(
                    f'Using .h5 aligned file ({h5_aligned_path})...'
                )
                channelDataFilePaths.append(h5_aligned_path)
            elif h5_path:
                self.logger.info(f'Using .h5 file ({h5_path})...')
                channelDataFilePaths.append(h5_path)
            elif npz_aligned_path:
                self.logger.info(
                    f'Using .npz aligned file ({npz_aligned_path})...'
                )
                channelDataFilePaths.append(npz_aligned_path)
            elif tif_path:
                self.logger.info(f'Using .tif file ({tif_path})...')
                channelDataFilePaths.append(tif_path)
            else:
                self.loadingDataAborted()
                self.criticalImgPathNotFound(images_path)
                return

        self.channelDataFilePaths = channelDataFilePaths

        self.initGlobalAttr()
        self.splitExpPaths()
        self.loadSelectedData()

    def clearAxes(self, side):
        self.axes[side].clear()

    def initGlobalAttr(self):
        self.segm_cmap = pg.colormap.getFromMatplotlib('viridis')
        self.itemsImageCleared = False
        self.expData[self.lastLoadedSide] = []

        self.gui_connectGraphicsEvents(self.lastLoadedSide, disconnect=True)

        bottomWidgets = self.bottomWidgets[self.lastLoadedSide]
        navigateScrollbar = bottomWidgets['navigateScrollbar']
        navigateScrollbar.setSliderPosition(0)

    def splitExpPaths(self):
        uniqueExpPaths = set(
            [f.parents[2] for f in self.channelDataFilePaths]
        )
        selectedPath = pathlib.Path(self.selectedPath)
        expPaths = dict()
        for expPath in uniqueExpPaths:
            try:
                expName = str(expPath.relative_to(selectedPath))
            except ValueError:
                expName = '.'
            expPaths[expName] = {
                'path': expPath,
                'channelDataPaths': []
            }
        for filePath in self.channelDataFilePaths:
            expPath = filePath.parents[2]
            try:
                expName = str(expPath.relative_to(selectedPath))
            except ValueError:
                expName = '.'
            expPaths[expName]['channelDataPaths'].append(filePath)

        self.logger.info('Sorting expPaths...')

        if len(expPaths) > 1:
            expPaths = utils.natSortExpPaths(expPaths)
        self.expPaths = expPaths

        self.updateExpNameCombobox()

    def updateExpNameCombobox(self):
        self.expNameCombobox.currentTextChanged.disconnect()
        items = list(self.expPaths.keys())
        currentText = self.expNameCombobox.currentText()
        self.expNameCombobox.clear()
        self.expNameCombobox.addItems(items)
        if len(items)==1:
            self.expNameLabelAction.setVisible(False)
            self.expNameAction.setVisible(False)
        else:
            self.expNameLabelAction.setVisible(True)
            self.expNameAction.setVisible(True)
            self.expNameAction.setEnabled(True)
            self.expNameLabelAction.setEnabled(True)
            if currentText in items:
                self.expNameCombobox.setCurrentText(currentText)
            else:
                self.warnLoadedDifferentPaths()
        self.expNameCombobox.currentTextChanged.connect(self.loadSelectedData)

    def warnLoadedDifferentPaths(self):
        pass

    def addPosCombobox(self, expName):
        try:
            self.topFileToolBar.removeAction(self.posNameAction)
            self.initiallyHiddenItems['top'].remove(self.posNameAction)
        except AttributeError:
            pass
        self.topFileToolBar.addWidget(QLabel('   Position: '))
        self.posNameCombobox = QComboBox(self.topFileToolBar)
        self.posNameCombobox.SizeAdjustPolicy(QComboBox.AdjustToContents)
        posFoldernames = [
            str(path.parents[1].name)
            for path in self.expPaths[expName]['channelDataPaths']
        ]
        self.posNameCombobox.addItems(posFoldernames)
        self.posNameCombobox.adjustSize()
        self.posNameAction = self.topFileToolBar.addWidget(self.posNameCombobox)
        self.initiallyHiddenItems['top'].append(self.posNameAction)

    def readMetadata(self, expName, posIdx):
        expPath = self.expPaths[expName]['path']
        channelDataPaths = self.expPaths[expName]['channelDataPaths']

        self.logger.info('Reading meatadata...')
        # Load first pos to read metadata
        channelDataPath = channelDataPaths[0]
        posDataRef = io.loadData(
            channelDataPath, self.user_ch_name, QParent=self
        )
        posDataRef.getBasenameAndChNames()
        posDataRef.buildPaths()
        posDataRef.loadChannelData()
        posDataRef.loadOtherFiles(load_metadata=True)
        proceed = posDataRef.askInputMetadata(
            self.numPos,
            ask_SizeT=True,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            save=True
        )
        if not proceed:
            return '', None

        return channelDataPath, posDataRef

    def warnMemoryNotSufficient(self, total_ram, available_ram, required_ram):
        total_ram = utils._bytes_to_MB(total_ram)
        available_ram = utils._bytes_to_MB(available_ram)
        required_ram = utils._bytes_to_MB(required_ram)

        msg = acdc_widgets.myMessageBox(self)
        txt = html_func.paragraph(
            'The total amount of data that you requested to load is about '
            f'<b>{required_ram} MB</b> but there are only '
            f'<b>{available_ram} MB</b> available.<br><br>'
            'For optimal operation, we recommend loading <b>maximum 40%</b> '
            'of the available memory. To do so, try to close open apps to '
            'free up some memory or consider using .h5 files and load only '
            'a portion of the file. Another option is to crop the images '
            'using the data prep module.<br><br>'
            'If you choose to continue, the <b>system might freeze</b> '
            'or your OS could simply kill the process.<br><br>'
            'What do you want to do?'
        )
        continueButton, abortButton = msg.warning(
            self, 'Memory not sufficient', txt,
            buttonsTexts=('Continue', 'Abort')
        )
        if msg.clickedButton == continueButton:
            return True
        else:
            return False

    def checkMemoryRequirements(self, required_ram):
        memory = psutil.virtual_memory()
        total_ram = memory.total
        available_ram = memory.available
        if required_ram/available_ram > 0.4:
            proceed = self.warnMemoryNotSufficient(
                total_ram, available_ram, required_ram
            )
            return proceed
        else:
            return True

    def loadSelectedData(self):
        # self.titleLabel.setText('Loading data...', color='w')
        expName = self.expNameCombobox.currentText()
        expPath = self.expPaths[expName]['path']
        self.expData[self.lastLoadedSide] = []

        selectedExpName = expName
        self.numPos = len(self.expPaths[expName]['channelDataPaths'])

        channelDataPaths = self.expPaths[expName]['channelDataPaths']
        required_ram = utils.getMemoryFootprint(channelDataPaths)
        proceed = self.checkMemoryRequirements(required_ram)
        if not proceed:
            self.logger.info(
                'Loading process cancelled by the user because '
                'memory not sufficient.'
            )
            self.loadingDataAborted()
            return

        fisrtChannelDataPath, posDataRef = self.readMetadata(expName, 0)
        if not fisrtChannelDataPath:
            self.logger.info('Loading process cancelled by the user.')
            self.loadingDataAborted()
            return

        required_ram = posDataRef.checkH5memoryFootprint()*posDataRef.loadSizeS
        if required_ram > 0:
            proceed = self.checkMemoryRequirements(required_ram)
            if not proceed:
                self.logger.info(
                    'Loading process cancelled by the user because '
                    'memory not sufficient.'
                )
                self.loadingDataAborted()
                return

        if posDataRef.SizeT > 1:
            self.isTimeLapse = True
            self.addPosCombobox(expName)
            selectedPos = self.posNameCombobox.currentText()
        else:
            self.isTimeLapse = False
            selectedPos = None
        
        existingSegmEndNames = self.getExistingSegmEndNames(selectedExpName)
        self.selectedSegmEndame = ''
        if len(existingSegmEndNames) > 1:
            win = acdc_apps.QDialogMultiSegmNpz(
                existingSegmEndNames, self.expPaths[selectedExpName]['path'], 
                parent=self, addNewFileButton=False, basename=None
            )
            win.exec_()
            if win.cancel:
                self.logger.info(
                    'Loading process cancelled by the user.'
                )
                self.loadingDataAborted()
            self.selectedSegmEndame = win.selectedItemText
        elif len(existingSegmEndNames) == 1:
            self.selectedSegmEndame = list(existingSegmEndNames)[0]

        self.progressWin = acdc_apps.QDialogWorkerProgress(
            title='Loading data...', parent=self,
            pbarDesc=f'Loading "{fisrtChannelDataPath}"...'
        )
        self.progressWin.show(self.app)
        self.posDataRef = posDataRef

        func = partial(self.startLoadDataWorker, selectedPos, selectedExpName)
        QTimer.singleShot(50, func)
    
    def getExistingSegmEndNames(self, selectedExpName):
        expInfo = self.expPaths[selectedExpName]
        channelDataPaths = expInfo['channelDataPaths']
        existingSegmEndNames = set()
        for channelDataPath in channelDataPaths:
            imagesPath = os.path.dirname(channelDataPath)
            basename, _ = io.get_basename_and_ch_names(imagesPath)
            segmFiles = io.get_segm_files(imagesPath)
            endnames = io.get_existing_segm_endnames(basename, segmFiles)
            existingSegmEndNames.update(endnames)
        return existingSegmEndNames

    def startLoadDataWorker(self, selectedPos, selectedExpName):
        self.funcDescription = 'loading data'
        worker = qtworkers.loadDataWorker(self, selectedPos, selectedExpName)
        worker.signals.finished.connect(self.loadDataWorkerFinished)
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)

    @exception_handler
    def loadingDataFinished(self):
        # self.clearAxes(self.lastLoadedSide)
        self.gui_removeAllItems(self.lastLoadedSide)
        self.gui_createGraphicsItems()
        self.histItems[self.lastLoadedSide]['actionGroup'] = QActionGroup(self)

        self.gui_addChannelNameQActionGroup(
            self.lastLoadedSide, self.user_ch_name
        )
        self.gui_addGradientLevels(self.lastLoadedSide)
        self.gui_addPlotItems(self.lastLoadedSide)
        self.gui_connectGraphicsEvents(self.lastLoadedSide)
        self.gui_connectBottomWidgetsSignals(self.lastLoadedSide)
        self.gui_connectSideToolbarsSignals(self.lastLoadedSide)
        self.areActionsConnected[self.lastLoadedSide] = True

        self.gui_setItemsDataShape(self.lastLoadedSide)
        self.gui_setVisibleItems(self.lastLoadedSide)
        self.gui_enableLoadButtons()
        self.updateImage(self.lastLoadedSide)
        self.updateSegmVisuals(self.lastLoadedSide)

        self.drawAnnotCombobox_to_options(self.lastLoadedSide)

        self.setParams()

        # self.titleLabel.setText('', color='w')
        self.dataLoaded[self.lastLoadedSide] = True

        areBothPlotsVisible = (
            self.dataLoaded['left'] and self.dataLoaded['right']
        )
        if areBothPlotsVisible:
            self.axes['left'].setYLink(self.axes['right'])
            self.axes['left'].setXLink(self.axes['right'])

        QTimer.singleShot(300, self.axes['left'].autoRange)

    def setParams(self):
        params = self.computeDockWidget.widget().parametersQGBox.params

        section = 'File paths and channels'
        if self.lastLoadedSide == 'left':
            anchor = 'spotsEndName'
        else:
            anchor = 'refChEndName'

        posData = self.currentPosData(self.lastLoadedSide)
        params[section][anchor]['widget'].setText(posData.channelDataPath)

        params['METADATA']['SizeT']['widget'].setValue(posData.SizeT)
        params['METADATA']['SizeZ']['widget'].setValue(posData.SizeZ)

    def loadDataWorkerFinished(self):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.loadingDataFinished()

    def z(self, side, layer=0):
        bottomWidgets = self.bottomWidgets[side]
        z = bottomWidgets[f'zSliceScrollbarLayer{layer}'].sliderPosition()-1
        return z

    def zStack_to_2D(self, side, zStack, layer=0):
        bottomWidgets = self.bottomWidgets[side]
        posData = self.currentPosData(side)
        zProjHow = bottomWidgets[f'zProjComboboxLayer{layer}'].currentText()
        z = self.z(side, layer=layer)
        if zProjHow == 'same as above':
            zProjHow = bottomWidgets[f'zProjComboboxLayer0'].currentText()
            z = self.z(side, layer=0) - posData.z0_window

        if zProjHow == 'single z-slice':
            img = zStack[z]
        elif zProjHow == 'max z-projection':
            img = zStack.max(axis=0)
        elif zProjHow == 'mean z-projection':
            img = zStack.mean(axis=0)
        elif zProjHow == 'median z-proj.':
            img = np.median(zStack, axis=0)
        return img

    def frame_i(self, side):
        posData = self.currentPosData(side)
        if posData.SizeT == 1:
            return 0

        bottomWidgets = self.bottomWidgets[side]
        frame_i = bottomWidgets['navigateScrollbar'].sliderPosition()-1
        return frame_i

    def overlayClicked(self, button, event):
        viewToolbar = self.sideToolbar['left']['viewToolbar']
        side = self.side(viewToolbar['overlayButton'])

        if not button.isChecked():
            self.checkIfDisableOverlayWidgets(side)
            self.updateImage(side)
            return

        posData = self.currentPosData(side)

        # Check if user already loaded data to merge and did not dblclick
        if posData.loadedMergeRelativeFilenames and not button.isDoubleClick:
            self.setEnabledOverlayWidgets(side, True)
            self.updateImage(side)
            return

        selectChannelWin = dialogs.QDialogListbox(
            'Select channel(s) to merge',
            'Select one or more channels to merge',
            posData.allRelFilenames, moreButtonFuncText='Browse...',
            parent=self
        )
        selectChannelWin.exec_()
        if selectChannelWin.cancel:
            button.setChecked(False)
            self.logger.info(f'Loading merge channel data aborted.')
            return

        if button.isDoubleClick:
            for posData in self.expData[side]:
                posData.loadedMergeRelativeFilenames = {}

        selectedRelFilenames = selectChannelWin.selectedItemsText
        shouldLoad = any([
            relFilename not in posData.loadedRelativeFilenamesData
            for relFilename in selectedRelFilenames
        ])

        if shouldLoad:
            self.progressWin = acdc_apps.QDialogWorkerProgress(
                title='Loading data...', parent=self,
                pbarDesc=f'Loading "{selectedRelFilenames[0]}"...'
            )
            self.progressWin.mainPbar.setMaximum(len(self.expData[side]))
            self.progressWin.show(self.app)
            self.startRelFilenameDataWorker(selectedRelFilenames, side, '')
        else:
            for i, relFilename in enumerate(selectedRelFilenames):
                posData.loadedMergeRelativeFilenames[relFilename] = i+1
            self.updateImage(side)

        self.setEnabledOverlayWidgets(side, True)

    def startRelFilenameDataWorker(self, selectedRelFilenames, side, nextStep):
        self.funcDescription = 'loading overlay data'
        worker = qtworkers.load_relFilenameData_Worker(
            self.expData, selectedRelFilenames, side, nextStep
        )
        worker.signals.sigLoadedData.connect(self.addWorkerData)
        worker.signals.finishedNextStep.connect(self.workerFinishedNextStep)
        worker.signals.progress.connect(self.workerProgress)
        worker.signals.initProgressBar.connect(self.workerInitProgressbar)
        worker.signals.progressBar.connect(self.workerUpdateProgressbar)
        worker.signals.critical.connect(self.workerCritical)
        self.threadPool.start(worker)

    def addWorkerData(self, posData, mergeData, relFilename, isNextStep):
        posData.loadedRelativeFilenamesData[relFilename] = mergeData
        if not isNextStep:
            posData.loadedMergeRelativeFilenames[relFilename] = None

    def loadMergeDataWorkerFinished(self, side):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.updateImage(side)

    def setEnabledOverlayWidgets(self, side, enabled):
        posData = self.currentPosData(side)
        bottomWidgets = self.bottomWidgets[side]

        if posData.SizeZ > 1:
            bottomWidgets['zProjComboboxLayer1'].setEnabled(enabled)
            how = bottomWidgets['zProjComboboxLayer1'].currentText()
            if how.find('max') != -1 or how == 'same as above':
                bottomWidgets['zSliceScrollbarLayer1'].setDisabled(True)

        bottomWidgets['alphaScrollbar'].setEnabled(enabled)

    def layerImage(self, side, relFilename=''):
        posData = self.currentPosData(side)

        if not relFilename:
            data = posData.chData
            frame_i = self.frame_i(side) - posData.t0_window
            layer = 0
        else:
            data = posData.loadedRelativeFilenamesData[relFilename]
            frame_i = self.frame_i(side)
            layer = 1

        if posData.SizeT > 1 and posData.SizeZ > 1:
            # 4D data
            zStack = data[frame_i]
            img = self.zStack_to_2D(side, zStack, layer=layer)
        elif posData.SizeT == 1 and posData.SizeZ > 1:
            # 3D z-stacks data
            img = self.zStack_to_2D(side, data, layer=layer)
        elif posData.SizeT > 1 and posData.SizeZ == 1:
            # 2D timelapse data
            img = data[frame_i]
        else:
            img = data
        return img

    def currentImage(self, side):
        layer0 = self.layerImage(side)
        img = layer0
        viewToolbar = self.sideToolbar[side]['viewToolbar']
        overlayButton = viewToolbar['overlayButton']

        if overlayButton.isChecked():
            posData = self.currentPosData(side)
            relFilenames = list(posData.loadedMergeRelativeFilenames)
            layer1_filename = relFilenames[0]
            layer1 = self.layerImage(side, relFilename=layer1_filename)
            color = self.colorItems[side]['Overlay image']
            alphaScrollbar = self.bottomWidgets[side]['alphaScrollbar']
            alpha = alphaScrollbar.value()/alphaScrollbar.maximum()
            merge = utils.mergeChannels(img, layer1, color, alpha)

            # for filename in relFilenames[1:]:
            #     layer_n = self.layerImage(side, relFilename=filename)
            #     merge = utils.mergeChannels(merge, layer_n, color, alpha)

            img = merge

        return img

    def clearSegmVisuals(self, side):
        allItems = zip(
            self.axes[side].ContoursCurves,
            self.axes[side].LabelItemsIDs,
            self.axes[side].BudMothLines
        )
        for idx, items_ID in enumerate(allItems):
            contCurve, labelItem, BudMothLine = items_ID
            if contCurve.getData()[0] is not None:
                contCurve.setData([], [])
            if BudMothLine.getData()[0] is not None:
                BudMothLine.setData([], [])
            labelItem.setText('')

    def clearOverlayVisuals(self, side):
        if self.axes[side].spotsScatterItem.getData()[0] is not None:
            self.axes[side].spotsScatterItem.setData([], [])
        if self.axes[side].contourScatterItem.getData()[0] is not None:
            self.axes[side].contourScatterItem.setData([], [])
        if self.axes[side].skelScatterItem.getData()[0] is not None:
            self.axes[side].skelScatterItem.setData([], [])

    @exception_handler
    def updateImage(self, side):
        img = self.currentImage(side)
        self.imgItems[side].setImage(img)

        viewToolbar = self.sideToolbar[side]['viewToolbar']
        plotSpotsCoordsButton = viewToolbar['plotSpotsCoordsButton']
        if plotSpotsCoordsButton.isChecked():
            self.plotSpotsCoords(side)

        viewToolbar = self.sideToolbar[side]['viewToolbar']
        plotSkeletonButton = viewToolbar['plotSkeletonButton']
        if plotSkeletonButton.isChecked():
            self.plotSkeleton(side)

        viewToolbar = self.sideToolbar[side]['viewToolbar']
        plotContourButton = viewToolbar['plotContourButton']
        if plotContourButton.isChecked():
            self.plotContours(side)

    def updateSegmVisuals(self, side):
        self.clearSegmVisuals(side)

        posData = self.currentPosData(side)
        if posData.rp is not None:
            for obj in posData.regionprops(self.frame_i(side)):
                self.drawID_and_Contour(obj, side)

    def addInspectResultsTab(self, side):
        posData = self.currentPosData(side)
        tabControl = self.computeDockWidget.widget()
        tabControl.addInspectResultsTab(posData)

    def updateHistogramItem(self, side):
        """
        Function called every time the image changes (updateImage).
        Perform the following:

        1. Set the gradient slider tick positions
        2. Set the region max and min levels
        3. Plot the histogram
        """
        histItem = self.histItems[side]['hist']
        imageItem = self.imgItems[side]
        try:
            histItem.sigLookupTableChanged.disconnect()
            connect = True
        except TypeError:
            connect = False
            pass

        posData = self.currentPosData(side)
        action = self.checkedLUTaction(side)
        chName = action.text()
        filename = self.filename(chName, side)
        min, max = posData.gradientLevels[filename]

        minTick = histItem.gradient.getTick(0)
        maxTick = histItem.gradient.getTick(1)
        histItem.gradient.setTickValue(minTick, min)
        histItem.gradient.setTickValue(maxTick, max)
        histItem.setLevels(
            min=imageItem.image.min(),
            max=imageItem.image.max()
        )
        h = imageItem.getHistogram()
        histItem.plot.setData(*h)
        if connect:
            histItem.sigLookupTableChanged.connect(self.histLUTchanged)

    def histLUTchanged(self, histItem):
        side = self.side(self.histItems['left']['hist'], sender=histItem)
        action = self.checkedLUTaction(side)
        chName = action.text()
        filename = self.filename(chName, side)
        self.gui_addGradientLevels(side, filename=filename)

    def checkedLUTaction(self, side):
        actions = self.histItems[side]['actionGroup'].actions()
        action = [action for action in actions if action.isChecked()][0]
        return action

    def drawID_and_Contour(self, obj, side, drawContours=True):
        posData = self.currentPosData(side)
        frame_i = self.frame_i(side)
        bottomWidgets = self.bottomWidgets[side]
        how = bottomWidgets['howDrawSegmCombobox'].currentText()
        IDs_and_cont = how == 'Draw IDs and contours'
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'

        # Draw LabelItems for IDs on ax2
        y, x = obj.centroid
        idx = obj.label-1


        # Draw LabelItems for IDs on ax1 if requested
        if IDs_and_cont or onlyIDs or only_ccaInfo or ccaInfo_and_cont:
            # Draw LabelItems for IDs on ax2
            self.setTextID(obj, side)

        # Draw line connecting mother and buds
        drawLines = only_ccaInfo or ccaInfo_and_cont or onlyMothBudLines
        if drawLines and posData.cca_df(frame_i) is not None:
            ID = obj.label
            BudMothLine = self.axes[side].BudMothLines[ID-1]
            cca_df_ID = posData.cca_df(frame_i).loc[ID]
            ccs_ID = cca_df_ID['cell_cycle_stage']
            relationship = cca_df_ID['relationship']
            if ccs_ID == 'S' and relationship=='bud':
                emerg_frame_i = cca_df_ID['emerg_frame_i']
                if emerg_frame_i == frame_i:
                    pen = self.NewBudMoth_Pen
                else:
                    pen = self.OldBudMoth_Pen
                relative_ID = cca_df_ID['relative_ID']
                if relative_ID in posData.IDs(frame_i):
                    relative_rp_idx = posData.IDs(frame_i).index(relative_ID)
                    relative_ID_obj = posData.regionprops(frame_i)[relative_rp_idx]
                    y1, x1 = obj.centroid
                    y2, x2 = relative_ID_obj.centroid
                    BudMothLine.setData([x1, x2], [y1, y2], pen=pen)
            else:
                BudMothLine.setData([], [])

        if not drawContours:
            return

        # Draw contours on ax1 if requested
        if IDs_and_cont or onlyCont or ccaInfo_and_cont:
            ID = obj.label
            cont = utils.objContours(obj)
            curveID = self.axes[side].ContoursCurves[idx]
            newIDs = posData.getNewIDs(frame_i)
            pen = self.newIDs_cpen if ID in newIDs else self.oldIDs_cpen
            curveID.setData(cont[:,0], cont[:,1], pen=pen)

    def side(self, leftWidget, sender=None):
        if sender is None:
            sender = self.sender()

        if sender == leftWidget:
            side = 'left'
        else:
            side = 'right'
        return side

    def updateLinkedCombobox(self, side, labelKey, text):
        linkedSide = 'left' if side=='right' else 'right'
        bottomWidgets = self.bottomWidgets[linkedSide]
        combobox = bottomWidgets[labelKey]
        if combobox.checkBox.isChecked():
            return

        combobox.setCurrentText(text)
    
    def resizeComputeDockWidget(self):
        paramsGroupbox = self.computeDockWidget.widget().parametersQGBox
        paramsScrollArea = self.computeDockWidget.widget().parametersTab
        verticalScrollbar = paramsScrollArea.verticalScrollBar()
        groupboxWidth = paramsGroupbox.size().width()
        scrollbarWidth = verticalScrollbar.size().width()
        minWidth = groupboxWidth + scrollbarWidth + 13
        self.resizeDocks([self.computeDockWidget], [minWidth], Qt.Horizontal)
        self.showParamsDockButton.click()
    
    def _waitLazyLoaderWorkerClosed(self):
        if self.worker.isFinished:
            self.waitLazyLoaderWorkerClosedLoop.stop()

    def closeEvent(self, event):
        # Close the inifinte loop of the thread
        self.worker.exit = True
        self.waitCond.wakeAll()
        self.waitReadH5cond.wakeAll()

        progressWin = acdc_apps.QDialogWorkerProgress(
            title='Closing lazy loader', parent=self,
            pbarDesc='Closing lazy loader...'
        )
        progressWin.show(self.app)
        progressWin.mainPbar.setMaximum(0)
        self.waitLazyLoaderWorkerClosedLoop = acdc_qutils.QWhileLoop(
            self._waitLazyLoaderWorkerClosed, period=250
        )
        self.waitLazyLoaderWorkerClosedLoop.exec_()
        progressWin.workerFinished = True
        progressWin.close()

        self.saveWindowGeometry()

        with open(colorItems_path, mode='w') as file:
            json.dump(self.colorItems, file, indent=2)

        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.ccaTableWin is not None:
            self.ccaTableWin.close()

        askSave = (
            self.saveAction.isEnabled()
            and (self.expData['left'] is None or self.expData['right'] is None)
        )
        if askSave:
            msg = acdc_widgets.myMessageBox()
            saveButton, noButton, cancelButton = msg.question(
                self, 'Save?', 'Do you want to save?',
                buttonsTexts=('Yes', 'No', 'Cancel')
            )
            if msg.clickedButton == saveButton:
                self.saveData()
                event.accept()
            elif msg.clickedButton == noButton:
                event.accept()
            else:
                event.ignore()

        # Close .h5 files
        for expData in self.expData.values():
            for posData in expData:
                if not posData.h5_path:
                    continue
                if posData.hdf_store is None:
                    continue
                self.logger.info(f'Closing opened h5 files...')
                posData.hdf_store.close()
                try:
                    posData.h5_dset.close()
                except AttributeError:
                    pass

        self.logger.info('Closing logger...')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

        if self.buttonToRestore is not None and event.isAccepted():
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()
        
        self.sigClosed.emit(self)

    def showInFileManager(self):
        posData = self.currentPosData('left')
        if not posData:
            posData = self.currentPosData('right')
        systems = {
            'nt': os.startfile,
            'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
            'os2': lambda foldername: os.system('open "%s"' % foldername)
             }

        systems.get(os.name, os.startfile)(posData.images_path)

    def addToRecentPaths(self, selectedPath):
        if not os.path.exists(selectedPath):
            return
        recentPaths_path = os.path.join(
            settings_path, 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if selectedPath in recentPaths:
                pop_idx = recentPaths.index(selectedPath)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, selectedPath)
            openedOn.insert(0, datetime.datetime.now())
            # Keep max 30 recent paths
            if len(recentPaths) > 30:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        else:
            recentPaths = [selectedPath]
            openedOn = [datetime.datetime.now()]
        df = pd.DataFrame({
            'path': recentPaths,
            'opened_last_on': pd.Series(openedOn, dtype='datetime64[ns]')
        })
        df.index.name = 'index'
        df.to_csv(recentPaths_path)

    def storeDefaultAndCustomColors(self):
        c = self.overlayButton.palette().button().color().name()
        self.defaultToolBarButtonColor = c
        self.doublePressKeyButtonColor = '#fa693b'

    def saveWindowGeometry(self):
        settings = QSettings('schmollerlab', 'spotmax_gui')
        settings.setValue("geometry", self.saveGeometry())

    def readSettings(self):
        settings = QSettings('schmollerlab', 'spotmax_gui')
        if settings.value('geometry') is not None:
            self.restoreGeometry(settings.value("geometry"))

    def show(self):
        # self.storeDefaultAndCustomColors()
        QMainWindow.show(self)

        h = self.bottomWidgets['left']['zProjComboboxLayer0'].size().height()
        self.bottomWidgets['left']['navigateScrollbar'].setFixedHeight(h)
        self.bottomWidgets['left']['zSliceScrollbarLayer0'].setFixedHeight(h)
        self.bottomWidgets['left']['zSliceScrollbarLayer1'].setFixedHeight(h)
        self.bottomWidgets['left']['alphaScrollbar'].setFixedHeight(h)

        self.bottomWidgets['right']['navigateScrollbar'].setFixedHeight(h)
        self.bottomWidgets['right']['zSliceScrollbarLayer0'].setFixedHeight(h)
        self.bottomWidgets['right']['zSliceScrollbarLayer1'].setFixedHeight(h)
        self.bottomWidgets['right']['alphaScrollbar'].setFixedHeight(h)

        viewToolbar = self.sideToolbar['left']['viewToolbar']
        overlayButton = viewToolbar['overlayButton']
        h = overlayButton.size().height()

        viewToolbar['colorButton'].setFixedHeight(h)

        w = self.showComputeDockButton.width()
        h = self.showComputeDockButton.height()
        self.showComputeDockButton.setFixedWidth((int(w/2)))
        self.showComputeDockButton.setFixedHeight(h*2)

        self.gui_hideInitItems()

        self.readSettings()

        # Dynamically resize dock widget until horizontalScrollBar is not visible
        self.computeDockWidgetMinWidth = None
        self.count = 0
        self.resizeTimer = QTimer()
        self.resizeTimer.timeout.connect(self.getDockWidgetWidth)
        self.resizeTimer.start(1)
