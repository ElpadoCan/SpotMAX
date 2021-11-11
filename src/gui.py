# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPyTop HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""spotMAX GUI"""
print('Importing modules...')
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
from pprint import pprint
from functools import partial, wraps
from tqdm import tqdm
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
    Qt, QFile, QTextStream, QSize, QRect, QRectF, QEventLoop, QTimer, QEvent,
    QThreadPool, QRunnable, pyqtSignal, QObject
)
from PyQt5.QtGui import (
    QIcon, QKeySequence, QCursor, QKeyEvent, QFont, QPixmap,
    QColor, QPainter, QPen
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
    QColorDialog
)

import pyqtgraph as pg

# Private modules
import load, dialogs, utils, widgets, qtworkers

# NOTE: Enable icons
import qrc_resources

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

print('Initializing...')

# Interpret image data as row-major instead of col-major
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
np.random.seed(1568)

def qt_debug_trace():
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    import pdb; pdb.set_trace()

def exception_handler(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                func(self, *args)
            else:
                func(self, *args, **kwargs)
        except Exception as e:
            self.logger.exception(e)
            msg = QMessageBox(self)
            msg.setWindowTitle('Critical error')
            msg.setIcon(msg.Critical)
            err_msg = (f"""
            <p style="font-size:10pt">
                Error in function <b>{func.__name__}</b> when trying to
                {self.funcDescription}.<br><br>
                More details below or in the terminal/console.<br><br>
                Note that the error details are also saved in the file
                spotMAX/stdout.log
            </p>
            """)
            msg.setText(err_msg)
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
            self.titleLabel.setText(
                'Error occured. See spotMAX/stdout.log file for more details',
                color='r'
            )
            self.loadingDataAborted()
            return inner_function
    return inner_function


class spotMAX_Win(QMainWindow):
    """Main Window."""

    def __init__(self, app, parent=None, buttonToRestore=None, mainWin=None):
        """Initializer."""
        super().__init__(parent)

        self.setupLogger()
        self.loadLastSessionSettings()

        self.buttonToRestore = buttonToRestore
        self.mainWin = mainWin

        self.app = app
        self.num_screens = len(app.screens())

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

        self.setWindowIcon(QIcon(":icon.svg"))
        self.setAcceptDrops(True)

        self.rightClickButtons = []
        self.leftClickButtons = []
        self.initiallyHiddenItems = []

        self.lastSelectedColorItem = None

        self.countKeyPress = 0
        self.xHoverImg, self.yHoverImg = None, None

        self.areActionsConnected = {'left': False, 'right': False}
        self.dataLoaded = {'left': False, 'right': False}

        # List of load.loadData class for each position of the experiment
        self.expData = {'left': [], 'right': []}

        self.bottomWidgets = {'left': {}, 'right': {}}
        self.sideToolbar = {'left': {}, 'right': {}}
        self.axes = {'left': None, 'right': None}
        self.imgItems = {'left': None, 'right': None}
        self.histItems = {'left': {}, 'right': {}}

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

        self.gui_connectActions()
        self.gui_createStatusBar()

        self.graphLayout = pg.GraphicsLayoutWidget()
        self.gui_createGraphicsPlots()

        self.gui_createBottomLayout()
        self.gui_addBottomLeftWidgets()
        self.gui_addBottomRightWidgets()
        self.gui_addGraphicsItems()

        self.gui_createThreadPool()

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QGridLayout()
        # mainLayout.addWidget(self.computeDockWidget, 0, 0)
        mainLayout.addWidget(self.graphLayout, 0, 0)
        mainLayout.addLayout(self.bottomLayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.gui_init(first_call=True)

    def setupLogger(self):
        logger = logging.getLogger('spotMAX')
        logger.setLevel(logging.INFO)

        src_path = os.path.dirname(os.path.abspath(__file__))
        logs_path = os.path.join(src_path, 'logs')
        if not os.path.exists(logs_path):
            os.mkdir(logs_path)
        else:
            # Keep 20 most recent logs
            ls = os.listdir(logs_path)
            if len(ls)>20:
                ls = [os.path.join(logs_path, f) for f in ls]
                ls.sort(key=lambda x: os.path.getmtime(x))
                for file in ls[:-20]:
                    os.remove(file)

        date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'{date_time}_stdout.log'
        log_path = os.path.join(logs_path, log_filename)

        output_file_handler = logging.FileHandler(log_path, mode='w')
        stdout_handler = logging.StreamHandler(sys.stdout)

        # Format your logs (optional)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s:\n'
            '------------------------\n'
            '%(message)s\n'
            '------------------------\n',
            datefmt='%d-%m-%Y, %H:%M:%S')
        output_file_handler.setFormatter(formatter)

        logger.addHandler(output_file_handler)
        logger.addHandler(stdout_handler)

        self.logger = logger

    def loadLastSessionSettings(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        settings_path = os.path.join(src_path, 'settings')
        self.colorItems_path = os.path.join(settings_path, 'colorItems.json')
        csv_path = os.path.join(settings_path, 'gui_settings.csv')
        self.settings_csv_path = csv_path
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
            if 'spots_transparency' not in self.df_settings.index:
                self.df_settings.at['spots_transparency', 'value'] = '0.3'
            if 'spots_pen_width' not in self.df_settings.index:
                self.df_settings.at['spots_pen_width', 'value'] = '2.0'
            if 'spots_size' not in self.df_settings.index:
                self.df_settings.at['spots_size', 'value'] = '3'
        else:
            idx = [
                'is_bw_inverted',
                'fontSize',
                'overlayColor',
                'how_normIntensities',
                'spots_transparency',
                'spots_pen_width',
                'spots_size'
            ]
            values = ['No', '10pt', '255-255-0', 'raw', '0.3', '2.0', '3']
            self.df_settings = pd.DataFrame({'setting': idx,
                                             'value': values}
                                           ).set_index('setting')

    def dragEnterEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(file_path):
            selectedPath = file_path
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

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_P:
            for curve in self.axes['left'].ContoursCurves:
                print(curve.getData())
            # self.logger.info('ciao')
        elif ev.key() == Qt.Key_Left:
            if not self.dataLoaded['left'] and not self.dataLoaded['right']:
                event.ignore()
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
                event.ignore()
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
        self.initiallyHiddenItems.append(self.showInFileManagerAction)
        fileToolBar.addAction(self.undoAction)
        fileToolBar.addAction(self.redoAction)
        fileToolBar.addWidget(QLabel('   Experiment name:'))
        self.expNameCombobox = QComboBox(fileToolBar)
        self.expNameCombobox.SizeAdjustPolicy(QComboBox.AdjustToContents)
        self.expNameAction = fileToolBar.addWidget(self.expNameCombobox)
        self.initiallyHiddenItems.append(self.expNameAction)

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

        self.initiallyHiddenItems.append(viewerToolbar)
        self.topViewerToolbar = viewerToolbar

        modeToolBar = QToolBar("Mode", self)
        self.addToolBar(modeToolBar)
        self.modeToolBar = modeToolBar
        self.modeComboBox = QComboBox()
        self.modeComboBox.addItems(['Viewer', 'Compute'])
        modeComboBoxLabel = QLabel('    Mode: ')
        modeComboBoxLabel.setBuddy(self.modeComboBox)
        modeToolBar.addWidget(modeComboBoxLabel)
        modeToolBar.addWidget(self.modeComboBox)
        self.initiallyHiddenItems.append(modeToolBar)

        self.topModeToolbar = modeToolBar

    def gui_createSideFileToolbar(self, side):
        self.sideToolbar[side]['fileToolbar'] = {}
        toolbarActions = self.sideToolbar[side]['fileToolbar']

        openFolderAction = QAction(
            QIcon(":folder-open.svg"),
            "&Load data from folder onto the left image...", self
        )
        toolbarActions['openFolderAction'] = openFolderAction

        openFileAction = QAction(
            QIcon(":image.svg"),
            "&Load single image/video file onto the left image...", self
        )
        toolbarActions['openFileAction'] = openFileAction

        reloadAction = QAction(
            QIcon(":reload.svg"), "Reload left image file", self
        )
        toolbarActions['reloadAction'] = reloadAction

        openFolderAction = QAction(
            QIcon(":folder-open.svg"),
            "Load data from folder onto the right image...", self
        )
        toolbarActions['openFolderAction'] = openFolderAction

        openFileAction = QAction(
            QIcon(":image.svg"),
            "Load single image/video file onto the right image...", self
        )
        toolbarActions['openFileAction'] = openFileAction

        reloadAction= QAction(
            QIcon(":reload.svg"), "Reload right image file", self
        )
        toolbarActions['reloadAction'] = reloadAction

        fileToolBar = QToolBar(f"{side} image file toolbar", self)
        fileToolBar.setMovable(False)

        QtSide = Qt.LeftToolBarArea if side == 'left' else Qt.RightToolBarArea
        self.addToolBar(QtSide, fileToolBar)

        fileToolBar.addAction(openFolderAction)
        fileToolBar.addAction(openFileAction)
        fileToolBar.addAction(reloadAction)
        self.initiallyHiddenItems.append(reloadAction)

    def gui_createSideViewerToolbar(self, side):
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

        try:
            # padding is new from pyqtgraph 0.12.3
            colorButton = pg.ColorButton(
                self, color=(0,255,255), padding=3
            )
        except Exception as e:
            colorButton = pg.ColorButton(
                self, color=(0,255,255)
            )
        colorButton.clicked.disconnect()
        colorButton.setToolTip('Edit colors')
        colorButton.setFlat(True)
        viewToolbarDict['colorButton'] = colorButton

        plotSpotsCoordsButton = widgets.DblClickQToolButton(self)
        plotSpotsCoordsButton.setIcon(QIcon(":plotSpots.svg"))
        plotSpotsCoordsButton.setCheckable(True)
        plotSpotsCoordsButton.setToolTip('Plot spots coordinates')
        viewToolbarDict['plotSpotsCoordsButton'] = plotSpotsCoordsButton

        plotSkeletonButton = widgets.DblClickQToolButton(self)
        plotSkeletonButton.setIcon(QIcon(":skeletonize.svg"))
        plotSkeletonButton.setCheckable(True)
        plotSkeletonButton.setToolTip('Plot skeleton of selected data')
        viewToolbarDict['plotSkeletonButton'] = plotSkeletonButton

        plotContourButton = widgets.DblClickQToolButton(self)
        plotContourButton.setIcon(QIcon(":contour.svg"))
        plotContourButton.setCheckable(True)
        plotContourButton.setToolTip('Plot contour of selected data')
        viewToolbarDict['plotContourButton'] = plotContourButton

        viewToolbar = QToolBar("Left image controls", self)
        viewToolbar.setMovable(False)
        QtSide = Qt.LeftToolBarArea if side == 'left' else Qt.RightToolBarArea
        self.addToolBar(Qt.LeftToolBarArea, viewToolbar)

        overlayAction = viewToolbar.addWidget(overlayButton)
        viewToolbarDict['overlayAction'] = overlayAction
        self.initiallyHiddenItems.append(overlayAction)

        colorAction = viewToolbar.addWidget(colorButton)
        viewToolbarDict['colorAction'] = colorAction
        self.initiallyHiddenItems.append(colorAction)

        plotSpotsCoordsAction = viewToolbar.addWidget(plotSpotsCoordsButton)
        viewToolbarDict['plotSpotsCoordsAction'] = plotSpotsCoordsAction
        self.initiallyHiddenItems.append(plotSpotsCoordsAction)

        plotSkeletonAction = viewToolbar.addWidget(plotSkeletonButton)
        viewToolbarDict['plotSkeletonAction'] = plotSkeletonAction
        self.initiallyHiddenItems.append(plotSkeletonAction)

        plotContourAction = viewToolbar.addWidget(plotContourButton)
        viewToolbarDict['plotContourAction'] = plotContourAction
        self.initiallyHiddenItems.append(plotContourAction)

    def gui_createLeftToolBars(self):
        self.gui_createSideFileToolbar('left')
        self.gui_createSideViewerToolbar('left')

    def gui_createRightToolBars(self):
        self.gui_createSideFileToolbar('right')
        self.gui_createSideViewerToolbar('right')

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

    def gui_createComputeDockWidget(self):
        self.computeDockWidget = QDockWidget('spotMAX analysis inputs', self)
        # self.initiallyHiddenItems.append(self.computeDockWidget)
        frame = dialogs.analysisInputsQFrame(self.computeDockWidget)

        self.computeDockWidget.setWidget(frame)
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
        self.editClickedSpotColor = QAction(
            'Clicked spot', self.editColorMenu)
        self.editInsideRefColor = QAction(
            'Spots inside ref. channel', self.editColorMenu
        )
        self.editOutsideRefColor = QAction(
            'Spots outside ref. channel', self.editColorMenu
        )
        self.editAllSpotsColor = QAction(
            'All spots', self.editColorMenu
        )
        self.editColorMenu.addAction(self.editClickedSpotColor)
        self.editColorMenu.addAction(self.editInsideRefColor)
        self.editColorMenu.addAction(self.editOutsideRefColor)
        self.editColorMenu.addAction(self.editAllSpotsColor)

        self.spotStyleAction = QAction(
            'Style...', self
        )

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

        openFolderActionRight = fileToolbarRight['openFolderAction']
        openFolderActionRight.triggered.connect(self.openFileRight)

        self.saveAction.triggered.connect(self.saveData)
        self.showInFileManagerAction.triggered.connect(self.showInFileManager)
        self.exitAction.triggered.connect(self.close)

        self.undoAction.triggered.connect(self.undo)
        self.redoAction.triggered.connect(self.redo)

        # Connect Open Recent to dynamically populate it
        self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)

        self.checkableQButtonsGroup.buttonClicked.connect(self.uncheckQButton)

        self.modeComboBox.currentTextChanged.connect(self.modeChanged)

        self.showOnlyInsideRefAction.triggered.connect(self.updateSpots)
        self.showOnlyOutsideRefAction.triggered.connect(self.updateSpots)
        self.showAllSpotsAction.triggered.connect(self.updateSpots)

        self.editClickedSpotColor.triggered.connect(self.selectSpotsColor)
        self.editInsideRefColor.triggered.connect(self.selectSpotsColor)
        self.editOutsideRefColor.triggered.connect(self.selectSpotsColor)
        self.editAllSpotsColor.triggered.connect(self.selectSpotsColor)

        self.spotStyleAction.triggered.connect(self.selectSpotStyle)

    def gui_connectBottomWidgetsSignals(self, side):
        if self.areActionsConnected[side]:
            return

        bottomWidgets = self.bottomWidgets[side]

        bottomWidgets['howDrawSegmCombobox'].currentTextChanged.connect(
            self.howDrawSegmCombobox_cb
        )

        bottomWidgets['navigateScrollbar'].actionTriggered.connect(
            self.navigateScrollbarTriggered
        )
        bottomWidgets['navigateScrollbar'].sliderReleased.connect(
            self.navigateScrollbarReleased
        )

        bottomWidgets['zSliceScrollbarLayer0'].actionTriggered.connect(
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
        bottomWidgets['zSliceScrollbarLayer1'].actionTriggered.connect(
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

        viewToolbar = self.sideToolbar[side]['viewToolbar']

        colorButton = viewToolbar['colorButton']
        colorButton.clicked.connect(self.gui_selectColor)
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
        side = self.side(viewToolbar['colorButton'])

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
        colorButton.side = side
        colorButton.key = win.selectedItems[0].text()
        colorButton.setColor(self.colorItems[side][colorButton.key])
        colorButton.selectColor()

    def gui_setColor(self, colorButton):
        side = colorButton.side
        key = colorButton.key
        self.colorItems[side][key] = list(colorButton.color(mode='byte'))
        self.gui_createSpotsBrushesPens()
        self.updateImage(side)

    def gui_createFonts(self):
        self.font10pt = QFont()
        self.font10pt.setPointSize(10)

    def gui_setListItems(self):
        self.drawSegmComboboxItems = [
            'Draw only contours',
            'Draw IDs and contours',
            'Draw only cell cycle info',
            'Draw cell cycle info and contours',
            'Draw only mother-bud lines',
            'Draw only IDs',
            'Draw nothing'
        ]

        self.zProjItems = [
            'single z-slice',
            'max z-projection',
            'mean z-projection',
            'median z-proj.'
        ]

        with open(self.colorItems_path) as file:
            self.colorItems = json.load(file)

    def gui_createBottomLayout(self):
        self.bottomLayout = QHBoxLayout()
        self.bottomLayout.addSpacing(100)

    def gui_createBottomWidgets(self, side):
        frame = QGroupBox()
        layout = QGridLayout()
        bottomWidgets = {}

        row = 0
        col = 1
        howDrawSegmCombobox = QComboBox()
        howDrawSegmCombobox.addItems(self.drawSegmComboboxItems)
        # Always adjust combobox width to largest item
        howDrawSegmCombobox.setSizeAdjustPolicy(
            howDrawSegmCombobox.AdjustToContents
        )
        layout.addWidget(
            howDrawSegmCombobox, row, col, alignment=Qt.AlignCenter
        )
        bottomWidgets['howDrawSegmCombobox'] = howDrawSegmCombobox

        row = 1
        col = 0
        navigateScrollbar_label = QLabel('frame n.  ')
        navigateScrollbar_label.setFont(self.font10pt)
        layout.addWidget(
            navigateScrollbar_label, row, col, alignment=Qt.AlignRight
        )
        bottomWidgets['navigateScrollbar_label'] = navigateScrollbar_label

        row = 1
        col = 1
        navigateScrollbar = QScrollBar(Qt.Horizontal)
        navigateScrollbar.setMinimum(1)
        layout.addWidget(navigateScrollbar, row, col)
        bottomWidgets['navigateScrollbar'] = navigateScrollbar

        row = 2
        col = 0
        zSliceSbL0_label = QLabel('First layer z-slice  ')
        zSliceSbL0_label.setFont(self.font10pt)
        layout.addWidget(
            zSliceSbL0_label, row, col, alignment=Qt.AlignRight
        )
        bottomWidgets['zSliceSbL0_label'] = zSliceSbL0_label

        row = 2
        col = 1
        zSliceScrollbarLayer0 = QScrollBar(Qt.Horizontal)
        zSliceScrollbarLayer0.setMinimum(1)
        zSliceScrollbarLayer0.layer = 0
        zSliceScrollbarLayer0.side = side
        layout.addWidget(zSliceScrollbarLayer0, row, col)
        bottomWidgets['zSliceScrollbarLayer0'] = zSliceScrollbarLayer0

        row = 2
        col = 2
        zProjComboboxLayer0 = QComboBox()
        zProjComboboxLayer0.layer = 0
        zProjComboboxLayer0.side = side
        zProjComboboxLayer0.addItems(self.zProjItems)
        layout.addWidget(zProjComboboxLayer0, row, col, alignment=Qt.AlignLeft)
        bottomWidgets['zProjComboboxLayer0'] = zProjComboboxLayer0

        row = 3
        col = 0
        zSliceSbL1_label = QLabel('Second layer z-slice  ')
        zSliceSbL1_label.setFont(self.font10pt)
        layout.addWidget(
            zSliceSbL1_label, row, col, alignment=Qt.AlignRight
        )
        bottomWidgets['zSliceSbL1_label'] = zSliceSbL1_label

        row = 3
        col = 1
        zSliceScrollbarLayer1 = QScrollBar(Qt.Horizontal)
        zSliceScrollbarLayer1.setMinimum(1)
        zSliceScrollbarLayer1.layer = 1
        zSliceScrollbarLayer1.side = side
        layout.addWidget(zSliceScrollbarLayer1, row, col)
        bottomWidgets['zSliceScrollbarLayer1'] = zSliceScrollbarLayer1

        row = 3
        col = 2
        zProjComboboxLayer1 = QComboBox()
        zProjComboboxLayer1.layer = 1
        zProjComboboxLayer1.side = side
        zProjComboboxLayer1.addItems(self.zProjItems)
        zProjComboboxLayer1.addItems(['same as above'])
        zProjComboboxLayer1.setCurrentIndex(1)
        layout.addWidget(zProjComboboxLayer1, row, col, alignment=Qt.AlignLeft)
        bottomWidgets['zProjComboboxLayer1'] = zProjComboboxLayer1

        row = 4
        col = 0
        alphaScrollbar_label = QLabel('Overlay alpha  ')
        alphaScrollbar_label.setFont(self.font10pt)
        layout.addWidget(
            alphaScrollbar_label, row, col, alignment=Qt.AlignRight
        )
        bottomWidgets['alphaScrollbar_label'] = alphaScrollbar_label

        row = 4
        col = 1
        alphaScrollbar = QScrollBar(Qt.Horizontal)
        alphaScrollbar.setMinimum(0)
        alphaScrollbar.setMaximum(40)
        alphaScrollbar.setValue(20)
        alphaScrollbar.setToolTip(
            'Control the alpha value of the overlay.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only fluorescent data visible'
        )
        layout.addWidget(alphaScrollbar, row, col)
        bottomWidgets['alphaScrollbar'] = alphaScrollbar

        layout.setColumnStretch(0,1)
        layout.setColumnStretch(1,3)
        layout.setColumnStretch(2,1)
        frame.setLayout(layout)

        # sp = frame.sizePolicy()
        # sp.setRetainSizeWhenHidden(True)
        # frame.setSizePolicy(sp)

        return frame, bottomWidgets

    def gui_addBottomLeftWidgets(self):
        frame, bottomLeftWidgets = self.gui_createBottomWidgets('left')
        self.bottomWidgets['left'] = bottomLeftWidgets
        self.bottomWidgets['left']['frame'] = frame
        self.bottomLayout.addWidget(frame)
        self.initiallyHiddenItems.append(frame)

    def gui_addBottomRightWidgets(self):
        frame, bottomRightWidgets = self.gui_createBottomWidgets('right')
        self.bottomWidgets['right'] = bottomRightWidgets
        self.bottomWidgets['right']['frame'] = frame
        self.bottomLayout.addWidget(frame)
        self.bottomLayout.addSpacing(100)
        self.initiallyHiddenItems.append(frame)

    def gui_createThreadPool(self):
        self.threadPool = QThreadPool.globalInstance()

    def gui_createGraphicsPlots(self):
        # Left plot
        ax1 = pg.PlotItem()
        ax1.invertY(True)
        ax1.setAspectLocked(True)
        ax1.hideAxis('bottom')
        ax1.hideAxis('left')
        self.graphLayout.addItem(ax1, row=1, col=1)
        self.axes['left'] = ax1

        # Left image instructions
        html = ("""
        <p style="font-size:26pt; color:rgb(70,70,70); text-align:center">
            Drag and drop experiment folder
            or single image/video file here to start.<br><br>
            Alternatively, you can use the buttons
            on the left-side toolbar.
        </p>
        <p style="font-size:20pt; color:rgb(70,70,70); text-align:center">
            Typically, you load on the left image the spots channel.<br><br>
            Note that it is not mandatory to load two signals. You can load
            only one (either spots or reference channel).
        </p>
        """)
        text = pg.TextItem(anchor=(0.5,0))
        text.setHtml(html)
        text.textItem.adjustSize()
        self.axes['left'].addItem(text)

        # Right plot
        ax2 = pg.PlotItem()
        ax2.setAspectLocked(True)
        ax2.invertY(True)
        ax2.hideAxis('bottom')
        ax2.hideAxis('left')
        self.graphLayout.addItem(ax2, row=1, col=2)
        self.axes['right'] = ax2

        # Right image instructions
        html = html.replace('left-side', 'right-side')
        html = html.replace('left image', 'right image')
        html = html.replace('spots channel', 'reference channel')
        text = pg.TextItem(anchor=(0.5,0))
        text.setHtml(html)
        text.textItem.adjustSize()
        self.axes['right'].addItem(text)

        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1)

    def gui_addGraphicsItems(self):
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
        histLeft = widgets.HistogramLUTItem()

        self.graphLayout.addItem(histLeft, row=1, col=0)
        histLeft.hide()
        self.histItems['left']['hist'] = histLeft

        # Right image histogram
        try:
            histRight = widgets.HistogramLUTItem(gradientPosition='left')
        except TypeError:
            histRight = widgets.HistogramLUTItem()

        self.graphLayout.addItem(histRight, row=1, col=3)
        histRight.hide()
        self.histItems['right']['hist'] = histRight

    def gui_addImageItem(self, side):
        # Blank image
        self.blank = np.zeros((256,256), np.uint8)

        imgItem = widgets.ImageItem(self.blank)
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
            symbol='o', size=3, pxMode=False,
            pen=self.spotsPens[side]['All spots'],
            brush=self.spotsBrushes[side]['All spots'][0],
            hoverable=True, hoverBrush=self.spotsBrushes[side]['All spots'][1]
        )
        self.axes[side].spotsScatterItem = spotsScatterItem
        self.axes[side].addItem(spotsScatterItem)

    def gui_addSkeletonScatterItem(self, side):
        skeletonScatterItem = widgets.ScatterPlotItem(
            symbol='o', size=3, pxMode=False,
            brush=pg.mkBrush(self.colorItems[side]['Skeleton']),
            pen=pg.mkPen(self.colorItems[side]['Skeleton'], width=2)
        )
        self.axes[side].skeletonScatterItem = skeletonScatterItem
        self.axes[side].addItem(skeletonScatterItem)

    def gui_addContourPlotItem(self, side):
        contourPlotItem = pg.PlotCurveItem(
            brush=pg.mkBrush(self.colorItems[side]['Contour']),
            pen=pg.mkPen(self.colorItems[side]['Contour'], width=2)
        )
        self.axes[side].contourPlotItem = contourPlotItem
        self.axes[side].addItem(contourPlotItem)


    def gui_addPlotItems(self, side):
        self.gui_addImageItem(side)
        self.gui_addSegmVisuals()
        self.gui_addRulerItems(side)
        self.gui_addSpotsScatterItem(side)
        self.gui_addSkeletonScatterItem(side)
        self.gui_addContourPlotItem(side)

    def gui_removeAllItems(self, side):
        self.axes[side].clear()

    def gui_createSpotsBrushesPens(self):
        # Spots pens and brushes
        alpha = float(self.df_settings.at['spots_transparency', 'value'])
        penWidth = float(self.df_settings.at['spots_pen_width', 'value'])
        self.spotsPens = {'left': {}, 'right': {}}
        self.spotsBrushes = {'left': {}, 'right': {}}
        for side, colors in self.colorItems.items():
            for key, color in colors.items():
                if key.lower().find('spot')==-1:
                    continue
                self.spotsPens[side][key] = pg.mkPen(color, width=penWidth)
                brushColor = color.copy()
                brushColor[-1] = int(color[-1]*alpha)
                self.spotsBrushes[side][key] = (
                    pg.mkBrush(brushColor), pg.mkBrush(color)
                )

    def gui_createGraphicsItems(self):
        self.gui_createSpotsBrushesPens()

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


    def gui_addSegmVisuals(self):
        # Temporary line item connecting bud to new mother
        self.BudMothTempLine = pg.PlotDataItem(pen=self.NewBudMoth_Pen)
        self.axes['left'].addItem(self.BudMothTempLine)


        # Create enough PlotDataItems and LabelItems to draw contours and IDs.
        maxID = 0
        for posData in self.expData['left']:
            if posData.segm_data is not None and posData.segm_data.max()>maxID:
                maxID = posData.segm_data.max()

        numItems = maxID+10
        self.axes['left'].ContoursCurves = []
        self.axes['right'].ContoursCurves = []

        self.axes['left'].BudMothLines = []
        self.axes['right'].BudMothLines = []

        self.axes['left'].LabelItemsIDs = []
        self.axes['right'].LabelItemsIDs = []
        for i in range(numItems):
            # Contours on ax1
            ContCurve = pg.PlotDataItem()
            self.axes['left'].ContoursCurves.append(ContCurve)
            self.axes['left'].addItem(ContCurve)

            # Bud mother line on ax1
            BudMothLine = pg.PlotDataItem()
            self.axes['left'].BudMothLines.append(BudMothLine)
            self.axes['left'].addItem(BudMothLine)

            # LabelItems on ax1
            ax1_IDlabel = pg.LabelItem()
            self.axes['left'].LabelItemsIDs.append(ax1_IDlabel)
            self.axes['left'].addItem(ax1_IDlabel)

            # LabelItems on ax2
            ax2_IDlabel = pg.LabelItem()
            self.axes['right'].LabelItemsIDs.append(ax2_IDlabel)
            self.axes['right'].addItem(ax2_IDlabel)

            # Contours on ax2
            ContCurve = pg.PlotDataItem()
            self.axes['right'].ContoursCurves.append(ContCurve)
            self.axes['right'].addItem(ContCurve)

            # Bud mother line on ax1
            BudMothLine = pg.PlotDataItem()
            self.axes['right'].BudMothLines.append(BudMothLine)
            self.axes['right'].addItem(BudMothLine)

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
        histItem.sigContextMenu.connect(self.gui_raiseContextMenuLUT)
        histItem.sigLookupTableChanged.connect(self.histLUTchanged)
        self.axes[side].spotsScatterItem.sigClicked.connect(self.spotsClicked)


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
            self.rulerButton.isChecked() and self.rulerHoverON
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
                self.wcLabel.setText(
                    f'(x={x:.2f}, y={y:.2f}, value={val:.0f}, max={layer0.max():.3f})'
                )
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

    def gui_hideInitItems(self):
        for item in self.initiallyHiddenItems:
            item.setVisible(False)
            item.setEnabled(False)

    def gui_setItemsDataShape(self, side):
        if self.dataLoaded[side]:
            return

        bottomWidgets = self.bottomWidgets[side]
        bottomWidgets['frame'].setVisible(True)
        bottomWidgets['frame'].setDisabled(False)
        posData = self.currentPosData(side)
        numPos = len(self.expData[side])
        overlayButton = self.sideToolbar[side]['viewToolbar']['overlayButton']

        if posData.SizeZ > 1:
            midZslice = int(posData.SizeZ/2)
            bottomWidgets['zSliceSbL0_label'].setStyleSheet('color: black')
            bottomWidgets['zSliceScrollbarLayer0'].setDisabled(False)
            bottomWidgets['zSliceScrollbarLayer0'].setMaximum(posData.SizeZ)
            bottomWidgets['zSliceScrollbarLayer1'].setMaximum(posData.SizeZ)
            bottomWidgets['zSliceScrollbarLayer0'].setSliderPosition(midZslice)
            zSliceSbL0_label = bottomWidgets['zSliceSbL0_label']
            zSliceSbL0_label.setText(
                f'First layer z-slice {midZslice}/{posData.SizeZ}'
            )
            bottomWidgets['zProjComboboxLayer0'].setDisabled(False)
        else:
            bottomWidgets['zSliceSbL0_label'].setStyleSheet('color: gray')
            bottomWidgets['zSliceScrollbarLayer0'].setDisabled(True)
            bottomWidgets['zProjComboboxLayer0'].setDisabled(True)
        if posData.SizeT > 1:
            numFrames = len(posData.chData)
            bottomWidgets['navigateScrollbar'].setSliderPosition(0)
            bottomWidgets['navigateScrollbar_label'].setText(
                f'frame n. {1}/{numFrames}  '
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
            bottomWidgets['zSliceSbL1_label'].setStyleSheet('color: gray')
            bottomWidgets['zSliceScrollbarLayer1'].setDisabled(True)
            bottomWidgets['zProjComboboxLayer1'].setDisabled(True)
            bottomWidgets['alphaScrollbar_label'].setStyleSheet('color: gray')
            bottomWidgets['alphaScrollbar'].setDisabled(True)
        elif overlayButton.isChecked() and posData.SizeZ > 1:
            bottomWidgets['zSliceSbL1_label'].setStyleSheet('color: black')
            bottomWidgets['zSliceScrollbarLayer1'].setDisabled(False)
            bottomWidgets['zSliceScrollbarLayer1'].setSliderPosition(midZslice)
            bottomWidgets['zSliceScrollbarLayer1'].setMaximum(posData.SizeZ)
            zSliceSbL0_label = bottomWidgets['zSliceSbL1_label']
            zSliceSbL0_label.setText(
                f'First layer z-slice {midZslice}/{posData.SizeZ}'
            )
            bottomWidgets['zProjComboboxLayer1'].setDisabled(False)
            bottomWidgets['alphaScrollbar_label'].setStyleSheet('color: black')
            bottomWidgets['alphaScrollbar'].setDisabled(False)

        if overlayButton.isChecked():
            bottomWidgets['alphaScrollbar_label'].setStyleSheet('color: black')
            bottomWidgets['alphaScrollbar'].setDisabled(False)

    def gui_setVisibleItems(self, side):
        if self.dataLoaded[side]:
            return

        self.topModeToolbar.setVisible(True)
        self.topModeToolbar.setDisabled(False)

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
        for side in ['left', 'right']:
            fileToolbar = self.sideToolbar[side]['fileToolbar']
            openFolderAction = fileToolbar['openFolderAction']
            openFileAction = fileToolbar['openFileAction']
            openFolderAction.setEnabled(True)
            openFileAction.setEnabled(True)

    def gui_addTitleLabel(self, colspan=None):
        self.graphLayout.removeItem(self.titleLabel)
        if colspan is None:
            colspan = 2 if self.expData['left'] and self.expData['right'] else 1
        self.titleLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=colspan)

    def modeChanged(self, mode):
        if mode == 'Viewer':
            self.computeDockWidget.setVisible(False)
        elif mode == 'Compute':
            self.computeDockWidget.setVisible(True)
            self.computeDockWidget.setEnabled(True)

    def loadingDataAborted(self):
        self.gui_addTitleLabel(colspan=2)
        self.titleLabel.setText('Loading data aborted.')
        self.gui_enableLoadButtons()
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
            cca_df_ID = df.loc[ID]
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
            if updateColor:
                LabelItemID.setText(txt, size=self.fontSize)
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
            self.axes[side].skeletonScatterItem.setData([], [])
            return

        posData = self.currentPosData(side)

        if posData.skelCoords and not button.isDoubleClick:
            self.updateImage(side)
            return

        # Check if user already loaded data to merge and did not dblclick
        # Use first loaded merged data to skeletonize
        if posData.loadedMergeRelativeFilenames and not button.isDoubleClick:
            relFilename = list(posData.loadedRelativeFilenamesData)[0]
            self.progressWin = dialogs.QDialogWorkerProcess(
                title='Skeletonizing...', parent=self,
                pbarDesc=f'Skeletonizing {relFilename}...'
            )
            self.progressWin.mainPbar.setMaximum(len(self.expData[side]))
            self.progressWin.show(self.app)
            self.startSkeletonizeWorker(side, initFilename=True)
            return

        selectChannelWin = dialogs.QDialogListbox(
            'Select data to skeletonize',
            'Select one of the following channels to load and skeletonize',
            posData.allRelFilenames, moreButtonFuncText='Cancel',
            parent=self
        )
        selectChannelWin.exec_()
        if selectChannelWin.cancel:
            button.setChecked(False)
            self.logger.info(f'Loading data to skeletonize aborted.')
            return

        selectedRelFilenames = selectChannelWin.selectedItemsText
        self.progressWin = dialogs.QDialogWorkerProcess(
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
            self.axes[side].contourPlotItem.setData([], [])
            return

        posData = self.currentPosData(side)

        if posData.skelCoords and not button.isDoubleClick:
            self.updateImage(side)
            return

        # Check if user already loaded data to merge and did not dblclick
        if posData.loadedMergeRelativeFilenames and not button.isDoubleClick:
            relFilename = list(posData.loadedRelativeFilenamesData)[0]
            self.progressWin = dialogs.QDialogWorkerProcess(
                title='Skeletonizing...', parent=self,
                pbarDesc=f'Skeletonizing {relFilename}...'
            )
            self.progressWin.mainPbar.setMaximum(len(self.expData[side]))
            self.progressWin.show(self.app)
            self.startContoursWorker(side, initFilename=True)
            return

        selectChannelWin = dialogs.QDialogListbox(
            'Select data to contour',
            'Select one of the following channels to load and display contour',
            posData.allRelFilenames, moreButtonFuncText='Cancel',
            parent=self
        )
        selectChannelWin.exec_()
        if selectChannelWin.cancel:
            button.setChecked(False)
            self.logger.info(f'Loading data to contour aborted.')
            return

        selectedRelFilenames = selectChannelWin.selectedItemsText
        self.progressWin = dialogs.QDialogWorkerProcess(
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
        win = dialogs.selectSpotsH5File(runsInfo, parent=self)
        win.show()
        win.exec_()
        if win.selectedFile is None:
            self.sender().setChecked(False)
            return

        h5_path = os.path.join(posData.spotmaxOutPath, win.selectedFile)
        if h5_path == posData.h5_path:
            self.plotSpotsCoords(side)
            return

        self.progressWin = dialogs.QDialogWorkerProcess(
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

        self.axes[side].skeletonScatterItem.setData(xx_skel, yy_skel)

    @exception_handler
    def plotContours(self, side):
        posData = self.currentPosData(side)
        if posData.SizeT > 1:
            frame_i = self.frame_i(side)
            contCoords = posData.contCoords[frame_i]
        else:
            contCoords = posData.contCoords

        if posData.SizeZ > 1:
            objContours = contCoords['proj']
            bottomWidgets = self.bottomWidgets[side]

            zProjHow = bottomWidgets[f'zProjComboboxLayer1'].currentText()
            z = bottomWidgets[f'zSliceScrollbarLayer1'].sliderPosition()-1
            if zProjHow == 'same as above':
                zProjHow = bottomWidgets[f'zProjComboboxLayer0'].currentText()
                z = bottomWidgets[f'zSliceScrollbarLayer0'].sliderPosition()-1

            if zProjHow == 'single z-slice':
                objContours = contCoords[z]
        else:
            objContours = contCoords['proj']

        for objID, contours_li in objContours.items():
            for cont in contours_li:
                yy_cont, xx_cont = cont[:,1], cont[:,0]
                self.axes[side].contourPlotItem.setData(xx_cont, yy_cont)

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

        if self.showOnlyInsideRefAction.isChecked():
            if 'is_spot_inside_ref_ch' in df.columns:
                

        yy, xx = df['y']+0.5, df['x']+0.5
        data = df['|abs|_spot'].round(4)

        # Add brushes and pens
        brushes = self.spotsBrushes[side]
        pens = self.spotsPens[side]
        df['brush'] = brushes['All spots'][0]
        df['pen'] = pens['All spots']

        if 'is_spot_inside_ref_ch' in df.columns:
            in_ref_mask = df['is_spot_inside_ref_ch']==0
            out_ref_mask = ~in_ref_mask
            in_ref_key = "Spots inside ref. channel"
            out_ref_key = "Spots outside ref. channel"
            df.loc[in_ref_mask, 'brush'] = brushes[in_ref_key][0]
            df.loc[out_ref_mask, 'brush'] = brushes[out_ref_key][0]

            df.loc[in_ref_mask, 'pen'] = pens[in_ref_key]
            df.loc[out_ref_mask, 'pen'] = pens[out_ref_key]

            xClicked, yClicked = spotsScatterItem.clickedSpot
            clickedMask = (df['x']==xClicked) & (df['y']==yClicked)
            clickedKey = "Clicked spot"
            df.loc[clickedMask, 'brush'] = brushes[clickedKey][0]
            df.loc[clickedMask, 'pen'] = pens[clickedKey]

        spotsScatterItem.setData(
            xx, yy, data=data,
            size=int(self.df_settings.at['spots_size', 'value']),
            pen=df['pen'].to_list(),
            brush=df['brush'].to_list(),
            hoverBrush=brushes['All spots'][1]
        )

    @exception_handler
    def spotsClicked(self, scatterItem, spotItems, event):
        side = self.side(self.axes['left'].spotsScatterItem, sender=scatterItem)

        xClicked = int(spotItems[0].pos().x()-0.5)
        yClicked = int(spotItems[0].pos().y()-0.5)
        scatterItem.clickedSpot = (xClicked, yClicked)

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
    def selectSpotsColor(self):
        """Callback of the actions from spotsClicked right-click QMenu"""

        side = self.sender().parent().side
        key = self.sender().text()
        viewToolbar = self.sideToolbar[side]['viewToolbar']

        # Trigger color button on the side toolbar which is connected to
        # gui_setColor
        colorButton = viewToolbar['colorButton']
        colorButton.side = side
        colorButton.key = key
        colorButton.setColor(self.colorItems[side][key])
        colorButton.selectColor()

    @exception_handler
    def selectSpotStyle(self):
        """Callback of the spotStyleAction from spotsClicked right-click QMenu"""
        side = self.sender().parent().side

        alpha = float(self.df_settings.at['spots_transparency', 'value'])
        penWidth = float(self.df_settings.at['spots_pen_width', 'value'])
        size = int(self.df_settings.at['spots_size', 'value'])

        transparencyVal = int(alpha*100)
        penWidthVal = int(penWidth*2)

        self.origAlpha = alpha
        self.origWidth = penWidth
        self.origSize = size

        self.spotStileWin = dialogs.spotStyleDock(
            'Spots style', parent=self
        )
        self.spotStileWin.side = side

        self.spotStileWin.transpSlider.setValue(transparencyVal)
        self.spotStileWin.transpSlider.sigValueChange.connect(
            self.gui_setSpotsTransparency
        )

        self.spotStileWin.penWidthSlider.setValue(penWidthVal)
        self.spotStileWin.penWidthSlider.sigValueChange.connect(
            self.gui_setSpotsPenWidth
        )

        self.spotStileWin.sizeSlider.setValue(size)
        self.spotStileWin.sizeSlider.sigValueChange.connect(
            self.gui_setSpotsSize
        )

        # self.spotStileWin.sigOk.connect(self.gui_setSpotsTransparency)
        self.spotStileWin.sigCancel.connect(self.gui_spotStyleCanceled)

        self.spotStileWin.show()

    def gui_spotStyleCanceled(self):
        self.df_settings.at['spots_transparency', 'value'] = self.origAlpha
        self.df_settings.at['spots_pen_width', 'value'] = self.origWidth
        self.df_settings.at['spots_size', 'value'] = self.origSize

        self.gui_createSpotsBrushesPens()
        self.plotSpotsCoords(self.spotStileWin.side)


    def gui_setSpotsTransparency(self, transparencyVal):
        alpha = transparencyVal/100
        self.df_settings.at['spots_transparency', 'value'] = alpha

        self.gui_createSpotsBrushesPens()
        self.plotSpotsCoords(self.spotStileWin.side)

    def gui_setSpotsPenWidth(self, penWidth):
        penWidthVal = penWidth/2
        self.df_settings.at['spots_pen_width', 'value'] = penWidthVal

        self.gui_createSpotsBrushesPens()
        self.plotSpotsCoords(self.spotStileWin.side)

    def gui_setSpotsSize(self, size):
        self.df_settings.at['spots_size', 'value'] = size

        self.gui_createSpotsBrushesPens()
        self.plotSpotsCoords(self.spotStileWin.side)

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
        self.openFile('left')

    def openFileRight(self, checked=True):
        self.openFile('right')

    @exception_handler
    def openFile(self, side, file_path=''):
        self.funcDescription = 'load data'

        if not file_path:
            self.getMostRecentPath()
            file_path = QFileDialog.getOpenFileName(
                self, 'Select image file', self.MostRecentPath,
                "Images/Videos (*.png *.tif *.tiff *.jpg *.jpeg *.mov *.avi *.mp4)"
                ";;All Files (*)")[0]
            if file_path == '':
                return file_path
        dirpath = os.path.dirname(file_path)
        dirname = os.path.basename(dirpath)
        if dirname != 'Images':
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            acdc_folder = f'{timestamp}_acdc'
            selectedPath = os.path.join(dirpath, acdc_folder, 'Images')
            os.makedirs(selectedPath)
        else:
            selectedPath = dirpath

        filename, ext = os.path.splitext(os.path.basename(file_path))
        if ext == '.tif' or ext == '.npz':
            self.openFolder(side, selectedPath=selectedPath, imageFilePath=file_path)
        else:
            self.logger.info('Copying file to .tif format...')
            data = load.loadData(file_path, '')
            data.loadImgData()
            img = data.img_data
            if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 4):
                self.logger.info('Converting RGB image to grayscale...')
                data.img_data = skimage.color.rgb2gray(data.img_data)
                data.img_data = skimage.img_as_ubyte(data.img_data)
            tif_path = os.path.join(selectedPath, f'{filename}.tif')
            if data.img_data.ndim == 3:
                SizeT = data.img_data.shape[0]
                SizeZ = 1
            elif data.img_data.ndim == 4:
                SizeT = data.img_data.shape[0]
                SizeZ = data.img_data.shape[1]
            else:
                SizeT = 1
                SizeZ = 1
            is_imageJ_dtype = (
                data.img_data.dtype == np.uint8
                or data.img_data.dtype == np.uint16
                or data.img_data.dtype == np.float32
            )
            if not is_imageJ_dtype:
                data.img_data = skimage.img_as_ubyte(data.img_data)

            myutils.imagej_tiffwriter(
                tif_path, data.img_data, {}, SizeT, SizeZ
            )
            self.openFolder(side, selectedPath=selectedPath, imageFilePath=tif_path)

    def openFolderRight(self, checked=True):
        self.openFolder('right')

    def openFolderLeft(self, checked=True):
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
        self.funcDescription = 'load data'
        self.lastLoadedSide = side

        self.gui_init(side=side)

        fileToolbar = self.sideToolbar[side]['fileToolbar']
        openFolderAction = fileToolbar['openFolderAction']

        if self.slideshowWin is not None:
            self.slideshowWin.close()

        if not selectedPath:
            self.getMostRecentPath()
            title = (
                'Select experiment folder containing Position_n folders '
                'or specific Position_n folder'
            )
            selectedPath = QFileDialog.getExistingDirectory(
                self, title, self.MostRecentPath
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

        self.channelNameUtil = load.channelName(which_channel=side)
        user_ch_name = None
        if imageFilePath:
            images_paths = [pathlib.Path(selectedPath)]
            filenames = os.listdir(selectedPath)
            ch_names, basenameNotFound = (
                channelNameUtil.getChannels(filenames, selectedPath)
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
                f for f in os.listdir(selectedPath)
                if f.find('Position_')!=-1
                and os.path.isdir(os.path.join(selectedPath, f))
            ]
            if len(pos_foldernames) == 1:
                pos = pos_foldernames[0]
                images_paths = [pathlib.Path(selectedPath) / pos / 'Images']
            else:
                self.progressWin = dialogs.QDialogWorkerProcess(
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
        pathScanner.input(app=app)
        if pathScanner.selectedPaths:
            self.images_paths = pathScanner.selectedPaths
            self.getChannelName()
        else:
            self.loadingDataAborted()
            self.logger.info('Loading data process aborted by the user.')
            self.titleLabel.setText('Loading data process aborted by the user.')

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
            self.startSkeletonizeWorker(side)
        elif nextStep == 'startContoursWorker':
            for posData in self.expData[side]:
                posData.contouredRelativeFilename = selectedRelFilename
            self.startContoursWorker(side)

    def workerProgress(self, text, loggerLevel):
        self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)

    @exception_handler
    def workerCritical(self, error):
        self.progressWin.workerFinished = True
        raise error

    def getChannelName(self):
        self.funcDescription = 'retrieving channel names'
        channelNameUtil = self.channelNameUtil
        images_path = self.images_paths[0]
        abortedByUser = False
        user_ch_name = None
        filenames = os.listdir(images_path)
        ch_names, basenameNotFound = (
            channelNameUtil.getChannels(filenames, images_path)
        )
        if not ch_names:
            self.logger.exception(
                f'No valid channels found in the folder {images_path}.'
            )
            self.titleLabel.setText(
                f'No valid channels found in the folder {images_path}.'
            )
            self.loadingDataAborted()
            return

        if len(ch_names) > 1:
            channelNameUtil.askSelectChannel(self, ch_names)
            if channelNameUtil.was_aborted:
                self.logger.info('Channel selection aborted by the User')
                self.titleLabel.setText('Channel selection aborted by the User')
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
            tif_found = False
            files = os.listdir(images_path)
            for file in files:
                channelDataPath = images_path / file
                if file.endswith(f'{self.user_ch_name}_aligned.npz'):
                    self.logger.info('Aligned file found, using it...')
                    channelDataFilePaths.append(channelDataPath)
                    break
                elif file.endswith(f'{self.user_ch_name}.tif'):
                    tif_found = True
                    tifPath = channelDataPath
            else:
                self.logger.info(
                    f'Aligned file not found in {images_path}, using .tif file...'
                )
                if tif_found:
                    channelDataFilePaths.append(tifPath)
                else:
                    self.loadingDataAborted()
                    self.criticalImgPathNotFound(images_path)
                    return

        self.initGlobalAttr()

        self.channelDataFilePaths = channelDataFilePaths
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
        uniqueExpPaths = set([f.parents[2] for f in self.channelDataFilePaths])
        expPaths = {}
        for expPath in uniqueExpPaths:
            expName = f'...{expPath.parent.name}{os.sep}{expPath.name}'
            expPaths[expName] = {
                'path': expPath,
                'channelDataPaths': []
            }
        for filePath in self.channelDataFilePaths:
            expPath = filePath.parents[2]
            expName = f'...{expPath.parent.name}{os.sep}{expPath.name}'
            expPaths[expName]['channelDataPaths'].append(filePath)

        self.expPaths = expPaths
        self.addExpNameCombobox()

    def addExpNameCombobox(self):
        self.topFileToolBar.removeAction(self.expNameAction)
        self.initiallyHiddenItems.remove(self.expNameAction)
        self.expNameCombobox = QComboBox(self.topFileToolBar)
        self.expNameCombobox.SizeAdjustPolicy(QComboBox.AdjustToContents)
        self.expNameCombobox.addItems(list(self.expPaths.keys()))
        self.expNameCombobox.adjustSize()
        self.expNameAction = self.topFileToolBar.addWidget(self.expNameCombobox)
        self.initiallyHiddenItems.append(self.expNameAction)

    def addPosCombobox(self, expName):
        try:
            self.topFileToolBar.removeAction(self.posNameAction)
            self.initiallyHiddenItems.remove(self.posNameAction)
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

    def readMetadata(self, expName, posIdx):
        expPath = self.expPaths[expName]['path']
        channelDataPaths = self.expPaths[expName]['channelDataPaths']

        self.logger.info('Reading meatadata...')
        # Load first pos to read metadata
        channelDataPath = channelDataPaths[0]
        posDataRef = load.loadData(
            channelDataPath, self.user_ch_name, QParent=self
        )
        posDataRef.getBasenameAndChNames()
        posDataRef.buildPaths()
        posDataRef.loadChannelData()
        posDataRef.loadOtherFiles(load_metadata=True)
        proceed = posDataRef.askInputMetadata(
            ask_SizeT=True,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            save=True
        )
        if not proceed:
            return '', None

        return channelDataPath, posDataRef

    def loadSelectedData(self):
        self.titleLabel.setText('Loading data...', color='w')
        expName = self.expNameCombobox.currentText()
        expPath = self.expPaths[expName]['path']

        selectedExpName = expName

        fisrtChannelDataPath, posDataRef = self.readMetadata(expName, 0)
        if not fisrtChannelDataPath:
            self.logger.info('Loading process cancelled by the user.')
            self.loadingDataAborted()
            return

        if posDataRef.SizeT > 1:
            self.isTimeLapse = True
            self.addPosCombobox(expName)
            selectedPos = self.posNameCombobox.currentText()
        else:
            self.isTimeLapse = False
            selectedPos = None

        self.progressWin = dialogs.QDialogWorkerProcess(
            title='Loading data...', parent=self,
            pbarDesc=f'Loading "{fisrtChannelDataPath}"...'
        )
        self.progressWin.show(self.app)
        self.posDataRef = posDataRef
        self.startLoadDataWorker(selectedPos, selectedExpName)

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

        self.titleLabel.setText('', color='w')
        self.dataLoaded[self.lastLoadedSide] = True

        areBothPlotsVisible = (
            self.dataLoaded['left'] and self.dataLoaded['right']
        )
        if areBothPlotsVisible:
            self.axes['left'].disableAutoRange()
            self.axes['left'].disableAutoRange()

            self.axes['left'].setYLink(self.axes['right'])
            self.axes['left'].setXLink(self.axes['right'])

            self.axes['left'].enableAutoRange()
            self.axes['right'].enableAutoRange()

    def loadDataWorkerFinished(self):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.loadingDataFinished()

    def zStack_to_2D(self, side, zStack, layer=0):
        bottomWidgets = self.bottomWidgets[side]
        zProjHow = bottomWidgets[f'zProjComboboxLayer{layer}'].currentText()
        z = bottomWidgets[f'zSliceScrollbarLayer{layer}'].sliderPosition()-1
        if zProjHow == 'same as above':
            zProjHow = bottomWidgets[f'zProjComboboxLayer0'].currentText()
            z = bottomWidgets[f'zSliceScrollbarLayer0'].sliderPosition()-1

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

        selectedRelFilenames = selectChannelWin.selectedItemsText
        shouldLoad = any([
            relFilename not in posData.loadedRelativeFilenamesData
            for relFilename in selectedRelFilenames
        ])

        if shouldLoad:
            self.progressWin = dialogs.QDialogWorkerProcess(
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

    def addWorkerData(self, posData, mergeData, relFilename):
        posData.loadedRelativeFilenamesData[relFilename] = mergeData
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
                bottomWidgets['zSliceSbL1_label'].setStyleSheet('color: gray')
                bottomWidgets['zSliceScrollbarLayer1'].setDisabled(True)
            elif enabled:
                bottomWidgets['zSliceSbL1_label'].setStyleSheet('color: black')
            else:
                bottomWidgets['zSliceSbL1_label'].setStyleSheet('color: gray')

        bottomWidgets['alphaScrollbar'].setEnabled(enabled)
        if enabled:
            bottomWidgets['alphaScrollbar_label'].setStyleSheet('color: black')
        else:
            bottomWidgets['alphaScrollbar_label'].setStyleSheet('color: gray')

    def layerImage(self, side, relFilename=''):
        posData = self.currentPosData(side)

        if not relFilename:
            data = posData.chData
            layer = 0
        else:
            data = posData.loadedRelativeFilenamesData[relFilename]
            layer = 1

        if posData.SizeT > 1 and posData.SizeZ > 1:
            # 4D data
            frame_i = self.frame_i(side)
            zStack = data[frame_i]
            img = self.zStack_to_2D(side, zStack, layer=layer)
        elif posData.SizeT == 1 and posData.SizeZ > 1:
            # 3D z-stacks data
            img = self.zStack_to_2D(side, data, layer=layer)
        elif posData.SizeT > 1 and posData.SizeZ == 1:
            # 2D timelapse data
            frame_i = self.frame_i(side)
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

    def howDrawSegmCombobox_cb(self, how):
        onlyIDs = how == 'Draw only IDs'
        nothing = how == 'Draw nothing'
        onlyCont = how == 'Draw only contours'
        only_ccaInfo = how == 'Draw only cell cycle info'
        ccaInfo_and_cont = how == 'Draw cell cycle info and contours'
        onlyMothBudLines = how == 'Draw only mother-bud lines'

        side = self.side(self.bottomWidgets['left']['howDrawSegmCombobox'])

        # Clear contours if requested
        if how.find('contours') == -1 or nothing:
            for axContCurve in self.axes[side].ContoursCurves:
                if axContCurve.getData()[0] is not None:
                    axContCurve.setData([], [])

        # Clear LabelItems IDs if requested (draw nothing or only contours)
        if onlyCont or nothing or onlyMothBudLines:
            for _IDlabel1 in self.axes[side].LabelItemsIDs:
                _IDlabel1.setText('')

        # Clear mother-bud lines if Requested
        drawLines = only_ccaInfo or ccaInfo_and_cont or onlyMothBudLines
        if not drawLines:
            for BudMothLine in self.axes[side].BudMothLines:
                if BudMothLine.getData()[0] is not None:
                    BudMothLine.setData([], [])

        self.updateSegmVisuals(side)

    def navigateScrollbarTriggered(self, action):
        side = self.side(self.bottomWidgets['left']['navigateScrollbar'])
        bottomWidgets = self.bottomWidgets[side]
        posData = self.currentPosData(side)
        posName = posData.pos_foldername
        bottomWidgets['navigateScrollbar_label'].setText(f'{posName} ')
        isSliderDrag = (
            action == QAbstractSlider.SliderMove
            and self.sender().isSliderDown()
        )
        if isSliderDrag:
            self.clearSegmVisuals(side)
            # Slider is being dragged --> simply set the image
            self.axes[side].spotsScatterItem.setData([], [])
            img = self.currentImage(side)
            self.imgItems[side].setImage(img)
        else:
            self.updateImage(side)

    def navigateScrollbarReleased(self):
        side = self.side(self.bottomWidgets['left']['navigateScrollbar'])
        self.updateImage(side)
        self.updateSegmVisuals(side)

    def zSliceScrollbarLayerTriggered(self, action):
        side = self.sender().side
        layer = self.sender().layer

        posData = self.currentPosData(side)
        z = self.sender().sliderPosition()
        bottomWidgets = self.bottomWidgets[side]
        zSliceSbL_label = bottomWidgets[f'zSliceSbL{layer}_label']
        zSliceSbL_label.setText(f'First layer z-slice {z+1}/{posData.SizeZ}')

        isSliderDrag = (
            action == QAbstractSlider.SliderMove
            and self.sender().isSliderDown()
        )
        if isSliderDrag:
            # Slider is being dragged --> simply set the image
            self.updateImage(side)
        else:
            self.updateImage(side)
            self.updateSegmVisuals(side)

    def zSliceScrollbarLayerReleased(self):
        side = self.sender().side
        self.updateImage(side)
        self.updateSegmVisuals(side)

    def updateZprojLayer(self, how):
        side = self.sender().side
        layer = self.sender().layer
        bottomWidgets = self.bottomWidgets[side]

        if how.find('max') != -1 or how == 'same as above':
            bottomWidgets[f'zSliceSbL{layer}_label'].setStyleSheet('color: gray')
            bottomWidgets[f'zSliceScrollbarLayer{layer}'].setDisabled(True)
        else:
            bottomWidgets[f'zSliceSbL{layer}_label'].setStyleSheet('color: black')
            bottomWidgets[f'zSliceScrollbarLayer{layer}'].setDisabled(False)

        self.updateImage(side)

    def updateAlphaOverlay(self, action):
        side = self.side(self.bottomWidgets['left']['alphaScrollbar'])
        self.updateImage(side)

    def currentPosData(self, side):
        if self.isTimeLapse:
            posIdx = self.posNameCombobox.currentIndex()
        else:
            bottomWidgets = self.bottomWidgets[side]
            navigateScrollbar = bottomWidgets['navigateScrollbar']
            posIdx = navigateScrollbar.sliderPosition()-1
        posData = self.expData[side][posIdx]
        return posData

    def criticalValidChannelDataNotFound(self, images_path):
        err_title = 'Valid channel data not found'
        err_msg = (f"""
            The folder \"{images_path}\" <b>does not contain valid data</b>.<br><br>
            Valid data is either a file ending with \"{self.user_ch_name}.tif\"
            or ending with \"{self.user_ch_name}_aligned.npz\".<br><br>
            Sorry about that.
        """)
        msg = QMessageBox()
        msg.critical(self, err_title, err_msg, msg.Ok)
        self.logger.exception(
            f'Folder "{images_path}" does NOT contain valid data.'
        )


    def criticalNoTifFound(self, images_path):
        err_title = 'No .tif files found in folder.'
        err_msg = (
            f'The folder "{images_path}" does not contain .tif files.\n\n'
            'Only .tif files can be loaded with "Open Folder" button.\n\n'
            'Try with "File --> Open image/video file..." and directly select '
            'the file you want to load.'
        )
        msg = QMessageBox()
        msg.critical(self, err_title, err_msg, msg.Ok)
        self.logger.exception(f'No .tif files found in folder "{images_path}"')


    def saveData(self):
        pass

    def populateOpenRecent(self):
        # Step 0. Remove the old options from the menu
        self.openRecentMenu.clear()
        # Step 1. Read recent Paths
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'settings', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            recentPaths = df['path'].to_list()
        else:
            recentPaths = []
        # Step 2. Dynamically create the actions
        actions = []
        for path in recentPaths:
            action = QAction(path, self)
            action.triggered.connect(partial(self.openRecentFile, path))
            actions.append(action)
        # Step 3. Add the actions to the menu
        self.openRecentMenu.addActions(actions)

    def getMostRecentPath(self):
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'settings', 'recentPaths.csv'
        )
        if os.path.exists(recentPaths_path):
            df = pd.read_csv(recentPaths_path, index_col='index')
            if 'opened_last_on' in df.columns:
                df = df.sort_values('opened_last_on', ascending=False)
            self.MostRecentPath = df.iloc[0]['path']
            if not isinstance(self.MostRecentPath, str):
                self.MostRecentPath = ''
        else:
            self.MostRecentPath = ''

    def openRecentFile(self, path):
        self.logger.info(f'Opening recent folder: {path}')
        self.openFolder('left', selectedPath=path)

    def closeEvent(self, event):
        self.saveWindowGeometry()

        with open(self.colorItems_path, mode='w') as file:
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
            msg = QMessageBox()
            save = msg.question(
                self, 'Save?', 'Do you want to save?',
                msg.Yes | msg.No | msg.Cancel
            )
            if save == msg.Yes:
                self.saveData()
                event.accept()
            elif save == msg.No:
                event.accept()
            else:
                event.ignore()

        for expData in self.expData.values():
            for posData in expData:
                if not posData.h5_path:
                    continue
                if posData.hdf_store is None:
                    continue
                posData.hdf_store.close()

        if self.buttonToRestore is not None and event.isAccepted():
            button, color, text = self.buttonToRestore
            button.setText(text)
            button.setStyleSheet(
                f'QPushButton {{background-color: {color};}}')
            self.mainWin.setWindowState(Qt.WindowNoState)
            self.mainWin.setWindowState(Qt.WindowActive)
            self.mainWin.raise_()

    def saveWindowGeometry(self):
        left = self.geometry().left()
        top = self.geometry().top()
        width = self.geometry().width()
        height = self.geometry().height()
        try:
            screenName = '/'.join(self.screen().name().split('\\'))
        except AttributeError:
            screenName = 'None'
            self.logger.info(
                'WARNING: could not retrieve screen name.'
                'Please update to PyQt5 version >= 5.14'
            )
        self.df_settings.at['geometry_left', 'value'] = left
        self.df_settings.at['geometry_top', 'value'] = top
        self.df_settings.at['geometry_width', 'value'] = width
        self.df_settings.at['geometry_height', 'value'] = height
        self.df_settings.at['screenName', 'value'] = screenName
        isMaximised = self.windowState() == Qt.WindowMaximized
        isMaximised = 'Yes' if isMaximised else 'No'
        self.df_settings.at['isMaximised', 'value'] = isMaximised
        self.df_settings.to_csv(self.settings_csv_path)
        # print('Window screen name: ', screenName)

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
        src_path = os.path.dirname(os.path.realpath(__file__))
        recentPaths_path = os.path.join(
            src_path, 'settings', 'recentPaths.csv'
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
            # Keep max 20 recent paths
            if len(recentPaths) > 20:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        else:
            recentPaths = [selectedPath]
            openedOn = [datetime.datetime.now()]
        df = pd.DataFrame({'path': recentPaths,
                           'opened_last_on': pd.Series(openedOn,
                                                       dtype='datetime64[ns]')})
        df.index.name = 'index'
        df.to_csv(recentPaths_path)

    def storeDefaultAndCustomColors(self):
        c = overlayButton.palette().button().color().name()
        self.defaultToolBarButtonColor = c
        self.doublePressKeyButtonColor = '#fa693b'

    def show(self):
        # self.storeDefaultAndCustomColors()
        QMainWindow.show(self)

        screenNames = []
        for screen in self.app.screens():
            name = '/'.join(screen.name().split('\\'))
            screenNames.append(name)

        if 'isMaximised' in self.df_settings.index:
            self.df_settings.loc['isMaximised'] = (
                self.df_settings.loc['isMaximised'].astype(str)
            )

        if 'geometry_left' in self.df_settings.index:
            isMaximised = self.df_settings.at['isMaximised', 'value'] == 'Yes'
            left = int(self.df_settings.at['geometry_left', 'value'])
            screenName = self.df_settings.at['screenName', 'value']
            if isMaximised:
                g = self.geometry()
                top, width, height = g.top(), g.width(), g.height()
            else:
                top = int(self.df_settings.at['geometry_top', 'value'])+10
                width = int(self.df_settings.at['geometry_width', 'value'])
                height = int(self.df_settings.at['geometry_height', 'value'])
            if screenName in screenNames:
                self.setGeometry(left, top, width, height)
            if isMaximised:
                self.showMaximized()

        h = self.bottomWidgets['left']['howDrawSegmCombobox'].size().height()
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


        self.gui_hideInitItems()


if __name__ == "__main__":
    print('Loading application...')
    # Handle high resolution displays:
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Create the application
    app = QApplication(sys.argv)

    # Apply style
    app.setStyle(QStyleFactory.create('Fusion'))
    # src_path = os.path.dirname(os.path.abspath(__file__))
    # styles_path = os.path.join(src_path, 'styles')
    # dark_orange_path = os.path.join(styles_path, '01_buttons.qss')
    # with open(dark_orange_path, mode='r') as txt:
    #     styleSheet = txt.read()
    # app.setStyleSheet(styleSheet)


    win = spotMAX_Win(app)
    win.show()

    # Run the event loop
    win.logger.info('Lauching application...')
    win.logger.info(
        'Done. If application GUI is not visible, it is probably minimized, '
        'behind some other open window, or on second screen.'
    )
    sys.exit(app.exec_())
