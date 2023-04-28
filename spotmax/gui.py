import os
import shutil
import datetime
import subprocess
from queue import Queue

import numpy as np
import pandas as pd

from PyQt5.QtCore import (
    Qt, QTimer, QThreadPool, QThread, QMutex, QWaitCondition
)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDockWidget, QToolBar, QAction, QFileDialog

from cellacdc import gui as acdc_gui
from cellacdc import widgets as acdc_widgets
from cellacdc import exception_handler
from cellacdc import apps as acdc_apps

from . import qtworkers, io, printl, dialogs
from . import logs_path, html_path, html_func

from . import qrc_resources

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
    
    def gui_createRegionPropsDockWidget(self):
        super().gui_createRegionPropsDockWidget(side=Qt.RightDockWidgetArea)
        self.gui_createParamsDockWidget()
    
    def gui_createParamsDockWidget(self):
        self.computeDockWidget = QDockWidget('spotMAX Tab Control', self)
        computeTabControl = dialogs.guiTabControl(
            parent=self.computeDockWidget, logging_func=self.logger.info
        )

        self.computeDockWidget.setWidget(computeTabControl)
        self.computeDockWidget.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )
        self.computeDockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )

        self.addDockWidget(Qt.LeftDockWidgetArea, self.computeDockWidget)
    
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
        
    def _setWelcomeText(self):
        html_filepath = os.path.join(html_path, 'gui_welcome.html')
        with open(html_filepath) as html_file:
            htmlText = html_file.read()
        self.ax1.infoTextItem.setHtml(htmlText)
    
    def _disableAcdcActions(self, *actions):
        for action in actions:
            action.setVisible(False)
            action.setDisabled(True)
    
    def reinitGui(self):
        super().reinitGui()
        self.showParamsDockButton.setDisabled(False)

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
            # if self.computeDockWidgetMinWidth is not None:
            #     self.resizeDocks([self.computeDockWidget], [w+5], Qt.Horizontal)
            try:
                self.addInspectResultsTab(self.lastLoadedSide)
            except Exception as e:
                pass

    def _loadFromExperimentFolder(self, selectedPath):
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
        self.addSpotsCoordinatesAction.setIcon(QIcon(":plotSpots.svg"))
        self.addSpotsCoordinatesAction.setToolTip('Plot spots coordinates')
    
    def gui_createToolBars(self):
        super().gui_createToolBars()

        self.addToolBarBreak(Qt.LeftToolBarArea)
        self.spotmaxToolbar = QToolBar("spotMAX toolbar", self)
        self.spotmaxToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(Qt.LeftToolBarArea, self.spotmaxToolbar)
        self.spotmaxToolbar.addAction(self.addSpotsCoordinatesAction)
        self.spotmaxToolbar.setVisible(False)
    
    def loadingDataCompleted(self):
        super().loadingDataCompleted()
        self.spotmaxToolbar.setVisible(True)
    
    def resizeComputeDockWidget(self):
        paramsGroupbox = self.computeDockWidget.widget().parametersQGBox
        paramsScrollArea = self.computeDockWidget.widget().parametersTab
        verticalScrollbar = paramsScrollArea.verticalScrollBar()
        groupboxWidth = paramsGroupbox.size().width()
        scrollbarWidth = verticalScrollbar.size().width()
        minWidth = groupboxWidth + scrollbarWidth + 20
        self.resizeDocks([self.computeDockWidget], [minWidth], Qt.Horizontal)
        self.showParamsDockButton.click()
    
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
