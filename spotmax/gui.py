import os
import pathlib

import numpy as np
import pandas as pd

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog

from cellacdc import gui as acdc_gui
from cellacdc import exception_handler

from . import utils, io, printl
from . import logs_path, html_path, settings_path

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
    
    def _setWelcomeText(self):
        html_filepath = os.path.join(html_path, 'gui_welcome.html')
        with open(html_filepath) as html_file:
            htmlText = html_file.read()
        self.ax1.infoTextItem.setHtml(htmlText)
    
    def _disableAcdcActions(self, *actions):
        for action in actions:
            action.setVisible(False)
            action.setDisabled(True)

    def initGui(self):
        self._setWelcomeText()
        self._disableAcdcActions(
            self.newAction, self.manageVersionsAction, self.openFileAction
        )
    
    # def getMostRecentPath(self):
    #     return utils.getMostRecentPath()
    
    # def addToRecentPaths(self, path, logger=None):
    #     io.addToRecentPaths(path)
    
    # def readRecentPaths(self):
    #     recentPaths_path = os.path.join(settings_path, 'recentPaths.csv')
    #     printl(recentPaths_path)
    #     super().readRecentPaths(recentPaths_path=recentPaths_path)
    
    def loadingDataCompleted(self):
        super().loadingDataCompleted()
        if not self.isSnapshot:
            self.modeComboBox.setCurrentText('Segmentation and Tracking')
            self.modeComboBox.setDisabled(True)
            self.modeMenu.menuAction().setVisible(False)
