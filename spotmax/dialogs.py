import os
import datetime
import re
import pathlib
import time
import traceback
from pprint import pprint

import numpy as np
import pandas as pd
from natsort import natsorted

from collections import defaultdict

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal, QEventLoop
from PyQt5.QtGui import (
    QFont, QFontMetrics, QTextDocument, QPalette, QColor,
    QIcon
)
from PyQt5.QtWidgets import (
    QDialog, QComboBox, QVBoxLayout, QHBoxLayout, QLabel, QApplication,
    QPushButton, QStyleFactory, QCheckBox, QTreeWidget, QTreeWidgetItem,
    QTreeWidgetItemIterator, QAbstractItemView, QFrame, QMessageBox,
    QMainWindow, QWidget, QTableView, QTextEdit, QGridLayout,
    QProgressBar, QSpinBox, QDoubleSpinBox, QListWidget, QGroupBox,
    QFileDialog, QDockWidget, QTabWidget, QScrollArea, QScrollBar
)

from cellacdc import apps as acdc_apps
from cellacdc import widgets as acdc_widgets

from . import html_func, io, widgets, utils, config
from . import core, dock_params_callbacks

# NOTE: Enable icons
from . import qrc_resources, printl, font
from . import is_mac, is_win, is_linux

class QBaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def exec_(self):
        self.show(block=True)

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class GopFeaturesAndThresholdsDialog(QBaseDialog):
    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle('Features and thresholds for filtering true spots')

        mainLayout = QVBoxLayout()

        self.setFeaturesGroupbox = widgets.GopFeaturesAndThresholdsGroupbox()
        mainLayout.addWidget(self.setFeaturesGroupbox)

        buttonsLayout = acdc_widgets.CancelOkButtonsLayout()
        buttonsLayout.cancelButton.clicked.connect(self.close)
        buttonsLayout.okButton.clicked.connect(self.ok_cb)

        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
    
    def show(self, block=False) -> None:
        super().show(block=False)
        firstButton = self.setFeaturesGroupbox.selectors[0].selectButton
        featuresNeverSet = firstButton.text().find('Click') != -1
        if featuresNeverSet:
            self.setFeaturesGroupbox.selectors[0].selectButton.click()
        super().show(block=block)
    
    def configIniParam(self):
        paramsText = ''
        for selector in self.setFeaturesGroupbox.selectors:
            selectButton = selector.selectButton
            column_name = selectButton.toolTip()
            if not column_name:
                continue
            lowValue = selector.lowRangeWidgets.value()
            highValue = selector.highRangeWidgets.value()
            if lowValue is None and highValue is None:
                self.warnRangeNotSelected(selectButton.text())
                return False
            paramsText = f'{paramsText}  * {column_name}, {lowValue}, {highValue}\n'
        tooltip = f'Features and ranges set:\n\n{paramsText}'
        return tooltip
    
    def warnRangeNotSelected(self, buttonText):
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph(
            'The following feature<br><br>'
            f'<code>{buttonText}</code><br><br>'
            'does <b>not have a valid range</b>.<br><br>'
            'Make sure you select <b>at least one</b> of the lower and higher '
            'range values.'
        )
        msg.critical(self, 'Invalid selection', txt)
    
    def ok_cb(self):
        isSelectionValid = self.configIniParam()
        if not isSelectionValid:
            return
        self.cancel = False
        self.close()


class measurementsQGroupBox(QGroupBox):
    def __init__(self, names, parent=None):
        QGroupBox.__init__(self, 'Single cell measurements', parent)
        self.formWidgets = []

        self.setCheckable(True)
        layout = widgets.myFormLayout()

        for row, item in enumerate(names.items()):
            key, labelTextRight = item
            widget = widgets.formWidget(
                QCheckBox(), labelTextRight=labelTextRight,
                parent=self, key=key
            )
            layout.addFormWidget(widget, row=row)
            self.formWidgets.append(widget)

        row += 1
        buttonsLayout = QHBoxLayout()
        self.selectAllButton = QPushButton('Deselect all', self)
        self.selectAllButton.setCheckable(True)
        self.selectAllButton.setChecked(True)
        helpButton = widgets.acdc_widgets.helpPushButton('Help', self)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(self.selectAllButton)
        buttonsLayout.addWidget(helpButton)
        layout.addLayout(buttonsLayout, row, 0, 1, 4)

        row += 1
        layout.setRowStretch(row, 1)
        layout.setColumnStretch(3, 1)

        layout.setVerticalSpacing(10)
        self.setFont(widget.labelRight.font())
        self.setLayout(layout)

        self.toggled.connect(self.checkAll)
        self.selectAllButton.clicked.connect(self.checkAll)

        for _formWidget in self.formWidgets:
            _formWidget.widget.setChecked(True)

    def checkAll(self, isChecked):
        for _formWidget in self.formWidgets:
            _formWidget.widget.setChecked(isChecked)
        if isChecked:
            self.selectAllButton.setText('Deselect all')
        else:
            self.selectAllButton.setText('Select all')

class guiBottomWidgets(QGroupBox):
    def __init__(
            self, side, drawSegmComboboxItems, zProjItems,
            isCheckable=False, checked=False, parent=None,
            font=None
        ):
        if font is None:
            self._font = QFont()
            self._font.setPixelSize(11)
        else:
            self._font = font

        QGroupBox.__init__(self, parent)

        layout = QGridLayout()

        bottomWidgets = {}

        initialCol = 0
        howToDrawCheckbox = None
        navigateCheckbox = None
        zSliceL0checkbox = None
        zSliceL1checkbox = None
        if isCheckable:
            howToDrawCheckbox = QCheckBox()
            bottomWidgets['howToDrawCheckbox'] = howToDrawCheckbox
            howToDrawCheckbox.stateChanged.connect(self.setHowToDrawEnabled)
            layout.addWidget(howToDrawCheckbox, 0, 0, alignment=Qt.AlignCenter)

            navigateCheckbox = QCheckBox()
            bottomWidgets['navigateCheckbox'] = navigateCheckbox
            navigateCheckbox.stateChanged.connect(self.setNavigateEnabled)
            layout.addWidget(navigateCheckbox, 1, 0, alignment=Qt.AlignCenter)

            zSliceL0checkbox = QCheckBox()
            bottomWidgets['zSliceL0checkbox'] = zSliceL0checkbox
            zSliceL0checkbox.stateChanged.connect(self.setZsliceL0Enabled)
            layout.addWidget(zSliceL0checkbox, 2, 0, alignment=Qt.AlignCenter)

            zSliceL1checkbox = QCheckBox()
            bottomWidgets['zSliceL1checkbox'] = zSliceL1checkbox
            zSliceL1checkbox.stateChanged.connect(self.setZsliceL1Enabled)
            layout.addWidget(zSliceL1checkbox, 3, 0, alignment=Qt.AlignCenter)
            initialCol = 1

        row = 0
        col = initialCol +1
        howDrawSegmCombobox = widgets.myQComboBox(checkBox=howToDrawCheckbox)
        howDrawSegmCombobox.addItems(drawSegmComboboxItems)
        # Always adjust combobox width to largest item
        howDrawSegmCombobox.setSizeAdjustPolicy(
            howDrawSegmCombobox.AdjustToContents
        )
        # layout.addWidget(
        #     howDrawSegmCombobox, row, col, alignment=Qt.AlignCenter
        # )
        bottomWidgets['howDrawSegmCombobox'] = howDrawSegmCombobox
        howDrawSegmCombobox.hide()
        checkboxes_text = (
            'IDs', 'Cell cycle info', 'Only mother-daughter line', '|', 
            'Contours', 'Segm. masks', '|' , 'Do not annotate'
        )
        self.checkboxes = {}
        checkboxesLayout = QHBoxLayout()
        checkboxesLayout.addStretch(1)
        checkboxesLayout.addWidget(QLabel(' | '))
        for text in checkboxes_text:
            if text == '|':
                checkboxesLayout.addWidget(QLabel(' | '))
                continue
            checkbox = QCheckBox(text, self)
            self.checkboxes[text] = checkbox
            checkboxesLayout.addWidget(checkbox)
            checkbox.clicked.connect(self.annotOptionClicked)
        checkboxesLayout.addWidget(QLabel(' | '))
        checkboxesLayout.addStretch(1)
        layout.addLayout(checkboxesLayout, row, col)

        row = 1
        col = initialCol +0
        navigateScrollbar_label = QLabel('frame n.  ')
        navigateScrollbar_label.setFont(self._font)
        layout.addWidget(
            navigateScrollbar_label, row, col, alignment=Qt.AlignRight
        )
        bottomWidgets['navigateScrollbar_label'] = navigateScrollbar_label

        row = 1
        col = initialCol +1
        navigateScrollbar = widgets.myQScrollBar(
            Qt.Horizontal,
            checkBox=navigateCheckbox,
            label=navigateScrollbar_label
        )
        navigateScrollbar.setMinimum(1)
        layout.addWidget(navigateScrollbar, row, col)
        bottomWidgets['navigateScrollbar'] = navigateScrollbar

        row = 2
        col = initialCol +0
        zSliceSbL0_label = QLabel('First layer z-slice  ')
        zSliceSbL0_label.setFont(self._font)
        layout.addWidget(
            zSliceSbL0_label, row, col, alignment=Qt.AlignRight
        )
        bottomWidgets['zSliceSbL0_label'] = zSliceSbL0_label

        row = 2
        col = initialCol +1
        zSliceScrollbarLayer0 = widgets.myQScrollBar(
            Qt.Horizontal,
            checkBox=zSliceL0checkbox,
            label=zSliceSbL0_label
        )
        zSliceScrollbarLayer0.setMinimum(1)
        zSliceScrollbarLayer0.layer = 0
        zSliceScrollbarLayer0.side = side
        layout.addWidget(zSliceScrollbarLayer0, row, col)
        bottomWidgets['zSliceScrollbarLayer0'] = zSliceScrollbarLayer0

        row = 2
        col = initialCol +2
        zProjComboboxLayer0 = widgets.myQComboBox(checkBox=zSliceL0checkbox)
        zProjComboboxLayer0.layer = 0
        zProjComboboxLayer0.side = side
        zProjComboboxLayer0.addItems(zProjItems)
        layout.addWidget(zProjComboboxLayer0, row, col, alignment=Qt.AlignLeft)
        bottomWidgets['zProjComboboxLayer0'] = zProjComboboxLayer0

        row = 3
        col = initialCol +0
        zSliceSbL1_label = QLabel('Second layer z-slice  ')
        zSliceSbL1_label.setFont(self._font)
        layout.addWidget(
            zSliceSbL1_label, row, col, alignment=Qt.AlignRight
        )
        bottomWidgets['zSliceSbL1_label'] = zSliceSbL1_label

        row = 3
        col = initialCol +1
        zSliceScrollbarLayer1 = widgets.myQScrollBar(
            Qt.Horizontal,
            checkBox=zSliceL1checkbox,
            label=zSliceSbL1_label
        )
        zSliceScrollbarLayer1.setMinimum(1)
        zSliceScrollbarLayer1.layer = 1
        zSliceScrollbarLayer1.side = side
        layout.addWidget(zSliceScrollbarLayer1, row, col)
        bottomWidgets['zSliceScrollbarLayer1'] = zSliceScrollbarLayer1

        row = 3
        col = initialCol +2
        zProjComboboxLayer1 = widgets.myQComboBox(checkBox=zSliceL1checkbox)
        zProjComboboxLayer1.layer = 1
        zProjComboboxLayer1.side = side
        zProjComboboxLayer1.addItems(zProjItems)
        zProjComboboxLayer1.addItems(['same as above'])
        zProjComboboxLayer1.setCurrentIndex(1)
        layout.addWidget(zProjComboboxLayer1, row, col, alignment=Qt.AlignLeft)
        bottomWidgets['zProjComboboxLayer1'] = zProjComboboxLayer1

        row = 4
        col = initialCol +0
        alphaScrollbar_label = QLabel('Overlay alpha  ')
        alphaScrollbar_label.setFont(self._font)
        layout.addWidget(
            alphaScrollbar_label, row, col, alignment=Qt.AlignRight
        )
        bottomWidgets['alphaScrollbar_label'] = alphaScrollbar_label

        row = 4
        col = initialCol +1
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

        layout.setColumnStretch(0,0)
        layout.setColumnStretch(initialCol+0,0)
        layout.setColumnStretch(initialCol+1,3)
        layout.setColumnStretch(initialCol+2,0)
        self.setLayout(layout)

        self.bottomWidgets = bottomWidgets

        if isCheckable:
            howToDrawCheckbox.setChecked(~checked)
            howToDrawCheckbox.setChecked(checked)

            navigateCheckbox.setChecked(~checked)
            navigateCheckbox.setChecked(checked)

            zSliceL0checkbox.setChecked(~checked)
            zSliceL0checkbox.setChecked(checked)

            zSliceL1checkbox.setChecked(~checked)
            zSliceL1checkbox.setChecked(checked)
    
    def annotOptionClicked(self):
        # First manually set exclusive with uncheckable
        clickedIDs = self.sender() == self.checkboxes['IDs']
        clickedCca = self.sender() == self.checkboxes['Cell cycle info']
        clickedMBline = self.sender() == self.checkboxes['Only mother-daughter line']
        if self.checkboxes['IDs'].isChecked() and clickedIDs:
            if self.checkboxes['Cell cycle info'].isChecked():
                self.checkboxes['Cell cycle info'].setChecked(False)
            if self.checkboxes['Only mother-daughter line'].isChecked():
                self.checkboxes['Only mother-daughter line'].setChecked(False)
        
        if self.checkboxes['Cell cycle info'].isChecked() and clickedCca:
            if self.checkboxes['IDs'].isChecked():
                self.checkboxes['IDs'].setChecked(False)
            if self.checkboxes['Only mother-daughter line'].isChecked():
                self.checkboxes['Only mother-daughter line'].setChecked(False)
        
        if self.checkboxes['Only mother-daughter line'].isChecked() and clickedMBline:
            if self.checkboxes['IDs'].isChecked():
                self.checkboxes['IDs'].setChecked(False)
            if self.checkboxes['Cell cycle info'].isChecked():
                self.checkboxes['Cell cycle info'].setChecked(False)
        
        clickedCont = self.sender() == self.checkboxes['Contours']
        clickedSegm = self.sender() == self.checkboxes['Segm. masks']
        if self.checkboxes['Contours'].isChecked() and clickedCont:
            if self.checkboxes['Segm. masks'].isChecked():
                self.checkboxes['Segm. masks'].setChecked(False)
        
        if self.checkboxes['Segm. masks'].isChecked() and clickedSegm:
            if self.checkboxes['Contours'].isChecked():
                self.checkboxes['Contours'].setChecked(False)
        
        clickedDoNot = self.sender() == self.checkboxes['Do not annotate']
        if clickedDoNot:
            self.checkboxes['IDs'].setChecked(False)
            self.checkboxes['Cell cycle info'].setChecked(False)
            self.checkboxes['Contours'].setChecked(False)
            self.checkboxes['Segm. masks'].setChecked(False)
            self.checkboxes['Only mother-daughter line'].setChecked(False)
        else:
            self.checkboxes['Do not annotate'].setChecked(False)
        
        self.annotOptionsToComboboxText()
    
    def annotOptionsToComboboxText(self):
        if self.checkboxes['IDs'].isChecked():
            if self.checkboxes['Contours'].isChecked():
                t = 'Draw IDs and contours'
                self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            elif self.checkboxes['Segm. masks'].isChecked():
                t = 'Draw IDs and overlay segm. masks'
                self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            else:
                t = 'Draw only IDs'
                self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            return
        
        if self.checkboxes['Cell cycle info'].isChecked():
            if self.checkboxes['Contours'].isChecked():
                t = 'Draw cell cycle info and contours'
                self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            elif self.checkboxes['Segm. masks'].isChecked():
                t = 'Draw cell cycle info and overlay segm. masks'
                self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            else:
                t = 'Draw only cell cycle info'
                self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            return
        
        if self.checkboxes['Segm. masks'].isChecked():
            t = 'Draw only overlay segm. masks'
            self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            return

        if self.checkboxes['Contours'].isChecked():
            t = 'Draw only contours'
            self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            return
        
        if self.checkboxes['Only mother-daughter line'].isChecked():
            t = 'Draw only mother-bud lines'
            self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            return
        
        if self.checkboxes['Do not annotate'].isChecked():
            t = 'Draw nothing'
            self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
            return
        
        t = 'Draw nothing'
        self.bottomWidgets['howDrawSegmCombobox'].setCurrentText(t)
    
    def setAnnotOptionsChecked(self, checked):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(checked)

    def setHowToDrawEnabled(self, state):
        bottomWidgets = self.bottomWidgets
        for checkbox in self.checkboxes.values():
            checkbox.setDisabled(not state)

    def setNavigateEnabled(self, state):
        bottomWidgets = self.bottomWidgets
        scrollbar = bottomWidgets['navigateScrollbar']
        if state:
            scrollbar.setDisabled(False, applyToCheckbox=False)
        else:
            scrollbar.setDisabled(True, applyToCheckbox=False)

    def setZsliceL0Enabled(self, state):
        bottomWidgets = self.bottomWidgets
        scrollbar = bottomWidgets['zSliceScrollbarLayer0']
        combobox = bottomWidgets['zProjComboboxLayer0']
        if state:
            scrollbar.setDisabled(False, applyToCheckbox=False)
            combobox.setDisabled(False, applyToCheckbox=False)
        else:
            scrollbar.setDisabled(True, applyToCheckbox=False)
            combobox.setDisabled(True, applyToCheckbox=False)

    def setZsliceL1Enabled(self, state):
        bottomWidgets = self.bottomWidgets
        scrollbar = bottomWidgets['zSliceScrollbarLayer1']
        combobox = bottomWidgets['zProjComboboxLayer1']
        if state:
            scrollbar.setDisabled(False, applyToCheckbox=False)
            combobox.setDisabled(False, applyToCheckbox=False)
        else:
            scrollbar.setDisabled(True, applyToCheckbox=False)
            combobox.setDisabled(True, applyToCheckbox=False)

class guiTabControl(QTabWidget):
    def __init__(self, parent=None, logging_func=print):
        super().__init__(parent)

        self.loadedFilename = ''

        self.parametersTab = QScrollArea(self)
        self.parametersQGBox = analysisInputsQGBox(self.parametersTab)
        self.logging_func = logging_func
        containerWidget = QWidget()
        containerLayout = QVBoxLayout()

        buttonsLayout = QHBoxLayout()
        
        self.saveParamsButton = acdc_widgets.savePushButton(
            'Save parameters to configuration file...'
        )
        self.loadPreviousParamsButton = acdc_widgets.browseFileButton(
            'Load parameters from a previous analysis', 
            ext={'Configuration files': ['.ini', '.csv']},
            start_dir=utils.getMostRecentPath(), 
            title='Select analysis parameters file'
        )
        buttonsLayout.addWidget(self.loadPreviousParamsButton)
        buttonsLayout.addWidget(self.saveParamsButton)
        buttonsLayout.addStretch(1)

        containerLayout.addLayout(buttonsLayout)

        containerLayout.addWidget(self.parametersQGBox)
        containerWidget.setLayout(containerLayout)

        self.parametersTab.setWidget(containerWidget)

        self.loadPreviousParamsButton.sigPathSelected.connect(
            self.loadPreviousParams
        )
        self.saveParamsButton.clicked.connect(self.saveParamsFile)

        self.addTab(self.parametersTab, 'Analysis paramenters')
    
    def setValuesFromParams(self, params):
        for section, anchorOptions in self.parametersQGBox.params.items():
            for anchor, options in anchorOptions.items():
                formWidget = options['formWidget']
                try:
                    val = params[section][anchor]['loadedVal']
                except Exception as e:
                    continue
                groupbox = options['groupBox']
                try:
                    groupbox.setChecked(True)
                except Exception as e:
                    pass
                formWidget.setValue(val)
    
    def loadPreviousParams(self, filePath):
        self.logging_func(f'Loading analysis parameters from "{filePath}"...')
        io.addToRecentPaths(os.path.dirname(filePath))
        self.loadedFilename, ext = os.path.splitext(os.path.basename(filePath))
        params = config.analysisInputsParams(filePath)
        self.setValuesFromParams(params)
        
    def saveParamsFile(self):
        if self.loadedFilename:
            entry = self.loadedFilename
        else:
            now = datetime.datetime.now().strftime(r'%Y-%m-%d')
            entry = f'{now}_analysis_parameters'
        txt = (
            'Insert <b>filename</b> for the parameters file.<br><br>'
            'After confirming, you will be asked to <b>choose the folder</b> '
            'where to save the file.'
        )
        while True:
            filenameWindow = acdc_apps.filenameDialog(
                parent=self, title='Insert file name for the parameters file', 
                allowEmpty=False, defaultEntry=entry, ext='.ini', hintText=txt
            )
            filenameWindow.exec_()
            if filenameWindow.cancel:
                return
            
            folder_path = QFileDialog.getExistingDirectory(
                    self, 'Select folder where to save the parameters file', 
                    utils.getMostRecentPath()
                )
            if not folder_path:
                return
            
            filePath = os.path.join(folder_path, filenameWindow.filename)
            if not os.path.exists(filePath):
                break
            else:
                msg = acdc_widgets.myMessageBox(wrapText=False)
                txt = (
                    'The following file already exists:<br><br>'
                    f'<code>{filePath}</code><br><br>'
                    'Do you want to continue?'
                )
                _, noButton, yesButton = msg.warning(
                    self, 'File exists', txt, 
                    buttonsTexts=(
                        'Cancel',
                        'No, let me choose a different path',
                        'Yes, overwrite existing file'
                    )
                )
                if msg.cancel:
                    return
                if msg.clickedButton == yesButton:
                    break
        self.parametersQGBox.saveToIniFile(filePath)

    def addInspectResultsTab(self, posData):
        self.inspectResultsTab = QScrollArea(self)

        self.inspectResultsQGBox = inspectResults(
            posData, parent=self.inspectResultsTab
        )

        self.inspectResultsTab.setWidget(self.inspectResultsQGBox)

        self.removeTab(1)
        self.addTab(self.inspectResultsTab, 'Inspect results')
        self.inspectResultsQGBox.resizeSelector()

class inspectResults(QGroupBox):
    def __init__(self, posData, parent=None):
        QGroupBox.__init__(self, parent)
        self.row = 0

        self.setFont(font)

        runsInfo = self.runsInfo(posData)

        self.H5_selectorLayout = selectSpotsH5FileLayout(
            runsInfo, font=font, parent=self
        )

        self.setLayout(self.H5_selectorLayout)

        self.setMyStyleSheet()

    def runsInfo(self, posData):
        run_nums = posData.validRuns()
        runsInfo = {}
        for run in run_nums:
            h5_files = posData.h5_files(run)
            if not h5_files:
                continue
            runsInfo[run] = h5_files
        return runsInfo

    def setMyStyleSheet(self):
        self.setStyleSheet("""
            QTreeWidget::item:hover {background-color:#E6E6E6;}
            QTreeWidget::item:selected {
                background-color:#CFEB9B;
                color:black;
            }
            QTreeView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
                outline: 0;
            }
            QTreeWidget::item {padding: 5px;}
        """)

    def resizeSelector(self):
        longestText = '3: Spots after goodness-of-peak AND ellipsoid test'
        w = (
            QFontMetrics(self.font())
            .boundingRect(longestText)
            .width()+120
        )
        self.setMinimumWidth(w)

class analysisInputsQGBox(QGroupBox):
    def __init__(self, *args):
        QGroupBox.__init__(self, *args)

        # mainLayout = QGridLayout(self)
        mainLayout = QVBoxLayout()

        font = config.font()

        _params = config.analysisInputsParams()
        self.params = {}
        for section, section_params in _params.items():
            formLayout = widgets.myFormLayout()
            self.params[section] = {}
            groupBox = QGroupBox(section)
            isNotCheckableGroup = (
                section == 'File paths and channels' or section == 'METADATA'
                or section == 'Pre-processing'
            )
            if isNotCheckableGroup:
                groupBox.setCheckable(False)
            else:
                groupBox.setCheckable(True)
            groupBox.setFont(font)
            for row, (anchor, paramValues) in enumerate(section_params.items()):
                self.params[section][anchor] = paramValues.copy()
                widgetFuncName = paramValues.get('formWidgetFunc', None)
                if widgetFuncName is not None:
                    module_name, attr = widgetFuncName.split('.')
                    widgets_module = globals()[module_name]
                    widgetFunc = getattr(widgets_module, attr)
                formWidget = widgets.formWidget(
                    widgetFunc(),
                    anchor=anchor,
                    labelTextLeft=paramValues.get('desc', ''),
                    initialVal=paramValues.get('initialVal', None),
                    stretchWidget=paramValues.get('stretchWidget', True),
                    addInfoButton=paramValues.get('addInfoButton', True),
                    addComputeButton=paramValues.get('addComputeButton', False),
                    addApplyButton=paramValues.get('addApplyButton', False),
                    addBrowseButton=paramValues.get('addBrowseButton', False),
                    addAutoButton=paramValues.get('addAutoButton', False),
                    addEditButton=paramValues.get('addEditButton', False),
                    addLabel=paramValues.get('addLabel', True),
                    valueSetter=paramValues.get('valueSetter'),
                    parent=self
                )
                formWidget.section = section
                formWidget.sigLinkClicked.connect(self.infoLinkClicked)
                self.connectFormWidgetButtons(formWidget, paramValues)
                formLayout.addFormWidget(formWidget, row=row)
                self.params[section][anchor]['widget'] = formWidget.widget
                self.params[section][anchor]['formWidget'] = formWidget
                self.params[section][anchor]['groupBox'] = groupBox

                isGroupChecked = paramValues.get('isSectionInConfig', True)
                groupBox.setChecked(isGroupChecked)

                actions = paramValues.get('actions', None)
                if actions is None:
                    continue

                for action in actions:
                    signal = getattr(formWidget.widget, action[0])
                    signal.connect(getattr(self, action[1]))

            groupBox.setLayout(formLayout)
            mainLayout.addWidget(groupBox)

        mainLayout.addStretch()

        self.setLayout(mainLayout)
        self.updateMinSpotSize()
    
    def _getCallbackFunction(self, callbackFuncPath):
        moduleName, functionName = callbackFuncPath.split('.')
        module = globals()[moduleName]
        return getattr(module, functionName)
    
    def connectFormWidgetButtons(self, formWidget, paramValues):
        editButtonCallback = paramValues.get('editButtonCallback')        
        if editButtonCallback is not None:
            function = self._getCallbackFunction(editButtonCallback)
            formWidget.sigEditClicked.connect(function)

    def infoLinkClicked(self, link):
        try:
            # Stop previously blinking controls, if any
            self.blinker.stopBlinker()
            self.labelBlinker.stopBlinker()
        except Exception as e:
            pass

        try:
            section, anchor, *option = link.split(';')
            formWidget = self.params[section][anchor]['formWidget']
            if option:
                option = option[0]
                widgetToBlink = getattr(formWidget, option)
            else:
                widgetToBlink = formWidget.widget
            self.blinker = utils.widgetBlinker(widgetToBlink)
            label = formWidget.labelLeft
            self.labelBlinker = utils.widgetBlinker(
                label, styleSheetOptions=('color',)
            )
            self.blinker.start()
            self.labelBlinker.start()
        except Exception as e:
            traceback.print_exc()

    def connectActions(self):
        self.pixelWidthWidget.widget.valueChanged.connect(
            self.updateMinSpotSize
        )
        self.pixelHeightWidget.widget.valueChanged.connect(
            self.updateMinSpotSize
        )
        self.voxelDepthWidget.widget.valueChanged.connect(
            self.updateMinSpotSize
        )
        self.emWavelenWidget.widget.valueChanged.connect(
            self.updateMinSpotSize
        )
        self.numApertureWidget.widget.valueChanged.connect(
            self.updateMinSpotSize
        )
        self.yxResolLimitMultiplierWidget.widget.valueChanged.connect(
            self.updateMinSpotSize
        )

    def updateMinSpotSize(self, value=0.0):
        metadata = self.params['METADATA']
        physicalSizeX = metadata['pixelWidth']['widget'].value()
        physicalSizeY = metadata['pixelHeight']['widget'].value()
        physicalSizeZ = metadata['voxelDepth']['widget'].value()
        emWavelen = metadata['emWavelen']['widget'].value()
        NA = metadata['numAperture']['widget'].value()
        zResolutionLimit_um = metadata['zResolutionLimit']['widget'].value()
        yxResolMultiplier = metadata['yxResolLimitMultiplier']['widget'].value()
        zyxMinSize_pxl, zyxMinSize_um = core.calcMinSpotSize(
            emWavelen, NA, physicalSizeX, physicalSizeY, physicalSizeZ,
            zResolutionLimit_um, yxResolMultiplier
        )
        zyxMinSize_pxl_txt = (f'{[round(val, 4) for val in zyxMinSize_pxl]} pxl'
            .replace(']', ')')
            .replace('[', '(')
        )
        zyxMinSize_um_txt = (f'{[round(val, 4) for val in zyxMinSize_um]} μm'
            .replace(']', ')')
            .replace('[', '(')
        )
        spotMinSizeLabels = metadata['spotMinSizeLabels']['widget']
        spotMinSizeLabels.pixelLabel.setText(zyxMinSize_pxl_txt)
        spotMinSizeLabels.umLabel.setText(zyxMinSize_um_txt)
    
    def configIniParams(self):
        ini_params = {}
        for section, section_params in self.params.items():
            ini_params[section] = {}
            for anchor, options in section_params.items():
                groupbox = options['groupBox']
                initalVal = options['initialVal']
                widget = options['widget']
                if groupbox.isCheckable() and not groupbox.isChecked():
                    value = initalVal
                elif isinstance(initalVal, bool):
                    value = widget.isChecked()
                elif isinstance(initalVal, str):
                    try:
                        value = widget.currentText()
                    except AttributeError:
                        value = widget.text()
                elif isinstance(initalVal, float) or isinstance(initalVal, int):
                    value = widget.value()
                
                if not value:
                    continue
                ini_params[section][anchor] = {
                    'desc': options['desc'], 'loadedVal': value
                }
        return ini_params

    def saveToIniFile(self, ini_filepath):
        params = self.configIniParams()
        io.writeConfigINI(params, ini_filepath)
        print('-'*60)
        print(f'Configuration file saved to: "{ini_filepath}"')
        print('*'*60)

    def showInfo(self):
        print(self.sender().label.text())

class spotStyleDock(QDockWidget):
    sigOk = pyqtSignal(int)
    sigCancel = pyqtSignal()

    def __init__(self, title, parent=None):
        super().__init__(title, parent)

        frame = QFrame()

        mainLayout = QVBoxLayout()
        slidersLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        row = 0
        self.transpSlider = widgets.sliderWithSpinBox(title='Opacity')
        self.transpSlider.setMaximum(100)
        slidersLayout.addWidget(self.transpSlider, row, 0)

        row += 1
        self.penWidthSlider = widgets.sliderWithSpinBox(title='Contour thickness')
        self.penWidthSlider.setMaximum(20)
        self.penWidthSlider.setMinimum(1)
        slidersLayout.addWidget(self.penWidthSlider, row, 0)

        row += 1
        self.sizeSlider = widgets.sliderWithSpinBox(title='Size')
        self.sizeSlider.setMaximum(100)
        self.sizeSlider.setMinimum(1)
        slidersLayout.addWidget(self.sizeSlider, row, 0)

        okButton = acdc_widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = acdc_widgets.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)
        
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(slidersLayout)
        mainLayout.addLayout(buttonsLayout)

        frame.setLayout(mainLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.setWidget(frame)
        self.setFloating(True)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )

    def ok_cb(self):
        self.hide()

    def cancel_cb(self):
        self.sigCancel.emit()
        self.hide()

    def show(self):
        QDockWidget.show(self)
        self.resize(int(self.width()*1.5), self.height())
        self.setFocus(True)
        self.activateWindow()


class QDialogMetadata(QBaseDialog):
    def __init__(
            self, SizeT, SizeZ, TimeIncrement,
            PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX,
            ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes, numPos,
            parent=None, font=None, imgDataShape=None, PosData=None,
            fileExt='.tif'
        ):
        self.cancel = True
        self.ask_TimeIncrement = ask_TimeIncrement
        self.ask_PhysicalSizes = ask_PhysicalSizes
        self.imgDataShape = imgDataShape
        self.PosData = PosData
        self.fileExt = fileExt
        super().__init__(parent)
        self.setWindowTitle('Image properties')

        mainLayout = QVBoxLayout()
        loadingSizesGroupbox = QGroupBox()
        loadingSizesGroupbox.setTitle('Parameters for loading')
        metadataGroupbox = QGroupBox()
        metadataGroupbox.setTitle('Image Properties')
        buttonsLayout = QGridLayout()

        loadingParamLayout = QGridLayout()
        row = 0
        loadingParamLayout.addWidget(
            QLabel('Number of Positions to load'), row, 0,
            alignment=Qt.AlignRight
        )
        self.loadSizeS_SpinBox = widgets.QSpinBoxOdd(acceptedValues=(numPos,))
        self.loadSizeS_SpinBox.setMinimum(1)
        self.loadSizeS_SpinBox.setMaximum(numPos)
        self.loadSizeS_SpinBox.setValue(numPos)
        if numPos == 1:
            self.loadSizeS_SpinBox.setDisabled(True)
        self.loadSizeS_SpinBox.setAlignment(Qt.AlignCenter)
        loadingParamLayout.addWidget(self.loadSizeS_SpinBox, row, 1)

        row += 1
        loadingParamLayout.addWidget(
            QLabel('Number of frames to load'), row, 0, alignment=Qt.AlignRight
        )
        self.loadSizeT_SpinBox = widgets.QSpinBoxOdd(acceptedValues=(SizeT,))
        self.loadSizeT_SpinBox.setMinimum(1)
        if ask_SizeT:
            self.loadSizeT_SpinBox.setMaximum(SizeT)
            self.loadSizeT_SpinBox.setValue(SizeT)
            if fileExt != '.h5':
                self.loadSizeT_SpinBox.setDisabled(True)
        else:
            self.loadSizeT_SpinBox.setMaximum(1)
            self.loadSizeT_SpinBox.setValue(1)
            self.loadSizeT_SpinBox.setDisabled(True)
        self.loadSizeT_SpinBox.setAlignment(Qt.AlignCenter)
        loadingParamLayout.addWidget(self.loadSizeT_SpinBox, row, 1)

        row += 1
        loadingParamLayout.addWidget(
            QLabel('Number of z-slices to load'), row, 0,
            alignment=Qt.AlignRight
        )
        self.loadSizeZ_SpinBox = widgets.QSpinBoxOdd(acceptedValues=(SizeZ,))
        self.loadSizeZ_SpinBox.setMinimum(1)
        if SizeZ > 1:
            self.loadSizeZ_SpinBox.setMaximum(SizeZ)
            self.loadSizeZ_SpinBox.setValue(SizeZ)
            if fileExt != '.h5':
                self.loadSizeZ_SpinBox.setDisabled(True)
        else:
            self.loadSizeZ_SpinBox.setMaximum(1)
            self.loadSizeZ_SpinBox.setValue(1)
            self.loadSizeZ_SpinBox.setDisabled(True)
        self.loadSizeZ_SpinBox.setAlignment(Qt.AlignCenter)
        loadingParamLayout.addWidget(self.loadSizeZ_SpinBox, row, 1)

        loadingParamLayout.setColumnMinimumWidth(1, 100)
        loadingSizesGroupbox.setLayout(loadingParamLayout)

        gridLayout = QGridLayout()
        row = 0
        gridLayout.addWidget(
            QLabel('Number of frames (SizeT)'), row, 0, alignment=Qt.AlignRight
        )
        self.SizeT_SpinBox = QSpinBox()
        self.SizeT_SpinBox.setMinimum(1)
        self.SizeT_SpinBox.setMaximum(2147483647)
        if ask_SizeT:
            self.SizeT_SpinBox.setValue(SizeT)
        else:
            self.SizeT_SpinBox.setValue(1)
            self.SizeT_SpinBox.setDisabled(True)
        self.SizeT_SpinBox.setAlignment(Qt.AlignCenter)
        self.SizeT_SpinBox.valueChanged.connect(self.TimeIncrementShowHide)
        gridLayout.addWidget(self.SizeT_SpinBox, row, 1)

        row += 1
        gridLayout.addWidget(
            QLabel('Number of z-slices (SizeZ)'), row, 0, alignment=Qt.AlignRight
        )
        self.SizeZ_SpinBox = QSpinBox()
        self.SizeZ_SpinBox.setMinimum(1)
        self.SizeZ_SpinBox.setMaximum(2147483647)
        self.SizeZ_SpinBox.setValue(SizeZ)
        self.SizeZ_SpinBox.setAlignment(Qt.AlignCenter)
        self.SizeZ_SpinBox.valueChanged.connect(self.SizeZvalueChanged)
        gridLayout.addWidget(self.SizeZ_SpinBox, row, 1)

        row += 1
        self.TimeIncrementLabel = QLabel('Time interval (s)')
        gridLayout.addWidget(
            self.TimeIncrementLabel, row, 0, alignment=Qt.AlignRight
        )
        self.TimeIncrementSpinBox = QDoubleSpinBox()
        self.TimeIncrementSpinBox.setDecimals(7)
        self.TimeIncrementSpinBox.setMaximum(2147483647.0)
        self.TimeIncrementSpinBox.setValue(TimeIncrement)
        self.TimeIncrementSpinBox.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.TimeIncrementSpinBox, row, 1)

        if SizeT == 1 or not ask_TimeIncrement:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()

        row += 1
        self.PhysicalSizeZLabel = QLabel('Physical Size Z (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeZLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeZSpinBox = QDoubleSpinBox()
        self.PhysicalSizeZSpinBox.setDecimals(7)
        self.PhysicalSizeZSpinBox.setMaximum(2147483647.0)
        self.PhysicalSizeZSpinBox.setValue(PhysicalSizeZ)
        self.PhysicalSizeZSpinBox.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.PhysicalSizeZSpinBox, row, 1)

        if SizeZ==1 or not ask_PhysicalSizes:
            self.PhysicalSizeZSpinBox.hide()
            self.PhysicalSizeZLabel.hide()

        row += 1
        self.PhysicalSizeYLabel = QLabel('Physical Size Y (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeYLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeYSpinBox = QDoubleSpinBox()
        self.PhysicalSizeYSpinBox.setDecimals(7)
        self.PhysicalSizeYSpinBox.setMaximum(2147483647.0)
        self.PhysicalSizeYSpinBox.setValue(PhysicalSizeY)
        self.PhysicalSizeYSpinBox.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.PhysicalSizeYSpinBox, row, 1)

        if not ask_PhysicalSizes:
            self.PhysicalSizeYSpinBox.hide()
            self.PhysicalSizeYLabel.hide()

        row += 1
        self.PhysicalSizeXLabel = QLabel('Physical Size X (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeXLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeXSpinBox = QDoubleSpinBox()
        self.PhysicalSizeXSpinBox.setDecimals(7)
        self.PhysicalSizeXSpinBox.setMaximum(2147483647.0)
        self.PhysicalSizeXSpinBox.setValue(PhysicalSizeX)
        self.PhysicalSizeXSpinBox.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.PhysicalSizeXSpinBox, row, 1)

        if not ask_PhysicalSizes:
            self.PhysicalSizeXSpinBox.hide()
            self.PhysicalSizeXLabel.hide()

        self.SizeZvalueChanged(SizeZ)

        gridLayout.setColumnMinimumWidth(1, 100)
        metadataGroupbox.setLayout(gridLayout)

        if numPos == 1:
            okTxt = 'Apply only to this Position'
        else:
            okTxt = 'Ok for loaded Positions'
        okButton = acdc_widgets.okPushButton(okTxt)
        okButton.setToolTip(
            'Save metadata only for current positionh'
        )
        okButton.setShortcut(Qt.Key_Enter)
        self.okButton = okButton

        if ask_TimeIncrement or ask_PhysicalSizes:
            okAllButton = QPushButton('Apply to ALL Positions')
            okAllButton.setToolTip(
                'Update existing Physical Sizes, Time interval, cell volume (fl), '
                'cell area (um^2), and time (s) for all the positions '
                'in the experiment folder.'
            )
            self.okAllButton = okAllButton

            selectButton = QPushButton('Select the Positions to be updated')
            selectButton.setToolTip(
                'Ask to select positions then update existing Physical Sizes, '
                'Time interval, cell volume (fl), cell area (um^2), and time (s)'
                'for selected positions.'
            )
            self.selectButton = selectButton
        else:
            self.okAllButton = None
            self.selectButton = None
            okButton.setText('Ok')

        cancelButton = acdc_widgets.cancelPushButton('Cancel')

        buttonsLayout.addWidget(okButton, 0, 0)
        if ask_TimeIncrement or ask_PhysicalSizes:
            buttonsLayout.addWidget(okAllButton, 0, 1)
            buttonsLayout.addWidget(selectButton, 1, 0)
            buttonsLayout.addWidget(cancelButton, 1, 1)
        else:
            buttonsLayout.addWidget(cancelButton, 0, 1)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        if imgDataShape is not None:
            label = QLabel(html_func.paragraph(
                    f'<i>Image data shape</i> = <b>{imgDataShape}</b><br>'
                )
            )
            mainLayout.addWidget(label, alignment=Qt.AlignCenter)
        mainLayout.addWidget(loadingSizesGroupbox)
        mainLayout.addStretch(1)
        mainLayout.addSpacing(10)
        mainLayout.addWidget(metadataGroupbox)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        if ask_TimeIncrement or ask_PhysicalSizes:
            okAllButton.clicked.connect(self.ok_cb)
            selectButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.setLayout(mainLayout)

    def SizeZvalueChanged(self, val):
        if len(self.imgDataShape) < 3:
            return
        if val > 1 and self.imgDataShape is not None:
            maxSizeZ = self.imgDataShape[-3]
            self.SizeZ_SpinBox.setMaximum(maxSizeZ)
            if self.fileExt == '.h5':
                self.loadSizeZ_SpinBox.setDisabled(False)
        else:
            self.SizeZ_SpinBox.setMaximum(2147483647)
            self.loadSizeZ_SpinBox.setValue(1)
            self.loadSizeZ_SpinBox.setDisabled(True)

        if not self.ask_PhysicalSizes:
            return
        if val > 1:
            self.PhysicalSizeZSpinBox.show()
            self.PhysicalSizeZLabel.show()
        else:
            self.PhysicalSizeZSpinBox.hide()
            self.PhysicalSizeZLabel.hide()

    def TimeIncrementShowHide(self, val):
        if not self.ask_TimeIncrement:
            return
        if val > 1:
            self.TimeIncrementSpinBox.show()
            self.TimeIncrementLabel.show()
            if self.fileExt == '.h5':
                self.loadSizeT_SpinBox.setDisabled(False)
        else:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()
            self.loadSizeT_SpinBox.setDisabled(True)
            self.loadSizeT_SpinBox.setValue(1)

    def ok_cb(self, event):
        self.cancel = False
        self.SizeT = self.SizeT_SpinBox.value()
        self.SizeZ = self.SizeZ_SpinBox.value()

        self.loadSizeS = self.loadSizeS_SpinBox.value()
        self.loadSizeT = self.loadSizeT_SpinBox.value()
        self.loadSizeZ = self.loadSizeZ_SpinBox.value()
        self.TimeIncrement = self.TimeIncrementSpinBox.value()
        self.PhysicalSizeX = self.PhysicalSizeXSpinBox.value()
        self.PhysicalSizeY = self.PhysicalSizeYSpinBox.value()
        self.PhysicalSizeZ = self.PhysicalSizeZSpinBox.value()
        valid4D = True
        valid3D = True
        valid2D = True
        if self.imgDataShape is None:
            self.close()
        elif len(self.imgDataShape) == 4:
            T, Z, Y, X = self.imgDataShape
            valid4D = self.SizeT == T and self.SizeZ == Z
        elif len(self.imgDataShape) == 3:
            TZ, Y, X = self.imgDataShape
            valid3D = self.SizeT == TZ or self.SizeZ == TZ
        elif len(self.imgDataShape) == 2:
            valid2D = self.SizeT == 1 and self.SizeZ == 1
        valid = all([valid4D, valid3D, valid2D])
        if not valid4D:
            txt = html_func.paragraph(
                'You loaded <b>4D data</b>, hence the number of frames MUST be '
                f'<b>{T}</b><br> nd the number of z-slices MUST be <b>{Z}</b>.'
                '<br><br> What do you want to do?'
            )
        if not valid3D:
            txt = html_func.paragraph(
                'You loaded <b>3D data</b>, hence either the number of frames is '
                f'<b>{TZ}</b><br> or the number of z-slices can be <b>{TZ}</b>.<br><br>'
                'However, if the number of frames is greater than 1 then the<br>'
                'number of z-slices MUST be 1, and vice-versa.<br><br>'
                'What do you want to do?'
            )

        if not valid2D:
            txt = html_func.paragraph(
                'You loaded <b>2D data</b>, hence the number of frames MUST be <b>1</b> '
                'and the number of z-slices MUST be <b>1</b>.<br><br>'
                'What do you want to do?'
            )

        if not valid:
            msg = acdc_widgets.myMessageBox(self)
            continueButton, cancelButton = msg.warning(
                self, 'Invalid entries', txt,
                buttonsTexts=('Continue', 'Let me correct')
            )
            if msg.clickedButton == cancelButton:
                return

        if self.PosData is not None and self.sender() != self.okButton:
            exp_path = self.PosData.exp_path
            pos_foldernames = natsorted(utils.listdir(exp_path))
            pos_foldernames = [
                pos for pos in pos_foldernames
                if pos.find('Position_')!=-1
                and os.path.isdir(os.path.join(exp_path, pos))
            ]
            if self.sender() == self.selectButton:
                select_folder = io.select_exp_folder()
                select_folder.pos_foldernames = pos_foldernames
                select_folder.QtPrompt(
                    self, pos_foldernames, allow_abort=False, toggleMulti=True
                )
                pos_foldernames = select_folder.selected_pos
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, 'Images')
                ls = utils.listdir(images_path)
                search = [file for file in ls if file.find('metadata.csv')!=-1]
                metadata_df = None
                if search:
                    fileName = search[0]
                    metadata_csv_path = os.path.join(images_path, fileName)
                    metadata_df = pd.read_csv(
                        metadata_csv_path
                        ).set_index('Description')
                if metadata_df is not None:
                    metadata_df.at['TimeIncrement', 'values'] = self.TimeIncrement
                    metadata_df.at['PhysicalSizeZ', 'values'] = self.PhysicalSizeZ
                    metadata_df.at['PhysicalSizeY', 'values'] = self.PhysicalSizeY
                    metadata_df.at['PhysicalSizeX', 'values'] = self.PhysicalSizeX
                    metadata_df.to_csv(metadata_csv_path)

                search = [file for file in ls if file.find('acdc_output.csv')!=-1]
                acdc_df = None
                if search:
                    fileName = search[0]
                    acdc_df_path = os.path.join(images_path, fileName)
                    acdc_df = pd.read_csv(acdc_df_path)
                    yx_pxl_to_um2 = self.PhysicalSizeY*self.PhysicalSizeX
                    vox_to_fl = self.PhysicalSizeY*(self.PhysicalSizeX**2)
                    if 'cell_vol_fl' not in acdc_df.columns:
                        continue
                    acdc_df['cell_vol_fl'] = acdc_df['cell_vol_vox']*vox_to_fl
                    acdc_df['cell_area_um2'] = acdc_df['cell_area_pxl']*yx_pxl_to_um2
                    acdc_df['time_seconds'] = acdc_df['frame_i']*self.TimeIncrement
                    try:
                        acdc_df.to_csv(acdc_df_path, index=False)
                    except PermissionError:
                        err_msg = (
                            'The below file is open in another app '
                            '(Excel maybe?).\n\n'
                            f'{acdc_df_path}\n\n'
                            'Close file and then press "Ok".'
                        )
                        msg = acdc_widgets.myMessageBox()
                        msg.critical(self, 'Permission denied', err_msg)
                        acdc_df.to_csv(acdc_df_path, index=False)

        elif self.sender() == self.selectButton:
            pass

        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

class QDialogCombobox(QBaseDialog):
    def __init__(
            self, title, ComboBoxItems, informativeText,
            CbLabel='Select value:  ', parent=None,
            defaultChannelName=None, iconPixmap=None
        ):
        self.cancel = True
        self.selectedItemText = ''
        self.selectedItemIdx = None
        super().__init__(parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        infoLayout = QHBoxLayout()
        topLayout = QHBoxLayout()
        bottomLayout = QHBoxLayout()

        if iconPixmap is not None:
            label = QLabel()
            # padding: top, left, bottom, right
            # label.setStyleSheet("padding:5px 0px 10px 0px;")
            label.setPixmap(iconPixmap)
            infoLayout.addWidget(label)

        if informativeText:
            infoLabel = QLabel(informativeText)
            infoLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        if CbLabel:
            label = QLabel(CbLabel)
            topLayout.addWidget(label, alignment=Qt.AlignRight)

        combobox = QComboBox()
        combobox.addItems(ComboBoxItems)
        if defaultChannelName is not None and defaultChannelName in ComboBoxItems:
            combobox.setCurrentText(defaultChannelName)
        self.ComboBox = combobox
        topLayout.addWidget(combobox)
        topLayout.setContentsMargins(0, 10, 0, 0)

        okButton = acdc_widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        bottomLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = acdc_widgets.cancelPushButton('Cancel')
        bottomLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(infoLayout)
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)


    def ok_cb(self, event):
        self.cancel = False
        self.selectedItemText = self.ComboBox.currentText()
        self.selectedItemIdx = self.ComboBox.currentIndex()
        self.close()

class QDialogListbox(QBaseDialog):
    def __init__(
            self, title, text, items, moreButtonFuncText='Cancel',
            multiSelection=True, currentItem=None,
            filterItems=(), parent=None
        ):
        self.cancel = True
        super().__init__(parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        bottomLayout = QHBoxLayout()

        label = QLabel(text)

        label.setFont(font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

        if filterItems:
            filteredItems = []
            for item in items:
                for textToFind in filterItems:
                    if item.find(textToFind) != -1:
                        filteredItems.append(item)
            items = filteredItems

        listBox = acdc_widgets.listWidget()
        listBox.setFont(font)
        listBox.addItems(items)
        if multiSelection:
            listBox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        else:
            listBox.setSelectionMode(QAbstractItemView.SingleSelection)
        if currentItem is None:
            listBox.setCurrentRow(0)
        else:
            listBox.setCurrentItem(currentItem)
        self.listBox = listBox
        listBox.itemDoubleClicked.connect(self.ok_cb)
        topLayout.addWidget(listBox)

        okButton = acdc_widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        bottomLayout.addWidget(okButton, alignment=Qt.AlignRight)

        moreButton = QPushButton(moreButtonFuncText)
        # cancelButton.setShortcut(Qt.Key_Escape)
        bottomLayout.addWidget(moreButton, alignment=Qt.AlignLeft)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        if moreButtonFuncText.lower().find('cancel') != -1:
            moreButton.clicked.connect(self.cancel_cb)
        elif moreButtonFuncText.lower().find('browse') != -1:
            moreButton.clicked.connect(self.browse)

        listBox.setFocus(True)
        self.setMyStyleSheet()

    def setMyStyleSheet(self):
        self.setStyleSheet("""
            QListWidget::item:hover {background-color:#E6E6E6;}
            QListWidget::item:selected {
                background-color:#CFEB9B;
                color:black;
                border-left:none;
                border-top:none;
                border-right:none;
                border-bottom:none;
            }
            QListWidget::item {padding: 5px;}
            QListView  {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
                outline: 0;
            }
        """)

    def browse(self, event):
        pass

    def ok_cb(self, event):
        self.cancel = False
        selectedItems = self.listBox.selectedItems()
        self.selectedItems = selectedItems
        self.selectedItemsText = [item.text() for item in selectedItems]
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.selectedItemsText = None
        self.close()

class selectedPathsSummaryDialog(acdc_apps.TreeSelectorDialog):
    def __init__(self) -> None:
        super().__init__()

class selectPathsSpotmax(QBaseDialog):
    def __init__(self, paths, homePath, parent=None, app=None):
        super().__init__(parent)

        self.cancel = True

        self.selectedPaths = []
        self.paths = paths
        runs = sorted(list(self.paths.keys()))
        self.runs = runs
        self.isCtrlDown = False
        self.isShiftDown = False

        self.setWindowTitle('Select experiments to load/analyse')

        infoLabel = QLabel()
        text = (
            'Select <b>one or more folders</b> to load<br><br>'
            '<code>Click</code> on experiment path <i>to select all positions</i><br>'
            '<code>Ctrl+Click</code> <i>to select multiple items</i><br>'
            '<code>Shift+Click</code> <i>to select a range of items</i><br>'
            '<code>Ctrl+A</code> <i>to select all</i><br>'
        )
        htmlText = html_func.paragraph(text, center=True)
        infoLabel.setText(htmlText)

        runNumberLayout = QHBoxLayout()
        runNumberLabel = QLabel()
        text = 'Number of pos. analysed for run number: '
        htmlText = html_func.paragraph(text)
        runNumberLabel.setText(htmlText)
        runNumberCombobox = QComboBox()
        runNumberCombobox.addItems([f'  {r}  ' for r in runs])
        runNumberCombobox.setCurrentIndex(len(runs)-1)
        self.runNumberCombobox = runNumberCombobox
        showAnalysisTableButton = widgets.showPushButton(
            'Show analysis inputs for selected run and selected experiment'
        )

        runNumberLayout.addStretch(1)
        runNumberLayout.addWidget(runNumberLabel, alignment=Qt.AlignRight)
        runNumberLayout.addWidget(runNumberCombobox, alignment=Qt.AlignRight)
        runNumberLayout.addWidget(showAnalysisTableButton)
        runNumberLayout.addStretch(1)

        checkBoxesLayout = QHBoxLayout()
        hideSpotCountCheckbox = QCheckBox('Hide fully spotCOUNTED')
        hideSpotSizeCheckbox = QCheckBox('Hide fully spotSIZED')
        checkBoxesLayout.addStretch(1)
        checkBoxesLayout.addWidget(
            hideSpotCountCheckbox, alignment=Qt.AlignCenter
        )
        checkBoxesLayout.addWidget(
            hideSpotSizeCheckbox, alignment=Qt.AlignCenter
        )
        checkBoxesLayout.addStretch(1)
        self.hideSpotCountCheckbox = hideSpotCountCheckbox
        self.hideSpotSizeCheckbox = hideSpotSizeCheckbox

        pathSelector = QTreeWidget()
        self.pathSelector = pathSelector
        pathSelector.setSelectionMode(QAbstractItemView.ExtendedSelection)
        pathSelector.setHeaderHidden(True)
        homePath = pathlib.Path(homePath)
        self.homePath = homePath
        self.populatePathSelector()

        buttonsLayout = QHBoxLayout()
        cancelButton = acdc_widgets.cancelPushButton('Cancel')
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)

        okButton = acdc_widgets.okPushButton('Ok')
        # okButton.setShortcut(Qt.Key_Enter)
        buttonsLayout.addWidget(okButton, alignment=Qt.AlignRight)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addLayout(runNumberLayout)
        runNumberLayout.setContentsMargins(0, 0, 0, 10)
        mainLayout.addLayout(checkBoxesLayout)
        mainLayout.addWidget(pathSelector)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        self.setLayout(mainLayout)

        pathSelector.itemSelectionChanged.connect(self.selectAllChildren)
        hideSpotCountCheckbox.stateChanged.connect(self.hideSpotCounted)
        hideSpotSizeCheckbox.stateChanged.connect(self.hideSpotSized)
        runNumberCombobox.currentIndexChanged.connect(self.updateRun)
        showAnalysisTableButton.clicked.connect(self.showAnalysisInputsTable)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        if app is not None:
            app.focusChanged.connect(self.on_focusChanged)

        self.pathSelector.setFocus(True)

        self.setFont(font)

        self.setMyStyleSheet()

    def on_focusChanged(self):
        self.isCtrlDown = False
        self.isShiftDown = False

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.pathSelector.clearSelection()
        elif ev.key() == Qt.Key_Control:
            self.isCtrlDown = True
        elif ev.key() == Qt.Key_Shift:
            self.isShiftDown = True

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Control:
            self.isCtrlDown = False
        elif ev.key() == Qt.Key_Shift:
            self.isShiftDown = False

    def selectAllChildren(self, label=None):
        self.pathSelector.itemSelectionChanged.disconnect()
        if label is not None:
            if not self.isCtrlDown and not self.isShiftDown:
                self.pathSelector.clearSelection()
            label.item.setSelected(True)
            if self.isShiftDown:
                selectionStarted = False
                it = QTreeWidgetItemIterator(self.pathSelector)
                while it:
                    item = it.value()
                    if item is None:
                        break
                    if item.isSelected():
                        selectionStarted = not selectionStarted
                    if selectionStarted:
                        item.setSelected(True)
                    it += 1

        for item in self.pathSelector.selectedItems():
            if item.parent() is None:
                for i in range(item.childCount()):
                    item.child(i).setSelected(True)
        self.pathSelector.itemSelectionChanged.connect(self.selectAllChildren)

    def showAnalysisInputsTable(self):
        idx = self.runNumberCombobox.currentIndex()
        run = self.runs[idx]

        selectedItems = self.pathSelector.selectedItems()

        if not selectedItems:
            self.warnNoPathSelected()
            return

        doc = QTextDocument()
        item = selectedItems[0]
        label = self.pathSelector.itemWidget(item, 0)
        doc.setHtml(label.text())
        plainText = doc.toPlainText()
        parent = item.parent()
        if parent is None:
            relPath1 = re.findall('...(.+) \(', plainText)[0]
            relPath1 = pathlib.Path(relPath1)
            relPath = pathlib.Path(*relPath1.parts[2:])
            if str(relPath) == '.':
                relPath = ''
            exp_path = os.path.join(self.homePath, relPath)

            selectedRunPaths = self.paths[run]
            df = selectedRunPaths[os.path.normpath(exp_path)].get('analysisInputs')
        else:
            posFoldername = re.findall('(.+) \(', plainText)[0]
            parentLabel = label = self.pathSelector.itemWidget(parent, 0)
            doc.setHtml(parentLabel.text())
            parentText = doc.toPlainText()
            relPath1 = re.findall('...(.+) \(', parentText)[0]
            relPath1 = pathlib.Path(relPath1)
            relPath = pathlib.Path(*relPath1.parts[2:])
            relPath1 = relPath / posFoldername
            exp_path = self.homePath / relPath / posFoldername
            spotmaxOutPath = exp_path / 'spotMAX_output'
            if os.path.exists(spotmaxOutPath):
                df = io.expFolderScanner().loadAnalysisInputs(spotmaxOutPath, run)
            else:
                df = None

        if df is None:
            self.warnAnalysisInputsNone(exp_path, run)
            return

        title = f'Analysis inputs table'
        infoText = html_func.paragraph(
            f'Analysis inputs used to analyse <b>run number {run}</b> '
            f'of experiment:<br>"{relPath1}"<br>'
        )
        self.analysisInputsTableWin = pdDataFrameWidget(
            df.reset_index(), title=title, infoText=infoText, parent=self
        )
        self.analysisInputsTableWin.show()

    def updateRun(self, idx):
        self.pathSelector.clear()
        self.populatePathSelector()
        self.resizeSelector()

    def populatePathSelector(self):
        addSpotCounted = not self.hideSpotCountCheckbox.isChecked()
        addSpotSized = not self.hideSpotSizeCheckbox.isChecked()
        pathSelector = self.pathSelector
        idx = self.runNumberCombobox.currentIndex()
        run = self.runs[idx]
        selectedRunPaths = self.paths[run]
        relPathItem = None
        posItem = None
        for exp_path, expInfo in selectedRunPaths.items():
            exp_path = pathlib.Path(exp_path)
            rel = exp_path.relative_to(self.homePath)
            if str(rel) == '.':
                rel = ''
            relPath = (
                f'...{self.homePath.parent.name}{os.path.sep}'
                f'{self.homePath.name}{os.path.sep}{rel}'
            )

            numPosSpotCounted = expInfo['numPosSpotCounted']
            numPosSpotSized = expInfo['numPosSpotSized']
            posFoldernames = expInfo['posFoldernames']
            totPos = len(posFoldernames)
            if numPosSpotCounted < totPos and numPosSpotCounted>0:
                nPSCtext = f'N. of spotCOUNTED pos. = {numPosSpotCounted}'
            elif numPosSpotCounted>0:
                nPSCtext = f'All pos. spotCOUNTED'
                if not addSpotCounted:
                    continue
            else:
                nPSCtext = 'Never spotCOUNTED'

            if numPosSpotSized < totPos and numPosSpotSized>0:
                nPSStext = f'Number of spotSIZED pos. = {numPosSpotSized}'
            elif numPosSpotSized>0:
                nPSStext = f'All pos. spotSIZED'
                if not addSpotSized:
                    continue
            elif numPosSpotCounted>0:
                nPSStext = 'NONE of the pos. spotSIZED'
            else:
                nPSStext = 'Never spotSIZED'

            relPathItem = QTreeWidgetItem()
            pathSelector.addTopLevelItem(relPathItem)
            relPathLabel = acdc_widgets.QClickableLabel()
            relPathLabel.item = relPathItem
            relPathLabel.clicked.connect(self.selectAllChildren)
            relPathText = f'{relPath} (<i>{nPSCtext}, {nPSStext}</i>)'
            relPathLabel.setText(html_func.paragraph(relPathText))
            pathSelector.setItemWidget(relPathItem, 0, relPathLabel)

            for pos in posFoldernames:
                posInfo = expInfo[pos]
                isPosSpotCounted = posInfo['isPosSpotCounted']
                isPosSpotSized = posInfo['isPosSpotSized']
                posText = pos
                if isPosSpotCounted and isPosSpotSized:
                    posText = f'{posText} (spotCOUNTED, spotSIZED)'
                    if not addSpotSized or not addSpotCounted:
                        continue
                elif isPosSpotCounted:
                    posText = f'{posText} (spotCOUNTED, NOT spotSIZED)'
                    if not addSpotCounted:
                        continue
                else:
                    posText = f'{posText} (NOT spotCOUNTED, NOT spotSIZED)'
                posItem = QTreeWidgetItem()
                posLabel = acdc_widgets.QClickableLabel()
                posLabel.item = posItem
                posLabel.clicked.connect(self.selectAllChildren)
                posLabel.setText(html_func.paragraph(posText))
                relPathItem.addChild(posItem)
                pathSelector.setItemWidget(posItem, 0, posLabel)
        if relPathItem is not None and len(selectedRunPaths) == 1:
            relPathItem.setExpanded(True)

    def warnAnalysisInputsNone(self, exp_path, run):
        text = (
            f'The selected experiment "{exp_path}" '
            f'does not have the <b>"{run}_analysis_inputs.csv"</b> file.<br><br>'
            'Sorry about that.'
        )
        msg = acdc_widgets.myMessageBox()
        msg.addShowInFileManagerButton(exp_path)
        msg.warning(
            self, 'Analysis inputs not found!',
            html_func.paragraph(text)
        )

    def setMyStyleSheet(self):
        self.setStyleSheet("""
            QTreeWidget::item:hover {background-color:#E6E6E6;}
            QTreeWidget::item:selected {background-color:#CFEB9B;}
            QTreeWidget::item:selected {color:black;}
            QTreeView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
                outline: 0;
            }
            QTreeWidget::item {padding: 5px;}
        """)

    def ok_cb(self, checked=True):
        selectedItems = self.pathSelector.selectedItems()
        doc = QTextDocument()
        for item in selectedItems:
            label = self.pathSelector.itemWidget(item, 0)
            doc.setHtml(label.text())
            plainText = doc.toPlainText()
            parent = item.parent()
            if parent is None:
                continue
            try:
                posFoldername = re.findall('(.+) \(', plainText)[0]
            except IndexError:
                posFoldername = plainText
            parentLabel = label = self.pathSelector.itemWidget(parent, 0)
            doc.setHtml(parentLabel.text())
            parentText = doc.toPlainText()
            relPath = re.findall('...(.+) \(', parentText)[0]
            relPath = pathlib.Path(relPath)
            relPath = pathlib.Path(*relPath.parts[2:])
            absPath = self.homePath / relPath / posFoldername
            imagesPath = absPath / 'Images'
            self.selectedPaths.append(imagesPath)

        doClose = True
        if not self.selectedPaths:
            doClose = self.warningNotPathsSelected()

        if doClose:
            self.close()

    def warnNoPathSelected(self):
        text = (
            'You didn\'t select <b>any experiment path!</b><br><br>'
            'To visualize the analysis inputs I need to know '
            'the experiment path you want me to show you.<br><br>'
            '<i>Note that if you select multiple experiments I will show you '
            'only the first one that you selected.</i>'
        )
        msg = acdc_widgets.myMessageBox()
        msg.warning(
            self, 'No path selected!', html_func.paragraph(text)
        )

    def warningNotPathsSelected(self):
        text = (
            '<b>You didn\'t select any path!</b> Do you want to cancel loading data?'
        )
        msg = acdc_widgets.myMessageBox()
        doClose, _ = msg.warning(
            self, 'No paths selected!', html_func.paragraph(text),
            buttonsTexts=(' Yes ', 'No')
        )
        return msg.clickedButton==doClose

    def cancel_cb(self, event):
        self.close()

    def hideSpotCounted(self, state):
        self.pathSelector.clear()
        self.populatePathSelector()

    def hideSpotSized(self, state):
        self.pathSelector.clear()
        self.populatePathSelector()

    def resizeSelector(self):
        w = 0
        for i in range(self.pathSelector.topLevelItemCount()):
            item = self.pathSelector.topLevelItem(i)
            label = self.pathSelector.itemWidget(item, 0)
            doc = QTextDocument()
            doc.setHtml(label.text())
            plainText = doc.toPlainText()
            currentW = label.fontMetrics().boundingRect(plainText).width()+60
            if currentW > w:
                w = currentW

        self.pathSelector.setMinimumWidth(w)

    def show(self, block=False):
        super().show(block=False)
        self.resizeSelector()
        if block:
            super().show(block=True)

class DataFrameModel(QtCore.QAbstractTableModel):
    # https://stackoverflow.com/questions/44603119/how-to-display-a-pandas-data-frame-with-pyqt5-pyside2
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.pyqtProperty(pd.DataFrame, fget=dataFrame,
                                    fset=setDataFrame)

    @QtCore.pyqtSlot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int,
                   orientation: QtCore.Qt.Orientation,
                   role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() \
            and 0 <= index.column() < self.columnCount()):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles

class pdDataFrameWidget(QMainWindow):
    def __init__(self, df, title='Table', infoText='', parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(title)

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        layout = QVBoxLayout()

        if infoText:
            infoLabel = QLabel(infoText)
            infoLabel.setAlignment(Qt.AlignCenter)
            layout.addWidget(infoLabel)

        self.tableView = QTableView(self)
        layout.addWidget(self.tableView)
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        for i in range(len(df.columns)):
            self.tableView.resizeColumnToContents(i)
        mainContainer.setLayout(layout)

    def updateTable(self, df):
        if df is None:
            df = self.parent.getBaseCca_df()
        df = df.reset_index()
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        for i in range(len(df.columns)):
            self.tableView.resizeColumnToContents(i)

    def show(self, maxWidth=1024):
        QMainWindow.show(self)


        width = self.tableView.verticalHeader().width() + 28
        for j in range(self.tableView.model().columnCount()):
            width += self.tableView.columnWidth(j) + 4

        height = self.tableView.horizontalHeader().height() + 4
        h = height + (self.tableView.rowHeight(0) + 4)*15
        w = width if width<maxWidth else maxWidth
        self.setGeometry(100, 100, w, h)

        # Center window
        parent = self.parent
        if parent is not None:
            # Center the window on main window
            mainWinGeometry = parent.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinCenterX = int(mainWinLeft + mainWinWidth/2)
            mainWinCenterY = int(mainWinTop + mainWinHeight/2)
            winGeometry = self.geometry()
            winWidth = winGeometry.width()
            winHeight = winGeometry.height()
            winLeft = int(mainWinCenterX - winWidth/2)
            winRight = int(mainWinCenterY - winHeight/2)
            self.move(winLeft, winRight)

    def closeEvent(self, event):
        self.parent.ccaTableWin = None

class selectSpotsH5FileDialog(QBaseDialog):
    def __init__(self, runsInfo, parent=None, app=None):
        QDialog.__init__(self, parent)

        self.setWindowTitle('Select analysis to load')

        self.parent = parent
        self.app = app
        self.runsInfo = runsInfo
        self.selectedFile = None

        self.setFont(font)

        mainLayout = selectSpotsH5FileLayout(
            runsInfo, font=font, parent=self, app=app
        )

        buttonsLayout = QHBoxLayout()
        okButton = acdc_widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        buttonsLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = acdc_widgets.cancelPushButton('Cancel')
        buttonsLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        buttonsLayout.setContentsMargins(0, 20, 0, 0)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.mainLayout = mainLayout
        self.setLayout(mainLayout)

        self.setMyStyleSheet()

    def setMyStyleSheet(self):
        self.setStyleSheet("""
            QTreeWidget::item:hover {background-color:#E6E6E6;}
            QTreeWidget::item:selected {
                background-color:#CFEB9B;
                color:black;
            }
            QTreeView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
                outline: 0;
            }
            QTreeWidget::item {padding: 5px;}
        """)

    def ok_cb(self, checked=True):
        selectedItems = self.mainLayout.treeSelector.selectedItems()
        if not selectedItems:
            doClose = self.warningNoFilesSelected()
            if doClose:
                self.close()
            return
        self.cancel = False
        selectedItem = selectedItems[0]
        runItem = selectedItem.parent()
        runNumber = int(re.findall('(\d+)', runItem.text(0))[0])
        idx = selectedItem.parent().indexOfChild(selectedItem)
        self.selectedFile = self.runsInfo[runNumber][idx]
        self.close()

    def warningNoFilesSelected(self):
        text = (
            'You didn\'t select <b>any analysis run!</b><br><br>'
            'Do you want to cancel the process?'
        )
        msg = acdc_widgets.myMessageBox()
        doClose, _ = msg.warning(
            self, 'No files selected!', html_func.paragraph(text),
            buttonsTexts=(' Yes ', 'No')
        )
        return msg.clickedButton==doClose

    def cancel_cb(self, checked=True):
        self.close()

    def resizeSelector(self):
        longestText = '3: Spots after goodness-of-peak AND ellipsoid test'
        w = (
            QFontMetrics(self.font())
            .boundingRect(longestText)
            .width()+120
        )
        self.mainLayout.treeSelector.setMinimumWidth(w)

    def show(self, block=False):
        super().show(block=False)
        self.resizeSelector()
        if block:
            super().show(block=True)

class selectSpotsH5FileLayout(QVBoxLayout):
    def __init__(self, runsInfo, font=None, parent=None, app=None):
        super().__init__(parent)
        self.runsInfo = runsInfo
        self.selectedFile = None
        self.font = font

        infoLabel = QLabel()
        text = 'Select which analysis to load <br>'
        htmlText = html_func.paragraph(text)
        infoLabel.setText(htmlText)

        treeSelector = QTreeWidget()
        self.treeSelector = treeSelector
        treeSelector.setHeaderHidden(True)
        self.populateSelector()

        self.addWidget(infoLabel, alignment=Qt.AlignCenter)
        self.addWidget(treeSelector)
        treeSelector.itemClicked.connect(self.expandTopLevel)

        treeSelector.setFocus(True)

    def populateSelector(self):
        for run, files in self.runsInfo.items():
            runItem = QTreeWidgetItem(self.treeSelector)
            runItem.setText(0, f'Analysis run number {run}')
            if self.font is not None:
                runItem.setFont(0, self.font)
            self.treeSelector.addTopLevelItem(runItem)
            for file in files:
                if file.find('0_Orig_data') != -1:
                    txt = '0: All detected spots'
                elif file.find('1_ellip_test') != -1:
                    txt = '1: Spots after ellipsoid test'
                elif file.find('2_p-_test') != -1:
                    txt = '2: Spots after goodness-of-peak test'
                elif file.find('3_p-_ellip_test') != -1:
                    txt = '3: Spots after goodness-of-peak AND ellipsoid test'
                elif file.find('4_spotFIT') != -1:
                    txt = '4: Spots after size test (spotFIT)'
                fileItem = QTreeWidgetItem(runItem)
                fileItem.setText(0, txt)
                if self.font is not None:
                    fileItem.setFont(0, self.font)
                runItem.addChild(fileItem)

    def expandTopLevel(self, item):
        if item.parent() is None:
            item.setExpanded(True)
            item.setSelected(False)

def getSelectedExpPaths(utilityName, parent=None):
    msg = acdc_widgets.myMessageBox()
    txt = html_func.paragraph("""
        After you click "Ok" on this dialog you will be asked
        to <b>select the experiment folders</b>, one by one.<br><br>
        Next, you will be able to <b>choose specific Positions</b>
        from each selected experiment.
    """)
    msg.information(
        parent, f'{utilityName}', txt,
        buttonsTexts=('Cancel', 'Ok')
    )
    if msg.cancel:
        return

    expPaths = {}
    mostRecentPath = utils.getMostRecentPath()
    while True:
        exp_path = QFileDialog.getExistingDirectory(
            parent, 'Select experiment folder containing Position_n folders',
            mostRecentPath
        )
        if not exp_path:
            break
        io.addToRecentPaths(exp_path)
        pathScanner = io.expFolderScanner(homePath=exp_path)
        _exp_paths = pathScanner.getExpPathsWithPosFoldernames()
        
        expPaths = {**expPaths, **_exp_paths}
        mostRecentPath = exp_path
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph("""
            Do you want to select <b>additional experiment folders</b>?
        """)
        noButton, yesButton = msg.question(
            parent, 'Select additional experiments?', txt,
            buttonsTexts=('No', 'Yes')
        )
        if msg.clickedButton == noButton:
            break
    
    if not expPaths:
        return

    multiplePos = any([len(posFolders) > 1 for posFolders in expPaths.values()])

    if len(expPaths) > 1 or multiplePos:
        # infoPaths = io.getInfoPosStatus(expPaths)
        selectPosWin = acdc_apps.selectPositionsMultiExp(expPaths)
        selectPosWin.exec_()
        if selectPosWin.cancel:
            return
        selectedExpPaths = selectPosWin.selectedPaths
    else:
        selectedExpPaths = expPaths
    
    return selectedExpPaths