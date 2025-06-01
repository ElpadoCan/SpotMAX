import sys
import os
import platform
import datetime
import re
import pathlib
import time
import shutil
import tempfile
import traceback
from pprint import pprint
from functools import partial

import numpy as np
import pandas as pd
from math import floor
from natsort import natsorted
import skimage.draw

from collections import defaultdict

from qtpy import QtCore
from qtpy.QtCore import Qt, Signal, QEventLoop, QPointF, QTimer
from qtpy.QtGui import (
    QFont, QFontMetrics, QTextDocument, QPalette, QColor,
    QIcon, QResizeEvent, QPixmap
)
from qtpy.QtWidgets import (
    QDialog, QComboBox, QVBoxLayout, QHBoxLayout, QLabel, QApplication,
    QPushButton, QPlainTextEdit, QCheckBox, QTreeWidget, QTreeWidgetItem,
    QTreeWidgetItemIterator, QAbstractItemView, QFrame, QFormLayout,
    QMainWindow, QWidget, QTableView, QTextEdit, QGridLayout,
    QSpacerItem, QSpinBox, QDoubleSpinBox, QButtonGroup, QGroupBox,
    QFileDialog, QDockWidget, QTabWidget, QScrollArea, QScrollBar, 
    QRadioButton, QLineEdit
)
from qtpy.compat import getexistingdirectory

import matplotlib
import pyqtgraph as pg

from cellacdc import apps as acdc_apps
from cellacdc import widgets as acdc_widgets
from cellacdc import myutils as acdc_myutils
from cellacdc import html_utils as acdc_html
from cellacdc import load as acdc_load
from cellacdc import _palettes as acdc_palettes

from . import html_func, io, widgets, utils, config
from . import core, features
from . import printl, font
from . import tune, docs
from . import gui_settings_csv_path as settings_csv_path
from . import last_selection_meas_filepath
from . import palettes
from . import prompts
from . import rng
from . import spotmax_path, icon_path
from . import read_version

LINEEDIT_INVALID_ENTRY_STYLESHEET = (
    acdc_palettes.lineedit_invalid_entry_stylesheet()
)

GIST_RAINBOW_CMAP = matplotlib.colormaps['gist_rainbow']
SIX_RGBs_RAINBOW = (
    [round(c*255) for c in GIST_RAINBOW_CMAP(0.0)][:3], 
    [round(c*255) for c in GIST_RAINBOW_CMAP(0.6)][:3], 
    [round(c*255) for c in GIST_RAINBOW_CMAP(0.2)][:3], 
    [round(c*255) for c in GIST_RAINBOW_CMAP(0.8)][:3], 
    [round(c*255) for c in GIST_RAINBOW_CMAP(0.4)][:3], 
    [round(c*255) for c in GIST_RAINBOW_CMAP(1.0)][:3],     
)

class QBaseDialog(acdc_apps.QBaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

class GopFeaturesAndThresholdsDialog(QBaseDialog):
    def __init__(self, parent=None, category='spots'):
        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle(
            f'Features and thresholds for filtering valid {category}')

        mainLayout = QVBoxLayout()

        scrollArea = QScrollArea()
        scrollAreaLayout = QVBoxLayout()
        scrollAreaContainer = QWidget()
        scrollAreaContainer.setContentsMargins(0, 0, 0, 0)
        scrollArea.setWidgetResizable(True)
        self.setFeaturesGroupbox = widgets.GopFeaturesAndThresholdsGroupbox(
            category=category
        )
        scrollAreaLayout.addWidget(self.setFeaturesGroupbox)
        scrollAreaLayout.addStretch(1)
        scrollAreaContainer.setLayout(scrollAreaLayout)
        scrollArea.setWidget(scrollAreaContainer)
        
        width = (
            self.setFeaturesGroupbox.sizeHint().width()
            + scrollArea.verticalScrollBar().sizeHint().width()
            + 10
        )
        scrollArea.setMinimumWidth(width)
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        height = int(
            self.setFeaturesGroupbox.sizeHint().height()*2.5
            + scrollArea.horizontalScrollBar().sizeHint().height()
        )
        scrollArea.setMinimumHeight(height)
        
        mainLayout.addWidget(scrollArea)
        
        clearAllButton = acdc_widgets.eraserPushButton(' Clear all')
        clearAllLayout = QHBoxLayout()
        clearAllLayout.addStretch(1)
        clearAllLayout.addWidget(clearAllButton)
        mainLayout.addLayout(clearAllLayout)
        
        mainLayout.addWidget(QLabel('Current features and ranges expression:'))
        self.textEdit = QPlainTextEdit()
        self.textEdit.setReadOnly(True)
        mainLayout.addWidget(self.textEdit)
        
        mainLayout.addSpacing(20)

        buttonsLayout = acdc_widgets.CancelOkButtonsLayout()
        buttonsLayout.cancelButton.clicked.connect(self.close)
        buttonsLayout.okButton.clicked.connect(self.ok_cb)

        mainLayout.addLayout(buttonsLayout)
        
        mainLayout.setStretch(0, 1)
        mainLayout.setStretch(1, 0)
        mainLayout.setStretch(2, 0)
        mainLayout.setStretch(3, 0)
        mainLayout.setStretch(4, 0)
        mainLayout.setStretch(5, 0)

        self.setLayout(mainLayout)
        
        self.updateExpression()
        self.setFeaturesGroupbox.sigValueChanged.connect(self.updateExpression)
        clearAllButton.clicked.connect(self.clearAll)
    
    def clearAll(self):
        self.setFeaturesGroupbox.clearAll()
        self.textEdit.setPlainText('')
    
    def show(self, block=False) -> None:
        super().show(block=False)
        firstButton = self.setFeaturesGroupbox.selectors[0].selectButton
        featuresNeverSet = firstButton.text().find('Click') != -1
        if featuresNeverSet:
            self.setFeaturesGroupbox.selectors[0].selectButton.click()
        super().show(block=block)
    
    def updateExpression(self):
        expr = self.expression(validate=False)
        self.textEdit.setPlainText(expr)
    
    def expression(self, validate=True):
        exprs = []
        for s, selector in enumerate(self.setFeaturesGroupbox.selectors):
            selectButton = selector.selectButton
            column_name = selectButton.toolTip()
            if not column_name:
                continue
            
            lowValue = selector.lowRangeWidgets.value()
            highValue = selector.highRangeWidgets.value()
            if lowValue is None and highValue is None and validate:
                self.warnRangeNotSelected(selectButton.text())
                return ''
            
            if s > 0:
                logicStatement = selector.logicStatementCombobox.currentText()
                logicStatement = f'{logicStatement} '
            else:
                logicStatement = ''
            
            openParenthesis = selector.openParenthesisCombobox.currentText()
            closeParenthesis = selector.closeParenthesisCombobox.currentText()
            
            expr = (
                f'{logicStatement}{openParenthesis}'
                f'{column_name}, {lowValue}, {highValue}{closeParenthesis}'
            )
            exprs.append(expr)
        
        expr = '\n'.join(exprs)
        return expr
    
    def configIniParam(self):
        expr = self.expression()
        tooltip = f'Features and ranges set:\n\n{expr}'
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
        expr = self.expression()
        if not expr:
            return
        self.cancel = False
        self.close()


class measurementsQGroupBox(QGroupBox):
    def __init__(self, names, parent=None):
        QGroupBox.__init__(self, 'Single cell measurements', parent)
        self.formWidgets = []

        self.setCheckable(True)
        layout = widgets.FormLayout()

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

class guiQuickSettingsGroupbox(QGroupBox):
    sigPxModeToggled = Signal(bool, bool)
    sigChangeFontSize = Signal(int)

    def __init__(self, df_settings, parent=None):
        super().__init__(parent)
        self.setTitle('Quick settings')

        formLayout = QFormLayout()
        formLayout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        formLayout.setFormAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.autoSaveToggle = acdc_widgets.Toggle()
        autoSaveTooltip = (
            'Automatically store a copy of the segmentation data and of '
            'the annotations in the `.recovery` folder after every edit.'
        )
        self.autoSaveToggle.setChecked(True)
        self.autoSaveToggle.setToolTip(autoSaveTooltip)
        autoSaveLabel = QLabel('Autosave')
        autoSaveLabel.setToolTip(autoSaveTooltip)
        formLayout.addRow(autoSaveLabel, self.autoSaveToggle)

        self.highLowResToggle = acdc_widgets.Toggle()
        self.highLowResToggle.setShortcut('w')
        highLowResTooltip = (
            'Resolution of the text annotations. High resolution results '
            'in slower update of the annotations.\n'
            'Not recommended with a number of segmented objects > 500.\n\n'
            'SHORTCUT: "W" key'
        )
        highResLabel = QLabel('High resolution')
        highResLabel.setToolTip(highLowResTooltip)
        self.highLowResToggle.setToolTip(highLowResTooltip)
        formLayout.addRow(highResLabel, self.highLowResToggle)

        self.realTimeTrackingToggle = acdc_widgets.Toggle()
        self.realTimeTrackingToggle.setChecked(True)
        self.realTimeTrackingToggle.setDisabled(True)
        label = QLabel('Real-time tracking')
        label.setDisabled(True)
        self.realTimeTrackingToggle.label = label
        formLayout.addRow(label, self.realTimeTrackingToggle)

        self.pxModeToggle = acdc_widgets.Toggle()
        self.pxModeToggle.setChecked(True)
        pxModeTooltip = (
            'With "Pixel mode" active, the text annotations scales relative '
            'to the object when zooming in/out (fixed size in pixels).\n'
            'This is typically faster to render, but it makes annotations '
            'smaller/larger when zooming in/out, respectively.\n\n'
            'Try activating it to speed up the annotation of many objects '
            'in high resolution mode.\n\n'
            'After activating it, you might need to increase the font size '
            'from the menu on the top menubar `Edit --> Font size`.'
        )
        pxModeLabel = QLabel('Pixel mode')
        self.pxModeToggle.label = pxModeLabel
        pxModeLabel.setToolTip(pxModeTooltip)
        self.pxModeToggle.setToolTip(pxModeTooltip)
        self.pxModeToggle.clicked.connect(self.pxModeToggled)
        formLayout.addRow(pxModeLabel, self.pxModeToggle)

        # Font size
        self.fontSizeSpinBox = acdc_widgets.SpinBox()
        self.fontSizeSpinBox.setMinimum(1)
        self.fontSizeSpinBox.setMaximum(99)
        formLayout.addRow('Font size', self.fontSizeSpinBox) 
        savedFontSize = str(df_settings.at['fontSize', 'value'])
        if savedFontSize.find('pt') != -1:
            savedFontSize = savedFontSize[:-2]
        self.fontSize = int(savedFontSize)
        if 'pxMode' not in df_settings.index:
            # Users before introduction of pxMode had pxMode=False, but now 
            # the new default is True. This requires larger font size.
            self.fontSize = 2*self.fontSize
            df_settings.at['pxMode', 'value'] = 1
            df_settings.to_csv(settings_csv_path)

        self.fontSizeSpinBox.setValue(self.fontSize)
        self.fontSizeSpinBox.editingFinished.connect(self.changeFontSize) 
        self.fontSizeSpinBox.sigUpClicked.connect(self.changeFontSize)
        self.fontSizeSpinBox.sigDownClicked.connect(self.changeFontSize)

        formLayout.addWidget(self.quickSettingsGroupbox)
        formLayout.addStretch(1)

        self.setLayout(formLayout)
    
    def pxModeToggled(self, checked):
        self.sigPxModeToggled.emit(checked, self.highLowResToggle.isChecked())
    
    def changeFontSize(self):
        self.sigChangeFontSize.emit(self.fontSizeSpinBox.value())

class guiTabControl(QTabWidget):
    sigRunAnalysis = Signal(str, bool)
    sigSetMeasurements = Signal()
    sigParametersLoaded = Signal(str)

    def __init__(self, parent=None, logging_func=print):
        super().__init__(parent)

        self.loadedFilename = ''
        self.lastEntry = None
        self.lastSavedIniFilePath = ''
        self.posData = None

        self.parametersTab = QScrollArea(self)
        self.parametersTab.setWidgetResizable(True)
        self.parametersQGBox = ParamsGroupBox(
            parent=self.parametersTab, 
            debug=True,
            logging_func=logging_func,
        )
        self.parametersQGBox.sigLoadMetadataFromAcdc.connect(
            self.loadMetadataFromAcdc
        )
        self.parametersTab.setWidget(self.parametersQGBox)        
        
        self.logging_func = logging_func
        containerWidget = QWidget()
        containerLayout = QVBoxLayout()

        buttonsContainerWidget = QWidget()
        buttonsScrollArea = widgets.InvisibleScrollArea()
        buttonsScrollArea.setWidget(buttonsContainerWidget)
        buttonsScrollArea.setWidgetResizable(True)
        buttonsLayout = QHBoxLayout()
        self.saveParamsButton = acdc_widgets.savePushButton(
            'Save parameters to file...'
        )
        self.loadPreviousParamsButton = acdc_widgets.browseFileButton(
            'Load parameters from previous analysis...', 
            ext={'Configuration files': ['.ini', '.csv']},
            start_dir=acdc_myutils.getMostRecentPath(), 
            title='Select analysis parameters file'
        )
        self.showInFileMangerButton = acdc_widgets.showInFileManagerButton(
            'Browse loaded/saved file'
        )
        self.showInFileMangerButton.setDisabled(True)
        buttonsLayout.addWidget(self.loadPreviousParamsButton)
        buttonsLayout.addWidget(self.showInFileMangerButton)
        buttonsLayout.addWidget(self.saveParamsButton)
        buttonsLayout.addStretch(1)

        self.runSpotMaxButton = widgets.RunSpotMaxButton('  Run analysis...')
        buttonsLayout.addWidget(self.runSpotMaxButton)
        
        buttonsLayout.setContentsMargins(0, 0, 0, 0)
        buttonsContainerWidget.setLayout(buttonsLayout)
        buttonsScrollArea.setFixedHeight(
            buttonsContainerWidget.sizeHint().height()+4)
        
        buttonsBottomLayout = QHBoxLayout()
        
        self.setMeasurementsButton = acdc_widgets.setPushButton(
            'Set measurements to save...'
        )
        buttonsBottomLayout.addWidget(self.setMeasurementsButton)
        buttonsBottomLayout.addStretch(1)
        self.paramsTabButtonsContainerWidget = buttonsContainerWidget

        # containerLayout.addLayout(buttonsLayout)
        containerLayout.addWidget(buttonsScrollArea)
        containerLayout.addWidget(self.parametersTab)
        containerLayout.addLayout(buttonsBottomLayout)
        containerLayout.setStretch(0, 0)
        containerLayout.setStretch(1, 1)
        containerLayout.setStretch(2, 0)
        
        containerWidget.setLayout(containerLayout)

        self.loadPreviousParamsButton.sigPathSelected.connect(
            self.loadPreviousParams
        )
        self.showInFileMangerButton.clicked.connect(
            self.browseLoadedParamsFile
        )
        self.saveParamsButton.clicked.connect(self.saveParamsFile)
        self.runSpotMaxButton.clicked.connect(self.runAnalysis)
        self.setMeasurementsButton.clicked.connect(self.setMeasurementsClicked)

        self.addTab(containerWidget, 'Analysis parameters')
    
    def setLoadedPosData(self, posData: acdc_load.loadData):
        self.posData = posData
    
    def addLeftClickButtons(self, buttonsGroup):
        buttonsGroup.addButton(
            self.inspectResultsTab.editResultsGroupbox.editResultsToggle
        )
        buttonsGroup.addButton(
            self.autoTuneTabWidget.addAutoTunePointsButton
        )
    
    def confirmMeasurementsSet(self):
        self.setMeasurementsButton.setText('Measurements are set. View or edit...')
        QTimer.singleShot(100, self.setMeasurementsButton.confirmAction)
    
    def runAnalysis(self):
        txt = html_func.paragraph("""
            Do you want to <b>save the current parameters</b> 
            to a configuration file?<br><br>
            A configuration file can be used to run the analysis again with 
            same parameters.
        """)
        msg = acdc_widgets.myMessageBox()
        _, yesButton, noButton = msg.question(
            self, 'Save parameters?', txt, 
            buttonsTexts=('Cancel', 'Yes', 'No')
        )
        if msg.cancel:
            return
        if msg.clickedButton == yesButton:
            ini_filepath = self.saveParamsFile()
            if not ini_filepath:
                return
            is_tempinifile = False
        else:
            # Save temp ini file
            ini_filepath, temp_dirpath = io.get_temp_ini_filepath()
            self.parametersQGBox.saveToIniFile(ini_filepath)
            is_tempinifile = True
            if self.lastSavedIniFilePath:
                with open(self.lastSavedIniFilePath, 'r') as ini:
                    saved_ini_text = ini.read()
                with open(ini_filepath, 'r') as ini_temp:
                    temp_ini_text = ini_temp.read()
                if saved_ini_text == temp_ini_text:
                    ini_filepath = self.lastSavedIniFilePath
                    is_tempinifile = False 
        
        cancel, ini_filepath = prompts.informationSpotmaxAnalysisStart(
            ini_filepath, qparent=self
        )
        if cancel:
            try:
                shutil.rmtree(temp_dirpath)
            except Exception as e:
                pass
            return

        self.sigRunAnalysis.emit(ini_filepath, is_tempinifile)
    
    def initState(self, isDataLoaded):
        self.isDataLoaded = isDataLoaded
        self.autoTuneTabWidget.autoTuneGroupbox.setDisabled(not isDataLoaded)
        if isDataLoaded:
            self.autoTuneTabWidget.autoTuningButton.clicked.disconnect()
        else:
            self.autoTuneTabWidget.autoTuningButton.clicked.connect(
                self.warnDataNotLoadedYet
            )
        if isDataLoaded:
            self.autoTuneTabWidget.addAutoTunePointsButton.clicked.disconnect()
        else:
            self.autoTuneTabWidget.addAutoTunePointsButton.clicked.connect(
                self.warnDataNotLoadedYet
            )

    def warnDataNotLoadedYet(self):
        txt = html_func.paragraph("""
            Before computing any of the analysis steps you need to <b>load some 
            image data</b>.<br><br>
            To do so, click on the <code>Open folder</code> button on the left of 
            the top toolbar (Ctrl+O) and choose an experiment folder to load. 
        """)
        msg = acdc_widgets.myMessageBox()
        msg.warning(self, 'Data not loaded', txt)
        self.sender().setChecked(False)
    
    def loadAcdcMetadataDf(self, params, askLoad=True):
        if self.posData is None and askLoad:
            metadata_csv_filepath = acdc_load.askOpenCsvFile(
                title='Select Cell-ACDC _metadata.csv file'
            )
        elif not hasattr(self.posData, 'metadata_csv_path'):
            return None, None
        elif not os.path.exists(self.posData.metadata_csv_path):
            return None, None
        else:
            metadata_csv_filepath = self.posData.metadata_csv_path
            
        if metadata_csv_filepath is None or not metadata_csv_filepath:
            return None, None
            
        acdc_metadata_df = pd.read_csv(
            metadata_csv_filepath, index_col='Description'
        )
        spots_ch_name = io.get_spots_channel_name_from_params(params)
        anchors_mapper = config.ini_metadata_anchor_to_acdc_metadata_mapper(
            acdc_metadata_df, spots_ch_name
        )
        return acdc_metadata_df, anchors_mapper
    
    def loadMetadataFromAcdc(self):
        params = self.parametersQGBox.params
        acdc_metadata_df, anchors_mapper = self.loadAcdcMetadataDf(params)
        if acdc_metadata_df is None:
            return True
        
        section = 'METADATA'
        anchorOptions = self.parametersQGBox.params[section]
        for anchor, options in anchorOptions.items():
            try:
                acdc_desc, dtype = anchors_mapper[anchor]
                acdc_value = dtype(acdc_metadata_df.at[acdc_desc, 'values'])
            except Exception as err:
                continue
            
            formWidget = options['formWidget']
            valueSetter = params[section][anchor].get('valueSetter')
            formWidget.setValue(acdc_value, valueSetter=valueSetter)
        
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph("""
            Metadata loaded!<br>
        """)
        msg.information(self, 'Metadata loaded', txt)
    
    def askDifferentValuesIniParamsAcdcMetadata(
            self, ini_filepath, acdc_metadata_csv_filepath, 
            ini_value, acdc_value, anchor, params
        ):
        desc = params['METADATA'][anchor]['desc']
        html_table = (f"""
<table cellspacing=0 cellpadding=5 width=100% border: 2px solid rgb(140 140 140)>
    <tr>
        <th style="text-align: left; vertical-align: middle;"><b>Parameter</b></th>
        <th><b>Cell-ACDC metadata.csv value</b></th>
        <th><b>Parameters file value</b></th>
    </tr>
    <tr>
        <td>{desc}</td>
        <td style="text-align: center; vertical-align: middle;">{acdc_value}</td>
        <td style="text-align: center; vertical-align: middle;">{ini_value}</td>
    </tr>
</table>
""")
        txt = html_func.paragraph(f"""
    The metadata in the loaded parameters file is <b>different</b> 
    from the metadata in the Cell-ACDC <code>metadata.csv</code> file!<br><br>
    {html_table}<br><br><br>
    How do you want to <b>proceed?</b><br><br>
    Loaded parameters file path:
""")
        msg = acdc_widgets.myMessageBox(wrapText=False)
        
        browseIniFile = acdc_widgets.showInFileManagerButton(
            'Show loaded parameters file in File manager...'
        )
        
        browseAcdcFile = acdc_widgets.showInFileManagerButton(
            'Show Cell-ACDC metadata file in File manager...'
        )
        
        keepMetadataButton = widgets.AcdcLogoPushButton(
            'Keep the metadata from the Cell-ACDC metadata.csv file'
        )
        yesButton = widgets.LoadFromFilePushButton(
            'Load metadata from the parameters file'
        )
        msg.warning(
            self, 'Loaded metadata different from Cell-ACDC values!', txt,
            buttonsTexts=(
                browseIniFile, browseAcdcFile, keepMetadataButton, yesButton
            ),
            commands=(ini_filepath,), showDialog=False
        )
        browseIniFile.clicked.disconnect()
        browseAcdcFile.clicked.disconnect()
        
        browseIniFile.setPathToBrowse(ini_filepath)
        browseAcdcFile.setPathToBrowse(acdc_metadata_csv_filepath)
        msg.exec_()
        return msg.clickedButton == yesButton
    
    def checkLoadedMetadata(self, anchorOptions, params, ini_filepath):
        acdc_metadata_df, anchors_mapper = self.loadAcdcMetadataDf(
            params, askLoad=False
        )
        if acdc_metadata_df is None:
            return True
        
        asked_about_different_values = False
        section = 'METADATA'
        for anchor, options in anchorOptions.items():
            try:
                ini_value = params[section][anchor]['loadedVal']
            except Exception as err:
                ini_value = None
            
            try:
                acdc_desc, dtype = anchors_mapper[anchor]
                acdc_value = dtype(acdc_metadata_df.at[acdc_desc, 'values'])
            except Exception as err:
                acdc_value = None
            
            if ini_value is None and acdc_value is not None:
                option = params[section][anchor]['desc']
                use_acdc_value = self.askMissingMetadataValue(
                    option, acdc_value
                )
                if use_acdc_value:
                    params[section][anchor]['loadedVal'] = acdc_value
                continue
            
            if acdc_value is None or acdc_value == 0:
                continue
            
            if asked_about_different_values:
                continue
            
            if ini_value != acdc_value:
                proceed = self.askDifferentValuesIniParamsAcdcMetadata(
                    ini_filepath, self.posData.metadata_csv_path, 
                    ini_value, acdc_value, anchor, params
                )
                if not proceed:
                    return False
                asked_about_different_values = True
        
        return True
    
    def askMissingMetadataValue(self, missing_option, acdc_value):
        txt = html_func.paragraph(f"""
            The metadata value <code>{missing_option}</code> is <b>missing</b> 
            in the loaded INI parameters file!<br><br>
            However, it was found in the Cell-ACDC <code>metadata.csv</code> 
            file with value <code>{acdc_value}</code><br><br>
            Do you want to use the Cell-ACDC value?
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        noButton, yesButton = msg.warning(
            self, 'Loaded metadata different from Cell-ACDC values!', txt,
            buttonsTexts=(
                'No, I will manually edit the value', 
                'Yes, load the Cell-ACDC value'
            )
        )
        return msg.clickedButton == yesButton
    
    def setValuesFromParams(self, params, ini_params_file_path):
        # Check if we need to add new widgets for sections with addFieldButton
        for section, section_options in params.items():
            for anchor, options in section_options.items():
                if not options.get('addAddFieldButton', False):
                    continue
                
                splitted = anchor.split('_')
                if len(splitted) == 1:
                    continue
                
                parentAnchor = splitted[0]
                fieldIdx = splitted[-1]
                try:
                    fieldIdx = int(fieldIdx)
                except Exception as err:
                    continue
                
                widget_options = (
                    self.parametersQGBox.params[section][parentAnchor]
                )
                formWidget = widget_options['formWidget']                
                formWidget.addField()
        
        for section, anchorOptions in self.parametersQGBox.params.items():
            if section == 'METADATA':
                proceed = self.checkLoadedMetadata(
                    anchorOptions, params, ini_params_file_path
                )
                if not proceed:
                    print('Metadata loading cancelled.')
                    continue
            
            for anchor, options in anchorOptions.items():
                formWidget = options['formWidget']
                if anchor == 'folderPathsToAnalyse':
                    val = config.parse_exp_paths(ini_params_file_path)
                    val = '\n'.join(val)
                else:
                    try:
                        val = params[section][anchor]['loadedVal']
                    except Exception as e:
                        continue
                groupbox = options['groupBox']
                try:
                    groupbox.setChecked(True)
                except Exception as e:
                    pass
                valueSetter = params[section][anchor].get('valueSetter')
                formWidget.setValue(val, valueSetter=valueSetter)
                if formWidget.useEditableLabel:
                    formWidget.labelLeft.setValue(
                        params[section][anchor]['desc']
                    )
        
        self.parametersQGBox.validatePaths()
        self.parametersQGBox.updateMinSpotSize()
        spotsParams = self.parametersQGBox.params['Spots channel']
        spotPredMethodWidget = spotsParams['spotPredictionMethod']['widget']
        try:
            spotPredMethodWidget.nnet_params_from_ini_sections(params)
            spotPredMethodWidget.bioimageio_params_from_ini_sections(params)
            spotPredMethodWidget.spotiflow_params_from_ini_sections(params)
        except Exception as err:
            print(f'[WARNING]: {err}')
            pass
    
    def validateIniFile(self, filePath):
        params = config.getDefaultParams()
        with open(filePath, 'r') as file:
            txt = file.read()
        isAnySectionPresent = any(
            [txt.find(f'[{section}]') != -1 for section in params.keys()]
        )
        if isAnySectionPresent:
            return True
        
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph(""" 
            The loaded INI file does <b>not contain any valid section</b>.<br><br>
            Please double-check that you are loading the correct file.<br><br>
            Loaded file:
        """)
        msg.warning(
            self, 'Invalid INI file', txt, 
            commands=(filePath,), 
            path_to_browse=os.path.dirname(filePath)
        )
        
        return False
    
    def removeAddedFields(self):
        sections = list(self.parametersQGBox.params.keys())
        for section in sections:
            anchorOptions = self.parametersQGBox.params[section]
            anchors = list(anchorOptions.keys())
            for anchor in anchors:
                formWidget = anchorOptions[anchor]['formWidget']
                if hasattr(formWidget, 'delFieldButton'):
                    formWidget.delFieldButton.click()
    
    def loadPreviousParams(self, filePath):
        self.logging_func(f'Loading analysis parameters from "{filePath}"...')
        acdc_myutils.addToRecentPaths(os.path.dirname(filePath))
        self.loadedFilename, ext = os.path.splitext(os.path.basename(filePath))
        proceed = self.validateIniFile(filePath)
        if not proceed:
            return
        self._lastLoadedParamsFilepath = filePath
        self.showInFileMangerButton.setDisabled(False)
        self.removeAddedFields()
        params = config.analysisInputsParams(filePath, cast_dtypes=False)
        self.setValuesFromParams(params, filePath)
        self.parametersQGBox.setSelectedMeasurements(filePath)
        self.showParamsLoadedMessageBox()
        self.sigParametersLoaded.emit(filePath)
        if self.parametersQGBox.selectedMeasurements is None:
            QTimer.singleShot(100, self.loadPreviousParamsButton.confirmAction)
            return
        self.confirmMeasurementsSet()
        QTimer.singleShot(100, self.loadPreviousParamsButton.confirmAction)
    
    def browseLoadedParamsFile(self):
        acdc_myutils.showInExplorer(self._lastLoadedParamsFilepath)
    
    def showParamsLoadedMessageBox(self):
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph("""
            Parameters loaded!<br>
        """)
        msg.information(self, 'Parameters loaded', txt)
    
    def askSetMeasurements(self):
        if self.setMeasurementsButton.text().find('are set.') != -1:
            msg_type = 'warning'
            txt = html_func.paragraph(
                'There are <b>measurements that have previously set</b> '
                'and will be saved along with the parameters.<br><br>'
                'Do you want to edit or view which <b>measurements will be '
                'saved</b>?<br>'
            )
            noText = 'No, save the set measurements'
        else:
            msg_type = 'question'
            txt = html_func.paragraph(
                'Do you want to select which <b>measurements to save?</b><br>'
            )
            noText = 'No, save all the measurements'

        msg = acdc_widgets.myMessageBox(wrapText=False)
        _, noButton, yesButton = getattr(msg, msg_type)(
            self, 'Set measurements?', txt, 
            buttonsTexts=('Cancel', noText, 'Yes, view set measurments.')
        )
        return msg.clickedButton == yesButton, msg.cancel
    
    def setMeasurementsClicked(self):
        parametersGroupBox = self.parametersQGBox
        
        spotsParams = parametersGroupBox.params['Spots channel']
        anchor = 'doSpotFit'
        isSpotFitRequested = spotsParams[anchor]['widget'].isChecked()
        
        win = SetMeasurementsDialog(
            parent=self, 
            selectedMeasurements=parametersGroupBox.selectedMeasurements,
            isSpotFitRequested=isSpotFitRequested
        )
        win.sigOk.connect(self.setSpotmaxMeasurements)
        win.exec_()
        return win.cancel
    
    def setSpotmaxMeasurements(self, selectedMeasurements):
        self.parametersQGBox.selectedMeasurements = selectedMeasurements
        self.confirmMeasurementsSet()
    
    def saveParamsFile(self):
        showSetMeas, cancel = self.askSetMeasurements()
        if cancel:
            return ''
        
        if showSetMeas:
            cancel = self.setMeasurementsClicked()
            if cancel:
                return ''
        
        if self.loadedFilename:
            entry = self.loadedFilename
        elif self.lastEntry is not None:
            entry = self.lastEntry
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
                return ''
            
            self.lastEntry = filenameWindow.entryText
            
            folder_path = QFileDialog.getExistingDirectory(
                self, 'Select folder where to save the parameters file', 
                acdc_myutils.getMostRecentPath()
            )
            if not folder_path:
                return ''
            
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
                    return ''
                if msg.clickedButton == yesButton:
                    break
        
        self._lastLoadedParamsFilepath = filePath
        self.showInFileMangerButton.setDisabled(False)
        self.loadedFilename, ext = os.path.splitext(os.path.basename(filePath))
        self.parametersQGBox.saveToIniFile(filePath)
        self.lastSavedIniFilePath = filePath
        self.savingParamsFileDone(filePath)
        return filePath

    def savingParamsFileDone(self, filePath):
        txt = html_func.paragraph(
            'Parameters file successfully <b>saved</b> at the following path:'
        )
        msg = acdc_widgets.myMessageBox()
        msg.addShowInFileManagerButton(os.path.dirname(filePath))
        msg.information(self, 'Saving done!', txt, commands=(filePath,))
        
    def addInspectResultsTab(self):
        self.inspectResultsTab = InspectEditResultsTabWidget()
        self.addTab(self.inspectResultsTab, 'Inspect and/or edit results')
    
    def addAutoTuneTab(self):
        self.autoTuneTabWidget = AutoTuneTabWidget()
        # self.autoTuneTabWidget.setDisabled(True)
        self.addTab(self.autoTuneTabWidget, 'Tune parameters')

class InspectEditResultsTabWidget(QWidget):
    sigEditResultsToggled = Signal(bool)
    sigSaveEditedResults = Signal(str, str)
    sigComputeFeatures = Signal(str, str, str) 
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        mainLayout = QVBoxLayout()
        
        buttonsContainerWidget = QWidget()
        buttonsScrollArea = widgets.InvisibleScrollArea()
        buttonsScrollArea.setWidget(buttonsContainerWidget)
        buttonsScrollArea.setWidgetResizable(True)
        buttonsLayout = QHBoxLayout()
        
        self.loadAnalysisButton = acdc_widgets.OpenFilePushButton(
            'Load results from previous analysis...'
        )
        buttonsLayout.addWidget(self.loadAnalysisButton)
        
        self.loadRefChDfButton = acdc_widgets.OpenFilePushButton(
            'Load ref. channel features table...'
        )
        buttonsLayout.addWidget(self.loadRefChDfButton)
        buttonsLayout.addStretch(1)
        
        helpButton = acdc_widgets.helpPushButton('Help...')
        buttonsLayout.addWidget(helpButton)
        
        buttonsLayout.setContentsMargins(0, 0, 0, 0)
        buttonsContainerWidget.setLayout(buttonsLayout)
        buttonsScrollArea.setFixedHeight(
            buttonsContainerWidget.sizeHint().height()+4)
        
        scrollArea = QScrollArea(self)
        scrollArea.setWidgetResizable(True)
        
        scrollAreaWidget = QWidget()
        scrollAreaLayout = QVBoxLayout()
        scrollAreaWidget.setLayout(scrollAreaLayout)
        
        self.editResultsGroupbox = EditResultsGropbox(
            parent=self
        )
        
        self.viewFeaturesGroupbox = AutoTuneViewSpotFeatures(
            parent=self, infoText=''
        )
        
        self.viewRefChFeaturesGroupbox = ViewRefChannelFeaturesGroupbox(
            parent=self
        )
        
        scrollAreaLayout.addWidget(self.editResultsGroupbox)
        scrollAreaLayout.addWidget(self.viewFeaturesGroupbox)
        scrollAreaLayout.addWidget(self.viewRefChFeaturesGroupbox)
        scrollAreaLayout.setStretch(0, 0)
        scrollAreaLayout.setStretch(1, 4)
        scrollAreaLayout.setStretch(2, 3)
        scrollArea.setWidget(scrollAreaWidget)
        
        mainLayout.addWidget(buttonsScrollArea)
        mainLayout.addWidget(scrollArea)
        mainLayout.setStretch(0, 0)
        mainLayout.setStretch(1, 1)
        
        self.setLayout(mainLayout)
        
        self.editResultsGroupbox.sigEditResultsToggled.connect(
            self.emitSigEditResultsToggled
        )
        self.editResultsGroupbox.sigSaveEditedResults.connect(
            self.emitSigSaveEditedResults
        )
        self.editResultsGroupbox.sigComputeFeatures.connect(
            self.emitSigComputeFeatures
        )
        helpButton.clicked.connect(self.helpClicked)
        
        self.nameToColMapper = features.feature_names_to_col_names_mapper(
            category='ref. channel objects'
        )
    
    def helpClicked(self):
        acdc_myutils.browse_url(docs.readthedocs_url)
    
    def setLoadedData(self, *args):
        self.editResultsGroupbox.setLoadedData(*args)
    
    def disconnectSignals(self):
        try:
            self.sigEditResultsToggled.disconnect()
            self.sigSaveEditedResults.disconnect()
            self.sigComputeFeatures.disconnect()
        except Exception as err:
            pass
    
    def reinitState(self):
        self.editResultsGroupbox.initState()
        self.disconnectSignals()
    
    def emitSigEditResultsToggled(self, checked):
        self.sigEditResultsToggled.emit(checked)
        
    def emitSigSaveEditedResults(self, *args):
        self.sigSaveEditedResults.emit(*args)

    def emitSigComputeFeatures(self, *args):
        self.sigComputeFeatures.emit(*args)
    
    def setInspectFeatures(self, point_features):
        if point_features is None:
            return
        self.viewFeaturesGroupbox.setFeatures(point_features)       
    
    def setLoadedRefChannelFeaturesFile(self, filename):
        self.viewRefChFeaturesGroupbox.infoLabel.setText(
            f'Loaded file: <code>{filename}</code>'
        )
    
    def resetLoadedRefChannelFeaturesFile(self):
        self.viewRefChFeaturesGroupbox.resetInfoLabel()
    
    def areFeaturesSelected(self):
        if len(self.viewRefChFeaturesGroupbox.featureButtons) > 1:
            return True
        
        selectButton = self.viewRefChFeaturesGroupbox.featureButtons[0]
        return not selectButton.text().startswith('Click to select feature')
    
    def setInspectedRefChFeatures(self, df, frame_i, ID, sub_obj_id):
        for selectButton in self.viewRefChFeaturesGroupbox.featureButtons:
            if selectButton.text().startswith('Click to select feature'):
                continue
            
            feature_name = selectButton.text()
            col = self.nameToColMapper[feature_name]
            
            value = df.at[(frame_i, ID, sub_obj_id), col]
            selectButton.entry.setValue(value)
    
    def isWholeObjRequested(self):
        for selectButton in self.viewRefChFeaturesGroupbox.featureButtons:
            if selectButton.text().find('whole object') != -1:
                return True
        
        return False
        

class AutoTuneGroupbox(QGroupBox):
    sigColorChanged = Signal(object, bool)
    sigFeatureSelected = Signal(object, str, str)
    sigYXresolMultiplChanged = Signal(float)
    sigZresolLimitChanged = Signal(float)
    sigYXresolMultiplActivated = Signal(bool)
    sigZresolLimitActivated = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)

        mainLayout = QVBoxLayout()
        font = config.font()

        params = config.analysisInputsParams()
        self.params = {}
        for section, section_params in params.items():
            groupBox = None
            row = 0
            for anchor, param in section_params.items():
                tuneWidget = param.get('autoTuneWidget')
                if tuneWidget is None:
                    continue
                if section not in self.params:
                    self.params[section] = {}
                    self.params[section]['groupBox'] = QGroupBox(section)
                    self.params[section]['formLayout'] = widgets.FormLayout()
                self.params[section][anchor] = param.copy()
                groupBox = self.params[section]['groupBox']
                formLayout = self.params[section]['formLayout']
                formWidget = widgets.ParamFormWidget(
                    anchor, param, self, use_tune_widget=True
                )
                formLayout.addFormWidget(formWidget, row=row)
                self.params[section][anchor]['widget'] = formWidget.widget
                self.params[section][anchor]['formWidget'] = formWidget
                self.params[section][anchor]['groupBox'] = groupBox
                if anchor == 'gopThresholds':
                    formWidget.widget.sigFeatureSelected.connect(
                        self.emitFeatureSelected
                    )
                elif anchor == 'yxResolLimitMultiplier':
                    formWidget.widget.valueChanged.connect(
                        self.emitYXresolMultiplSigChanged
                    )
                    formWidget.widget.sigActivated.connect(
                        self.emitYXresolMultiplSigActivated
                    )
                    formWidget.widget.activateCheckbox.setChecked(True)
                    formWidget.widget.setDisabled(False)
                elif anchor == 'zResolutionLimit':
                    formWidget.widget.valueChanged.connect(
                        self.emitZresolLimitSigChanged
                    )
                    formWidget.widget.sigActivated.connect(
                        self.emitZresolLimitSigActivated
                    )
                    formWidget.widget.activateCheckbox.setChecked(False)
                    formWidget.widget.setDisabled(True)
                row += 1
            if groupBox is None:
                continue
            groupBox.setLayout(formLayout)
            mainLayout.addWidget(groupBox)
        
        autoTuneSpotProperties = AutoTuneSpotProperties() 
        self.trueRadioButton = autoTuneSpotProperties.trueRadioButton
        self.trueColorButton= autoTuneSpotProperties.trueColorButton
        self.falseColorButton = autoTuneSpotProperties.falseColorButton
        
        self.falseColorButton.sigColorChanging.connect(self.setFalseColor)
        
        self.trueItem = autoTuneSpotProperties.trueItem
        self.falseItem = autoTuneSpotProperties.falseItem
        self.autoTuneSpotProperties = autoTuneSpotProperties
        
        self.trueItem.sigClicked.connect(self.truePointsClicked)
        self.falseItem.sigClicked.connect(self.falsePointsClicked)
        
        self.trueItem.sigHovered.connect(self.truePointsHovered)
        self.falseItem.sigHovered.connect(self.falsePointsHovered)
        
        self.viewFeaturesGroupbox = AutoTuneViewSpotFeatures()
        
        mainLayout.addWidget(autoTuneSpotProperties)
        mainLayout.addWidget(self.viewFeaturesGroupbox)
        mainLayout.addStretch(1)
        self.setLayout(mainLayout)
        self.setFont(font)
    
    def setInspectFeatures(self, point_features):
        self.viewFeaturesGroupbox.setFeatures(point_features)
    
    def clearInspectFeatures(self):
        self.viewFeaturesGroupbox.clearFeatures()
    
    def emitYXresolMultiplSigChanged(self, value):
        self.sigYXresolMultiplChanged.emit(value)
    
    def emitZresolLimitSigChanged(self, value):
        self.sigZresolLimitChanged.emit(value)
    
    def emitYXresolMultiplSigActivated(self, checked):
        self.sigYXresolMultiplActivated.emit(checked)

    def emitZresolLimitSigActivated(self, checked):
        self.sigZresolLimitActivated.emit(checked)
    
    def emitFeatureSelected(self, button, featureText, colName):
        self.sigFeatureSelected.emit(button, featureText, colName)
    
    def falsePointsClicked(self, item, points, event):
        pass
    
    def falsePointsHovered(self, item, points, event):
        pass
    
    def truePointsClicked(self, item, points, event):
        pass
    
    def truePointsHovered(self, item, points, event):
        pass
    
    def setFalseColor(self, colorButton):
        r, g, b, a = colorButton.color().getRgb()
        self.falseItem.setBrush(r, g, b, 50)
        self.falseItem.setPen(r, g, b, width=2)
        self.sigColorChanged.emit((r, g, b, a), False)
    
    def setTrueColor(self, colorButton):
        r, g, b, a = colorButton.color().getRgb()
        self.trueItem.setBrush(r, g, b, 50)
        self.trueItem.setPen(r, g, b, width=2)
        self.sigColorChanged.emit((r, g, b, a), True)

class AutoTuneSpotProperties(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setTitle('Spots properties')
        layout = QVBoxLayout()
        
        trueFalseToggleLayout = QHBoxLayout()
                
        trueFalseToggleLayout.addWidget(
            QLabel('True spots color'), alignment=Qt.AlignRight
        )
        self.trueColorButton = acdc_widgets.myColorButton(
            color=(255, 0, 0)
        )
        trueFalseToggleLayout.addWidget(
            self.trueColorButton, alignment=Qt.AlignCenter
        )
        trueFalseToggleLayout.addStretch(1)        
        
        trueFalseToggleLayout.addWidget(
            QLabel('False spots color'), alignment=Qt.AlignRight
        )
        self.falseColorButton = acdc_widgets.myColorButton(
            color=(0, 255, 255)
        )
        trueFalseToggleLayout.addWidget(
            self.falseColorButton, alignment=Qt.AlignCenter
        )
        trueFalseToggleLayout.addStretch(1) 
        
        trueRadioButton = QRadioButton('Add true spots')
        falseRadioButton = QRadioButton('Add false spots')
        trueRadioButton.setChecked(True)
        self.trueRadioButton = trueRadioButton
        
        trueFalseToggleLayout.addWidget(trueRadioButton)
        trueFalseToggleLayout.addWidget(falseRadioButton)
        layout.addLayout(trueFalseToggleLayout)
        
        clearPointsButtonsLayout = QHBoxLayout()
        
        clearFalsePointsButton = acdc_widgets.eraserPushButton('Clear false points')
        clearTruePointsButton = acdc_widgets.eraserPushButton('Clear true points')
        clearAllPointsButton = acdc_widgets.eraserPushButton('Clear all points')
        clearPointsButtonsLayout.addWidget(clearFalsePointsButton)
        clearPointsButtonsLayout.addWidget(clearTruePointsButton)
        clearPointsButtonsLayout.addWidget(clearAllPointsButton)
        layout.addSpacing(10)
        layout.addLayout(clearPointsButtonsLayout)
        
        clearFalsePointsButton.clicked.connect(self.clearFalsePoints)
        clearTruePointsButton.clicked.connect(self.clearTruePoints)
        clearAllPointsButton.clicked.connect(self.clearAllPoints)
        
        self.trueColorButton.sigColorChanging.connect(self.setTrueColor)
        self.falseColorButton.sigColorChanging.connect(self.setFalseColor)
        
        self.trueItem = widgets.TuneScatterPlotItem(
            symbol='o', size=3, pxMode=False,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2),
            hoverable=True, hoverPen=pg.mkPen((255,0,0), width=3),
            hoverBrush=pg.mkBrush((255,0,0)), tip=None
        )
        
        self.falseItem = widgets.TuneScatterPlotItem(
            symbol='o', size=3, pxMode=False,
            brush=pg.mkBrush((0,255,255,50)),
            pen=pg.mkPen((0,255,255), width=2),
            hoverable=True, hoverPen=pg.mkPen((0,255,255), width=3),
            hoverBrush=pg.mkBrush((0,255,255)), tip=None
        )
        self.setLayout(layout)
    
    def setFalseColor(self, colorButton):
        r, g, b, a = colorButton.color().getRgb()
        self.falseItem.setBrush(r, g, b, 50)
        self.falseItem.setPen(r, g, b, width=2)
        self.sigColorChanged.emit((r, g, b, a), False)
    
    def setTrueColor(self, colorButton):
        r, g, b, a = colorButton.color().getRgb()
        self.trueItem.setBrush(r, g, b, 50)
        self.trueItem.setPen(r, g, b, width=2)
        self.sigColorChanged.emit((r, g, b, a), True)
    
    def clearFalsePoints(self):
        self.trueItem.setVisible(False)
        self.trueItem.clear()
    
    def clearTruePoints(self):
        self.falseItem.setVisible(False)
        self.falseItem.clear()

    def clearAllPoints(self):
        self.trueItem.setVisible(False)
        self.falseItem.setVisible(False)
        self.trueItem.clear()
        self.falseItem.clear()

class ViewRefChannelFeaturesGroupbox(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setTitle('Features of the ref. ch. mask under mouse cursor')
        
        mainLayout = QVBoxLayout()
        
        layout = QGridLayout()
        
        txt = html_func.span(
            '<i>Load ref. ch. masks and image to view features</i>',
            font_color='red'
        )
        self._infoText = txt
        self.infoLabel = QLabel(txt)
        
        col = 0
        row = 0
        layout.addWidget(
            self.infoLabel, row, col, 1, 2, alignment=Qt.AlignCenter
        )
        
        row += 1
        self.selectButton = widgets.FeatureSelectorButton(
            'Click to select feature to view...  ', alignment='right'
        )
        self.selectButton.setSizeLongestText(
            'Spotfit intens. metric, Foregr. integral gauss. peak'
        )
        self.selectButton.clicked.connect(self.selectFeature)
        self.selectButton.entry = widgets.ReadOnlyLineEdit()
        self.addFeatureButton = acdc_widgets.addPushButton()
        layout.addWidget(self.selectButton, row, col)
        layout.addWidget(self.selectButton.entry, row, col+1)
        layout.addWidget(
            self.addFeatureButton, row, col+2, alignment=Qt.AlignLeft
        )
        self.featureButtons = [self.selectButton]
        self.addFeatureButton.clicked.connect(self.addFeatureEntry)
        
        self.nextRow = row + 1
        
        self._layout = layout
        
        mainLayout.addLayout(layout)
        mainLayout.addStretch(1)
        self.setLayout(mainLayout)
    
    def resetInfoLabel(self):
        self.infoLabel.setText(self._infoText)
    
    def addFeatureEntry(self):
        selectButton = widgets.FeatureSelectorButton(
            'Click to select feature to view...  ', alignment='right'
        )
        selectButton.setSizeLongestText(
            'Spotfit intens. metric, Foregr. integral gauss. peak'
        )
        selectButton.clicked.connect(self.selectFeature)
        selectButton.entry = widgets.ReadOnlyLineEdit()
        delButton = acdc_widgets.delPushButton()
        delButton.widgets = [selectButton, selectButton.entry]
        delButton.selector = selectButton
        delButton.clicked.connect(self.removeFeatureField)
        
        self._layout.addWidget(selectButton, self.nextRow, 0)
        self._layout.addWidget(selectButton.entry, self.nextRow, 1)
        self._layout.addWidget(
            delButton, self.nextRow, 2, alignment=Qt.AlignLeft
        )
        self.nextRow += 1
        
        self.featureButtons.append(selectButton)
    
    def removeFeatureField(self):
        delButton = self.sender()
        for widget in delButton.widgets:
            self._layout.removeWidget(widget)
        self._layout.removeWidget(delButton)
        self.featureButtons.remove(delButton.selector)
    
    def getFeatureGroup(self):
        if self.selectButton.text().find('Click') != -1:
            return ''

        text = self.selectButton.text()
        topLevelText, childText = text.split(', ')
        return {topLevelText: childText}
    
    def selectFeature(self):
        self.selectFeatureDialog = widgets.FeatureSelectorDialog(
            parent=self.sender(), category='ref. channel objects',
            multiSelection=False, expandOnDoubleClick=True, 
            isTopLevelSelectable=False, infoTxt='Select feature', 
            allItemsExpanded=False
        )
        self.selectFeatureDialog.setCurrentItem(self.getFeatureGroup())
        # self.selectFeatureDialog.resizeVertical()
        self.selectFeatureDialog.sigClose.connect(self.setFeatureText)
        self.selectFeatureDialog.show()
    
    def setFeatureText(self):
        if self.selectFeatureDialog.cancel:
            return
        selectButton = self.selectFeatureDialog.parent()
        selectButton.setFlat(True)
        selection = self.selectFeatureDialog.selectedItems()
        group_name = list(selection.keys())[0]
        feature_name = selection[group_name][0]
        featureText = f'{group_name}, {feature_name}'
        selectButton.setFeatureText(featureText)
        mapper = features.feature_names_to_col_names_mapper(
            category='ref. channel objects'
        )
        column_name = mapper[featureText]
        selectButton.setToolTip(f'{column_name}')

class AutoTuneViewSpotFeatures(QGroupBox):
    def __init__(self, parent=None, infoText=None):
        super().__init__(parent)
        
        self.setTitle('Features of the spot under mouse cursor')
        
        mainLayout = QVBoxLayout()
        
        layout = QGridLayout()
        
        col = 0
        row = 0
        if infoText is None:
            txt = html_func.span(
                '<i>Add some points and run autotuning to view spots features</i>',
                font_color='red'
            )
        else:
            txt = infoText
        self._infoText = txt
        self.infoLabel = QLabel(txt)
        layout.addWidget(
            self.infoLabel, row, col, 1, 2, alignment=Qt.AlignCenter
        )
        
        row += 1
        layout.addWidget(QLabel('Spot id'), row, col, alignment=Qt.AlignRight)
        self.spotIdEntry = widgets.ReadOnlyLineEdit()
        layout.addWidget(self.spotIdEntry, row, col+1)
        
        row += 1
        layout.addWidget(QLabel('x coordinate'), row, col, alignment=Qt.AlignRight)
        self.xLineEntry = widgets.ReadOnlyLineEdit()
        layout.addWidget(self.xLineEntry, row, col+1)
        
        row += 1
        layout.addWidget(QLabel('y coordinate'), row, col, alignment=Qt.AlignRight)
        self.yLineEntry = widgets.ReadOnlyLineEdit()
        layout.addWidget(self.yLineEntry, row, col+1)
        
        row += 1
        layout.addWidget(QLabel('z coordinate'), row, col, alignment=Qt.AlignRight)
        self.zLineEntry = widgets.ReadOnlyLineEdit()
        layout.addWidget(self.zLineEntry, row, col+1)
        
        row += 1
        self.selectButton = widgets.FeatureSelectorButton(
            'Click to select feature to view...  ', alignment='right'
        )
        self.selectButton.setSizeLongestText(
            'Spotfit intens. metric, Foregr. integral gauss. peak'
        )
        self.selectButton.clicked.connect(self.selectFeature)
        self.selectButton.entry = widgets.ReadOnlyLineEdit()
        self.addFeatureButton = acdc_widgets.addPushButton()
        layout.addWidget(self.selectButton, row, col)
        layout.addWidget(self.selectButton.entry, row, col+1)
        layout.addWidget(
            self.addFeatureButton, row, col+2, alignment=Qt.AlignLeft
        )
        self.featureButtons = [self.selectButton]
        self.addFeatureButton.clicked.connect(self.addFeatureEntry)
        
        self.nextRow = row + 1
        
        self._layout = layout
        
        mainLayout.addLayout(layout)
        mainLayout.addStretch(1)
        self.setLayout(mainLayout)
    
    def resetFeatures(self):
        self.infoLabel.setText(self._infoText)
    
    def setFeatures(self, point_features: pd.Series):
        pos_foldername, frame_i, z, y, x = point_features.name
        self.spotIdEntry.setText(str(point_features.loc['spot_id']))
        self.xLineEntry.setText(str(x))
        self.yLineEntry.setText(str(y))
        self.zLineEntry.setText(str(z))
        for selectButton in self.featureButtons:
            feature_colname = selectButton.toolTip()
            if feature_colname not in point_features.index:
                continue
            value = point_features.loc[feature_colname]
            selectButton.entry.setText(str(value))
        self.infoLabel.setText('<i>&nbsp;</i>')
    
    def clearFeatures(self):
        self.spotIdEntry.setText('')
        self.xLineEntry.setText('')
        self.yLineEntry.setText('')
        self.zLineEntry.setText('')
        for selectButton in self.featureButtons:
            selectButton.entry.setText('')
    
    def addFeatureEntry(self):
        selectButton = widgets.FeatureSelectorButton(
            'Click to select feature to view...  ', alignment='right'
        )
        selectButton.setSizeLongestText(
            'Spotfit intens. metric, Foregr. integral gauss. peak'
        )
        selectButton.clicked.connect(self.selectFeature)
        selectButton.entry = widgets.ReadOnlyLineEdit()
        delButton = acdc_widgets.delPushButton()
        delButton.widgets = [selectButton, selectButton.entry]
        delButton.selector = selectButton
        delButton.clicked.connect(self.removeFeatureField)
        
        self._layout.addWidget(selectButton, self.nextRow, 0)
        self._layout.addWidget(selectButton.entry, self.nextRow, 1)
        self._layout.addWidget(
            delButton, self.nextRow, 2, alignment=Qt.AlignLeft
        )
        self.nextRow += 1
        
        self.featureButtons.append(selectButton)
        
    def removeFeatureField(self):
        delButton = self.sender()
        for widget in delButton.widgets:
            self._layout.removeWidget(widget)
        self._layout.removeWidget(delButton)
        self.featureButtons.remove(delButton.selector)
    
    def getFeatureGroup(self):
        if self.selectButton.text().find('Click') != -1:
            return ''

        text = self.selectButton.text()
        topLevelText, childText = text.split(', ')
        return {topLevelText: childText}
    
    def selectFeature(self):
        self.selectFeatureDialog = widgets.FeatureSelectorDialog(
            parent=self.sender(), multiSelection=False, 
            expandOnDoubleClick=True, isTopLevelSelectable=False, 
            infoTxt='Select feature', allItemsExpanded=False
        )
        self.selectFeatureDialog.setCurrentItem(self.getFeatureGroup())
        # self.selectFeatureDialog.resizeVertical()
        self.selectFeatureDialog.sigClose.connect(self.setFeatureText)
        self.selectFeatureDialog.show()
    
    def setFeatureText(self):
        if self.selectFeatureDialog.cancel:
            return
        selectButton = self.selectFeatureDialog.parent()
        selectButton.setFlat(True)
        selection = self.selectFeatureDialog.selectedItems()
        group_name = list(selection.keys())[0]
        feature_name = selection[group_name][0]
        featureText = f'{group_name}, {feature_name}'
        selectButton.setFeatureText(featureText)
        column_name = features.feature_names_to_col_names_mapper()[featureText]
        selectButton.setToolTip(f'{column_name}')

class AutoTuneTabWidget(QWidget):
    sigStartAutoTune = Signal(object)
    sigStopAutoTune = Signal(object)
    sigTrueFalseToggled = Signal(object)
    sigColorChanged = Signal(object, bool)
    sigFeatureSelected = Signal(object, str, str)
    sigAddAutoTunePointsToggle = Signal(bool)
    sigYXresolMultiplChanged = Signal(float)
    sigZresolLimitChanged = Signal(float)
    sigYXresolMultiplActivated = Signal(bool)
    sigZresolLimitActivated = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout()
        
        self.df_features = None
        
        self.isYXresolMultiplActive = True
        self.isZresolLimitActive = False

        buttonsLayout = QHBoxLayout()
        
        buttonsContainerWidget = QWidget()
        buttonsScrollArea = widgets.InvisibleScrollArea()
        buttonsScrollArea.setWidget(buttonsContainerWidget)
        buttonsScrollArea.setWidgetResizable(True)
        
        # Start adding points autotune button
        self.addAutoTunePointsButton = widgets.AddAutoTunePointsButton()
        buttonsLayout.addWidget(self.addAutoTunePointsButton)
        
        autoTuningButton = widgets.AutoTuningButton()
        self.loadingCircle = acdc_widgets.LoadingCircleAnimation(size=16)
        self.loadingCircle.setVisible(False)
        buttonsLayout.addWidget(autoTuningButton)
        buttonsLayout.addWidget(self.loadingCircle)
        self.autoTuningButton = autoTuningButton
        
        buttonsLayout.addStretch(1)
        
        trainButton = widgets.TrainSpotmaxAIButton('Train spotMAX AI...')
        buttonsLayout.addWidget(trainButton)
        
        helpButton = acdc_widgets.helpPushButton('Help...')
        buttonsLayout.addWidget(helpButton)
        
        autoTuneScrollArea = QScrollArea(self)
        autoTuneScrollArea.setWidgetResizable(True)

        self.autoTuneGroupbox = AutoTuneGroupbox(parent=self)
        autoTuneScrollArea.setWidget(self.autoTuneGroupbox)
        
        buttonsLayout.setContentsMargins(0, 0, 0, 0)
        buttonsContainerWidget.setLayout(buttonsLayout)
        buttonsScrollArea.setFixedHeight(
            buttonsContainerWidget.sizeHint().height()+4)

        layout.addWidget(buttonsScrollArea)
        layout.addWidget(autoTuneScrollArea)
        # layout.addStretch(1)
        # layout.addWidget(self.autoTuneGroupbox)
        
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
        
        self.setLayout(layout)

        autoTuningButton.sigToggled.connect(self.emitAutoTuningSignal)
        self.addAutoTunePointsButton.sigToggled.connect(
            self.emitAddAutoTunePointsToggle
        )
        self.autoTuneGroupbox.trueRadioButton.toggled.connect(
            self.emitForegrBackrToggledSignal
        )
        self.autoTuneGroupbox.sigColorChanged.connect(
            self.emitColorChanged
        )
        self.autoTuneGroupbox.sigFeatureSelected.connect(
            self.emitFeatureSelected
        )
        self.autoTuneGroupbox.sigYXresolMultiplChanged.connect(
            self.emitYXresolMultiplSigChanged
        )
        self.autoTuneGroupbox.sigZresolLimitChanged.connect(
            self.emitZresolLimitSigChanged
        )
        self.autoTuneGroupbox.sigYXresolMultiplActivated.connect(
            self.emitYXresolMultiplSigActivated
        )
        self.autoTuneGroupbox.sigZresolLimitActivated.connect(
            self.emitZresolLimitSigActivated
        )
        helpButton.clicked.connect(self.showHelp)
    
    def emitYXresolMultiplSigChanged(self, value):
        self.sigYXresolMultiplChanged.emit(value)
    
    def emitZresolLimitSigChanged(self, value):
        self.sigZresolLimitChanged.emit(value)
    
    def emitYXresolMultiplSigActivated(self, checked):
        self.sigYXresolMultiplActivated.emit(checked)
        self.isYXresolMultiplActive = True
        self.isZresolLimitActive = False

    def emitZresolLimitSigActivated(self, checked):
        self.sigZresolLimitActivated.emit(checked)
        self.isYXresolMultiplActive = False
        self.isZresolLimitActive = True
    
    def emitFeatureSelected(self, button, featureText, colName):
        self.sigFeatureSelected.emit(button, featureText, colName)
    
    def emitAddAutoTunePointsToggle(self, button, checked):
        self.setAutoTuneItemsVisible(True)
        self.sigAddAutoTunePointsToggle.emit(checked)
        self.addAutoTunePointsButton.clearFocus()
    
    def emitColorChanged(self, color: tuple, true_spots: bool):
        self.sigColorChanged.emit(color, true_spots)
    
    def emitAutoTuningSignal(self, button, started):
        self.loadingCircle.setVisible(started)
        if started:
            self.sigStartAutoTune.emit(self)
        else:
            self.sigStopAutoTune.emit(self)
    
    def setAutoTuneItemsVisible(self, visible):
        self.autoTuneGroupbox.trueItem.setVisible(visible)
        self.autoTuneGroupbox.falseItem.setVisible(visible)
    
    def setInspectFeatures(self, points):
        if self.df_features is None:
            return
        point = points[0]
        ptz_idx = point.data()[:3]
        pos = point.pos()
        x, y = round(pos.x()), round(pos.y())
        point_features = self.df_features.loc[(*ptz_idx, y, x)]
        self.autoTuneGroupbox.setInspectFeatures(point_features)
    
    def emitForegrBackrToggledSignal(self, checked):
        self.sigTrueFalseToggled.emit(checked)
    
    def initAutoTuneColors(self, trueColor, falseColor):
        self.autoTuneGroupbox.trueColorButton.setColor(trueColor)
        self.autoTuneGroupbox.falseColorButton.setColor(falseColor)
    
    def selectedFeatures(self):
        SECTION = 'Spots channel'
        ANCHOR = 'gopThresholds'
        widget = self.autoTuneGroupbox.params[SECTION][ANCHOR]['widget']
        selectedFeatures = {
            groupbox.title(): [None, None] 
            for groupbox in widget.featureGroupboxes.values()
            if groupbox.title().find('Click') == -1
        }    
        return selectedFeatures
    
    def setTuneResult(self, tuneResult: tune.TuneResult):
        SECTION = 'Spots channel'
        ANCHOR = 'gopThresholds'
        widget = self.autoTuneGroupbox.params[SECTION][ANCHOR]['widget']
        for groupbox in widget.featureGroupboxes.values():
            feature_name = groupbox.title()
            if feature_name not in tuneResult.features_range:
                continue
            minimum, maximum = tuneResult.features_range[feature_name]
            groupbox.setRange(minimum, maximum)
        
        ANCHOR = 'spotThresholdFunc'
        widget = self.autoTuneGroupbox.params[SECTION][ANCHOR]['widget']
        widget.setText(tuneResult.threshold_method)
        
        self.autoTuneGroupbox.viewFeaturesGroupbox.infoLabel.setText(
            '<i>Hover mouse cursor on points to view features</i>'
        )
        self.df_features = (
            tuneResult.df_features.reset_index()
            .set_index(['Position_n', 'frame_i', 'z', 'y', 'x'])
        )
    
    def updatePos(self, posData, z):
        self.setPosData(posData)
        self.setVisibleAutoTunePoints(self.posFoldername(), 0, z)
        self.autoTuneGroupbox.clearInspectFeatures()
    
    def setPosData(self, posData):
        self._posData = posData
    
    def posFoldername(self):
        return self._posData.pos_foldername
    
    def getHoveredPoints(self, frame_i, z, y, x):
        items = [
            self.autoTuneGroupbox.trueItem, self.autoTuneGroupbox.falseItem
        ]
        hoveredPoints = []
        for item in items:
            hoveredMask = item._maskAt(QPointF(x, y))
            points = item.points()[hoveredMask][::-1]
            if len(points) == 0:
                continue
            for point in points:
                point_data = point.data()
                if point_data[0] != self.posFoldername():
                    continue
                if point_data[1] != frame_i:
                    continue
                if point_data[2] != z:
                    continue
                hoveredPoints.append(point)
        return hoveredPoints
    
    def addAutoTunePoint(self, frame_i, z, x, y):
        if self.autoTuneGroupbox.trueRadioButton.isChecked():
            item = self.autoTuneGroupbox.trueItem
            item.setVisible(True)
        else:
            item = self.autoTuneGroupbox.falseItem
            item.setVisible(True)
        hoveredMask = item._maskAt(QPointF(x, y))
        points = item.points()
        hoveredPoints = item.points()[hoveredMask][::-1]
        if len(hoveredPoints) > 0:
            for hoveredPoint in hoveredPoints:
                xref, yref = hoveredPoint._data['x'], hoveredPoint._data['y']
                zref = hoveredPoint.data()[2]
                # Remove same point in neigh z-slices
                for point in points:
                    point_data = point.data()
                    if point_data[0] != self.posFoldername():
                        continue

                    if point_data[1] != frame_i:
                        continue
                    
                    if point_data[3] != zref:
                        continue
                    
                    item.removePoint(point._index)
        else:
            point_data = (self.posFoldername(), frame_i, z, z)
            item.addPoints([x], [y], data=[point_data])
            for neigh_z in range(z-self.z_radius, z+self.z_radius+1):
                if neigh_z == z:
                    continue
                neigh_point_data = (self.posFoldername(), frame_i, neigh_z, z)
                item.addPoints(
                    [x], [y], data=[neigh_point_data], 
                    brush=[pg.mkBrush((0, 0, 0, 0))], 
                    pen=[pg.mkPen((0, 0, 0, 0))]
                )
        
        self.resetFeatures()
    
    def resetFeatures(self):
        self.df_features = None
        self.autoTuneGroupbox.viewFeaturesGroupbox.resetFeatures()
    
    def setVisibleAutoTunePoints(self, *args):
        items = [
            self.autoTuneGroupbox.trueItem, self.autoTuneGroupbox.falseItem
        ]

        for item in items:
            brushes = []
            pens = []
            visibilities = []
            for point in item.data['item']:
                point_data = point.data()
                visible = (point_data[0], point_data[1], point_data[2]) == args
                if not visible:
                    brush = pg.mkBrush((0, 0, 0, 0))
                    pen = pg.mkPen((0, 0, 0, 0))
                else:
                    brush = item.itemBrush()
                    pen = item.itemPen()
                brushes.append(brush)
                pens.append(pen)
                visibilities.append(visible)
            if not brushes:
                continue
            item.setBrush(brushes)
            item.setPen(pens)
            item.setPointsVisible(visibilities)
            
    def setAutoTunePointSize(self, yx_diameter, z_diameter):
        self.autoTuneGroupbox.trueItem.setSize(yx_diameter)
        self.autoTuneGroupbox.falseItem.setSize(yx_diameter)
        z_diameter = round(z_diameter)
        if z_diameter % 2:
            z_diameter -= 1
        self.z_radius = int(z_diameter/2)
    
    def showHelp(self):
        msg = acdc_widgets.myMessageBox()
        steps = [
    'Load images (<code>Open folder</code> button on the top toolbar).',
    'Select the features used to filter true spots.',
    'Click <code>Start autotuning</code> on the "Autotune parameters" tab.',
    'Choose whether to use the current spots segmentation mask.',
    'Adjust spot size with up/down arrow keys.',
    'Click on the true spots on the image.'
        ]
        txt = html_func.paragraph(f"""
            Autotuning can be used to interactively determine the 
            <b>optimal parameters</b> for the analysis.<br><br>
            Instructions:{acdc_html.to_list(steps, ordered=True)}<br>
            Select as many features as you want. The tuning process will then 
            optimise their values that will be used to filter true spots.<br><br>
            The more true spots you add, the better the optimisation process 
            will be. However, adding the spots that are 
            <b>more difficult to detect</b> (e.g., out-of-focus or dim) 
            should yield <b>better results</b>.
        """)
        msg.information(self, 'Autotuning instructions', txt)
    
    def setDisabled(self, disabled: bool) -> None:
        self.autoTuneGroupbox.setDisabled(disabled)
        self.autoTuningButton.setDisabled(disabled)

class ParamsGroupBox(QGroupBox):
    sigResolMultiplValueChanged = Signal(float)
    sigLoadMetadataFromAcdc = Signal()
    
    def __init__(self, parent=None, debug=False, logging_func=print):
        super().__init__(parent)

        self.selectedMeasurements = None
        # mainLayout = QGridLayout(self)
        mainLayout = QVBoxLayout()

        self.logging_func = logging_func
        
        section_option_to_desc_mapper = docs.get_params_desc_mapper()
        
        font = config.font()

        _params = config.analysisInputsParams()
        self.params = {}
        for section, section_params in _params.items():
            formLayout = widgets.FormLayout()
            self.params[section] = {}
            isNotCheckableGroup = (
                section == 'File paths and channels' or section == 'METADATA'
                or section == 'Pre-processing'
            )
            
            if section == 'SpotFIT':
                groupBox = widgets.ExpandableGroupbox(section)
                groupBox.setExpanded(False)
            else:
                groupBox = QGroupBox(section)
            
            if isNotCheckableGroup:
                groupBox.setCheckable(False)
            else:
                groupBox.setCheckable(True)
            groupBox.setFont(font)
            groupBox.formWidgets = []
            for row, (anchor, param) in enumerate(section_params.items()):
                self.params[section][anchor] = param.copy()
                formWidget = widgets.ParamFormWidget(
                    anchor, param, self, 
                    section_option_to_desc_mapper=section_option_to_desc_mapper
                )
                formWidget.section = section
                formWidget.sigLinkClicked.connect(self.infoLinkClicked)
                self.connectFormWidgetButtons(formWidget, param)
                formLayout.addFormWidget(formWidget, row=row)
                self.params[section][anchor]['widget'] = formWidget.widget
                self.params[section][anchor]['formWidget'] = formWidget
                self.params[section][anchor]['groupBox'] = groupBox
                if formWidget.useEditableLabel:
                    self.params[section][anchor]['desc'] = (
                        formWidget.labelLeft.text()
                    )
                    
                if formWidget.addFieldButton is not None:
                    formWidget.sigAddField.connect(
                        self.addFieldToParams
                    )
                    formWidget.sigRemoveField.connect(
                        self.removeFieldFromParams
                    )
                
                groupBox.formWidgets.append(formWidget)

                isGroupChecked = param.get('isSectionInConfig', False)
                groupBox.setChecked(isGroupChecked)

                if param.get('editSlot') is not None:
                    editSlot = param.get('editSlot')
                    slot = getattr(self, editSlot)
                    formWidget.sigEditClicked.connect(slot)
                actions = param.get('actions', None)
                if actions is None:
                    continue

                for action in actions:
                    signal = getattr(formWidget.widget, action[0])
                    signal.connect(getattr(self, action[1]))

            if section == 'METADATA':
                # loadMetadataFromAcdcButton = acdc_widgets.browseFileButton(
                #     'Load metadata from Cell-ACDC metadata.csv file...',
                #     title='Select Cell-ACDC metadata.csv file',
                #     ext={'CSV': ['.csv']},
                #     start_dir=acdc_myutils.getMostRecentPath()
                # )
                # loadMetadataFromAcdcButton.sigPathSelected.connect(
                #     self.loadMetadataFromAcdc
                # )
                loadMetadataFromAcdcButton = acdc_widgets.LoadPushButton(
                    'Load metadata from Cell-ACDC metadata.csv file'
                )
                loadMetadataFromAcdcButton.clicked.connect(
                    self.loadMetadataFromAcdc
                )
                colSpan = formLayout.columnCount()
                formLayout.addWidget(
                    loadMetadataFromAcdcButton, row+1, 0, 1, colSpan, 
                    alignment=Qt.AlignRight
                )
            
            groupBox.setLayout(formLayout)
            mainLayout.addWidget(groupBox)

        # mainLayout.addStretch()
        
        metadata = self.params['METADATA']
        pixelSize = metadata['pixelWidth']['widget'].value()
        self.updateLocalBackgroundValue(pixelSize)

        self.setLayout(mainLayout)
        self.updateMinSpotSize()
        self.doSpotFitToggled(False)
    
    def loadMetadataFromAcdc(self):
        self.sigLoadMetadataFromAcdc.emit()
    
    def addFieldToParams(self, formWidget):
        if formWidget.fieldIdx == 0:
            return
        
        section = formWidget.section
        anchor = formWidget.anchor
        groupBox = self.params[section][anchor]['groupBox']

        defaultParams = config.getDefaultParams()
        added_anchor = f'{anchor}_{formWidget.fieldIdx}'
        self.params[section][added_anchor] = defaultParams[section][anchor]
        anchor = added_anchor
        
        self.params[section][anchor]['widget'] = formWidget.widget
        self.params[section][anchor]['formWidget'] = formWidget
        self.params[section][anchor]['groupBox'] = groupBox        
        if formWidget.useEditableLabel:
            self.params[section][anchor]['desc'] = (
                formWidget.labelLeft.text()
            )   
    
    def removeFieldFromParams(self, section, anchor, fieldIdx):
        if fieldIdx > 0:
            anchor = f'{anchor}_{fieldIdx}'
        self.params[section].pop(anchor)
    
    def addFoldersToAnalyse(self, formWidget):
        preSelectedPaths = formWidget.widget.text().split('\n')
        preSelectedPaths = [path for path in preSelectedPaths if path]
        if not preSelectedPaths:
            preSelectedPaths = None
        win = SelectFolderToAnalyse(
            preSelectedPaths=preSelectedPaths, scanFolderTree=True
        )
        win.exec_()
        if win.cancel:
            return
        selectedPathsList = win.paths
        
        self.validatePaths(selectedPathsList, formWidget)        
        selectedPaths = '\n'.join(selectedPathsList)
        formWidget.widget.setText(selectedPaths)
    
    def validatePaths(self, selectedPathsList=None, formWidget=None):
        if formWidget is None:
            section = 'File paths and channels'
            anchor = 'folderPathsToAnalyse'
            formWidget = self.params[section][anchor]['formWidget']
        
        if selectedPathsList is None:
            selectedPathsList = formWidget.widget.text().split('\n')
        
        warningButton = formWidget.warningButton
        warningButton.hide()
        formWidget.widget.setInvalidEntry(False)
        
        notExistingPaths = [
            path for path in selectedPathsList if not os.path.exists(path)
        ]
        if not notExistingPaths:
            return

        formWidget.widget.setInvalidEntry(True)
        warningButton.show()
        try:
            formWidget.sigWarningButtonClicked.disconnect()
        except Exception as err:
            pass
        formWidget.sigWarningButtonClicked.connect(
            partial(
                self.warnExpPathsNotExisting, 
                notExistingPaths=notExistingPaths
            )
        )
    
    def warnExpPathsNotExisting(self, notExistingPaths=None):
        if notExistingPaths is None:
            return
        
        txt = html_func.paragraph(f"""
            One or more selected <b>experiment paths do not exist</b>!<br><br>
            See below for more details.
        """)
        notExistingPathsStr = '\n'.join(notExistingPaths)
        detailsText = (f"""
            Not existing paths:\n\n{notExistingPathsStr}
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Experiment path(s) not existing', txt,
            detailsText=detailsText
        )
            
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

    def SizeZchanged(self, SizeZ):
        isZstack = SizeZ > 1
        metadata = self.params['METADATA']
        spotMinSizeLabels = metadata['spotMinSizeLabels']['widget']
        spotMinSizeLabels.setIsZstack(isZstack)
        self.updateMinSpotSize()
        
        preProcessParams = self.params['Pre-processing']
        extend3DsegmRangeFormWidget = (
            preProcessParams['extend3DsegmRange']['formWidget']
        )
        extend3DsegmRangeFormWidget.setDisabled(not isZstack)
    
    def zyxVoxelSize(self):
        metadata = self.params['METADATA']
        physicalSizeX = metadata['pixelWidth']['widget'].value()
        physicalSizeY = metadata['pixelHeight']['widget'].value()
        physicalSizeZ = metadata['voxelDepth']['widget'].value()
        return (physicalSizeZ, physicalSizeY, physicalSizeX)
    
    def updateLocalBackgroundValue(self, pixelSize):
        spotParams = self.params['Spots channel']
        localBkgrRingWidthWidget = spotParams['localBkgrRingWidth']['widget']
        localBkgrRingWidthWidget.setPixelSize(pixelSize)
    
    def doSpotFitToggled(self, checked):
        for section, section_params in self.params.items():
            for anchor, param in section_params.items():
                parentActivator = param.get('parentActivator')
                if parentActivator is None:
                    continue
                
                parentSection, parentAnchor = parentActivator
                parentParam = self.params[parentSection][parentAnchor]
                isDisabled = not parentParam['widget'].isChecked()
                param['formWidget'].setDisabled(isDisabled)
                if not isDisabled:
                    continue
                
                parentDesc = parentParam['desc']
                tooltip = (
                    'This parameter is disabled because it requires\n'
                    f'`{parentDesc}` to be activated.'
                )
                param['formWidget'].setToolTip(tooltip)
    
    def updateMinSpotSize(self, value=0.0):
        metadata = self.params['METADATA']
        physicalSizeX = metadata['pixelWidth']['widget'].value()
        physicalSizeY = metadata['pixelHeight']['widget'].value()
        physicalSizeZ = metadata['voxelDepth']['widget'].value()
        SizeZ = metadata['SizeZ']['widget'].value()
        emWavelen = metadata['emWavelen']['widget'].value()
        NA = metadata['numAperture']['widget'].value()
        zResolutionLimit_um = metadata['zResolutionLimit']['widget'].value()
        yxResolMultiplier = metadata['yxResolLimitMultiplier']['widget'].value()
        zyxMinSize_pxl, zyxMinSize_um = core.calcMinSpotSize(
            emWavelen, NA, physicalSizeX, physicalSizeY, physicalSizeZ,
            zResolutionLimit_um, yxResolMultiplier
        )
        if SizeZ == 1:
            zyxMinSize_pxl = (float('nan'), *zyxMinSize_pxl[1:])
            zyxMinSize_um = (float('nan'), *zyxMinSize_um[1:])
        
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
        
        self.sigResolMultiplValueChanged.emit(yxResolMultiplier)
        
        formWidget = metadata['spotMinSizeLabels']['formWidget']
        warningButton = formWidget.warningButton
        warningButton.hide()
        if any([val<2 for val in zyxMinSize_pxl]):
            warningButton.show()
            try:
                formWidget.sigWarningButtonClicked.disconnect()
            except Exception as err:
                pass
            formWidget.sigWarningButtonClicked.connect(
                self.warnSpotSizeMightBeTooLow
            )
        
        spot_yx_radius_pixel = zyxMinSize_pxl[-1]
        spotsParams = self.params['Spots channel']
        anchor = 'spotPredictionMethod'
        spotPredictionMethodWidget = spotsParams[anchor]['widget']
        spotPredictionMethodWidget.setExpectedYXSpotRadius(spot_yx_radius_pixel)
    
    def warnSpotSizeMightBeTooLow(self, formWidget):
        spotMinSizeLabels = formWidget.widget.pixelLabel.text()
        txt = html_func.paragraph(f"""
            One or more radii of the <code>{formWidget.text()}</code> are 
            <b>less than 2 pixels</b>.<br><br>
            This means that SpotMAX can detect spots that are 1 pixel away 
            along the dimension that is less than 2 pixels.<br><br>
            We recommend <b>increasing the radii to at least 3 pixels</b>.<br><br>
            Current <code>{formWidget.text()} = {spotMinSizeLabels}</code>
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Minimimum spot size potentially too low', txt)
    
    def configIniParams(self):
        ini_params = {}
        for section, section_params in self.params.items():
            ini_params[section] = {}
            for anchor, options in section_params.items():
                groupbox = options['groupBox']
                initialVal = options['initialVal']
                widget = options['widget']
                if groupbox.isCheckable() and not groupbox.isChecked():
                    # Use default value if the entire section is not checked
                    value = initialVal
                elif isinstance(initialVal, bool):
                    value = widget.isChecked()
                elif isinstance(initialVal, str):
                    try:
                        value = widget.currentText()
                    except AttributeError:
                        value = widget.text()
                elif isinstance(initialVal, float) or isinstance(initialVal, int):
                    value = widget.value()
                else:
                    value = widget.value()
                
                try:
                    # Editable labels (see widgets.FormWidget) have dynamic 
                    # text for the description
                    formWidget = options['formWidget']
                    desc = formWidget.labelLeft.text()
                except Exception as err:
                    desc = options['desc']
                
                if not desc:
                    continue
                
                ini_params[section][anchor] = {
                    'desc': desc, 
                    'loadedVal': value, 
                    'initialVal': initialVal
                }
        
        ini_params = self.addNNetParams(ini_params, 'spots')
        ini_params = self.addNNetParams(ini_params, 'ref_ch')
        return ini_params
    
    def addNNetParams(self, ini_params, channel):
        if channel == 'spots':
            params = self.params['Spots channel']
            anchor = 'spotPredictionMethod'
        else:
            params = self.params['Reference channel']
            anchor = 'refChSegmentationMethod'
        
        widget = params[anchor]['widget']
        nnet_params = widget.nnet_params_to_ini_sections()
        bioimageio_model_params = (
            widget.bioimageio_model_params_to_ini_sections()
        )
        spotiflow_model_params = (
            widget.spotiflow_model_params_to_ini_sections()
        )
        is_dl_model = (
            nnet_params is not None 
            or bioimageio_model_params is not None
            or spotiflow_model_params is not None
        )
        if not is_dl_model:
            return ini_params

        kwargs_model = None
        if nnet_params is not None:
            section_id_name = 'neural_network'
            model_params = nnet_params
            init_model_params, segment_model_params = model_params
        elif bioimageio_model_params is not None:
            section_id_name = 'bioimageio_model'
            model_params = bioimageio_model_params
            init_model_params, segment_model_params, kwargs_model = model_params
        elif spotiflow_model_params is not None:
            section_id_name = 'spotiflow'
            model_params = spotiflow_model_params
            init_model_params, segment_model_params = model_params
            
        SECTION = f'{section_id_name}.init.{channel}'
        for key, value in init_model_params.items():
            if SECTION not in ini_params:
                ini_params[SECTION] = {}
            ini_params[SECTION][key] = {
                'desc': key, 'loadedVal': value, 'isParam': True
            }
        
        SECTION = f'{section_id_name}.segment.{channel}'
        for key, value in segment_model_params.items():
            if SECTION not in ini_params:
                ini_params[SECTION] = {}
            ini_params[SECTION][key] = {
                'desc': key, 'loadedVal': value, 'isParam': True
            }
        
        if kwargs_model is not None:
            SECTION = f'{section_id_name}.kwargs.{channel}'
            for key, value in kwargs_model.items():
                if SECTION not in ini_params:
                    ini_params[SECTION] = {}
                ini_params[SECTION][key] = {
                    'desc': key, 'loadedVal': value, 'isParam': True
                }
        
        return ini_params
    
    def saveSelectedMeasurements(self, configPars, ini_filepath):
        if self.selectedMeasurements is None:
            return

        section = 'Single-spot measurements to save'
        configPars[section] = {}
        for key, value in self.selectedMeasurements['single_spot'].items():
            configPars[section][key] = value
        
        section = 'Aggregated measurements to save'
        configPars[section] = {}
        for key, value in self.selectedMeasurements['aggr'].items():
            configPars[section][key] = value
        
        with open(ini_filepath, 'w', encoding="utf-8") as file:
            configPars.write(file)
    
    def setSelectedMeasurements(self, ini_filepath):
        cp = config.ConfigParser()
        cp.read(ini_filepath)
        tabKeys_sections = [
            ('single_spot', 'Single-spot measurements to save'),
            ('aggr', 'Aggregated measurements to save')
        ]
        self.selectedMeasurements = {}
        for tabKey, section in tabKeys_sections:
            if not cp.has_section(section):
                continue
            
            self.selectedMeasurements[tabKey] = dict(cp[section])
            
        if not self.selectedMeasurements:
            self.selectedMeasurements = None
        
    
    def saveToIniFile(self, ini_filepath):
        params = self.configIniParams()
        configPars = io.writeConfigINI(params, ini_filepath)
        self.saveSelectedMeasurements(configPars, ini_filepath)
        print('-'*100)
        print(f'Configuration file saved to: "{ini_filepath}"')
        print('*'*100)

class spotStyleDock(QDockWidget):
    sigOk = Signal(int)
    sigCancel = Signal()

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
            QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

    def ok_cb(self):
        self.hide()

    def cancel_cb(self):
        self.sigCancel.emit()
        self.hide()

    def show(self):
        QDockWidget.show(self)
        self.resize(int(self.width()*1.5), self.height())
        self.setFocus()
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
            listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
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

        listBox.setFocus()
        self.setMyStyleSheet()

    def setMyStyleSheet(self):
        self.setStyleSheet("""
            QListWidget::item:hover {background-color:#E6E6E6;}
            QListWidget::item:hover {color:black;}
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
        mostRecentDateModifiedRun = 0
        mostRecentRun = runs[-1]
        for run, runInfo in self.paths.items():
            for exp_path, expInfo in runInfo.items():
                for pos in expInfo['posFoldernames']:
                    dateModified = expInfo[pos]['timeAnalysed']
                    if dateModified > mostRecentDateModifiedRun:
                        mostRecentRun = run
                        mostRecentDateModifiedRun = dateModified

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
        runNumberCombobox.setCurrentIndex(runs.index(mostRecentRun))
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

        pathSelector = acdc_widgets.TreeWidget(multiSelection=True)
        self.pathSelector = pathSelector
        pathSelector.setHeaderHidden(True)
        homePath = pathlib.Path(homePath)
        self.homePath = homePath
        self.populatePathSelector()

        buttonsLayout = QHBoxLayout()
        cancelButton = acdc_widgets.cancelPushButton('Cancel')
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)

        showInFileManagerButton = acdc_widgets.showInFileManagerButton(
            setDefaultText=True
        )
        showInFileManagerButton.clicked.connect(self.showInFileManager)
        buttonsLayout.addWidget(showInFileManagerButton)

        okButton = acdc_widgets.okPushButton('Ok')
        # okButton.setShortcut(Qt.Key_Enter)
        buttonsLayout.addWidget(okButton)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addLayout(runNumberLayout)
        runNumberLayout.setContentsMargins(0, 0, 0, 10)
        mainLayout.addLayout(checkBoxesLayout)
        mainLayout.addWidget(pathSelector)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        self.setLayout(mainLayout)

        hideSpotCountCheckbox.stateChanged.connect(self.hideSpotCounted)
        hideSpotSizeCheckbox.stateChanged.connect(self.hideSpotSized)
        runNumberCombobox.currentIndexChanged.connect(self.updateRun)
        showAnalysisTableButton.clicked.connect(self.showAnalysisInputsTable)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        pathSelector.itemClicked.connect(self.selectAllChildren)

        self.pathSelector.setFocus()

        self.setFont(font)
    
    def showInFileManager(self):
        selectedItems = self.pathSelector.selectedItems()
        doc = QTextDocument()
        firstItem = selectedItems[0]
        label = self.pathSelector.itemWidget(firstItem, 0)
        doc.setHtml(label.text())
        plainText = doc.toPlainText()
        parent = firstItem.parent()
        if parent is None:
            posFoldername = ''
            parentText = plainText
        else:
            try:
                posFoldername = re.findall(r'(.+) \(', plainText)[0]
            except IndexError:
                posFoldername = plainText
            parentLabel = self.pathSelector.itemWidget(parent, 0)
            doc.setHtml(parentLabel.text())
            parentText = doc.toPlainText()
        
        relPath = re.findall(r'...(.+) \(', parentText)[0]
        relPath = pathlib.Path(relPath)
        relPath = pathlib.Path(*relPath.parts[2:])
        absPath = self.homePath / relPath / posFoldername
        acdc_myutils.showInExplorer(str(absPath))

    def showAnalysisInputsTable(self):
        idx = self.runNumberCombobox.currentIndex()
        run = self.runs[idx]

        selectedItems = self.pathSelector.selectedItems()

        if not selectedItems:
            self.warnNoPathSelected()
            return

        doc = QTextDocument()
        item = selectedItems[0]
        text = item.text(0)
        doc.setHtml(text)
        plainText = doc.toPlainText()
        parent = item.parent()
        if parent is None:
            relPath1 = re.findall(r'...(.+) \(', plainText)[0]
            relPath1 = pathlib.Path(relPath1)
            relPath = pathlib.Path(*relPath1.parts[2:])
            if str(relPath) == '.':
                relPath = ''
            exp_path = os.path.join(self.homePath, relPath)

            selectedRunPaths = self.paths[run]
            analysisInputs = selectedRunPaths[os.path.normpath(exp_path)].get(
                'analysisInputs'
            )
        else:
            posFoldername = re.findall(r'(.+) \(', plainText)[0]
            doc.setHtml(parent.text(0))
            parentText = doc.toPlainText()
            relPath1 = re.findall(r'...(.+) \(', parentText)[0]
            relPath1 = pathlib.Path(relPath1)
            relPath = pathlib.Path(*relPath1.parts[2:])
            relPath1 = relPath / posFoldername
            exp_path = self.homePath / relPath / posFoldername
            spotmaxOutPath = exp_path / 'spotMAX_output'
            if os.path.exists(spotmaxOutPath):
                iniFilepath = io.get_analysis_params_filepath_from_run(
                    spotmaxOutPath, run
                )
                analysisInputs = io.read_ini(iniFilepath)
            else:
                analysisInputs = None

        if analysisInputs is None:
            self.warnAnalysisInputsNone(exp_path, run)
            return

        if isinstance(analysisInputs, pd.DataFrame):
            title = f'Analysis inputs table'
            infoText = html_func.paragraph(
                f'Analysis inputs used to analyse <b>run number {run}</b> '
                f'of experiment:<br>"{relPath1}"<br>'
            )
            self.analysisInputsTableWin = pdDataFrameWidget(
                analysisInputs.reset_index(), title=title, infoText=infoText, 
                parent=self
            )
        else:
            self.analysisInputsTableWin = iniFileWidget(
                analysisInputs, filename=analysisInputs.filename()
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
            relPathText = f'{relPath} ({nPSCtext}, {nPSStext})'
            relPathItem.setText(0, relPathText)
            
            # relPathLabel = acdc_widgets.QClickableLabel()
            # relPathLabel.item = relPathItem
            # relPathLabel.clicked.connect(self.selectAllChildren)

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
                posItem.setText(0, posText)
                # posLabel = acdc_widgets.QClickableLabel()
                # posLabel.item = posItem
                # posLabel.clicked.connect(self.selectAllChildren)
                # posLabel.setText(posText)
                relPathItem.addChild(posItem)
                # pathSelector.setItemWidget(posItem, 0, posLabel)
        if relPathItem is not None and len(selectedRunPaths) == 1:
            relPathItem.setExpanded(True)

    def selectAllChildren(self, label=None):
        self.pathSelector.selectAllChildren(label)
    
    def warnAnalysisInputsNone(self, exp_path, run):
        text = (
            f'The selected experiment "{exp_path}" '
            f'does not have the <b>"{run}_analysis_inputs.csv"</b> nor '
            f'the <b>"{run}_analysis_parameters.ini"</b> file.<br><br>'
            'Sorry about that.'
        )
        msg = acdc_widgets.myMessageBox()
        msg.addShowInFileManagerButton(exp_path)
        msg.warning(
            self, 'Analysis inputs not found!',
            html_func.paragraph(text)
        )

    def ok_cb(self, checked=True):
        selectedItems = self.pathSelector.selectedItems()
        doc = QTextDocument()
        for item in selectedItems:
            plainText = item.text(0)
            parent = item.parent()
            if parent is None:
                continue
            try:
                posFoldername = re.findall(r'(.+) \(', plainText)[0]
            except IndexError:
                posFoldername = plainText
            parentText = parent.text(0)
            relPath = re.findall(r'...(.+) \(', parentText)[0]
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
            labelText = item.text(0)
            currentW = item.sizeHint(0).width()
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

    dataFrame = QtCore.Property(pd.DataFrame, fget=dataFrame,
                                    fset=setDataFrame)

    @QtCore.Slot(int, QtCore.Qt.Orientation, result=str)
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

class iniFileWidget(QBaseDialog):
    def __init__(self, configPars, filename='', parent=None):
        self.cancel = True

        super().__init__(parent)

        self.setWindowTitle('Configuration file content')

        mainLayout = QVBoxLayout()

        if filename:
            label = QLabel()
            txt = html_func.paragraph(f'Filename: <code>{filename}</code><br>')
            label.setText(txt)
            mainLayout.addWidget(label)
        
        self.textWidget = QTextEdit()
        self.textWidget.setReadOnly(True)
        self.setIniText(configPars)
        
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)

        okButton = acdc_widgets.okPushButton(' Ok ')
        buttonsLayout.addWidget(okButton)

        okButton.clicked.connect(self.ok_cb)
        
        mainLayout.addWidget(self.textWidget)
        mainLayout.addLayout(buttonsLayout)
        self.setLayout(mainLayout)
    
    def setIniText(self, configPars):
        htmlText = ''
        palette = palettes.ini_hex_colors()
        section_hex = palette['section']
        option_hex = palette['option']
        for section in configPars.sections():
            sectionText = html_func.span(f'[{section}]', font_color=section_hex)
            htmlText = f'{htmlText}{sectionText}<br>'
            for option in configPars.options(section):
                value = configPars[section][option]
                # option = option.replace('Î¼', '&micro;')
                optionText = html_func.span(
                    f'<i>{option}</i> = ', font_color=option_hex
                )
                value = value.replace('\n', '<br>&nbsp;&nbsp;&nbsp;&nbsp;')
                htmlText = f'{htmlText}{optionText}{value}<br>'
            htmlText = f'{htmlText}<br>'
        self.textWidget.setHtml(html_func.paragraph(htmlText))
    
    def show(self, block=False):
        super().show(block=False)
        self.move(self.pos().x(), 20)
        height = int(self.screen().size().height()*0.7)
        width = round(height*0.85)
        self.resize(width, height)
        super().show(block=block)
    
    def ok_cb(self):
        self.cancel = False
        self.close()

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
            QTreeWidget::item:hover {color:black;}
            QTreeWidget::item:selected {
                background-color:#CFEB9B;
                color:black;
            }
            QTreeView {
                selection-background-color: #CFEB9B;
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
        runNumber = int(re.findall(r'(\d+)', runItem.text(0))[0])
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

        treeSelector.setFocus()

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
    mostRecentPath = acdc_myutils.getMostRecentPath()
    while True:
        exp_path = QFileDialog.getExistingDirectory(
            parent, 'Select experiment folder containing Position_n folders',
            mostRecentPath
        )
        if not exp_path:
            break
        acdc_myutils.addToRecentPaths(exp_path)
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

class SpotsItemPropertiesDialog(QBaseDialog):
    sigDeleteSelecAnnot = Signal(object)

    def __init__(
            self, df_spots_files, spotmax_out_path=None, parent=None, 
            state=None, color_idx=0, selected_file=None
        ):
        self.cancel = True
        self.loop = None
        self.clickedButton = None
        self.spotmax_out_path = spotmax_out_path

        super().__init__(parent)

        self.setWindowTitle('Load spots table to visualize or edit')

        layout = acdc_widgets.FormLayout()

        row = 0
        h5fileCombobox = QComboBox()
        h5fileCombobox.addItems(df_spots_files)
        
        if selected_file is not None:
            h5fileCombobox.setCurrentText(selected_file)
        
        if state is not None:
            h5fileCombobox.setCurrentText(state['selected_file'])
            h5fileCombobox.setDisabled(True)
        
        self.h5fileCombobox = h5fileCombobox
        body_txt = ("""
            Select which table you want to plot.
        """)
        h5FileInfoTxt = (f'{html_func.paragraph(body_txt)}')
        self.dfSpotsFileWidget = acdc_widgets.formWidget(
            h5fileCombobox, addInfoButton=True, labelTextLeft='Table to plot: ',
            parent=self, infoTxt=h5FileInfoTxt
        )
        layout.addFormWidget(self.dfSpotsFileWidget, row=row)
        self.h5fileCombobox.currentTextChanged.connect(self.setSizeFromTable)

        row += 1
        self.nameInfoLabel = QLabel()
        layout.addWidget(
            self.nameInfoLabel, row, 0, 1, 2, alignment=Qt.AlignCenter
        )

        row += 1
        symbolInfoTxt = ("""
        <b>Symbol</b> used to draw the spot.
        """)
        symbolInfoTxt = (f'{html_func.paragraph(symbolInfoTxt)}')
        self.symbolWidget = acdc_widgets.formWidget(
            acdc_widgets.pgScatterSymbolsCombobox(), addInfoButton=True,
            labelTextLeft='Symbol: ', parent=self, infoTxt=symbolInfoTxt
        )
        if state is not None:
            self.symbolWidget.widget.setCurrentText(state['symbol_text'])
        layout.addFormWidget(self.symbolWidget, row=row)

        row += 1
        shortcutInfoTxt = ("""
        <b>Shortcut</b> that you can use to <b>activate/deactivate</b> annotation
        of this spots item.<br><br> Leave empty if you don't need a shortcut.
        """)
        shortcutInfoTxt = (f'{html_func.paragraph(shortcutInfoTxt)}')
        self.shortcutWidget = acdc_widgets.formWidget(
            acdc_widgets.ShortcutLineEdit(), addInfoButton=True,
            labelTextLeft='Shortcut: ', parent=self, infoTxt=shortcutInfoTxt
        )
        if state is not None:
            self.shortcutWidget.widget.setText(state['shortcut'])
        layout.addFormWidget(self.shortcutWidget, row=row)

        row += 1
        descInfoTxt = ("""
        <b>Description</b> will be used as the <b>tool tip</b> that will be
        displayed when you hover with the mouse cursor on the toolbar button
        specific for this annotation.
        """)
        descInfoTxt = (f'{html_func.paragraph(descInfoTxt)}')
        self.descWidget = acdc_widgets.formWidget(
            QPlainTextEdit(), addInfoButton=True,
            labelTextLeft='Description: ', parent=self, infoTxt=descInfoTxt
        )
        if state is not None:
            self.descWidget.widget.setPlainText(state['description'])
        layout.addFormWidget(self.descWidget, row=row)

        row += 1
        self.colorButton = acdc_widgets.myColorButton(color=(255, 0, 0))
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        self.colorButton.setCursor(Qt.PointingHandCursor)
        self.colorWidget = acdc_widgets.formWidget(
            self.colorButton, addInfoButton=False, stretchWidget=False,
            labelTextLeft='Symbol color: ', parent=self, 
            widgetAlignment='left'
        )
        if state is not None:
            self.colorButton.setColor(state['symbolColor'])
        else:
            color_idx = color_idx%6
            rgb = SIX_RGBs_RAINBOW[color_idx]
            self.colorButton.setColor(rgb)
        layout.addFormWidget(self.colorWidget, row=row)
        row += 1
        self.sizeSpinBox = acdc_widgets.SpinBox()
        self.sizeSpinBox.setMinimum(1)
        self.sizeSpinBox.setValue(3)

        self.sizeWidget = acdc_widgets.formWidget(
            self.sizeSpinBox, addInfoButton=False, stretchWidget=False,
            labelTextLeft='Symbol size: ', parent=self, 
            widgetAlignment='left'
        )
        if state is not None:
            self.sizeSpinBox.setValue(state['size'])
        layout.addFormWidget(self.sizeWidget, row=row)

        row += 1
        self.opacitySlider = acdc_widgets.sliderWithSpinBox(
            isFloat=True, normalize=True
        )
        self.opacitySlider.setMinimum(0)
        self.opacitySlider.setMaximum(100)
        self.opacitySlider.setValue(0.3)

        self.opacityWidget = acdc_widgets.formWidget(
            self.opacitySlider, addInfoButton=False, stretchWidget=True,
            labelTextLeft='Symbol opacity: ', parent=self
        )
        if state is not None:
            self.opacitySlider.setValue(state['opacity'])
        layout.addFormWidget(self.opacityWidget, row=row)

        row += 1
        layout.addItem(QSpacerItem(5, 5), row, 0)

        row += 1
        noteText = (
            '<br><i>NOTE: you can change these options later with<br>'
            '<b>RIGHT-click</b> on the associated left-side <b>toolbar button<b>.</i>'
        )
        noteLabel = QLabel(html_func.paragraph(noteText, font_size='11px'))
        layout.addWidget(noteLabel, row, 1, 1, 3)

        buttonsLayout = QHBoxLayout()

        self.okButton = acdc_widgets.okPushButton('  Ok  ')
        cancelButton = acdc_widgets.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(self.okButton)

        cancelButton.clicked.connect(self.cancelCallBack)
        self.cancelButton = cancelButton
        self.okButton.clicked.connect(self.ok_cb)
        self.okButton.setFocus()

        mainLayout = QVBoxLayout()

        mainLayout.addLayout(layout)
        mainLayout.addStretch(1)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        
        if state is None:
            self.setSizeFromTable(self.h5fileCombobox.currentText())
    
    def setSizeFromTable(self, filename):
        from .core import ZYX_RESOL_COLS
        df = io.load_spots_table(self.spotmax_out_path, filename)
        try:
            size = round(df[ZYX_RESOL_COLS[1]].iloc[0])
        except Exception as err:
            return
        self.sizeSpinBox.setValue(size)

    def checkName(self, text):
        if not text:
            txt = 'Name cannot be empty'
            self.nameInfoLabel.setText(
                html_func.paragraph(
                    txt, font_size='11px', font_color='red'
                )
            )
            return
        for name in self.internalNames:
            if name.find(text) != -1:
                txt = (
                    f'"{text}" cannot be part of the name, '
                    'because <b>reserved<b>.'
                )
                self.nameInfoLabel.setText(
                    html_func.paragraph(
                        txt, font_size='11px', font_color='red'
                    )
                )
                break
        else:
            self.nameInfoLabel.setText('')

    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        self.colorButton.colorDialog.setWindowFlags(
            Qt.Window | Qt.WindowStaysOnTopHint
        )
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)

    def ok_cb(self, checked=True):
        self.cancel = False
        self.clickedButton = self.okButton
        self.toolTip = (
            f'Table name: {self.dfSpotsFileWidget.widget.currentText()}\n\n'
            f'Edit properties: right-click on button\n\n'
            f'Description: {self.descWidget.widget.toPlainText()}\n\n'
            f'SHORTCUT: "{self.shortcutWidget.widget.text()}"'
        )

        symbol = self.symbolWidget.widget.currentText()
        self.symbol = re.findall(r"\'(.+)\'", symbol)[0]

        self.state = {
            'selected_file': self.dfSpotsFileWidget.widget.currentText(),
            'symbol_text':  self.symbolWidget.widget.currentText(),
            'pg_symbol': self.symbol,
            'shortcut': self.shortcutWidget.widget.text(),
            'description': self.descWidget.widget.toPlainText(),
            'symbolColor': self.colorButton.color(),
            'size': self.sizeSpinBox.value(),
            'opacity': self.opacitySlider.value()
        }
        self.close()

    def cancelCallBack(self, checked=True):
        self.cancel = True
        self.clickedButton = self.cancelButton
        self.close()

class SelectFolderToAnalyse(QBaseDialog):
    def __init__(
            self, parent=None, preSelectedPaths=None, onlyExpPaths=False, 
            scanFolderTree=True
        ):
        super().__init__(parent)
        
        self.cancel = True
        self.onlyExpPaths = onlyExpPaths
        self.setWindowTitle('Select experiments to analyse')
        self.scanTree = scanFolderTree
        
        mainLayout = QVBoxLayout()
        
        instructionsText = html_func.paragraph(
            'Drag and drop folders or click on <code>Browse</code> button to '
            '<b>add</b> as many <b>paths</b> '
            'as needed.<br>', font_size='14px'
        )
        instructionsLabel = QLabel(instructionsText)
        instructionsLabel.setAlignment(Qt.AlignCenter)
        
        infoText = html_func.paragraph(            
            'A <b>valid folder</b> is either a <b>Position</b> folder, '
            'or an <b>experiment folder</b> (containing Position_n folders),<br>'
            'or any folder that contains <b>multiple experiment folders</b>.<br><br>'
            
            'In the last case, SpotMAX will automatically scan the entire tree of '
            'sub-directories<br>'
            'and will add all experiments having the right folder structure.<br>',
            font_size='12px'
        )
        infoLabel = QLabel(infoText)
        infoLabel.setAlignment(Qt.AlignCenter)
        
        self.listWidget = acdc_widgets.listWidget()
        self.listWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        if preSelectedPaths is not None:
            self.listWidget.addItems(preSelectedPaths)
        
        buttonsLayout = acdc_widgets.CancelOkButtonsLayout()

        delButton = acdc_widgets.delPushButton('Remove selected path(s)')
        browseButton = acdc_widgets.browseFileButton(
            'Add folder...', openFolder=True, 
            start_dir=acdc_myutils.getMostRecentPath()
        )
        
        buttonsLayout.insertWidget(3, delButton)
        buttonsLayout.insertWidget(4, browseButton)
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        browseButton.sigPathSelected.connect(self.addFolderPath)
        delButton.clicked.connect(self.removePaths)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addWidget(instructionsLabel)
        mainLayout.addWidget(infoLabel)
        mainLayout.addWidget(self.listWidget)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch(1)
        
        self.setLayout(mainLayout)
        
        self.setAcceptDrops(True)
        
        font = config.font()
        self.setFont(font)
    
    def dragEnterEvent(self, event):
        event.acceptProposedAction()
    
    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        for url in event.mimeData().urls():
            dropped_path = url.toLocalFile()
            if os.path.isfile(dropped_path):
                dropped_path = os.path.dirname(dropped_path)
            
            QTimer.singleShot(50, partial(self.addFolderPath, dropped_path))
    
    def pathsList(self):
        return [
            self.listWidget.item(i).text().replace('\\', '/') 
            for i in range(self.listWidget.count())
        ]
    
    def ok_cb(self):
        self.cancel = False
        self.paths = self.pathsList()
        self.close()
    
    def warnNoValidPathsFound(self, selected_path):
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph("""
            The selected path (see below) <b>does not contain any valid folder.</b><br><br>
            Please, make sure to select a Position folder, the Images folder 
            inside a Position folder, or any folder containing a Position folder 
            as a sub-directory.<br><br>
            Thank you for your patience!<br><br>
            Selected path:
        """)
        msg.warning(
            self, 'Training workflow generated', txt, 
            commands=(f'{selected_path}',),
            path_to_browse=selected_path
        )
    
    def addFolderPath(self, path):
        acdc_myutils.addToRecentPaths(path)
        
        folder_type = acdc_myutils.determine_folder_type(path)     
        is_pos_folder, is_images_folder, folder_path = folder_type 
        if is_pos_folder:
            paths = [path]
        elif is_images_folder:
            paths = [os.path.dirname(path)]
        elif self.scanTree:
            pathScanner = io.expFolderScanner(path)
            pathScanner.getExpPaths(path)
            paths = pathScanner.expPaths
        else:
            paths = [path]
        
        if not paths:
            self.warnNoValidPathsFound(path)
        
        for selectedPath in paths:
            if self.onlyExpPaths:
                selectedPath = acdc_load.get_exp_path(selectedPath)
            
            selectedPath = selectedPath.replace('\\', '/')
            if selectedPath in self.pathsList():
                print(
                    f'[WARNING]: The following path was already selected: '
                    f'"{selectedPath}"'
                )
                return
                
            self.listWidget.addItem(selectedPath)
    
    def removePaths(self):
        for item in self.listWidget.selectedItems():
            row = self.listWidget.row(item)
            self.listWidget.takeItem(row)

class SetMeasurementsDialog(QBaseDialog):
    sigOk = Signal(object)
    
    def __init__(
            self, parent=None, selectedMeasurements=None, 
            isSpotFitRequested=False
        ):
        self.cancel = True
        
        super().__init__(parent=parent)
        
        self.setWindowTitle('Set SpotMAX measurements to save')
        
        self.tabScrollbar = QScrollArea()
        self.tabWidget = QTabWidget()
        self.tabScrollbar.setWidget(self.tabWidget)
        self.tabScrollbar.setWidgetResizable(True)
        
        self.lastSelectionCp = None
        if os.path.exists(last_selection_meas_filepath):
            self.lastSelectionCp = config.ConfigParser()
            self.lastSelectionCp.read(last_selection_meas_filepath)
        
        mainLayout = QVBoxLayout()
        
        searchLineEdit = acdc_widgets.SearchLineEdit()
        
        showColNamesLabel = QLabel('Show column names')
        colNamesToggle = acdc_widgets.Toggle()
        
        searchLayout = QHBoxLayout()
        searchLayout.addWidget(showColNamesLabel)
        searchLayout.addWidget(colNamesToggle)
        searchLayout.addStretch(2)
        searchLayout.addWidget(searchLineEdit)
        searchLayout.setStretch(3, 1)
        
        self.groupBoxes = {'single_spot': {}, 'aggr': {}}
        
        self.singleSpotFeatGroups = features.get_features_groups()
        self.singleSpotFeatToColMapper = (
            features.feature_names_to_col_names_mapper()
        )
        singleSpotTab = self.buildTab(
            isSpotFitRequested, self.singleSpotFeatGroups, 'single_spot'
        )        
        self.tabWidget.addTab(singleSpotTab, 'Single-spot measurements')
        
        self.aggrFeatGroups = features.get_aggr_features_groups()
        self.aggrFeatToColMapper = (
            features.aggr_feature_names_to_col_names_mapper()
        )
        aggrTab = self.buildTab(
            isSpotFitRequested, self.aggrFeatGroups, 'aggr'
        )        
        self.tabWidget.addTab(aggrTab, 'Aggregated measurements')
        
        self.mappers = {
            'single_spot': self.singleSpotFeatToColMapper,
            'aggr': self.aggrFeatToColMapper
        }
        self.groups = {
            'single_spot': self.singleSpotFeatGroups,
            'aggr': self.aggrFeatGroups
        }
        
        self.setSelectedMeasurementsChecked(selectedMeasurements)
        
        additionalButtons = []
        self.selectAllButton = acdc_widgets.selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.setCheckedAll)
        additionalButtons.append(self.selectAllButton)
        
        if self.lastSelectionCp is not None:
            self.loadLastSelButton = acdc_widgets.reloadPushButton(
                '  Load last selection...  '
            )
            self.loadLastSelButton.clicked.connect(self.loadLastSelection)
            additionalButtons.append(self.loadLastSelButton)
            
        buttonsLayout = acdc_widgets.CancelOkButtonsLayout(
            additionalButtons=additionalButtons
        )
            
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(searchLayout)
        mainLayout.addSpacing(20)
        mainLayout.addWidget(self.tabScrollbar)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setFont(font)
        self.setLayout(mainLayout)       
        
        searchLineEdit.textEdited.connect(self.searchAndHighlight)
        colNamesToggle.toggled.connect(self.showColNamesToggled)
    
    def buildTab(self, isSpotFitRequested, featGroups, tabKey):
        maxNumElementsPerVBox = 16
        rowNumElements = 0
        row = 0
        groupBoxesHLayout = QHBoxLayout()
        groupBoxesVLayout = QVBoxLayout()
        for groupName, metrics in featGroups.items():
            rowSpan = len(metrics) + 1
            rowNumElements += rowSpan
            
            if tabKey == 'single_spot':
                infoUrl = docs.single_spot_feature_group_name_to_url(groupName)
            else:
                infoUrl = docs.aggr_feature_group_name_to_url(groupName)
                
            itemsInfoUrls = {name:infoUrl for name in metrics}
            
            lastSelection = self.getLastSelectionSection(
                self.lastSelectionCp, f'{tabKey};;{groupName}'
            )      
            
            groupbox = acdc_widgets.SetMeasurementsGroupBox(
                groupName, metrics, parent=self, lastSelection=lastSelection,
                itemsInfoUrls=itemsInfoUrls
            )
            groupBoxesVLayout.addWidget(groupbox)
            groupBoxesVLayout.setStretch(row, rowSpan)
            row += 1
            # printl(groupName, row, col, rowSpan)
            # groupBoxesLayout.addWidget(groupbox, row, col, rowSpan, 1)           
            self.groupBoxes[tabKey][groupName] = groupbox
            
            if not isSpotFitRequested and groupName.startswith('Spotfit'):
                groupbox.setChecked(False)
                groupbox.setDisabled(True)
                groupbox.setToolTip(
                    'Spotfit metrics cannot be saved because you did not '
                    'activate the parameter "Compute spots size".'
                )
            
            if rowNumElements >= maxNumElementsPerVBox:
                groupBoxesHLayout.addLayout(groupBoxesVLayout) 
                groupBoxesVLayout = QVBoxLayout()
                rowNumElements = 0
                row = 0
        
        # Add last layout
        groupBoxesHLayout.addLayout(groupBoxesVLayout)
        
        if tabKey == 'aggr':
            groupBoxesHLayout.setStretch(0, 1)
            groupBoxesHLayout.setStretch(1, 1)
            groupBoxesHLayout.addStretch(2)
        
        widget = QWidget()
        widget.setLayout(groupBoxesHLayout)
        return widget
    
    def setSelectedMeasurementsChecked(self, selectedMeasurements):
        if selectedMeasurements is None:
            return
        for tabKey, groupboxes in self.groupBoxes.items():
            if tabKey not in selectedMeasurements:
                continue
            
            mapper = self.mappers[tabKey]
            for groupName, groupbox in groupboxes.items():
                for checkbox in groupbox.checkboxes.values():
                    key = f'{groupName}, {checkbox.text()}'
                    colname = mapper[key]
                    checkbox.setChecked(
                        colname in selectedMeasurements[tabKey]
                    )
    
    def getLastSelectionSection(self, lastSelectionCp, sectionName):
        if lastSelectionCp is None:
            return
        
        if not lastSelectionCp.has_section(sectionName):
            return
        
        lastSelection = {}
        for option in lastSelectionCp.options(sectionName):
            lastSelection[option] = lastSelectionCp.getboolean(
                sectionName, option
            )
        
        return lastSelection
    
    def searchAndHighlight(self, text):
        if len(text) == 1:
            return
        
        for tabKey, groupboxes in self.groupBoxes.items():
            for groupName, groupbox in groupboxes.items():
                groupbox.highlightCheckboxesFromSearchText(text)
    
    def setCheckedAll(self, checked):
        for tabKey, groupboxes in self.groupBoxes.items():
            for groupName, groupbox in groupboxes.items():
                groupbox.selectAllButton.setChecked(checked)
    
    def loadLastSelection(self):
        for tabKey, groupboxes in self.groupBoxes.items():
            for groupName, groupbox in groupboxes.items():
                if not hasattr(groupbox, 'loadLastSelButton'):
                    continue
                groupbox.loadLastSelButton.click()
    
    def showColNamesToggled(self, checked):
        for tabKey, groupboxes in self.groupBoxes.items():
            mapper = self.mappers[tabKey]
            groups = self.groups[tabKey]
            for groupName, groupbox in groupboxes.items():
                for c, checkbox in enumerate(groupbox.checkboxes.values()):
                    if checked:
                        key = f'{groupName}, {checkbox.text()}'
                        colname = mapper[key]
                        newText = colname
                    else:
                        newText = groups[groupName][c]
                    checkbox.setText(newText)
        QTimer.singleShot(200, self.resizeGroupBoxes)
    
    def resizeGroupBoxes(self):
        for tabKey, groupboxes in self.groupBoxes.items():
            for groupName, groupbox in groupboxes.items():
                groupbox.resizeWidthNoScrollBarNeeded()
    
    def saveLastSelection(self):
        cp = config.ConfigParser()
        for tabKey, groupboxes in self.groupBoxes.items():
            for groupName, groupbox in groupboxes.items():
                if not groupbox.isChecked():
                    continue
                cp[f'{tabKey};;{groupName}'] = {}
                for name, checkbox in groupbox.checkboxes.items():
                    cp[f'{tabKey};;{groupName}'][name] = str(checkbox.isChecked())
        with open(last_selection_meas_filepath, 'w') as ini:
            cp.write(ini)
    
    def getSelectedMeasurements(self):
        selectedMeasurements = {}
        for tabKey, groupboxes in self.groupBoxes.items():
            selectedMeasurements[tabKey] = {}
            mapper = self.mappers[tabKey]
            for groupName, groupbox in groupboxes.items():
                if not groupbox.isChecked():
                    continue
                for c, checkbox in enumerate(groupbox.checkboxes.values()):
                    if not checkbox.isChecked():
                        continue
                    key = f'{groupName}, {checkbox.text()}'
                    colname = mapper[key]
                    selectedMeasurements[tabKey][colname] = key
        return selectedMeasurements
                
    def ok_cb(self):
        self.cancel = False
        self.saveLastSelection()
        selectedMeasurements = self.getSelectedMeasurements()
        self.close()
        self.sigOk.emit(selectedMeasurements)
    
    def show(self, block=False):
        super().show(block=False)
        topScreen = self.screen().geometry().top()
        leftScreen = self.screen().geometry().left()
        screenHeight = self.screen().size().height()
        screenWidth = self.screen().size().width()
        topWindow = round(topScreen + (0.15*screenHeight/2))
        leftWindow = round(leftScreen + (0.3*screenWidth/2))
        widthWindow = round(0.7*screenWidth)
        heightWindow = round(0.85*screenHeight)
        self.setGeometry(leftWindow, topWindow, widthWindow, heightWindow)
        QTimer.singleShot(200, self.resizeGroupBoxes)
        super().show(block=block)
        
class EditResultsGropbox(QGroupBox):
    sigEditResultsToggled = Signal(bool)
    sigSaveEditedResults = Signal(str, str)
    sigComputeFeatures = Signal(str, str, str)    
    
    def __init__(self, parent=None, infoText=None):
        super().__init__(parent)
        
        self.setTitle('Edit results')
        
        mainLayout = QVBoxLayout()
        
        gridLayout = QGridLayout()
        gridLayout.setColumnStretch(0, 1)
        gridLayout.setColumnStretch(1, 2)
        
        # editResultsFormWidget = widgets.formWidget(
        #     acdc_widgets.Toggle(), 
        #     labelTextLeft='Activate edits',
        #     stretchWidget=False
        # )
        
        '----------------------------------------------------------------------'
        row = 0
        self.editResultsToggle = acdc_widgets.Toggle()
        self.editResultsToggle.label = QLabel('Activate edits')
        gridLayout.addWidget(
            self.editResultsToggle.label, row, 0, alignment=Qt.AlignRight
        )
        gridLayout.addWidget(
            self.editResultsToggle, row, 1, alignment=Qt.AlignCenter
        )
        '======================================================================'
        
        '----------------------------------------------------------------------'
        row += 1
        self.inputTextLineEdit = widgets.CenteredAlphaNumericLineEdit()
        self.inputTextLineEdit.label = QLabel('Text to add to new filenames')
        gridLayout.addWidget(
            self.inputTextLineEdit.label, row, 0, alignment=Qt.AlignRight
        )
        gridLayout.addWidget(
            self.inputTextLineEdit, row, 1
        )
        '======================================================================'

        buttonsLayout = QHBoxLayout()
        computeButton = widgets.computePushButton(
            'Compute features of new spots...'
        )
        saveButton = acdc_widgets.savePushButton(
            'Save edited results'
        )
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(computeButton)
        buttonsLayout.addWidget(saveButton)
        
        
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
        
        self.editResultsToggle.toggled.connect(self.emitSigEditResultsToggled)
        computeButton.clicked.connect(self.emitSigComputeFeatures)
        saveButton.clicked.connect(self.emitSigSaveEditedResults)
        
        self.initState()
    
    def initState(self):
        if not hasattr(self, 'spotsItems'):
            self.spotsItems = None
            return
        
        if self.spotsItems is None:
            return
        
        self.spotsItems.setEditsEnabled(False)
        self.spotsItems = None
        
    def setLoadedData(self, spotsItems, img_data, segm_data):
        self.spotsItems = spotsItems
        spotsItems.initEdits(img_data, segm_data)
    
    def warnResultsNotLoaded(self, action='save the'):
        txt = html_func.paragraph(f"""
            In order to {action} results, you first need to load some :D<br><br>
            To do so, click on the <code>Load results from previous analysis...</code> 
            button on the top-left of the tab. 
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Results not loaded', txt)

    
    def warnLoadedSegmDoesNotCorrespondToAnalysisParams(
            self, loadedSegmEndname, analysisSegmEndname
        ):
        txt = html_func.paragraph(f"""
            <b>Editing results is not possible<\b> because you loaded the segmentation 
            file ending with <code>{loadedSegmEndname}</code>,<br>
            while the segmentation file used for the analysis of the loaded 
            spots table is <code>{analysisSegmEndname}</code>.<br><br>
            
            Load the segmentation file <code>{loadedSegmEndname}</code> to enable 
            edits.<br><br>
            Thank you for your patience!
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Segmentation files mismatch', txt)
    
    def emitSigEditResultsToggled(self, checked):
        if checked and self.spotsItems is None:
            self.warnResultsNotLoaded(action='edit the')
            self.editResultsToggle.setChecked(False)
            return

        if checked:
            loadedSegmEndname, analysisSegmEndname = (
                self.spotsItems.getLoadedSegmAndAnalysisSegm()
            )
            if loadedSegmEndname != analysisSegmEndname:
                self.warnLoadedSegmDoesNotCorrespondToAnalysisParams(
                    loadedSegmEndname, analysisSegmEndname
                )
                self.editResultsToggle.setChecked(False)    
                return
        
        self.sigEditResultsToggled.emit(checked)
    
    def warnInputTextEmpty(self):
        self.inputTextLineEdit.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
        txt = html_func.paragraph("""
            <code>Text to add to new filenames</code> cannot be empty.<br><br>
            Please provide some text to add to the filenames, thanks.  
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Text to add cannot be empty', txt)
        self.inputTextLineEdit.setStyleSheet('')
    
    def warnNothingToSave(self, action='save it'):
        txt = html_func.paragraph(f"""
            The loaded spots table was <b>not edited</b>.<br><br>  
            Are you sure you want to {action}?
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        noButton, yesButton = msg.warning(
            self, 'Loaded spots table was not edited', txt, 
            buttonsTexts=(
                f'No, do not {action}', f'Yes, {action} anyway'
            )
        )
        return msg.clickedButton == yesButton

    def warnFeaturesNotComputed(self):
        txt = html_func.paragraph("""
            The <b>features</b> of the newly added spots were 
            <b>not computed</b>.<br><br>  
            If you need these features, please click on the 
            <code>Compute features of new spots...</code><br> button 
            before saving the edited spots table.<br><br>
            Do you want to save the table without features?
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        noButton, yesButton = msg.warning(
            self, 'Loaded spots table was not edited', txt, 
            buttonsTexts=(
                'No, do not save it', 'Yes, save it without features'
            )
        )
        return msg.clickedButton == yesButton
    
    def emitSigSaveEditedResults(self):        
        if self.spotsItems is None:
            self.warnResultsNotLoaded()
            return
        
        button = self.spotsItems.getActiveButton()
        if 'edited' not in button.df.columns:
            saveAnyway = self.warnNothingToSave()
            if not saveAnyway:
                return
        
        if (button.df['x_local'] == -1).any():
            saveAnyway = self.warnFeaturesNotComputed()
            if not saveAnyway:
                return
        
        if not self.inputTextLineEdit.text():
            self.warnInputTextEmpty()
            return 
        
        text_to_add = self.inputTextLineEdit.text()
        src_df_filename = button.filename
        self.sigSaveEditedResults.emit(src_df_filename, text_to_add)

    def emitSigComputeFeatures(self):
        if self.spotsItems is None:
            self.warnResultsNotLoaded(action='compute features of the edited')
            return
        
        button = self.spotsItems.getActiveButton()
        if 'edited' not in button.df.columns:
            saveAnyway = self.warnNothingToSave(action='re-compute features')
            if not saveAnyway:
                return
        
        if not self.inputTextLineEdit.text():
            self.warnInputTextEmpty()
            return 
        
        text_to_add = self.inputTextLineEdit.text()
        src_df_filename = button.filename
        ini_filepath = self.spotsItems.getAnalysisParamsIniFilepath(button)
        self.sigComputeFeatures.emit(text_to_add, src_df_filename, ini_filepath)

class SetupUnetTrainingDialog(QBaseDialog):
    def __init__(self, parent=None):        
        self.cancel = True
        super().__init__(parent)
        
        self.selectedSpotsCoordsFiles = None
        self.selectedChannelNamesFiles = None
        self.selectedMasksFiles = None
        self.selectedSpotMaskSizes = None
        self.selectedPixelSizes = None
        self.selectedTrainPositions = None
        self.selectedValPositions = None
        
        self.setWindowTitle('Setup SpotMAX AI Training Workflow')
        
        mainLayout = QVBoxLayout()
                
        paramsLayout = QGridLayout()
        
        self.isSetButtonsMapper = {}
        
        row = 0
        paramsLayout.addWidget(QLabel('Ground-truth Experiments: '), row, 0)
        self.selectExpPathsButton = acdc_widgets.editPushButton(
            ' Select/view experiment folders... '
        )
        paramsLayout.addWidget(self.selectExpPathsButton, row, 1, 1, 2)
        selectPosInfoButton = acdc_widgets.infoPushButton()
        paramsLayout.addWidget(selectPosInfoButton, row, 3)
        self.areExpPathsSetButton = widgets.IsFieldSetButton()
        paramsLayout.addWidget(self.areExpPathsSetButton , row, 4)
        
        row += 1
        paramsLayout.addWidget(QLabel('Spots channel names: '), row, 0)
        self.selectChannelNamesButton = acdc_widgets.editPushButton(
            ' Select/view spots channel names... '
        )
        paramsLayout.addWidget(self.selectChannelNamesButton, row, 1, 1, 2)
        isFieldSetButton = widgets.IsFieldSetButton()
        paramsLayout.addWidget(isFieldSetButton , row, 4)
        self.isSetButtonsMapper[self.selectChannelNamesButton] = (
            isFieldSetButton, 'selectedChannelNamesFiles'
        )
        
        row += 1
        paramsLayout.addWidget(
            QLabel('Randomly choose train-validation: '), row, 0
        )
        self.randomSplitToggle = acdc_widgets.Toggle()
        paramsLayout.addWidget(
            self.randomSplitToggle, row, 1, 1, 2, alignment=Qt.AlignCenter
        )
        self.randomSplitToggle.setChecked(True)
        
        row += 1
        paramsLayout.addWidget(
            QLabel('Percentage of Positions for <b>training</b> '), row, 0
        )
        self.trainPercSpinbox = acdc_widgets.DoubleSpinBox()
        self.trainPercSpinbox.setMaximum(100)
        self.trainPercSpinbox.setSingleStep(5)
        self.trainPercSpinbox.setValue(80)
        paramsLayout.addWidget(self.trainPercSpinbox, row, 1, 1, 2)
        
        row += 1
        paramsLayout.addWidget(
            QLabel('Percentage of Positions for <b>validation</b>: '), row, 0
        )
        self.valPercSpinbox = acdc_widgets.DoubleSpinBox()
        self.valPercSpinbox.setMaximum(100)
        self.valPercSpinbox.setSingleStep(5)
        self.valPercSpinbox.setValue(20)
        paramsLayout.addWidget(self.valPercSpinbox, row, 1, 1, 2)
        
        row += 1
        paramsLayout.addWidget(QLabel('Training Positions: '), row, 0)
        self.selectTrainPosButton = acdc_widgets.editPushButton(
            ' Select/view experiment folders to use as training data... '
        )
        paramsLayout.addWidget(self.selectTrainPosButton, row, 1, 1, 2)
        isFieldSetButton = widgets.IsFieldSetButton()
        paramsLayout.addWidget(isFieldSetButton , row, 4)
        self.isSetButtonsMapper[self.selectExpPathsButton] = (
            isFieldSetButton, 'selectedTrainPositions'
        )
        
        row += 1
        paramsLayout.addWidget(QLabel('Validation Positions: '), row, 0)
        self.selectValPosButton = acdc_widgets.editPushButton(
            ' Select/view experiment folders to use as validation data... '
        )
        paramsLayout.addWidget(self.selectValPosButton, row, 1, 1, 2)
        isFieldSetButton = widgets.IsFieldSetButton()
        paramsLayout.addWidget(isFieldSetButton , row, 4)
        self.isSetButtonsMapper[self.selectValPosButton] = (
            isFieldSetButton, 'selectedValPositions'
        )
        
        row += 1
        paramsLayout.addWidget(QLabel('Ground-truth spots masks: '), row, 0)
        self.selectMasksFilesButton = acdc_widgets.editPushButton(
            ' Select/view spots masks files... '
        )
        paramsLayout.addWidget(self.selectMasksFilesButton, row, 1, 1, 2)
        isFieldSetButton = widgets.IsFieldSetButton()
        paramsLayout.addWidget(isFieldSetButton , row, 4)
        self.isSetButtonsMapper[self.selectMasksFilesButton] = (
            isFieldSetButton, 'selectedMasksFiles'
        )
        
        row += 1
        paramsLayout.addWidget(QLabel('Ground-truth spots coords: '), row, 0)
        self.selectCoordsFilesButton = acdc_widgets.editPushButton(
            ' Select/view spots coords files... '
        )
        paramsLayout.addWidget(self.selectCoordsFilesButton, row, 1, 1, 2)
        isFieldSetButton = widgets.IsFieldSetButton()
        paramsLayout.addWidget(isFieldSetButton , row, 4)
        self.isSetButtonsMapper[self.selectCoordsFilesButton] = (
            isFieldSetButton, 'selectedSpotsCoordsFiles'
        )
        
        row += 1
        paramsLayout.addWidget(QLabel('Spot mask size: '), row, 0)
        self.selectSpotMaskSizeButton = acdc_widgets.editPushButton(
            ' Select/view spot mask size... '
        )
        paramsLayout.addWidget(self.selectSpotMaskSizeButton, row, 1, 1, 2)
        isFieldSetButton = widgets.IsFieldSetButton()
        paramsLayout.addWidget(isFieldSetButton , row, 4)
        self.isSetButtonsMapper[self.selectSpotMaskSizeButton] = (
            isFieldSetButton, 'selectedSpotMaskSizes'
        )
        
        row += 1
        paramsLayout.addWidget(QLabel('Pixel size (XY): '), row, 0)
        self.selectedPixelSizesButton = acdc_widgets.editPushButton(
            ' Select/view XY pixel size... '
        )
        paramsLayout.addWidget(self.selectedPixelSizesButton, row, 1, 1, 2)
        isFieldSetButton = widgets.IsFieldSetButton()
        paramsLayout.addWidget(isFieldSetButton, row, 4)
        self.isSetButtonsMapper[self.selectedPixelSizesButton] = (
            isFieldSetButton, 'selectedPixelSizes'
        )
        
        row += 1
        paramsLayout.addWidget(QLabel('Model size: '), row, 0)
        self.modelSizeCombobox = QComboBox()
        self.modelSizeCombobox.addItems(('Large', 'Medium', 'Small'))
        paramsLayout.addWidget(self.modelSizeCombobox, row, 1, 1, 2)
        
        row += 1
        paramsLayout.addWidget(QLabel('Rescale images to pixel size: '), row, 0)
        self.rescaleToPixelSizeWidget = widgets.FloatLineEdit()
        paramsLayout.addWidget(self.rescaleToPixelSizeWidget, row, 1, 1, 2)
        
        row += 1
        paramsLayout.addWidget(QLabel('Augment images: '), row, 0)
        self.augmentToggle = acdc_widgets.Toggle()
        self.augmentToggle.setChecked(True)
        paramsLayout.addWidget(
            self.augmentToggle, row, 1, 1, 2, alignment=Qt.AlignCenter
        )
        augmentInfoButton = acdc_widgets.infoPushButton()
        paramsLayout.addWidget(augmentInfoButton, row, 3)
        
        row += 1
        paramsLayout.addWidget(QLabel('DoG filter (z,y,x) spot size: '), row, 0)
        self.dogFilterSpotSizeWidget = acdc_widgets.VectorLineEdit()
        paramsLayout.addWidget(self.dogFilterSpotSizeWidget, row, 1, 1, 2)
        paramsLayout.addWidget(QLabel('pixel'), row, 3)
        
        row += 1
        paramsLayout.addWidget(QLabel('Gaussian filter sigma(s) 1: '), row, 0)
        self.gaussSigmaOneWidget = widgets.Gaussian3SigmasLineEdit()
        paramsLayout.addWidget(self.gaussSigmaOneWidget, row, 1, 1, 2)
        paramsLayout.addWidget(QLabel('pixel'), row, 3)
        
        row += 1
        paramsLayout.addWidget(QLabel('Gaussian filter sigma(s) 2: '), row, 0)
        self.gaussSigmaOneTwoWidget = widgets.Gaussian3SigmasLineEdit()
        paramsLayout.addWidget(self.gaussSigmaOneTwoWidget, row, 1, 1, 2)
        paramsLayout.addWidget(QLabel('pixel'), row, 3)
        
        row += 1
        paramsLayout.addWidget(QLabel('Pre-trained weights file: '), row, 0)
        self.preTrainedWeightsFilepathControl = acdc_widgets.ElidingLineEdit()
        paramsLayout.addWidget(
            self.preTrainedWeightsFilepathControl, row, 1, 1, 2
        )
        browseModelWeightsButton = acdc_widgets.browseFileButton(
            ext={'PyTorch Model Weights': ['.pth']}, 
            title='Select spotMAX AI model weights',
            start_dir=os.path.expanduser('~/spotmax_appdata/unet_checkpoints'), 
        )
        browseModelWeightsButton.sigPathSelected.connect(
            self.preTrainedWeightsFilepathControl.setText
        )
        paramsLayout.addWidget(browseModelWeightsButton, row, 3)
        
        row += 1
        paramsLayout.addWidget(QLabel('Crop background: '), row, 0)
        self.cropBkgrToggle = acdc_widgets.Toggle()
        self.cropBkgrToggle.setChecked(True)
        paramsLayout.addWidget(
            self.cropBkgrToggle, row, 1, 1, 2, alignment=Qt.AlignCenter
        )
        cropBkgrInfoButton = acdc_widgets.infoPushButton()
        paramsLayout.addWidget(cropBkgrInfoButton, row, 3)
        
        row += 1
        paramsLayout.addWidget(QLabel('Crop background tolerance: '), row, 0)
        self.cropBkgrPadWidget = acdc_widgets.SpinBox()
        self.cropBkgrPadWidget.setValue(5)
        paramsLayout.addWidget(self.cropBkgrPadWidget, row, 1, 1, 2)
        paramsLayout.addWidget(QLabel('pixel'), row, 3)
        
        row += 1
        self.cropYspinbox = acdc_widgets.SpinBox()
        self.cropXspinbox = acdc_widgets.SpinBox()
        self.cropYspinbox.setValue(256)
        self.cropYspinbox.setMinimum(250)
        self.cropXspinbox.setValue(256)
        self.cropXspinbox.setMinimum(250)
        cropLayout = QHBoxLayout()
        cropLayout.addWidget(self.cropYspinbox)
        cropLayout.addWidget(self.cropXspinbox)
        paramsLayout.addWidget(QLabel('YX crops shape: '), row, 0)
        paramsLayout.addLayout(cropLayout, row, 1, 1, 2)
        cropsShapeInfoButton = acdc_widgets.infoPushButton()
        paramsLayout.addWidget(cropsShapeInfoButton, row, 3)
        
        row += 1
        self.maxNumCropsSpinbox = acdc_widgets.SpinBox()
        self.maxNumCropsSpinbox.setValue(-1)
        self.maxNumCropsSpinbox.setMinimum(-1)
        paramsLayout.addWidget(QLabel('Max number of crops per image: '), row, 0)
        paramsLayout.addWidget(self.maxNumCropsSpinbox, row, 1, 1, 2)
        maxNumCropsInfoButton = acdc_widgets.infoPushButton()
        paramsLayout.addWidget(maxNumCropsInfoButton, row, 3)
        
        row += 1
        paramsLayout.addWidget(QLabel('Folder where to save worflow: '), row, 0)
        self.folderPathWorflowWidget = acdc_widgets.ElidingLineEdit()
        paramsLayout.addWidget(
            self.folderPathWorflowWidget, row, 1, 1, 2
        )
        browseFolderWorkflowButton = acdc_widgets.browseFileButton(
            openFolder=True, 
            title='Select where to save training workflow file'
        )
        browseFolderWorkflowButton.sigPathSelected.connect(
            self.folderPathWorflowWidget.setText
        )
        paramsLayout.addWidget(browseFolderWorkflowButton, row, 3)
        
        row += 1
        paramsLayout.addWidget(QLabel('Workflow filename: '), row, 0)
        self.workflowFilenameWidget = acdc_widgets.alphaNumericLineEdit()
        self.workflowFilenameWidget.setAlignment(Qt.AlignCenter)
        now = datetime.datetime.now().strftime(r'%Y-%m-%d')
        default_filename = f'{now}_training_workflow'
        self.workflowFilenameWidget.setText(default_filename)
        paramsLayout.addWidget(self.workflowFilenameWidget, row, 1, 1, 2)
        paramsLayout.addWidget(QLabel('.ini'), row, 3)

        paramsLayout.setColumnStretch(0, 0)
        paramsLayout.setColumnStretch(1, 1)
        paramsLayout.setColumnStretch(3, 0)
        
        buttonsLayout = acdc_widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.setText('Done, generate worflow files')
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        saveToWorkflowFileButton = acdc_widgets.savePushButton(
            'Save to workflow file...'
        )
        buttonsLayout.insertWidget(3, saveToWorkflowFileButton)
        
        loadFromWorkflowFileButton = acdc_widgets.browseFileButton(
            'Load from workflow file...', 
            title='Select worflow INI file', 
            ext={'Training workflow file': ['.ini']},
            start_dir=acdc_myutils.getMostRecentPath(),
        )
        buttonsLayout.insertWidget(3, loadFromWorkflowFileButton)
        
        helpButton = acdc_widgets.helpPushButton('Help...')
        buttonsLayout.insertWidget(3, helpButton)
        
        mainLayout.addLayout(paramsLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        # mainLayout.addStretch(1)
        
        selectPosInfoButton.clicked.connect(self.showSelectPosInfo)
        self.selectExpPathsButton.clicked.connect(self.selectExperimentPaths)
        
        self.selectChannelNamesButton.clicked.connect(partial(
            self.selectEntries, 
            widget_name='widgets.EndnameLineEdit', 
            attrToSet='selectedChannelNamesFiles', 
            extensions={'TIFF': ['.tif']}, 
            use_value_as_widgets_value=True,
            allow_spotmax_output=False, 
            widgets_values_are_multiple_entries=True,
            allow_add_field=True, 
            rel_start_dir='Images', 
            enable_autofill=True
        ))
        
        self.selectCoordsFilesButton.clicked.connect(partial(
            self.selectEntries, 
            widget_name='widgets.EndnameLineEdit', 
            attrToSet='selectedSpotsCoordsFiles', 
            extensions={'Tables': ['.csv', '.h5']}, 
            allow_spotmax_output=True, 
            use_value_as_widgets_value=True,
            widgets_values_are_multiple_entries=True,
            depends_on_channels=True, 
            rel_start_dir='Position', 
            enable_autofill=True
        ))
        
        self.selectMasksFilesButton.clicked.connect(partial(
            self.selectEntries, 
            widget_name='widgets.EndnameLineEdit', 
            attrToSet='selectedMasksFiles', 
            extensions={'Masks': ['.tif', '.npz']}, 
            allow_spotmax_output=False, 
            use_value_as_widgets_value=True,
            widgets_values_are_multiple_entries=True,
            depends_on_channels=True, 
            rel_start_dir='Images',
            enable_autofill=True
        ))
        
        self.selectSpotMaskSizeButton.clicked.connect(partial(
            self.selectEntries, 
            attrToSet='selectedSpotMaskSizes', 
            widget_name='widgets.VoxelSizeWidget', 
            add_browse_button=False, 
            use_tooltip_as_widget_kwargs=True, 
            use_value_as_widgets_value=True, 
            entry_header='Spot mask radius', 
            add_apply_to_all_buttons=True
        ))
        
        self.selectedPixelSizesButton.clicked.connect(partial(
            self.selectEntries, 
            attrToSet='selectedPixelSizes', 
            widget_name='widgets.FloatLineEdit', 
            add_browse_button=False, 
            use_tooltip_as_widget_kwargs=False, 
            use_value_as_widgets_value=True, 
            entry_header='Pixel size (um/pixel)'
        ))
        
        self.selectTrainPosButton.clicked.connect(partial(
            self.selectEntries, 
            attrToSet='selectedTrainPositions', 
            widget_name='widgets.SelectPosFoldernamesButton', 
            add_browse_button=False, 
            use_tooltip_as_widget_kwargs=True, 
            use_value_as_widgets_value=True, 
            entry_header='Training Positions'
        ))
        
        self.selectValPosButton.clicked.connect(partial(
            self.selectEntries, 
            attrToSet='selectedValPositions', 
            widget_name='widgets.SelectPosFoldernamesButton', 
            add_browse_button=False, 
            use_tooltip_as_widget_kwargs=True, 
            use_value_as_widgets_value=True, 
            entry_header='Validation Positions'
        ))
        
        cropsShapeInfoButton.clicked.connect(self.showCropsShapeInfo)
        helpButton.clicked.connect(self.showHelp)
        maxNumCropsInfoButton.clicked.connect(self.showMaxNumCropsInfo)
        self.randomSplitToggle.toggled.connect(self.randomSplitToggled)
        loadFromWorkflowFileButton.sigPathSelected.connect(
            self.loadFromWorkflowFile
        )
        saveToWorkflowFileButton.clicked.connect(self.saveToWorkflowFile)
        augmentInfoButton.clicked.connect(self.showAugmentInfo)
        self.augmentToggle.toggled.connect(self.augmentToggled)
        
        cropBkgrInfoButton.clicked.connect(self.showCropBkgrInfo)
        self.cropBkgrToggle.toggled.connect(self.cropBkgrToggled)
        
        self.setLayout(mainLayout)
        
        self.checkWhichFieldsAreSet()
    
    def checkWhichFieldsAreSet(self):
        selectedPaths = self.selectExpPathsButton.toolTip().split('\n')
        selectedPaths = [path for path in selectedPaths if path]
        self.areExpPathsSetButton.setSelected(len(selectedPaths)>0)
        
        for items in self.isSetButtonsMapper.items():
            selectButton, (isFieldSetButton, selectedMapperName) = items
            selectedMapper = getattr(self, selectedMapperName)
            isFieldSetButton.setSelected(selectedMapper is not None)
    
    def augmentToggled(self, checked):
        if checked:
            self.gaussSigmaOneWidget.setValue(0.75)
            self.gaussSigmaOneTwoWidget.setValue(1.5)
        else:
            self.dogFilterSpotSizeWidget.setValue(0)
            self.gaussSigmaOneWidget.setValue(0)
            self.gaussSigmaOneTwoWidget.setValue(0)
            
        self.dogFilterSpotSizeWidget.setEnabled(checked)
        self.gaussSigmaOneWidget.setEnabled(checked)
        self.gaussSigmaOneTwoWidget.setEnabled(checked)
    
    def setDataAugmentParamFromIniSection(self, cp, section):
        params_mapper = {
            'data_augmentation_1;spotmax.filters.DoG_spots': (
                'spots_zyx_radii_pxl', self.dogFilterSpotSizeWidget
            ),
            'data_augmentation_2;spotmax.filters.gaussian': (
                'sigma', self.gaussSigmaOneWidget
            ),
            'data_augmentation_3;spotmax.filters.DoG_spots': (
                'sigma', self.gaussSigmaOneTwoWidget
            ),
        }
        param = params_mapper.get(section)
        if param is None:
            return
        
        kwarg, widget = param
        value = eval(cp.get(section, kwarg))
        widget.setValue(value)
    
    def loadFromWorkflowFile(self, ini_filepath):
        cp = config.ConfigParser()
        cp.read(ini_filepath)
        folderpath = os.path.dirname(ini_filepath)
        self.folderPathWorflowWidget.setText(folderpath)
        
        filename = os.path.basename(ini_filepath)
        self.workflowFilenameWidget.setText(filename[:-4])
        
        expPaths = []       
        for section in cp.sections():
            if section == 'training_params':
                continue
            
            if section.startswith('data_augmentation'):
                self.setDataAugmentParamFromIniSection(cp, section)
                continue
            
            exp_path = section
            
            if not os.path.exists(exp_path):
                continue
            
            expPaths.append(exp_path)
            
            channels = cp[exp_path].get('channels', '').split('\n')
            channel_names = []
            for channel_name in channels:
                if not channel_name:
                    continue
                
                filepath = acdc_load.search_filepath_from_endname(
                    exp_path, channel_name
                )
                if os.path.dirname(filepath).endswith('spotMAX_output'):
                    tooltip = 'spotMAX_output'
                else:
                    tooltip = 'Images'
                
                if self.selectedChannelNamesFiles is None:
                    self.selectedChannelNamesFiles = {}
                
                if exp_path not in self.selectedChannelNamesFiles:
                    self.selectedChannelNamesFiles[exp_path] = {}
                
                self.selectedChannelNamesFiles[exp_path][channel_name] = (
                    tooltip, channel_name
                )
                channel_names.append(channel_name)
            
            m = 0
            masks_endnames = cp[exp_path].get('masks_endnames', '').split('\n')
            for masks_endname in masks_endnames:
                if not masks_endname:
                    continue
                    
                filepath = acdc_load.search_filepath_from_endname(
                    exp_path, masks_endname
                )
                if os.path.dirname(filepath).endswith('spotMAX_output'):
                    tooltip = 'spotMAX_output'
                else:
                    tooltip = 'Images'
                
                if self.selectedMasksFiles is None:
                    self.selectedMasksFiles = {}
                
                if exp_path not in self.selectedMasksFiles:
                    self.selectedMasksFiles[exp_path] = {}
                
                channel_name = channel_names[m]
                self.selectedMasksFiles[exp_path][channel_name] = (
                    tooltip, masks_endname
                )
                m += 1
                
            if self.selectedPixelSizes is None:
                self.selectedPixelSizes = {}
            pixel_size = cp[exp_path]['pixel_size']
            self.selectedPixelSizes[exp_path] = (r'{}', pixel_size)
            
            spot_masks_size = cp[exp_path].get('spot_masks_size')
            if spot_masks_size is not None:
                spot_masks_size = eval(spot_masks_size)
                unit = 'pixel'
                kwargs = {
                    'um_to_pixel': pixel_size, 
                    'unit': unit
                }  
                tooltip = str(kwargs)
                
                if self.selectedSpotMaskSizes is None:
                    self.selectedSpotMaskSizes = {}
                self.selectedSpotMaskSizes[exp_path] = (
                    tooltip, spot_masks_size
                )
            
            m = 0
            spots_coords_endnames = (
                cp[exp_path].get('spots_coords_endnames', '').split('\n')
            )
            for spots_coords_endname in spots_coords_endnames:
                if not spots_coords_endname:
                    continue
                    
                filepath = acdc_load.search_filepath_from_endname(
                    exp_path, spots_coords_endname
                )
                if os.path.dirname(filepath).endswith('spotMAX_output'):
                    tooltip = 'spotMAX_output'
                else:
                    tooltip = 'Images'
                
                if self.selectedSpotsCoordsFiles is None:
                    self.selectedSpotsCoordsFiles = {}
                
                if exp_path not in self.selectedSpotsCoordsFiles:
                    self.selectedSpotsCoordsFiles[exp_path] = {}
                
                channel_name = channel_names[m]
                self.selectedSpotsCoordsFiles[exp_path][channel_name] = (
                    tooltip, spots_coords_endname
                )
                m += 1
            
            training_positions = cp[exp_path].get('training_positions')
            training_positions = training_positions.split('\n')
            training_positions = [pos for pos in training_positions if pos]
            if self.selectedTrainPositions is None:
                self.selectedTrainPositions = {}
            self.selectedTrainPositions[exp_path] = (
                str({'exp_path': exp_path}), training_positions
            )
            
            validation_positions = cp[exp_path].get('validation_positions')
            validation_positions = validation_positions.split('\n')
            validation_positions = [pos for pos in validation_positions if pos]
            if self.selectedValPositions is None:
                self.selectedValPositions = {}
            self.selectedValPositions[exp_path] = (
                str({'exp_path': exp_path}), validation_positions
            )
        
        self.selectExpPathsButton.setToolTip('\n'.join(expPaths))
        
        training_params = cp['training_params']
        self.modelSizeCombobox.setCurrentText(training_params['model_size'])
        
        crop_shapes = eval(training_params['crops_shapes'])
        self.cropYspinbox.setValue(crop_shapes[0])
        self.cropXspinbox.setValue(crop_shapes[1])
        
        max_number_of_crops = training_params.getint('max_number_of_crops')
        self.maxNumCropsSpinbox.setValue(max_number_of_crops)

        try:
            rescale_to_pixel_size = training_params.getfloat(
                'rescale_to_pixel_size'
            )
            self.rescaleToPixelSizeWidget.setValue(rescale_to_pixel_size)
        except Exception as err:
            pass
        
        try:
            crop_background = training_params.getboolean('crop_background')
            self.cropBkgrToggle.setChecked(crop_background)
        except Exception as err:
            pass
        
        try:
            crop_background_pad = training_params.getint('crop_background_pad')
            self.cropBkgrPadWidget.setValue(crop_background_pad)
        except Exception as err:
            pass
        
        if self.selectedTrainPositions is not None:
            self.randomSplitToggle.toggled.disconnect()
            self.randomSplitToggle.setChecked(False)
            self.randomSplitToggle.toggled.connect(self.randomSplitToggled)
        else:
            self.randomSplitToggle.setChecked(True)
            self.randomSplitToggled(True)
            
        self.checkWhichFieldsAreSet()
        
        self._showInfo(
            'Parameters loaded', 
            'Done! Parameters loaded from the following workflow file:<br>',
            commands=(ini_filepath,)
        )
    
    def saveToWorkflowFile(self):
        workflowFolderpath = self.folderPathWorflowWidget.text()
        if not workflowFolderpath or not os.path.exists(workflowFolderpath):
            workflowFolderpath = getexistingdirectory(
                parent=self,
                caption='Select folder where to save workflow file', 
                basedir=acdc_myutils.getMostRecentPath()
            )
            if not workflowFolderpath:
                return
            self.folderPathWorflowWidget.setText(workflowFolderpath)
        
        self.generateWorkflow(doNotCreateDatasets=True)
    
    def randomSplitToggled(self, checked):
        if checked:
            selectedPaths = self.selectExpPathsButton.toolTip().split('\n')
            selectedPaths = [path for path in selectedPaths if path]
            expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
            for exp_path in expPaths:
                pos_foldernames = acdc_myutils.get_pos_foldernames(exp_path)
                train_positions, val_positions = self._randomChoiceTrainValPos(
                    pos_foldernames
                )
                if self.selectedTrainPositions is None:
                    self.selectedTrainPositions = {}
                self.selectedTrainPositions[exp_path] = (
                    str({'exp_path': exp_path}), train_positions
                )
                if self.selectedValPositions is None:
                    self.selectedValPositions = {}
                self.selectedValPositions[exp_path] = (
                    str({'exp_path': exp_path}), val_positions
                )
        else:
            self.selectedTrainPositions = None
            self.selectedValPositions = None
        
        self.checkWhichFieldsAreSet()
    
    def showHelp(self):
        title = 'Setup training worflow help'
        txt = ("""
        On this dialog you can setup the <b>parameters to train your own SpotMAX AI model</b>.<br><br>
        The training can be done from spot masks that you generated with SpotMAX or Cell-ACDC,<br> 
        or from a table of spot coordindates.<br><br>
        
        Note that this table must contain the columns <code>x, y</code>, with <code>z</code> for z-stack data,<br>
        and <code>t</code> or <code>frame_i</code> for timelapse data.<br><br>
        
        Once you set all the parameters, these will be saved to a configuration file.<br><br>
        
        Alongside the configuration file, SpotMAX will also save the spot channel images<br>
        and the ground-truth spot masks to HDF database files (.h5), one for each unique<br>
        experiment folder.<br><br>
        
        <b>The configuration file and the database files are all you need to train</b> your own model.<br><br>
        You can move these files wherever you like (as long as they all stay in the same folder)<br>
        and you can then run the training process with the following command:<br>
        """)
        commands = ('spotmax -t "path/to/configuration/file"',)
        self._showInfo(title, txt, commands)
    
    def selectExperimentPaths(self):
        preSelectedPaths = self.selectExpPathsButton.toolTip().split('\n')
        preSelectedPaths = [path for path in preSelectedPaths if path]
        if not preSelectedPaths:
            preSelectedPaths = None
        win = SelectFolderToAnalyse(
            preSelectedPaths=preSelectedPaths, onlyExpPaths=True,
            scanFolderTree=True
        )
        win.exec_()
        if win.cancel:
            return
        
        selectedPathsList = win.paths
        selectedPaths = '\n'.join(selectedPathsList)
        self.selectExpPathsButton.setToolTip(selectedPaths)

        defaultSpotSize = (2.0, 3.0)
        expPaths = acdc_load.get_unique_exp_paths(selectedPathsList)
        for exp_path in expPaths:
            exp_path = exp_path.replace('\\', '/')
            
            if self.selectedSpotMaskSizes is None:
                self.selectedSpotMaskSizes = {}
            
            pos_foldernames = acdc_myutils.get_pos_foldernames(exp_path)
            sample_pos = pos_foldernames[0]
            images_path = os.path.join(exp_path, sample_pos, 'Images')
            df_metadata = acdc_load.load_metadata_df(images_path)         
            try:
                PhysicalSizeZ = float(df_metadata.at['PhysicalSizeZ', 'values'])
                PhysicalSizeY = float(df_metadata.at['PhysicalSizeY', 'values'])
                PhysicalSizeX = float(df_metadata.at['PhysicalSizeX', 'values'])
                set_pixel_size = True
            except Exception as err:
                PhysicalSizeZ = 1.0
                PhysicalSizeY = 1.0
                PhysicalSizeX = 1.0 
                set_pixel_size = False
            
            um_to_pixel = (PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX)
            kwargs = {
                'um_to_pixel': um_to_pixel, 
                'unit': 'pixel'
            }
            tooltip = str(kwargs)
            if exp_path not in self.selectedSpotMaskSizes:
                self.selectedSpotMaskSizes[exp_path] = (tooltip, defaultSpotSize)
            
            if self.selectedPixelSizes is None:
                self.selectedPixelSizes = {}
            
            if exp_path not in self.selectedPixelSizes and set_pixel_size:
                self.selectedPixelSizes[exp_path] = ('', PhysicalSizeY)
            
            if self.randomSplitToggle.isChecked():
                train_positions, val_positions = self._randomChoiceTrainValPos(
                    pos_foldernames
                )
                if self.selectedTrainPositions is None:
                    self.selectedTrainPositions = {}
                
                if self.selectedValPositions is None:
                    self.selectedValPositions = {}
                
                if exp_path not in self.selectedTrainPositions:
                    self.selectedTrainPositions[exp_path] = (
                        str({'exp_path': exp_path}), train_positions
                    )
                    self.selectedValPositions[exp_path] = (
                        str({'exp_path': exp_path}), val_positions
                    )
        
        self.checkWhichFieldsAreSet()
        
    def _randomChoiceTrainValPos(self, pos_foldernames):
        train_perc = self.trainPercSpinbox.value()
        val_perc = self.valPercSpinbox.value()
        train_positions, val_positions = utils.random_choice_pos_foldernames(
            pos_foldernames, train_perc=train_perc, val_perc=val_perc
        )
        return train_positions, val_positions
    
    def showMaxNumCropsInfo(self):
        note_admon = html_func.to_admonition(
            'A value of -1 means no upper limit to the number of crops'
        )
        title = 'Max number of crops info'
        txt = (f"""
            To avoid GPU memory issues, images are cropped with patches 
            of the requested shape.<br><br>
            However, if the images are large compared to the crop size, 
            you might end up with too many images<br>
            and the <b>training process might take very long</b>.<br><br>
            To reduce the number of total images, set a maximum number of crops.<br><br>
            SpotMAX will then randomly select crops from each image.<br>
            {note_admon}
        """)
        self._showInfo(title, txt)      

    def showCropsShapeInfo(self):
        title = 'Crops shape info'
        txt = ("""
            To avoid GPU memory issues, images are cropped with patches 
            of the requested shape.<br><br>
            The default value of (256, 256) should work with a minimum of 16 GB 
            GPU memory.<br><br>
            However, if your images are divisible by a specific number, by 
            using that number as shape<br>
            you will avoid information redundancy.<br><br>
        """)
        self._showInfo(title, txt)
    
    def showAugmentInfo(self):
        title = 'Augment Images info'
        note = html_func.to_admonition(
            'To disable any of the filters, pass a value of 0 for the '
            'corresponding parameters'
        )
        txt = (f"""
            To increase the model generalization power, SpotMAX will 
            expand the training dataset with augmented images.<br><br>
            
            Data augmentation is the process of generating new images from 
            the existing ones.<br><br>
            
            For each image, SpotMAX will generate an additional 3 images by 
            applying a Difference of Gaussians filter (DoG) and two Gaussian 
            filters.<br><br>
            
            For the parameters of the filters, we recommend using the same 
            ones you will use to detect the spots.<br><br>
           
            This way, the network will already have seen filtered images, 
            potentially making the predictions more robust.<br>
            {note}
        """)
        self._showInfo(title, txt)
    
    def showCropBkgrInfo(self):
        title = 'Crop background info'
        txt = ("""
            If active, SpotMAX will crop the images around the spots masks.<br><br>
            Use the <code>Crop background tolerance</code> parameter to control 
            how many pixels of the background (away from the spots masks) to keep 
            in the image.<br><br>
            This parameter is useful if you annotated the images partially and 
            you want to remove signal that you did not annnotate.
        """)
        self._showInfo(title, txt)
    
    def cropBkgrToggled(self, checked):
        self.cropBkgrPadWidget.setEnabled(checked)
    
    def showSelectPosInfo(self):
        title = 'Select Positions info'
        txt = ("""
            Click on the <code>Select/view experiment folders...</code> to 
            select which Positions<br>
            to use as ground-truth for the training session.<br><br>
            
            SpotMAX will then generate a single database file (.h5 extension)<br>
            for each experiment folder (the parent folder of the Positions).<br><br>
            
            The image data in each database file will be normalized based<br>
            on the minimum and maximum intensity in the database file.<br><br>
            
            The database files will contain the intensity images, the spot masks,<br>
            and the pixel size.<br><br>
            
            Once you have generated these files, you will not need the Positions<br>
            folder for the training.<br>
            That means, if you are planning to run the training session on a<br>
            different system, you only need to move/copy the database files<br>
            and not the Position folders.
        """)
        self._showInfo(title, txt)
    
    def _showInfo(self, title, txt, commands=None):
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph(txt)
        msg.information(self, title, txt, commands=commands)
    
    def warnPathsNotSelected(self):
        txt = html_func.paragraph("""
            You did <b>not select any Position folder.</b><br><br>
            Click on <code>Select/view experiment folders...</code> button to 
            select the<br>
            Position folders to use as ground-truth for the 
            training session.                          
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Positions were not selected', txt)
    
    def warnDoGspotSizeNotValid(self):
        txt = html_func.paragraph("""
            The parameter <code>DoG filter (z,y,x) spot size</code> is <b>not valid</b>.<br>br>
            It must be either 0 (to disable the filter) or 3 values for z-, y-, and  
            z-dimensions.<br><br>
            If you are working with 2D images, write 1.0 for the z-dimension.
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Positions were not selected', txt)
    
    def warnWorflowFolderpathNotProvided(self):
        txt = html_func.paragraph("""
            You did not provide the <b>folder where to save the worflow</b>.</b>                     
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Worflow folder path invalid', txt)
    
    def warnWorflowFolderpathNotEmpty(self, workflow_folderpath):
        txt = html_func.paragraph("""
            The selected folder path of the <b>training workflow file is NOT empty</b>.<br><br>
            Please, choose an empty folder, thanks!                    
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Worflow folder path invalid', txt, 
            path_to_browse=workflow_folderpath
        )
    
    def warnWorflowFolderpathDoesNotExist(self):
        txt = html_func.paragraph("""
            The selected folder path of the <b>training workflow file does not exist.</b>                     
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Worflow folder path invalid', txt)
    
    def warnWorkflowFilepathEmpty(self):
        txt = html_func.paragraph("""
            The entered filename for the <b>training workflow file is empty.</b>                     
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Worflow filename empty', txt)
    
    def warnWorkflowFileExists(self, workflowFilepath):
        txt = html_func.paragraph("""
            The <b>training workflow file already exists.</b><br><br>
            How do you want to proceed?                     
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        _, overwriteButton = msg.warning(
            self, 'Worflow file exists', txt, 
            buttonsTexts=('Cancel', 'Overwrite existing file'), 
            commands=(workflowFilepath,), 
            path_to_browse=os.path.dirname(workflowFilepath)
        )
        if msg.cancel:
            return ''

        return workflowFilepath
    
    def warnDataPrepProcessStartsNow(self, workflowFilepath):
        important_text = (f"""
            Do not close the GUI nor the terminal during the process
        """)
        txt = html_func.paragraph(f"""
            SpotMAX will now <b>start the data prep process</b> that will generate 
            the HDF database files.<br><br>
            It might take some time, depending on the amount of data to process.<br><br>
            Progress will be displayed in the terminal, while the GUI 
            will be in a frozen state.<br> 
            {html_func.to_admonition(important_text, admonition_type='important')}                         
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Data prep process starting now', txt, 
            buttonsTexts=('Cancel', 'Ok, let\'s go!')
        )
        return msg.cancel
    
    def warnChannelsNotSelected(self):
        txt = html_func.paragraph("""
            You did <b>not select any channel for spots images.</b><br><br>
            Click on <code>Select/view spots channel names...</code> button to 
            select the<br>
            spots channel names to use for the training session.                          
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Spot channel names not selected', txt)
    
    def warnChannelNotSelectedForExpPath(self, exp_path):
        txt = html_func.paragraph("""
            You did <b>not select any spots channel name</b> for the following 
            experiment folder path:<br>                         
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Channel not selected', txt, 
            commands=(exp_path,), 
            path_to_browse=exp_path
        )
    
    def warnTrainValPosNotSelectedForExpPath(self, exp_path, category):
        txt = html_func.paragraph(f"""
            You did <b>not select any {category} Position</b> for the following 
            experiment folder path:<br>                         
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Channel not selected', txt, 
            commands=(exp_path,), 
            path_to_browse=exp_path
        )
    
    def warnSpotMaskSizeInvalidForExpPath(self, exp_path, value):
        txt = html_func.paragraph(f"""
            The selected <b>spot masks size</b> for the experiment folder below 
            is <b>invalid</b>.<br><br>     
            It must be either a single number for 2D images or two numbers, 
            one for z-direcation and one for xy-direction, for 3D data.<br><br>    
            Selected spot masks size = {value}<br><br>
            Experiment folder path:<br>                
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Channel not selected', txt, 
            commands=(exp_path,), 
            path_to_browse=exp_path
        )
    
    def warnTrainPosNotSelected(self):
        txt = html_func.paragraph("""
            You did <b>not select any Position to use as training data.</b><br><br>
            Click on <code>Select/view experiment folders to use as training data...</code> 
            to select them. 
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Spot masks parameters not selected', txt)
    
    def warnValPosNotSelected(self):
        txt = html_func.paragraph("""
            You did <b>not select any Position to use as validation data.</b><br><br>
            Click on <code>Select/view experiment folders to use as validation data...</code> 
            to select them.  
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Spot masks parameters not selected', txt)
    
    def warnMasksParamsNotSelected(self):
        txt = html_func.paragraph("""
            You did <b>not select any parameter for the spot masks</b> to use 
            as ground-truth for the training session<br><br>
            Click on either <code>Select/view spots coords files...</code> or 
            <code>Select/view spot mask size...</code> to fix this.    
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Spot masks parameters not selected', txt)
    
    def warnSpotMasksNotSelectedForExpPath(self, exp_path):
        txt = html_func.paragraph("""
            You did <b>not select any spots masks file</b> for the following 
            experiment folder path:<br>                         
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Spot masks file not selected', txt, 
            commands=(exp_path,), 
            path_to_browse=exp_path
        )
    
    def warnSpotsCoordsNotSelectedForExpPath(self, exp_path):
        txt = html_func.paragraph("""
            You did <b>not select any spots coords file</b> for the following 
            experiment folder path:<br>                         
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Spot coords file not selected', txt, 
            commands=(exp_path,), 
            path_to_browse=exp_path
        )
    
    def warnPixelSizeNotSelectedForExpPath(self, exp_path):
        txt = html_func.paragraph("""
            You did <b>not entered the pixel size</b> for the following 
            experiment folder path:<br>                         
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Spot coords file not selected', txt, 
            commands=(exp_path,), 
            path_to_browse=exp_path
        )
    
    def selectEntries(
            self, attrToSet='', extensions=None, allow_spotmax_output=False, 
            widget_name='widgets.LineEdit', use_tooltip_as_widget_kwargs=False, 
            add_browse_button=True, use_value_as_widgets_value=False, 
            entry_header='File endname', depends_on_channels=False, 
            allow_add_field=False, widgets_values_are_multiple_entries=False,
            rel_start_dir=None, enable_autofill=False, 
            add_apply_to_all_buttons=False
        ):
        selectedPaths = self._validateSelectedPaths()
        if not selectedPaths:
            return
        
        channel_names = None
        if depends_on_channels:
            selectedChannelNames = self._validateSelectedChannelNames(
                selectedPaths
            )
            if not selectedChannelNames:
                return 
            channel_names = list(selectedChannelNames.values())
        
        expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
        selectedValuesMapper = getattr(self, attrToSet)        
        widgets_kwargs = None
        if use_tooltip_as_widget_kwargs and selectedValuesMapper is not None:
            if allow_add_field or depends_on_channels:
                widgets_kwargs = []
                for exp_path in expPaths:
                    values_mapper = selectedValuesMapper[exp_path]
                    kwargs_list = [
                        eval(value[0]) for value in values_mapper.keys()
                    ]
                    widgets_kwargs.append(kwargs_list)
            elif selectedValuesMapper is not None:
                widgets_kwargs = [
                    eval(selectedValuesMapper[exp_path][0]) 
                    for exp_path in expPaths
                ]
        
        widgets_values = None
        if use_value_as_widgets_value and selectedValuesMapper is not None:
            if allow_add_field or depends_on_channels:
                widgets_values = []
                for exp_path in expPaths:
                    values_mapper = selectedValuesMapper[exp_path]
                    values_list = list(values_mapper.keys())
                    widgets_values.append(values_list)
            elif selectedValuesMapper is not None:
                widgets_values = [
                    selectedValuesMapper[exp_path][1] for exp_path in expPaths
                ]
        
        are_values_multiple_entries = widgets_values_are_multiple_entries
        expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
        win = SelectInfoForEachExperimentDialog(
            expPaths, 
            extensions=extensions, 
            widget_name=widget_name, 
            widgets_kwargs=widgets_kwargs, 
            widgets_values=widgets_values,
            allow_spotmax_output=allow_spotmax_output, 
            entry_header=entry_header, 
            channel_names=channel_names, 
            allow_add_field=allow_add_field, 
            widgets_values_are_multiple_entries=are_values_multiple_entries, 
            add_browse_button=add_browse_button, 
            rel_start_dir=rel_start_dir, 
            enable_autofill=enable_autofill,
            add_apply_to_all_buttons=add_apply_to_all_buttons,
            parent=self
        )
        win.exec_()
        if win.cancel:
            return
        
        setattr(self, attrToSet, win.selectedValues)
        self.checkWhichFieldsAreSet()
    
    def _validateSelectedPaths(self):
        selectedPaths = self.selectExpPathsButton.toolTip().split('\n')
        selectedPaths = [path for path in selectedPaths if path]
        if not selectedPaths:
            self.warnPathsNotSelected()
            return []
        return selectedPaths
    
    def _validateSelectedChannelNames(self, selectedPaths, warn=True):
        if self.selectedChannelNamesFiles is None:
            if warn:
                self.warnChannelsNotSelected()
            return {}
        
        channelNamesMapper = {}
        expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
        for exp_path in expPaths:
            exp_path = exp_path.replace('\\', '/')
            if exp_path not in self.selectedChannelNamesFiles:
                if warn:
                    self.warnChannelNotSelectedForExpPath(exp_path)
                    return {}
                continue

            channels = []
            channels_mapper = self.selectedChannelNamesFiles[exp_path]
            for key, (tooltip, channel_name) in channels_mapper.items():
                channels.append(channel_name)
                
            channelNamesMapper[exp_path] = channels
            
        return channelNamesMapper

    def _validateSelectedTrainPos(self, selectedPaths, warn=True):
        if self.selectedTrainPositions is None:
            if warn:
                self.warnTrainPosNotSelected()
            return {}
        
        trainingPosMapper = {}
        expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
        for exp_path in expPaths:
            exp_path = exp_path.replace('\\', '/')
            if exp_path not in self.selectedTrainPositions:
                if warn:
                    self.warnTrainValPosNotSelectedForExpPath(
                        exp_path, 'training'
                    )
                    return {}
                continue
            
            _, positions = self.selectedTrainPositions[exp_path]
            trainingPosMapper[exp_path] = positions
        
        return trainingPosMapper

    def _validateSelectedValPos(self, selectedPaths, warn=True):
        if self.selectedValPositions is None:
            if warn:
                self.warnValPosNotSelected()
            return {}
            
        validationPosMapper = {}
        expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
        for exp_path in expPaths:
            exp_path = exp_path.replace('\\', '/')
            if exp_path not in self.selectedValPositions:
                if warn:
                    self.warnTrainValPosNotSelectedForExpPath(
                        exp_path, 'validation'
                    )
                    return {}
                continue
            
            _, positions = self.selectedValPositions[exp_path]
            validationPosMapper[exp_path] = positions
        
        return validationPosMapper
    
    def _validateSelectedMasks(self, selectedPaths, warn=True):
        masksParamsNotSelected = (
            self.selectedMasksFiles is None 
            and self.selectedSpotsCoordsFiles is None
        )
        if masksParamsNotSelected:
            if warn:
                self.warnMasksParamsNotSelected()
            return {}
        
        if self.selectedMasksFiles is not None:
            spotMasksEndnames = {}
            expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
            for exp_path in expPaths:
                exp_path = exp_path.replace('\\', '/')
                if exp_path not in self.selectedMasksFiles:
                    if warn:
                        self.warnSpotMasksNotSelectedForExpPath(exp_path)
                        return {}
                    else:
                        continue

                values = []
                channels_mapper = self.selectedMasksFiles[exp_path]
                for channel, (tooltip, value) in channels_mapper.items():
                    values.append(value)
                spotMasksEndnames[exp_path] = values
            return spotMasksEndnames
    
    def _validateSelectedSpotsCoords(self, selectedPaths, warn=False):   
        if self.selectedSpotsCoordsFiles is None:
            return {}
             
        spotCoordsEndnames = {}
        expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
        for exp_path in expPaths:
            exp_path = exp_path.replace('\\', '/')
            if exp_path not in self.selectedSpotsCoordsFiles:
                if warn:
                    self.warnSpotsCoordsNotSelectedForExpPath(exp_path)
                    return {}
                else:
                    continue

            values = []
            channels_mapper = self.selectedSpotsCoordsFiles[exp_path]
            for channel, (tooltip, value) in channels_mapper.items():
                values.append(value)
            spotCoordsEndnames[exp_path] = values
        return spotCoordsEndnames
    
    def _validateSelectedPixelSizes(self, selectedPaths, warn=True):
        pixelSizes = {}
        expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
        for exp_path in expPaths:
            exp_path = exp_path.replace('\\', '/')
            if exp_path not in self.selectedPixelSizes:
                if warn:
                    self.warnPixelSizeNotSelectedForExpPath(exp_path)
                    return {}
                else:
                    continue

            value = self.selectedPixelSizes[exp_path][1]
            pixelSizes[exp_path] = value
        return pixelSizes
    
    def _validateWorflowFolderpath(self):
        workflowFolderpath = self.folderPathWorflowWidget.text()
        if not workflowFolderpath:
            self.warnWorflowFolderpathNotProvided()
            return ''
        
        if not os.path.exists(workflowFolderpath):
            self.warnWorflowFolderpathDoesNotExist()
            return ''

        ls_workflow_folder = os.listdir(workflowFolderpath)
        is_empty = (
            not ls_workflow_folder 
            or (
                len(ls_workflow_folder) == 1 
                and ls_workflow_folder[0].endswith('.ini')
            )
        )
        if not is_empty:
            self.warnWorflowFolderpathNotEmpty(workflowFolderpath)
            return ''
            
        return workflowFolderpath
    
    def _validateWorflowFilepath(self, workflow_folderpath):
        if not self.workflowFilenameWidget.text():
            self.warnWorkflowFilepathEmpty()
            return ''
        
        workflowFilepath = (
            f'{workflow_folderpath}/{self.workflowFilenameWidget.text()}.ini'
            .replace('\\', '/')
        )
        if os.path.exists(workflowFilepath):
            workflowFilepath = self.warnWorkflowFileExists(workflowFilepath)
 
        return workflowFilepath
    
    def _validateSpotMasksSize(self, selectedPaths):
        if self.selectedMasksFiles is not None:
            return
        
        if self.selectedSpotMaskSizes is None:
            return 
        
        spotMasksSizes = {}
        expPaths = acdc_load.get_unique_exp_paths(selectedPaths)
        for exp_path in expPaths:
            exp_path = exp_path.replace('\\', '/')
            value = self.selectedSpotMaskSizes[exp_path][1]
            try:
                valid = len(value) == 2
            except TypeError as err:
                valid = True

            if not valid:
                continue
            
            spotMasksSizes[exp_path] = value
        
        return spotMasksSizes
    
    def askRescaleToPixelSize(self):
        note_admon = html_func.to_admonition(
            'One option is to choose the maximum pixel size '
            'you have in all the images.'
        )
        txt = html_func.paragraph(f"""
            Rescaling to the same pixel size can help the model to 
            generalize to pixel sizes it was not trained on.<br><br>
            
            However, you left the field <code>Rescale images to pixel size</code> 
            empty.<br><br>                        
                                  
            Are you sure you don't want to rescale images to the same 
            pixel size?<br>
            
            {note_admon}
        """)
        msg = acdc_widgets.myMessageBox(wrapText=False)
        _, yesButton, noButton = msg.question(
            self, 'Rescale images to same pixel size?', txt, 
            buttonsTexts=(
                'Cancel',
                'I am sure, do not rescale images', 
                'Interesting, let me edit that'
            )
        )
        if msg.cancel or msg.clickedButton == noButton:
            return False
        
        return -1.0
    
    def _validateRescaleToPixelSize(self, warn=True):
        if self.rescaleToPixelSizeWidget.value():
            return self.rescaleToPixelSizeWidget.value()
        
        if warn:
            value = self.askRescaleToPixelSize()
        else:
            value = ''
        return value
    
    def _validateDataAugmentParams(self, warn=True):
        sigma1 = self.gaussSigmaOneWidget.value()
        sigma2 = self.gaussSigmaOneTwoWidget.value()
        
        if not self.augmentToggle.isChecked():
            spots_zyx_radii_pxl = 0
            sigma1 = 0
            sigma2 = 0
            valid = True
        elif self.dogFilterSpotSizeWidget.value() == 0:
            valid = True
            spots_zyx_radii_pxl = 0
            sigma1 = 0
            sigma2 = 0
        else:
            try:
                valid = len(self.dogFilterSpotSizeWidget.value()) == 3
                spots_zyx_radii_pxl = self.dogFilterSpotSizeWidget.value()
            except Exception as err:
                valid = False
        
        if not valid:
            if warn:
                self.warnDoGspotSizeNotValid()
            return False
        
        params = {
            '1;spotmax.filters.DoG_spots': {
                'spots_zyx_radii_pxl': spots_zyx_radii_pxl
            },
            '2;spotmax.filters.gaussian': {
                'sigma': self.gaussSigmaOneWidget.value()
            },
            '3;spotmax.filters.gaussian': {
                'sigma': self.gaussSigmaOneTwoWidget.value()
            },
        }
        return params
    
    def generateWorkflow(self, doNotCreateDatasets=False):
        selectedPaths = self._validateSelectedPaths()
        if not selectedPaths:
            return False

        warn = not doNotCreateDatasets
        
        selectedChannelNames = self._validateSelectedChannelNames(
            selectedPaths, warn=warn
        )
        if not selectedChannelNames and warn:
            return False
        
        selectedTrainPos = self._validateSelectedTrainPos(
            selectedPaths, warn=warn
        )
        if not selectedTrainPos and warn:
            return False
        
        selectedValPos = self._validateSelectedValPos(
            selectedPaths, warn=warn
        )
        if not selectedValPos and warn:
            return False
        
        selectedSpotsCoords = None
        selectedMasks = self._validateSelectedMasks(selectedPaths, warn=warn)
        if not selectedMasks:
            selectedSpotsCoords = self._validateSelectedSpotsCoords(
                selectedPaths, warn=warn
            )
            if not selectedSpotsCoords and warn:
                return False

        pixelSizes = self._validateSelectedPixelSizes(selectedPaths, warn=warn)
        if not pixelSizes and warn:
            return False
        
        rescaleToPixelSize = self._validateRescaleToPixelSize(warn=warn)
        if not rescaleToPixelSize and warn:
            return False
        
        workflowFolderpath = self._validateWorflowFolderpath()
        if not workflowFolderpath:
            return False
        
        workflowFilepath = self._validateWorflowFilepath(workflowFolderpath)
        if not workflowFilepath:
            return False
        
        spotMasksSize = self._validateSpotMasksSize(selectedPaths)
                
        dataAugmentParams = self._validateDataAugmentParams(warn=warn)
        if not dataAugmentParams and warn:
            return False
        
        cropsShape = (
            self.cropYspinbox.value(), self.cropXspinbox.value()
        )
        
        proceed = True
        if warn:
            cancel = self.warnDataPrepProcessStartsNow(workflowFilepath)
            if cancel:
                return False
        
        self.workflowFilepath = workflowFilepath
        # acdc_myutils.addToRecentPaths(workflowFolderpath)
        try:
            print('Generating SpotMAX AI training workflow files...')
            utils.generate_unet_training_workflow_files(
                selectedTrainPos, 
                selectedValPos,
                selectedChannelNames, 
                workflowFilepath, 
                pixelSizes,
                model_size=self.modelSizeCombobox.currentText(),
                rescale_to_pixel_size=rescaleToPixelSize,
                spots_coords_endnames=selectedSpotsCoords, 
                masks_endnames=selectedMasks,
                spot_masks_size=spotMasksSize, 
                crops_shapes=cropsShape, 
                max_number_of_crops=self.maxNumCropsSpinbox.value(),
                data_augment_params=dataAugmentParams, 
                crop_background=self.cropBkgrToggle.isChecked(), 
                crop_background_pad=self.cropBkgrPadWidget.value(),
                do_not_generate_datasets=doNotCreateDatasets
            )
        except Exception as err:
            traceback.print_exc()
            return False
        
        if doNotCreateDatasets:
            self.workflowFileCreated(workflowFilepath)
    
        return proceed
    
    def workflowFileCreated(self, workflowFilepath):
        title = 'Workflow file saved'
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = html_func.paragraph("""
            Done!<br><br>
            Workflow file saved at the following location:<br>
        """)
        msg.information(
            self, title, txt, 
            commands=workflowFilepath, 
            path_to_browse=os.path.dirname(workflowFilepath)
        )
    
    def ok_cb(self):
        proceed = self.generateWorkflow()
        if not proceed:
            return
        
        self.cancel = False
        self.close()

class SelectInfoForEachExperimentDialog(QBaseDialog):
    def __init__(
            self, experiment_paths, 
            title='Select files', 
            channel_names=None,
            widget_name='widgets.LineEdit',
            extensions=None,
            allow_spotmax_output=False,
            add_browse_button=True,
            widgets_kwargs=None, 
            widgets_values=None,
            widgets_values_are_multiple_entries=False,
            entry_header='File endname',
            allow_add_field=False,
            rel_start_dir=None,
            enable_autofill=False,
            add_apply_to_all_buttons=False,
            parent=None, 
        ):        
        self.cancel = True
        super().__init__(parent)
        
        self._allow_spotmax_output = allow_spotmax_output
        self._add_browse_button = add_browse_button
        self._allow_add_field = allow_add_field
        self._channel_names = channel_names
        self._enable_autofill = enable_autofill
        self._add_apply_to_all_buttons = add_apply_to_all_buttons
        self._last_col = 5
        
        self.setWindowTitle(title)
        
        mainLayout = QVBoxLayout()
        
        if self._enable_autofill:
            autoFillLayout = QHBoxLayout()
            autoFillToggle = acdc_widgets.Toggle()
            autoFillToggle.setChecked(True)
            autoFillLayout.addStretch(1)
            autoFillLayout.addWidget(QLabel('Autofill'))
            autoFillLayout.addWidget(autoFillToggle)
            mainLayout.addLayout(autoFillLayout)
            autoFillToggle.toggled.connect(self.autoFillToggled)
        
        gridLayout = QGridLayout()
        self.gridLayout = gridLayout
        gridLayout.setHorizontalSpacing(5)
        gridLayout.setVerticalSpacing(10)
        
        scrollArea = QScrollArea()
        scrollAreaContainer = QWidget()
        scrollAreaContainer.setContentsMargins(0, 0, 0, 0)
        scrollArea.setWidgetResizable(True)
        scrollAreaContainer.setLayout(gridLayout)
        scrollArea.setWidget(scrollAreaContainer)
        self.scrollArea = scrollArea
        
        gridLayout.addWidget(
            QLabel('<b>Experiment paths</b>'), 0, 0, alignment=Qt.AlignCenter
        )
        gridLayout.addWidget(
            QLabel(f'<b>{entry_header}</b>'), 0, 2, alignment=Qt.AlignCenter
        )
        
        module_name, attr = widget_name.split('.')
        try:
            widgets_module = globals()[module_name]
            widgetFunc = getattr(widgets_module, attr)
        except KeyError as e:
            widgetFunc = globals()[attr]
        
        depends_on_channels = channel_names is not None
        if depends_on_channels:
            allow_add_field = False
        else:
            channel_names = [[''] for _ in range(len(experiment_paths))]
        
        if widgets_values is None:
            widgets_values_are_multiple_entries = False
        
        self.expPathLabels = []
        self.widgets = {}
        row = 1
        for e, exp_path in enumerate(experiment_paths):
            channels_exp = channel_names[e]
            
            if widgets_values_are_multiple_entries and not depends_on_channels:
                channels_exp = ['']*len(widgets_values[e])
                
            expPathLabel = widgets.ReadOnlyElidingLineEdit(transparent=True)
            expPathLabel.setToolTip(exp_path)
            expPathLabel.setText(exp_path)
            expPathLabel.setFrame(False)
            self.expPathLabels.append(expPathLabel)
            rowSpan = len(channels_exp)
            gridLayout.addWidget(
                expPathLabel, row, 0, rowSpan, 1, alignment=Qt.AlignVCenter
            )
            expPathLabelRow = row
            
            if allow_add_field:
                addFieldButton = acdc_widgets.addPushButton()
                addFieldButton.setToolTip('Add new entry')
                addFieldButton.clicked.connect(self.addField)
                gridLayout.addWidget(addFieldButton, row, self._last_col)
            else:
                gridLayout.addWidget(QLabel(''), row, self._last_col)
            
            if add_apply_to_all_buttons:
                applyToAllButton = widgets.ArrowButtons(
                    order=('down', 'up'), 
                    tooltips=(
                        'Apply value to all entries below', 
                        'Apply value to all entries above'
                    )
                )
                gridLayout.addWidget(applyToAllButton, row, self._last_col-1)
                applyToAllButton.sigButtonClicked.connect(
                    self.applyToOtherFields
                )
                applyToAllButton.index = row-1
                delButtonWidget = applyToAllButton
            else:
                emptyLabel = QLabel('')
                gridLayout.addWidget(emptyLabel, row, self._last_col-1)
                delButtonWidget = emptyLabel
            
            widget_kwargs = [{} for _ in range(len(channels_exp))]
            if widgets_kwargs is not None:
                widget_kwargs = widgets_kwargs[e]
                if not widgets_values_are_multiple_entries:
                    widget_kwargs = [
                        widget_kwargs for _ in range(len(channels_exp))
                    ]
            
            widget_value = [None]*len(channels_exp)
            if widgets_values is not None:
                widget_value = widgets_values[e]
                if not widgets_values_are_multiple_entries:
                    widget_value = [
                        widget_value for _ in range(len(channels_exp))
                    ]
            
            channelsLayout = QGridLayout()
            for ch, channel in enumerate(channels_exp):
                if ch > 0 and allow_add_field:
                    delFieldButton = acdc_widgets.delPushButton()
                    delFieldButton.addFieldButton = addFieldButton
                    self.gridLayout.addWidget(
                        delFieldButton, row, self._last_col
                    )   
                    delFieldButton.widgets = []
                    delFieldButton.clicked.connect(self.removeField)
                    
                widget_kwargs_ch = widget_kwargs[ch]                    
                widget = widgetFunc(**widget_kwargs_ch)     
                if self._enable_autofill:
                    widget.sigValueChanged.connect(self.autoFill)
                          
                widget_value_ch = widget_value[ch]
                if widget_value_ch is not None:
                    widget.setValue(widget_value_ch)
                
                channel_text = ''
                if channel:
                    channel_text = f'| <i>{channel}:</i> '
                
                channelLabel = QLabel(channel_text)
                gridLayout.addWidget(
                    channelLabel, row, 1, alignment=Qt.AlignLeft
                )
                if ch > 0 and allow_add_field:
                    delFieldButton.widgets.append(channelLabel)
                
                # lineEdit.setReadOnly(True)
                gridLayout.addWidget(widget, row, 2)

                browseFileButtonKwargs = {}
                if add_browse_button:
                    start_dir = exp_path
                    pos_foldernames = acdc_myutils.get_pos_foldernames(exp_path)
                    if rel_start_dir == 'Position':
                        start_dir = os.path.join(exp_path, pos_foldernames[0])
                    elif rel_start_dir == 'Images':
                        pos_path = os.path.join(exp_path, pos_foldernames[0])
                        start_dir = os.path.join(pos_path, 'Images')
                    elif rel_start_dir == 'spotMAX_output':
                        pos_path = os.path.join(exp_path, pos_foldernames[0])
                        start_dir = os.path.join(pos_path, 'spotMAX_output')
                    
                    if not os.path.exists(start_dir):
                        start_dir = exp_path
                    
                    browseFileButtonKwargs = {
                        'ext': extensions, 
                        'title': title, 
                        'start_dir': start_dir, 
                    }
                    browseFileButton = acdc_widgets.browseFileButton(
                        **browseFileButtonKwargs
                    )                    
                    browseFileButton.sigPathSelected.connect(
                        partial(
                            self.pathSelected, 
                            lineEdit=widget, 
                            parentExpPath=exp_path
                        )
                    )
                    gridLayout.addWidget(browseFileButton, row, 3)
                    if ch > 0 and allow_add_field:
                        delFieldButton.widgets.append(browseFileButton)
                else:
                    emptyLabel = QLabel('')
                    gridLayout.addWidget(emptyLabel, row, 3)
                    if ch > 0 and allow_add_field:
                        delFieldButton.widgets.append(emptyLabel)
                
                if channel:
                    key = (exp_path, channel)
                else:
                    key = (exp_path, row)
                
                self.widgets[key] = widget
                if ch > 0 and allow_add_field:
                    delFieldButton.key = key
                
                row += 1           
            
            if allow_add_field:
                addFieldButton.row = row
                addFieldButton.browseFileButtonKwargs = browseFileButtonKwargs
                addFieldButton.widgetFunc = widgetFunc
                addFieldButton.widget_kwargs = widget_kwargs_ch
                addFieldButton.exp_path = exp_path
                addFieldButton.expPathLabel = expPathLabel
                addFieldButton.expPathLabelRow = expPathLabelRow
                addFieldButton.expPathRowSpan = rowSpan

        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        gridLayout.setColumnStretch(0, 2)
        gridLayout.setColumnStretch(1, 0)
        gridLayout.setColumnStretch(2, 1)
        gridLayout.setColumnStretch(3, 0)
        gridLayout.setColumnStretch(4, 0)
        gridLayout.setColumnStretch(5, 0)
        gridLayout.setRowStretch(0, 0)
        
        buttonsLayout = acdc_widgets.CancelOkButtonsLayout()
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.mainLayout = mainLayout
        
        self.setLayout(mainLayout)
    
    def resizeEvent(self, event: QResizeEvent):        
        self.isAnyLabelElided = False
        longestLineWidth = 0
        for lineEdit in self.expPathLabels:
            if lineEdit.isTextElided():
                self.isAnyLabelElided = True
            width = lineEdit.width()
            if width > longestLineWidth:
                longestLineWidth = width
                longestLineEdit = lineEdit
        
        columnStretch = 2 if self.isAnyLabelElided else 0
        currentColStretch = self.gridLayout.columnStretch(0)
        
        oldWidth = event.oldSize().width()
        width = event.size().width()
        
        if not self.isAnyLabelElided and width < oldWidth:
            self.gridLayout.setColumnStretch(0, 2)
            longestLineEdit.setMinimumWidth(5)
            columnStretch = 2
        
        if columnStretch == currentColStretch:
            return
        
        if not columnStretch:
            longestLineEdit.setMinimumWidth(longestLineWidth+5)
            longestLineEdit.setMaximumWidth(longestLineWidth+5)
            self.gridLayout.setColumnStretch(0, 0)
    
    def autoFillToggled(self, enabled: bool):
        if not enabled:
            return
        
        for key, widget in self.widgets.items():
            if not widget.value():
                continue
            
            self.autoFill(widget.value())
    
    def autoFill(self, value):
        for key, widget in self.widgets.items():
            if widget.value():
                continue
            
            exp_path, _ = key
            filepath = acdc_load.search_filepath_from_endname(
                exp_path, value
            )
            if filepath is None:
                continue
            
            folderpath = os.path.dirname(filepath)
            foldername = os.path.basename(folderpath)
            widget.setText(value)
            widget.setToolTip(foldername)
    
    def addField(self):
        self.prevScrollAreaHeight = int(
            self.scrollArea.widget().sizeHint().height()
            + self.scrollArea.horizontalScrollBar().sizeHint().height()
        )
        
        addFieldButton = self.sender()
        nextRow = addFieldButton.row
        
        # Move widgets one row down
        for i in range(nextRow, self.gridLayout.rowCount()):
            for j in range(self.gridLayout.columnCount()):
                item = self.gridLayout.itemAtPosition(i, j)
                if item is None:
                    continue
                self.gridLayout.addWidget(item.widget(), i+1, j)
        
        widget_kwargs = addFieldButton.widget_kwargs
        widgetFunc = addFieldButton.widgetFunc
        exp_path = addFieldButton.exp_path
        
        addFieldButton.expPathRowSpan += 1
        rowSpan = addFieldButton.expPathRowSpan
        self.gridLayout.addWidget(
            addFieldButton.expPathLabel, 
            addFieldButton.expPathLabelRow, 0, 
            rowSpan, 1, 
            alignment=Qt.AlignVCenter
        ) 
        
        delFieldButton = acdc_widgets.delPushButton()
        delFieldButton.addFieldButton = addFieldButton
        self.gridLayout.addWidget(delFieldButton, nextRow, self._last_col)   
        delFieldButton.widgets = []
        delFieldButton.clicked.connect(self.removeField)
        
        widget = widgetFunc(**widget_kwargs)
        if self._enable_autofill:
            widget.sigValueChanged.connect(self.autoFill)
        
        emptyLabel = QLabel('')
        self.gridLayout.addWidget(
            emptyLabel, nextRow, 1, alignment=Qt.AlignLeft
        )
        delFieldButton.widgets.append(emptyLabel)
        
        # lineEdit.setReadOnly(True)
        self.gridLayout.addWidget(widget, nextRow, 2)

        browseFileButtonKwargs = {}
        if self._add_browse_button:
            browseFileButton = acdc_widgets.browseFileButton(
                **addFieldButton.browseFileButtonKwargs
            )                    
            browseFileButton.sigPathSelected.connect(
                partial(
                    self.pathSelected, 
                    lineEdit=widget, 
                    parentExpPath=exp_path
                )
            )
            self.gridLayout.addWidget(browseFileButton, nextRow, 3)
            delFieldButton.widgets.append(browseFileButton)
        else:
            emptyLabel = QLabel('')
            self.gridLayout.addWidget(emptyLabel, nextRow, 3)
            delFieldButton.widgets.append(emptyLabel)
        
        if self._add_apply_to_all_buttons:
            applyToAllButton = widget.ArrowButtons(
                order=('down', 'up')
            )
            self.gridLayout.addWidget(
                applyToAllButton, nextRow, self._last_col-1
            )
            applyToAllButton.sigButtonClicked.connect(
                self.applyToOtherFields
            )
            applyToAllButton.row = nextRow
            delButtonWidget = applyToAllButton
        else:
            emptyLabel = QLabel('')
            self.gridLayout.addWidget(emptyLabel, nextRow, self._last_col-1)
            delButtonWidget = emptyLabel
        delFieldButton.widgets.append(delButtonWidget)
        
        self.widgets[(exp_path, nextRow)] = widget           
        
        delFieldButton.key = (exp_path, nextRow)
        addFieldButton.row = nextRow + 1
        
        QTimer.singleShot(50, self.resizeUponFieldAddedOrRemoved)
    
    def applyToOtherFields(self, direction):
        index = self.sender().index
        if direction == 'up':
            index_range = range(index-1, -1, -1)
        else:
            index_range = range(index+1, len(self.widgets))
        
        widgets = list(self.widgets.values())
        value = widgets[index].value()
        for i in index_range:
            widget = widgets[i]
            widget.setValue(value)
    
    def removeField(self):
        self.prevScrollAreaHeight = int(
            self.scrollArea.widget().sizeHint().height()
            + self.scrollArea.horizontalScrollBar().sizeHint().height()
        )
        
        delFieldButton = self.sender()
        for widget in delFieldButton.widgets:
            self.gridLayout.removeWidget(widget)

        widget = self.widgets[delFieldButton.key]
        self.gridLayout.removeWidget(widget)
        del self.widgets[delFieldButton.key]
        
        addFieldButton = delFieldButton.addFieldButton
        addFieldButton.expPathRowSpan -= 1
        rowSpan = addFieldButton.expPathRowSpan
        self.gridLayout.addWidget(
            addFieldButton.expPathLabel, 
            addFieldButton.expPathLabelRow, 0, 
            rowSpan, 1, 
            alignment=Qt.AlignVCenter
        ) 
        
        self.gridLayout.removeWidget(delFieldButton)
        
        addFieldButton.row -= 1
        
        QTimer.singleShot(50, self.resizeUponFieldAddedOrRemoved)
    
    def resizeUponFieldAddedOrRemoved(self):
        scrollArea = self.scrollArea
        height = int(
            self.scrollArea.widget().sizeHint().height()
            + scrollArea.horizontalScrollBar().sizeHint().height()
        )
        deltaHeight = height - self.prevScrollAreaHeight
        newHeight = self.height() + deltaHeight
        self.resize(self.width(), newHeight)
    
    def warnNotValidExpPathSelected(self, selectedExpPath, parentExpPath):
        txt = ("""
            The experiment path of the selected file does <b>not correspond 
            to the parent experiment folder</b>.<br><br>
            You need to select a file that is in the 
            <code>Position_n/Images</code>                      
        """)
        if self._allow_spotmax_output:
            txt = (f"""{txt}
                or in the <code>Position_n/spotMAX_output</code>                   
            """)
        
        txt = f'{txt} folder(s).<br><br>Parent experiment folder path:'
        txt = html_func.paragraph(txt)
        
        msg = acdc_widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Selected file not valid', txt, 
            commands=(parentExpPath,), 
            path_to_browse=parentExpPath
        )
    
    def pathSelected(self, filePath, lineEdit=None, parentExpPath=''):
        selectedExpPath = os.path.dirname(
            os.path.dirname(os.path.dirname(filePath))
        )
        selectedExpPath = selectedExpPath.replace('\\', '/')
        parentExpPath = parentExpPath.replace('\\', '/')
        if parentExpPath != selectedExpPath:
            self.warnNotValidExpPathSelected(selectedExpPath, parentExpPath)
            return
        
        lineEdit.setValue(filePath)
    
    def show(self, block=False):
        screenWidth = self.screen().size().width()
        windowWidth = int(screenWidth*0.5)
        self.resize(windowWidth, self.sizeHint().height())
        super().show(block=block)      
    
    def ok_cb(self):
        self.cancel = False
        
        self.selectedValues = {}
        for (exp_path, channel), widget in self.widgets.items():
            value = (widget.toolTip(), widget.value()) 
            if self._channel_names is not None or self._allow_add_field:
                if exp_path not in self.selectedValues:
                    self.selectedValues[exp_path] = {}
                self.selectedValues[exp_path][channel] = value
            else:
                self.selectedValues[exp_path] = value
                
        self.close()

def setupSpotmaxAiTraining(qparent=None):
    win = SetupUnetTrainingDialog(parent=qparent)
    win.exec_()
    if win.cancel:
        return False
    
    workflowFilepath = win.workflowFilepath
    workflowFolderpath = os.path.dirname(workflowFilepath)
    msg = acdc_widgets.myMessageBox(wrapText=False)
    txt = html_func.paragraph("""
        Done! Training workflow <b>generated successfully</b>.<br><br>
        To run the worflow and train the SpotMAX AI model, run the command 
        <code>spotmax -t "path to workflow" file</code>.<br><br>
        You can also copy/move the workflow folder wherever you like 
        and run the command with the new file path.<br><br>
        For example, to run the training workflow with the folder in the current 
        location, you would run the following command:<br>
    """)
    msg.information(
        qparent, 'Training workflow generated', txt, 
        commands=(f'spotmax -t "{workflowFilepath}"',),
        path_to_browse=workflowFolderpath
    )
    
    return True
    
class AboutSpotMAXDialog(QBaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle('About SpotMAX')
        
        mainLayout = QVBoxLayout()
        
        textLayout = QHBoxLayout()
        
        version = read_version()
        release_date = acdc_myutils.get_date_from_version(
            version, package='spotmax'
        )
        
        py_ver = sys.version_info
        python_version = f'{py_ver.major}.{py_ver.minor}.{py_ver.micro}'
        
        iconPixmap = QPixmap(icon_path)
        h = 128
        iconPixmap = iconPixmap.scaled(h,h, aspectRatioMode=Qt.KeepAspectRatio)
        iconLabel = QLabel()
        iconLabel.setPixmap(iconPixmap)
        
        titleLabel = QLabel()
        txt = (f"""
        <p style="font-size:20px; font-family:ubuntu">
            <b>SpotMAX</b>
            <span style="font-size:12pt; font-family:ubuntu">
                (Multi-dimensional spot detection and quantification)
            </span>
        </p>
        <p style="font-size:14px; font-family:ubuntu">
            Version {version}<br><br>
            Release date: {release_date}
        </p>
        <p style="font-size:13px; font-family:ubuntu">
            Qt {QtCore.__version__}<br>
            Python {python_version}<br>
        </p>
        <p style="font-size:13px; font-family:ubuntu">
            Platform: {platform.platform()}<br>
        </p>
        """)

        titleLabel.setText(txt)
        
        textLayout.addWidget(iconLabel, alignment=Qt.AlignTop)
        textLayout.addWidget(QLabel(txt), alignment=Qt.AlignTop)
        
        buttonsLayout = QHBoxLayout()
        
        buttonsLayout.addWidget(QLabel(
            f'Installed in: <code>{spotmax_path}</code>'
        ))
        buttonsLayout.addStretch(1)
        copyPathButton = acdc_widgets.copyPushButton('Copy path')
        copyPathButton.setTextToCopy(spotmax_path)
        buttonsLayout.addWidget(copyPathButton)
        
        browseButton = acdc_widgets.showInFileManagerButton(setDefaultText=True)
        browseButton.setPathToBrowse(os.path.dirname(spotmax_path))
        
        buttonsLayout.addWidget(browseButton)
        
        mainLayout.addLayout(textLayout)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)

class QDialogBioimageIOModelParams(acdc_apps.QDialogModelParams):
    def __init__(self, posData=None, df_metadata=None, parent=None):
        from spotmax.BioImageIO import model
        init_params, segment_params = acdc_myutils.getModelArgSpec(model)
        url = model.url_help()
        
        super().__init__(
            init_params,
            segment_params,
            'BioImageIO model', 
            url=url, 
            initLastParams=True, 
            posData=posData,
            df_metadata=df_metadata,
            force_postprocess_2D=False,
            addPostProcessParams=False,
            addPreProcessParams=False,
            model_module=model, 
            parent=parent
        )
        
        initGroupboxLayout = (
            self.scrollArea.widget().layout().itemAt(0).widget().layout()
        )
        reloadButton = acdc_widgets.reloadPushButton()
        reloadButton.setToolTip(
            'Load model-specific additional parameters.'
        )
        initGroupboxLayout.addWidget(reloadButton, 0, 4)
        reloadButton.clicked.connect(self.onModelSelected)
        self.reloadButton = reloadButton
        self.reloadButton.blinker = utils.widgetBlinker(reloadButton)
        
        modelSourceArgWidget = self.init_argsWidgets[0]
        modelSourceArgWidget.widget.editingFinished.connect(
            self.blinkReloadButton
        )
        
        self.additionalWidgetsAdded = False
        self.deltaHeight = 0
        # self.onModelSelected(resize=False)
    
    def blinkReloadButton(self):
        init_kwargs = self.argsWidgets_to_kwargs(self.init_argsWidgets)
        if not init_kwargs['model_doi_url_rdf_or_zip_path']:
            return
        
        self.reloadButton.blinker.start()
    
    def onModelSelected(self, checked=False, resize=True):
        init_kwargs = self.argsWidgets_to_kwargs(self.init_argsWidgets)
        if not init_kwargs['model_doi_url_rdf_or_zip_path']:
            return
        
        from spotmax.BioImageIO import model
        try:
            bioImageIOModel = model.Model(**init_kwargs)
            additionalKwargs = bioImageIOModel.kwargs
            additionalArgspecs = self.kwargsToArgspecs(additionalKwargs)
            additionalInitGroupBox, self.additionalInitArgsWidgets = (
                self.createGroupParams(
                    additionalArgspecs, 
                    'Additional initialization parameters'
                )
            )
            scrollAreaLayout = self.scrollArea.widget().layout()
            if self.additionalWidgetsAdded:
                scrollAreaLayout.removeWidget(1)
            scrollAreaLayout.insertWidget(1, additionalInitGroupBox)
            self.additionalWidgetsAdded = True
            if not resize:
                return
            
            QTimer.singleShot(100, self._resizeHeight)
        except Exception as err:
            self.additionalWidgetsAdded = False
            printl(err)
            return
    
    def _resizeHeight(self):
        newDeltaHeight = (
            self.scrollArea.minimumHeightNoScrollbar() + 70 - self.deltaHeight
        )
        self.resize(self.width(), newDeltaHeight)
        self.deltaHeight = newDeltaHeight
    
    def ok_cb(self, checked=False):
        try:
            self.additionalKwargs = self.argsWidgets_to_kwargs(
                self.additionalInitArgsWidgets
            )
        except Exception as err:
            self.additionalKwargs = None
            
        super().ok_cb(checked)
    
    def setValuesFromParams(self, init_kwargs, segment_kwargs, model_kwargs):
        super().__init__(init_kwargs, segment_kwargs)
        if not self.additionalWidgetsAdded:
            return
        
        for argWidget in self.additionalInitArgsWidgets:
            val = model_kwargs.get(argWidget.name)
            widget = argWidget.widget
            if val is None:
                continue
            casters = [lambda x: x, int, float, str, bool]
            for caster in casters:
                try:
                    argWidget.valueSetter(widget, caster(val))
                    break
                except Exception as e:
                    continue
    
    def kwargsToArgspecs(self, kwargs):
        argSpecs = []
        for key, value in kwargs.items():
            argspec = acdc_myutils.ArgSpec(
                name=key, 
                default=value, 
                type=type(value),
                desc='Additional BioImageIO model parameter', 
                docstring=None
            )
            argSpecs.append(argspec)
        return argSpecs
        
        
        