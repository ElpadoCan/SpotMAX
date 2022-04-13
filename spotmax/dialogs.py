import os
import sys
import re
import pathlib
import time
from pprint import pprint

import numpy as np
import pandas as pd
from natsort import natsorted

from collections import defaultdict

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal
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
    QSlider, QDockWidget, QTabWidget, QScrollArea, QScrollBar
)

from . import html_func, load, widgets, core, utils

# NOTE: Enable icons
from . import qrc_resources

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
        helpButton = QPushButton('Help', self)
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
            isCheckable=False, checked=False, parent=None
        ):
        self.font13px = QFont()
        self.font13px.setPixelSize(13)

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
        layout.addWidget(
            howDrawSegmCombobox, row, col, alignment=Qt.AlignCenter
        )
        bottomWidgets['howDrawSegmCombobox'] = howDrawSegmCombobox

        row = 1
        col = initialCol +0
        navigateScrollbar_label = QLabel('frame n.  ')
        navigateScrollbar_label.setFont(self.font13px)
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
        zSliceSbL0_label.setFont(self.font13px)
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
        zSliceSbL1_label.setFont(self.font13px)
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
        alphaScrollbar_label.setFont(self.font13px)
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

    def setHowToDrawEnabled(self, state):
        bottomWidgets = self.bottomWidgets
        combobox = bottomWidgets['howDrawSegmCombobox']
        if state:
            combobox.setDisabled(False, applyToCheckbox=False)
        else:
            combobox.setDisabled(True, applyToCheckbox=False)

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
    def __init__(self, *args):
        super().__init__(args[0])

        self.parametersTab = QScrollArea(self)
        self.parametersQGBox = analysisInputsQGBox(self.parametersTab)
        self.parametersTab.setWidget(self.parametersQGBox)

        self.addTab(self.parametersTab, 'Analysis paramenters')

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

        font = QFont()
        font.setPixelSize(13)
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
        mainLayout = widgets.myFormLayout()

        font = QFont()
        font.setPixelSize(13)

        row = 0
        self.loadRefChWidget = widgets.formWidget(
            widgets.Toggle(), anchor='loadRefCh',
            stretchWidget=False, addInfoButton=True,
            parent=self
        )
        mainLayout.addFormWidget(self.loadRefChWidget, row=row)

        row += 1
        self.refChSingleObjWidget = widgets.formWidget(
            widgets.Toggle(), anchor='refChSingleObj',
            stretchWidget=False, addInfoButton=True,
            addApplyButton=True, parent=self
        )
        mainLayout.addFormWidget(self.refChSingleObjWidget, row=row)

        row += 1
        self.keepPeaksInsideRefWidget = widgets.formWidget(
            widgets.Toggle(), anchor='keepPeaksInsideRef',
            stretchWidget=False, addInfoButton=True,
            addApplyButton=True, parent=self
        )
        mainLayout.addFormWidget(self.keepPeaksInsideRefWidget, row=row)

        row += 1
        self.filterPeaksInsideRefWidget = widgets.formWidget(
            widgets.Toggle(), anchor='filterPeaksInsideRef',
            stretchWidget=False, addInfoButton=True,
            addComputeButton=True, parent=self
        )
        mainLayout.addFormWidget(self.filterPeaksInsideRefWidget, row=row)

        row += 1
        self.sharpenSpotsWidget = widgets.formWidget(
            widgets.Toggle(), anchor='sharpenSpots',
            stretchWidget=False, addInfoButton=True,
            addComputeButton=True, parent=self
        )
        mainLayout.addFormWidget(self.sharpenSpotsWidget, row=row)

        row += 1
        self.aggregateWidget = widgets.formWidget(
            widgets.Toggle(), anchor='aggregate',
            stretchWidget=False, addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.aggregateWidget, row=row)

        row += 1
        self.pixelWidthWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='pixelWidth',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.pixelWidthWidget, row=row)

        row += 1
        self.pixelHeightWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='pixelHeight',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.pixelHeightWidget, row=row)

        row += 1
        self.voxelDepthWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='voxelDepth',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.voxelDepthWidget, row=row)

        row += 1
        self.numApertureWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='numAperture',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.numApertureWidget, row=row)

        row += 1
        self.emWavelenWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='emWavelen',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.emWavelenWidget, row=row)

        row += 1
        self.zResolutionLimitWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='zResolutionLimit',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.zResolutionLimitWidget, row=row)

        row += 1
        self.yxResolLimitMultiplierWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='yxResolLimitMultiplier',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.yxResolLimitMultiplierWidget, row=row)

        row += 1
        txt = 'Spot (z,y,x) minimum dimensions'
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, 2, 1, alignment=Qt.AlignRight)

        self.spotSize_um_label = QLabel()
        self.spotSize_um_label.setFont(font)
        mainLayout.addWidget(
            self.spotSize_um_label, row, 1, alignment=Qt.AlignCenter
        )

        row += 1
        self.spotSize_pxl_label = QLabel()
        self.spotSize_pxl_label.setFont(font)
        mainLayout.addWidget(
            self.spotSize_pxl_label, row, 1, alignment=Qt.AlignCenter
        )

        row += 1
        self.gaussSigmaWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='gaussSigma',
            addInfoButton=True, addComputeButton=True,
            parent=self
        )
        mainLayout.addFormWidget(self.gaussSigmaWidget, row=row)

        row += 1
        widget = widgets.myQComboBox()
        items = utils.skimageAutoThresholdMethods()
        widget.addItems(items)
        self.refChThresholdFuncWidget = widgets.formWidget(
            widget, anchor='refChThresholdFunc',
            addInfoButton=True, addComputeButton=True,
            parent=self
        )
        mainLayout.addFormWidget(self.refChThresholdFuncWidget, row=row)


        row += 1
        widget = widgets.myQComboBox()
        items = utils.skimageAutoThresholdMethods()
        widget.addItems(items)
        self.spotThresholdFuncWidget = widgets.formWidget(
            widget, anchor='spotThresholdFunc',
            addInfoButton=True, addComputeButton=True,
            parent=self
        )
        mainLayout.addFormWidget(self.spotThresholdFuncWidget, row=row)

        row += 1
        widget = widgets.myQComboBox()
        items = ['Effect size', 't-test (p-value)']
        widget.addItems(items)
        self.gopMethodWidget = widgets.formWidget(
            widget, anchor='gopMethod',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.gopMethodWidget, row=row)

        row += 1
        self.gopLimitWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='gopLimit',
            addInfoButton=True, addComputeButton=True,
            parent=self
        )
        mainLayout.addFormWidget(self.gopLimitWidget, row=row)

        row += 1
        self.doSpotFitWidget = widgets.formWidget(
            widgets.Toggle(), anchor='doSpotFit',
            stretchWidget=False, addInfoButton=True,
            addComputeButton=True, parent=self
        )
        mainLayout.addFormWidget(self.doSpotFitWidget, row=row)

        row += 1
        self.minSpotSizeWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='minSpotSize',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.minSpotSizeWidget, row=row)

        row += 1
        self.maxSpotSizeWidget = widgets.formWidget(
            widgets.floatLineEdit(), anchor='maxSpotSize',
            addInfoButton=True, parent=self
        )
        mainLayout.addFormWidget(self.maxSpotSizeWidget, row=row)

        row += 1
        self.calcRefChNetLenWidget = widgets.formWidget(
            widgets.Toggle(), anchor='calcRefChNetLen',
            stretchWidget=False, addInfoButton=True,
            addComputeButton=True, parent=self
        )
        mainLayout.addFormWidget(self.calcRefChNetLenWidget, row=row)

        row += 1
        mainLayout.setRowStretch(row, 1)

        self.setLayout(mainLayout)

        self.updateMinSpotSize()
        self.connectActions()

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
        physicalSizeX = self.pixelWidthWidget.widget.value()
        physicalSizeY = self.pixelHeightWidget.widget.value()
        physicalSizeZ = self.voxelDepthWidget.widget.value()
        emWavelen = self.emWavelenWidget.widget.value()
        NA = self.numApertureWidget.widget.value()
        zResolutionLimit_um = self.zResolutionLimitWidget.widget.value()
        yxResolMultiplier = self.yxResolLimitMultiplierWidget.widget.value()
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
        self.spotSize_pxl_label.setText(zyxMinSize_pxl_txt)
        self.spotSize_um_label.setText(zyxMinSize_um_txt)

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

        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = QPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okButton)
        buttonsLayout.addWidget(cancelButton)
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


class QDialogMetadata(QDialog):
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
        okButton = QPushButton(okTxt)
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

        cancelButton = QPushButton('Cancel')

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
        self.setModal(True)

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
            msg = widgets.myMessageBox(self)
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
                select_folder = load.select_exp_folder()
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
                        msg = widgets.myMessageBox()
                        msg.critical(self, 'Permission denied', err_msg)
                        acdc_df.to_csv(acdc_df_path, index=False)

        elif self.sender() == self.selectButton:
            pass

        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

class QDialogCombobox(QDialog):
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

        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        bottomLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        bottomLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(infoLayout)
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        self.setModal(True)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)


    def ok_cb(self, event):
        self.cancel = False
        self.selectedItemText = self.ComboBox.currentText()
        self.selectedItemIdx = self.ComboBox.currentIndex()
        self.close()

class QDialogListbox(QDialog):
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
        _font = QFont()
        _font.setPixelSize(13)
        label.setFont(_font)
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

        listBox = QListWidget()
        listBox.setFont(_font)
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

        okButton = QPushButton('Ok')
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
        self.setModal(True)
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

class selectPathsSpotmax(QDialog):
    def __init__(self, paths, homePath, parent=None, app=None):
        super().__init__(parent)

        self.selectedPaths = []
        self.paths = paths
        runs = sorted(list(self.paths.keys()))
        self.runs = runs
        self.isCtrlDown = False
        self.isShiftDown = False

        self.setWindowTitle('Select experiments to load')

        infoLabel = QLabel()
        text = 'Select <b>one or more folders</b> to load (Ctrl+A to select All) <br>'
        htmlText = html_func.paragraph(text)
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
        showAnalysisTableButton = QPushButton(
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
        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        buttonsLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        buttonsLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        buttonsLayout.setContentsMargins(0, 20, 0, 0)


        mainLayout = QVBoxLayout()
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addLayout(runNumberLayout)
        runNumberLayout.setContentsMargins(0, 0, 0, 10)
        mainLayout.addLayout(checkBoxesLayout)
        mainLayout.addWidget(pathSelector)
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

        font = QFont()
        font.setPixelSize(13)
        self.setFont(font)

        self.setMyStyleSheet()
        self.setModal(True)

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
            relPath = pathlib.Path(*relPath1.parts[1:])
            exp_path = os.path.join(self.homePath, relPath)

            selectedRunPaths = self.paths[run]
            df = selectedRunPaths[str(exp_path)].get('analysisInputs')
        else:
            posFoldername = re.findall('(.+) \(', plainText)[0]
            parentLabel = label = self.pathSelector.itemWidget(parent, 0)
            doc.setHtml(parentLabel.text())
            parentText = doc.toPlainText()
            relPath1 = re.findall('...(.+) \(', parentText)[0]
            relPath1 = pathlib.Path(relPath1)
            relPath = pathlib.Path(*relPath1.parts[1:])
            relPath1 = relPath / posFoldername
            exp_path = self.homePath / relPath / posFoldername
            spotmaxOutPath = exp_path / 'spotMAX_output'
            if os.path.exists(spotmaxOutPath):
                df = load.scanExpFolders().loadAnalysisInputs(spotmaxOutPath, run)
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
            relPath = f'...{self.homePath.name}{os.path.sep}{rel}'

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
                nPSCtext = ''

            if numPosSpotSized < totPos and numPosSpotSized>0:
                nPSStext = f'Number of spotSIZED pos. = {numPosSpotSized}'
            elif numPosSpotSized>0:
                nPSStext = f'All pos. spotSIZED'
                if not addSpotSized:
                    continue
            elif numPosSpotCounted>0:
                nPSStext = 'NONE of the pos. spotSIZED'
            else:
                nPSStext = ''

            relPathItem = QTreeWidgetItem()
            pathSelector.addTopLevelItem(relPathItem)
            relPathLabel = widgets.QClickableLabel()
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
                posItem = QTreeWidgetItem()
                posLabel = widgets.QClickableLabel()
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
        msg = widgets.myMessageBox()
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
            relPath = pathlib.Path(*relPath.parts[1:])
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
        msg = widgets.myMessageBox()
        msg.warning(
            self, 'No path selected!', html_func.paragraph(text)
        )

    def warningNotPathsSelected(self):
        text = (
            '<b>You didn\'t select any path!</b> Do you want to cancel loading data?'
        )
        msg = widgets.myMessageBox()
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

    def show(self):
        QDialog.show(self)
        self.resizeSelector()

class QDialogWorkerProcess(QDialog):
    def __init__(
            self, title='Progress', infoTxt='',
            showInnerPbar=False, pbarDesc='',
            parent=None
        ):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        pBarLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        if infoTxt:
            infoLabel = QLabel(infoTxt)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        self.progressLabel = QLabel(pbarDesc)

        self.mainPbar = widgets.QProgressBarWithETA(self)
        self.mainPbar.setValue(0)
        pBarLayout.addWidget(self.mainPbar, 0, 0)
        pBarLayout.addWidget(self.mainPbar.ETA_label, 0, 1)

        self.innerPbar = widgets.QProgressBarWithETA(self)
        self.innerPbar.setValue(0)
        pBarLayout.addWidget(self.innerPbar, 1, 0)
        pBarLayout.addWidget(self.innerPbar.ETA_label, 1, 1)
        if showInnerPbar:
            self.innerPbar.show()
        else:
            self.innerPbar.hide()

        self.logConsole = widgets.QLogConsole()

        abortButton = QPushButton('   Abort process   ')
        abortButton.clicked.connect(self.abort)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(abortButton)
        # buttonsLayout.addStretch(1)
        buttonsLayout.setContentsMargins(0,10,0,5)

        mainLayout.addWidget(self.progressLabel)
        mainLayout.addLayout(pBarLayout)
        mainLayout.addWidget(self.logConsole)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        self.setModal(True)

    def abort(self):
        self.clickCount += 1
        self.aborted = True
        if self.clickCount > 3:
            self.workerFinished = True
            self.close()

    def closeEvent(self, event):
        if not self.workerFinished:
            event.ignore()

    def show(self, app):
        QDialog.show(self)
        screen = app.primaryScreen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()
        parentGeometry = self.parent().geometry()
        mainWinLeft, mainWinWidth = parentGeometry.left(), parentGeometry.width()
        mainWinTop, mainWinHeight = parentGeometry.top(), parentGeometry.height()
        mainWinCenterX = int(mainWinLeft+mainWinWidth/2)
        mainWinCenterY = int(mainWinTop+mainWinHeight/2)

        width = int(screenWidth/3)
        height = int(screenHeight/3)
        left = int(mainWinCenterX - width/2)
        top = int(mainWinCenterY - height/2)

        self.setGeometry(left, top, width, height)

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
        # layout.addWidget(QPushButton('Ok', self))
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

class selectSpotsH5FileDialog(QDialog):
    def __init__(self, runsInfo, parent=None, app=None):
        QDialog.__init__(self, parent)

        self.setWindowTitle('Select analysis to load')

        self.parent = parent
        self.app = app
        self.runsInfo = runsInfo
        self.selectedFile = None

        font = QFont()
        font.setPixelSize(13)
        self.setFont(font)

        mainLayout = selectSpotsH5FileLayout(
            runsInfo, font=font, parent=self, app=app
        )

        buttonsLayout = QHBoxLayout()
        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        buttonsLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        buttonsLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        buttonsLayout.setContentsMargins(0, 20, 0, 0)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.mainLayout = mainLayout
        self.setLayout(mainLayout)
        self.setModal(True)

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
        msg = widgets.myMessageBox()
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

    def show(self):
        QDialog.show(self)
        self.resizeSelector()

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

if __name__ == '__main__':
    class Window(QMainWindow):
        def __init__(self):
            super().__init__()

            container = QWidget()
            layout = QVBoxLayout()

            self.tabControl = guiTabControl(self)

            bottomWidgets = guiBottomWidgets(
                'left', ['test'], ['test'], isCheckable=True, checked=False
            )

            layout.addWidget(self.tabControl)
            layout.addWidget(bottomWidgets)
            names = utils.singleSpotCountMeasurementsName()
            layout.addWidget(measurementsQGroupBox(names, parent=self))

            # layout.addStretch(1)
            container.setLayout(layout)
            self.setCentralWidget(container)

        def show(self):
            QMainWindow.show(self)

            parametersQGBox = self.tabControl.parametersQGBox
            parametersTab = self.tabControl.parametersTab
            horizontalScrollBar = parametersTab.horizontalScrollBar()
            i = 1
            while horizontalScrollBar.isVisible():
                self.resize(self.width()+i, self.height())
                i += 1

            channelDataPath = r"G:\My Drive\1_MIA_Data\Dimitra\test_spotMAX_nucleusSegm_1\TIFFs\Position_2\Images\20210526_DCY8_SCGE_M2_10_s02_mCitr.tif"
            user_ch_name = 'mCitr'
            posData = load.loadData(channelDataPath, user_ch_name)

            run_nums = posData.validRuns()
            runsInfo = {}
            for run in run_nums:
                h5_files = posData.h5_files(run)
                if not h5_files:
                    continue
                runsInfo[run] = h5_files

            # self.dialog = selectSpotsH5FileDialog(runsInfo)
            # self.dialog.show()
            self.tabControl.addInspectResultsTab(posData)


            # print(self.tabControl.inspectResultsQGBox.width())
            # self.tabControl.inspectResultsQGBox.setMinimumWidth(500)
            # print(self.tabControl.inspectResultsQGBox.width())

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    # win = QDialogMetadata(
    #     60, 40, 180, 0.35, 0.06, 0.06, True, True, True, 15,
    #     imgDataShape=(60,40,600,600)
    # )
    win = Window()
    win.show()
    app.exec_()

    # print(win.selectedPaths)
