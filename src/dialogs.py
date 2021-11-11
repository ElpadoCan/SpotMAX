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
from PyQt5.QtGui import QFont, QFontMetrics, QTextDocument, QPalette, QColor
from PyQt5.QtWidgets import (
    QDialog, QComboBox, QVBoxLayout, QHBoxLayout, QLabel, QApplication,
    QPushButton, QStyleFactory, QCheckBox, QTreeWidget, QTreeWidgetItem,
    QTreeWidgetItemIterator, QAbstractItemView, QFrame, QMessageBox,
    QMainWindow, QWidget, QTableView, QTextEdit, QGridLayout,
    QProgressBar, QSpinBox, QDoubleSpinBox, QListWidget, QGroupBox,
    QSlider, QDockWidget
)

import html, load, widgets

class analysisInputsQFrame(QGroupBox):
    def __init__(self, *args, **kwargs):
        QGroupBox.__init__(self, *args, **kwargs)

        mainLayout = QGridLayout(self)

        font = QFont()
        font.setPointSize(10)

        row = 0
        txt = 'Load and segment reference channel '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.loadRefChToggle = widgets.Toggle()
        toggleLayout = QHBoxLayout()
        toggleLayout.addStretch(1)
        toggleLayout.addWidget(self.loadRefChToggle)
        toggleLayout.addStretch(1)
        mainLayout.addLayout(toggleLayout, row, 1)

        row += 1
        txt = 'Ref. channel is single object (e.g., nucleus) '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.refChSingleObjToggle = widgets.Toggle()
        toggleLayout = QHBoxLayout()
        toggleLayout.addStretch(1)
        toggleLayout.addWidget(self.refChSingleObjToggle)
        toggleLayout.addStretch(1)
        mainLayout.addLayout(toggleLayout, row, 1)

        row += 1
        txt = 'Keep only peaks that are inside reference channel '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.keepPeaksInsideRefToggle = widgets.Toggle()
        toggleLayout = QHBoxLayout()
        toggleLayout.addStretch(1)
        toggleLayout.addWidget(self.keepPeaksInsideRefToggle)
        toggleLayout.addStretch(1)
        mainLayout.addLayout(toggleLayout, row, 1)

        row += 1
        txt = 'Sharpen spots signal prior detection '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.sharpenSpotsToggle = widgets.Toggle()
        toggleLayout = QHBoxLayout()
        toggleLayout.addStretch(1)
        toggleLayout.addWidget(self.sharpenSpotsToggle)
        toggleLayout.addStretch(1)
        mainLayout.addLayout(toggleLayout, row, 1)

        row += 1
        txt = 'Aggregate cells prior detection '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)
        self.aggregateToggle = widgets.Toggle()
        toggleLayout = QHBoxLayout()
        toggleLayout.addStretch(1)
        toggleLayout.addWidget(self.aggregateToggle)
        toggleLayout.addStretch(1)
        mainLayout.addLayout(toggleLayout, row, 1)

        row += 1
        txt = 'Pixel width (μm) '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.pixelWidthQDSB = widgets.floatLineEdit()
        mainLayout.addWidget(self.pixelWidthQDSB, row, 1)

        row += 1
        txt = 'Pixel height (μm) '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.pixelHeightQDSB = widgets.floatLineEdit()
        mainLayout.addWidget(self.pixelHeightQDSB, row, 1)

        row += 1
        txt = 'Voxel depth (μm) '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.voxelDepthQDSB = widgets.floatLineEdit()
        mainLayout.addWidget(self.voxelDepthQDSB, row, 1)

        row += 1
        txt = 'Numerical aperture '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.numApertureQDSB = widgets.floatLineEdit()
        mainLayout.addWidget(self.numApertureQDSB, row, 1)

        row += 1
        txt = 'Spots reporter emission wavelength (nm) '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.emWavelenQDSB = widgets.floatLineEdit()
        mainLayout.addWidget(self.emWavelenQDSB, row, 1)

        row += 1
        txt = 'Resolution limit in z-direction (μm) '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.zResolutionLimit = widgets.floatLineEdit()
        mainLayout.addWidget(self.zResolutionLimit, row, 1)

        row += 1
        txt = 'Resolution multiplier in y- and x- direction '
        label = QLabel(txt)
        label.setFont(font)
        mainLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)

        self.yxResolMultiplierLimit = widgets.floatLineEdit()
        mainLayout.addWidget(self.yxResolMultiplierLimit, row, 1)

        row += 1
        mainLayout.setRowStretch(row, 1)

        self.setLayout(mainLayout)

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
        self.transpSlider = widgets.sliderWithSpinBox(title='Transparency')
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
    def __init__(self, SizeT, SizeZ, TimeIncrement,
                 PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX,
                 ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes,
                 parent=None, font=None, imgDataShape=None, PosData=None,
                 singlePos=False):
        self.cancel = True
        self.ask_TimeIncrement = ask_TimeIncrement
        self.ask_PhysicalSizes = ask_PhysicalSizes
        self.imgDataShape = imgDataShape
        self.PosData = PosData
        super().__init__(parent)
        self.setWindowTitle('Image properties')

        mainLayout = QVBoxLayout()
        gridLayout = QGridLayout()
        # formLayout = QFormLayout()
        buttonsLayout = QGridLayout()

        if imgDataShape is not None:
            label = QLabel(
                f"""
                <p style="font-size:11pt">
                    <i>Image data shape</i> = <b>{imgDataShape}</b><br>
                </p>
                """)
            mainLayout.addWidget(label, alignment=Qt.AlignCenter)

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

        if singlePos:
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

        gridLayout.setColumnMinimumWidth(1, 100)
        mainLayout.addLayout(gridLayout)
        # mainLayout.addLayout(formLayout)
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
        else:
            self.SizeZ_SpinBox.setMaximum(2147483647)

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
        else:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()

    def ok_cb(self, event):
        self.cancel = False
        self.SizeT = self.SizeT_SpinBox.value()
        self.SizeZ = self.SizeZ_SpinBox.value()

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
            txt = (f"""
            <p style="font-size:10pt">
                You loaded <b>4D data</b>, hence the number of frames MUST be
                <b>{T}</b><br> nd the number of z-slices MUST be <b>{Z}</b>.<br><br>
                What do you want to do?
            </p>
            """)
        if not valid3D:
            txt = (f"""
            <p style="font-size:10pt">
                You loaded <b>3D data</b>, hence either the number of frames is
                <b>{TZ}</b><br> or the number of z-slices can be <b>{TZ}</b>.<br><br>
                However, if the number of frames is greater than 1 then the<br>
                number of z-slices MUST be 1, and vice-versa.<br><br>
                What do you want to do?
            </p>
            """)

        if not valid2D:
            txt = (f"""
            <p style="font-size:10pt">
                You loaded <b>2D data</b>, hence the number of frames MUST be <b>1</b>
                and the number of z-slices MUST be <b>1</b>.<br><br>
                What do you want to do?
            </p>
            """)

        if not valid:
            msg = QMessageBox(self)
            msg.setIcon(msg.Warning)
            msg.setWindowTitle('Invalid entries')
            msg.setText(txt)
            continueButton = QPushButton(
                f'Continue anyway'
            )
            cancelButton = QPushButton(
                f'Let me correct'
            )
            msg.addButton(continueButton, msg.YesRole)
            msg.addButton(cancelButton, msg.NoRole)
            msg.exec_()
            if msg.clickedButton() == cancelButton:
                return

        if self.PosData is not None and self.sender() != self.okButton:
            exp_path = self.PosData.exp_path
            pos_foldernames = natsorted(os.listdir(exp_path))
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
                ls = os.listdir(images_path)
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
                        msg = QMessageBox()
                        msg.critical(self, 'Permission denied', err_msg, msg.Ok)
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
            multiSelection=True, currentItem=None, parent=None
        ):
        self.cancel = True
        super().__init__(parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        bottomLayout = QHBoxLayout()

        label = QLabel(text)
        _font = QFont()
        _font.setPointSize(10)
        label.setFont(_font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

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
        htmlText = html.html_paragraph_11pt(text)
        infoLabel.setText(htmlText)

        runNumberLayout = QHBoxLayout()
        runNumberLabel = QLabel()
        text = 'Number of pos. analysed for run number: '
        htmlText = html.html_paragraph_10pt(text)
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
        font.setPointSize(10)
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
        infoText = html.html_paragraph_10pt(
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
            relPathLabel.setText(html.html_paragraph_10pt(relPathText))
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
                posLabel.setText(html.html_paragraph_10pt(posText))
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
        msg = QMessageBox()
        msg.warning(
            self, 'Analysis inputs not found!',
            html.html_paragraph_10pt(text),
            msg.Ok
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
        msg = QMessageBox()
        msg.warning(
            self, 'No path selected!', html.html_paragraph_10pt(text),
            msg.Ok
        )

    def warningNotPathsSelected(self):
        text = (
            '<b>You didn\'t select any path!</b> Do you want to cancel loading data?'
        )
        msg = QMessageBox()
        doClose = msg.warning(
            self, 'No paths selected!', html.html_paragraph_10pt(text),
            msg.Yes | msg.No
        )
        return doClose==msg.Yes

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

class selectSpotsH5File(QDialog):
    def __init__(self, runsInfo, parent=None, app=None):
        super().__init__(parent)
        self.runsInfo = runsInfo
        self.selectedFile = None

        self.setWindowTitle('Select analysis to load')

        font = QFont()
        font.setPointSize(10)
        self.setFont(font)

        infoLabel = QLabel()
        text = 'Select which analysis to load <br>'
        htmlText = html.html_paragraph_11pt(text)
        infoLabel.setText(htmlText)

        treeSelector = QTreeWidget()
        self.treeSelector = treeSelector
        treeSelector.setHeaderHidden(True)
        self.populateSelector()

        buttonsLayout = QHBoxLayout()
        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        buttonsLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        buttonsLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        buttonsLayout.setContentsMargins(0, 20, 0, 0)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addWidget(treeSelector)
        mainLayout.addLayout(buttonsLayout)
        self.setLayout(mainLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        treeSelector.itemClicked.connect(self.expandTopLevel)

        treeSelector.setFocus(True)

        self.setMyStyleSheet()
        self.setModal(True)

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

    def populateSelector(self):
        for run, files in self.runsInfo.items():
            runItem = QTreeWidgetItem(self.treeSelector)
            runItem.setText(0, f'Analysis run number {run}')
            runItem.setFont(0, self.font())
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
                fileItem.setFont(0, self.font())
                runItem.addChild(fileItem)

    def expandTopLevel(self, item):
        if item.parent() is None:
            item.setExpanded(True)
            item.setSelected(False)


    def ok_cb(self, checked=True):
        selectedItems = self.treeSelector.selectedItems()
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
        msg = QMessageBox()
        doClose = msg.warning(
            self, 'No files selected!', html.html_paragraph_10pt(text),
            msg.Yes | msg.No
        )
        return doClose==msg.Yes

    def cancel_cb(self, checked=True):
        self.close()

    def resizeSelector(self):
        longestText = '3: Spots after goodness-of-peak AND ellipsoid test'
        w = (
            QFontMetrics(self.font())
            .boundingRect(longestText)
            .width()+80
        )
        self.treeSelector.setMinimumWidth(w)

    def show(self):
        QDialog.show(self)
        self.resizeSelector()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    paths = {
    1: {
        r'C:\Users\Frank\SCD\replicate_num1\0nM': {
            'numPosSpotCounted': 4,
            'numPosSpotSized': 0,
            'posFoldernames': ['Position_1', 'Position_2', 'Position_3', 'Position_4'],
            'Position_1': {'isPosSpotCounted': True, 'isPosSpotSized': True},
            'Position_2': {'isPosSpotCounted': True, 'isPosSpotSized': False},
            'Position_3': {'isPosSpotCounted': True, 'isPosSpotSized': True},
            'Position_4': {'isPosSpotCounted': True, 'isPosSpotSized': False}
        },
        # r'C:\Users\Frank\SCD\replicate_num1\15nM': {
        #     'numPosSpotCounted': 4,
        #     'numPosSpotSized': 4,
        #     'posFoldernames': ['Position_1', 'Position_2', 'Position_3', 'Position_4'],
        #     'Position_1': {'isPosSpotCounted': True, 'isPosSpotSized': True},
        #     'Position_2': {'isPosSpotCounted': True, 'isPosSpotSized': True},
        #     'Position_3': {'isPosSpotCounted': True, 'isPosSpotSized': True},
        #     'Position_4': {'isPosSpotCounted': True, 'isPosSpotSized': True}
        # }
    },
    # 2: {
    #     r'C:\Users\Frank\SCD\replicate_num1\0nM': {
    #         'numPosSpotCounted': 0,
    #         'numPosSpotSized': 0,
    #         'posFoldernames': ['Position_1', 'Position_2', 'Position_3', 'Position_4'],
    #         'Position_1': {'isPosSpotCounted': True, 'isPosSpotSized': True},
    #         'Position_2': {'isPosSpotCounted': True, 'isPosSpotSized': False},
    #         'Position_3': {'isPosSpotCounted': True, 'isPosSpotSized': True},
    #         'Position_4': {'isPosSpotCounted': True, 'isPosSpotSized': False}
    #     }
    #}
    }

    # win = selectPathsSpotmax(paths, r'C:\Users\Frank\SCD')
    runsInfo = {
        1: [
            '1_0_Orig_data_v1.h5',
            '1_1_ellip_test_data_v1.h5',
            '1_2_p-_test_data_v1.h5',
            '1_3_p-_ellip_test_data_v1.h5'
        ]
    }
    # win = QDialogListbox('test', 'Selection test', runsInfo[1])
    win = spotStyleDock('Spots transparency')
    # win.show(app)
    win.show()
    app.exec_()

    # print(win.selectedPaths)
