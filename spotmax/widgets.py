import sys
import time
import re
import traceback
import typing
import webbrowser
from pprint import pprint
from functools import partial
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QWidget
from qtpy import QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from qtpy.QtCore import (
    Signal, QTimer, Qt, QRegularExpression, QEvent, QPropertyAnimation,
    QPointF
)
from qtpy.QtGui import (
    QFont,  QPainter, QRegularExpressionValidator, QIcon, QColor, QPalette
)
from qtpy.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QFormLayout, QWidget, QVBoxLayout, QMainWindow, QStyleFactory,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QDockWidget,
    QScrollArea, QSizePolicy, QComboBox, QPushButton, QScrollBar,
    QGroupBox, QAbstractSlider, QDialog, QStyle, QSpacerItem,
    QAction, QWidgetAction, QMenu, QActionGroup, QFileDialog, QFrame,
    QListWidget, QPlainTextEdit
)

import pyqtgraph as pg

from cellacdc import apps as acdc_apps
from cellacdc import widgets as acdc_widgets
from cellacdc._palettes import lineedit_invalid_entry_stylesheet
from cellacdc import myutils as acdc_myutils
from cellacdc.regex import float_regex

from . import is_mac, is_win, printl, font, font_small
from . import dialogs, config, html_func, _docs
from . import utils
from . import features, io

LINEEDIT_INVALID_ENTRY_STYLESHEET = lineedit_invalid_entry_stylesheet()

def removeHSVcmaps():
    hsv_cmaps = []
    for g, grad in pg.graphicsItems.GradientEditorItem.Gradients.items():
        if grad['mode'] == 'hsv':
            hsv_cmaps.append(g)
    for g in hsv_cmaps:
        del pg.graphicsItems.GradientEditorItem.Gradients[g]

def renamePgCmaps():
    Gradients = pg.graphicsItems.GradientEditorItem.Gradients
    try:
        Gradients['hot'] = Gradients.pop('thermal')
    except KeyError:
        pass
    try:
        Gradients.pop('greyclip')
    except KeyError:
        pass

def addGradients():
    Gradients = pg.graphicsItems.GradientEditorItem.Gradients
    Gradients['cividis'] = {
        'ticks': [
            (0.0, (0, 34, 78, 255)),
            (0.25, (66, 78, 108, 255)),
            (0.5, (124, 123, 120, 255)),
            (0.75, (187, 173, 108, 255)),
            (1.0, (254, 232, 56, 255))],
        'mode': 'rgb'
    }
    Gradients['cool'] = {
        'ticks': [
            (0.0, (0, 255, 255, 255)),
            (1.0, (255, 0, 255, 255))],
        'mode': 'rgb'
    }
    Gradients['sunset'] = {
        'ticks': [
            (0.0, (71, 118, 148, 255)),
            (0.4, (222, 213, 141, 255)),
            (0.8, (229, 184, 155, 255)),
            (1.0, (240, 127, 97, 255))],
        'mode': 'rgb'
    }
    cmaps = {}
    for name, gradient in Gradients.items():
        ticks = gradient['ticks']
        colors = [tuple([v/255 for v in tick[1]]) for tick in ticks]
        cmaps[name] = LinearSegmentedColormap.from_list(name, colors, N=256)
    return cmaps

renamePgCmaps()
removeHSVcmaps()
cmaps = addGradients()

def getMathLabels(text, parent=None):
    html_text = text
    untaggedParagraph, _ = html_func.untag(text, 'p')
    if untaggedParagraph:
        html_text = text
        text = untaggedParagraph[0]

    in_tag_texts, out_tag_texts = html_func.untag(text, 'math')
    if not in_tag_texts[0]:
        label = QLabel(parent)
        label.setText(html_text)

        return label,

    labels = []
    for out_tag_text, in_tag_text in zip(out_tag_texts, in_tag_texts):
        if out_tag_text:
            out_tag_text = html_func.paragraph(out_tag_text)
            labels.append(QLabel(out_tag_text, parent))
        if in_tag_text:
            tex_txt = fr'${in_tag_text}$'
            labels.append(mathTeXLabel(tex_txt, parent))
    return labels

class showPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':magnGlass.svg'))

class TunePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':tune.svg'))

class applyPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':magnGlass.svg'))

class computePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':compute.svg'))

class lessThanPushButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':less_than.svg'))
        flat = kwargs.get('flat')
        if flat is not None:
            self.setFlat(True)

class RunSpotMaxButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cog_play.svg'))

class QSpinBoxOdd(QSpinBox):
    def __init__(self, acceptedValues=(), parent=None):
        QSpinBox.__init__(self, parent)
        self.acceptedValues = acceptedValues
        self.valueChanged.connect(self.onValueChanged)
        self.setSingleStep(2)

    def onValueChanged(self, val):
        if val in self.acceptedValues:
            return
        if val % 2 == 0:
            self.setValue(val+1)

class AutoTuningButton(QPushButton):
    sigToggled = Signal(object, bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        self.setText('  Start autotuning  ')
        self.setIcon(QIcon(':tune.svg'))
        self.toggled.connect(self.onToggled)
    
    def onToggled(self, checked):
        if checked:
            self.setText('  Stop autotuning   ')
            self.setIcon(QIcon(':stop.svg'))
        else:
            self.setText('  Start autotuning  ')
            self.setIcon(QIcon(':tune.svg'))
        self.sigToggled.emit(self, checked)

class AddAutoTunePointsButton(acdc_widgets.CrossCursorPointButton):
    sigToggled = Signal(object, bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCheckable(True)
        self.setText(' Start adding points ')
        self.toggled.connect(self.onToggled)
        self._animationTimer = QTimer()
        self._animationTimer.setInterval(750)
        self._animationTimer.timeout.connect(self.toggledAnimation)
        self._counter = 1
        self.setMinimumWidth(self.sizeHint().width())
    
    def onToggled(self, checked):
        if checked:
            self.setText('   Adding points...   ')
            self._animationTimer.start()
        else:
            self._animationTimer.stop()
            self._counter = 1
            self.setText(' Start adding points ')
        self.sigToggled.emit(self, checked)
    
    def toggledAnimation(self):
        if self._counter == 4:
            self._counter = 1
        dots = '.'*self._counter
        spaces = ' '*(3-self._counter)
        self.setText(f'   Adding points{dots}{spaces}    ')
        self._counter += 1

class measurementsQGroupBox(QGroupBox):
    def __init__(self, names, parent=None):
        QGroupBox.__init__(self, 'Single cell measurements', parent)
        self.formWidgets = []

        self.setCheckable(True)
        layout = FormLayout()

        for row, item in enumerate(names.items()):
            key, labelTextRight = item
            widget = formWidget(
                QCheckBox(), labelTextRight=labelTextRight,
                parent=self, key=key
            )
            layout.addFormWidget(widget, row=row)
            self.formWidgets.append(widget)

        row += 1
        layout.setRowStretch(row, 1)
        layout.setColumnStretch(3, 1)

        layout.setVerticalSpacing(10)
        self.setFont(widget.labelRight.font())
        self.setLayout(layout)

        self.toggled.connect(self.checkAll)

    def checkAll(self, isChecked):
        for _formWidget in self.formWidgets:
            _formWidget.widget.setChecked(isChecked)

class tooltipLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.editingFinished.connect(self.setTextTooltip)

    def setText(self, text):
        QLineEdit.setText(self, text)
        self.setToolTip(text)

    def setTextTooltip(self):
        self.setToolTip(self.text())

class StretchableEmptyWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

class VerticalSpacerEmptyWidget(QWidget):
    def __init__(self, parent=None, height=5) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self.setFixedHeight(height)

class FeatureSelectorButton(QPushButton):
    def __init__(self, text, parent=None, alignment=''):
        super().__init__(text, parent=parent)
        self._isFeatureSet = False
        self._alignment = alignment
        self.setCursor(Qt.PointingHandCursor)
    
    def setFeatureText(self, text):
        self.setText(text)
        self.setFlat(True)
        self._isFeatureSet = True
        if self._alignment:
            self.setStyleSheet(f'text-align:{self._alignment};')
    
    def enterEvent(self, event) -> None:
        if self._isFeatureSet:
            self.setFlat(False)
        return super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._isFeatureSet:
            self.setFlat(True)
        self.update()
        return super().leaveEvent(event)

    def setSizeLongestText(self, longestText):
        currentText = self.text()
        self.setText(longestText)
        w, h = self.sizeHint().width(), self.sizeHint().height()
        self.setMinimumWidth(w+10)
        # self.setMinimumHeight(h+5)
        self.setText(currentText)

class FeatureSelectorDialog(acdc_apps.TreeSelectorDialog):
    sigClose = Signal()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        features_groups = features.get_features_groups()
        self.addTree(features_groups)

        self.setFont(font)
    
    def closeEvent(self, event):
        self.sigClose.emit()

class myQComboBox(QComboBox):
    def __init__(self, checkBox=None, parent=None):
        super().__init__(parent)

        # checkBox that controls if ComboBox can be enabled or not
        self.checkBox = checkBox
        self.activated.connect(self.clearFocus)
        self.installEventFilter(self)

    def eventFilter(self, object, event):
        # Disable wheel scroll on widgets to allow scroll only on scrollarea
        if event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return False

    def setEnabled(self, enabled, applyToCheckbox=True):
        if self.checkBox is None or self.checkBox.isChecked():
            QComboBox.setEnabled(self, enabled)
        else:
            QComboBox.setEnabled(self, False)
        if applyToCheckbox and self.checkBox is not None:
            self.checkBox.setEnabled(enabled)

    def setDisabled(self, disabled, applyToCheckbox=True):
        if self.checkBox is None or self.checkBox.isChecked():
            QComboBox.setDisabled(self, disabled)
        else:
            QComboBox.setDisabled(self, True)
        if applyToCheckbox and self.checkBox is not None:
            self.checkBox.setDisabled(disabled)

class CheckableSpinBoxWidgets:
    def __init__(self, isFloat=True):
        if isFloat:
            self.spinbox = FloatLineEdit()
        else:
            self.spinbox = acdc_widgets.SpinBox()
        self.checkbox = QCheckBox('Activate')
        self.spinbox.setEnabled(False)
        self.checkbox.toggled.connect(self.spinbox.setEnabled)
    
    def value(self):
        if not self.checkbox.isChecked():
            return
        return self.spinbox.value()

class FeatureRangeSelector:
    def __init__(self) -> None:
        self.lowRangeWidgets = CheckableSpinBoxWidgets()
        self.highRangeWidgets = CheckableSpinBoxWidgets()        
        
        self.selectButton = FeatureSelectorButton('Click to select feature...')
        self.selectButton.setSizeLongestText(
            'Spotfit intens. metric, Foregr. integral gauss. peak'
        )
        self.selectButton.clicked.connect(self.selectFeature)
        self.selectButton.setCursor(Qt.PointingHandCursor)

        self.widgets = [
            {'pos': (0, 0), 'widget': self.lowRangeWidgets.checkbox}, 
            {'pos': (1, 0), 'widget': self.lowRangeWidgets.spinbox}, 
            {'pos': (1, 1), 'widget': lessThanPushButton(flat=True)},
            {'pos': (1, 2), 'widget': self.selectButton},
            {'pos': (1, 3), 'widget': lessThanPushButton(flat=True)},
            {'pos': (0, 4), 'widget': self.highRangeWidgets.checkbox},
            {'pos': (1, 4), 'widget': self.highRangeWidgets.spinbox}, 
            {'pos': (2, 0), 'widget': VerticalSpacerEmptyWidget(height=10)}
        ]
    
    def setText(self, text):
        self.selectButton.setText(text)
    
    def getFeatureGroup(self):
        if self.selectButton.text().find('Click') != -1:
            return ''

        text = self.selectButton.text()
        topLevelText, childText = text.split(', ')
        return {topLevelText: childText}
    
    def selectFeature(self):
        self.selectFeatureDialog = FeatureSelectorDialog(
            parent=self.selectButton, multiSelection=False, 
            expandOnDoubleClick=True, isTopLevelSelectable=False, 
            infoTxt='Select feature', allItemsExpanded=False,
            title='Select feature'
        )
        self.selectFeatureDialog.setCurrentItem(self.getFeatureGroup())
        # self.selectFeatureDialog.resizeVertical()
        self.selectFeatureDialog.sigClose.connect(self.setFeatureText)
        self.selectFeatureDialog.show()
    
    def setFeatureText(self):
        if self.selectFeatureDialog.cancel:
            return
        self.selectButton.setFlat(True)
        selection = self.selectFeatureDialog.selectedItems()
        group_name = list(selection.keys())[0]
        feature_name = selection[group_name][0]
        featureText = f'{group_name}, {feature_name}'
        self.selectButton.setFeatureText(featureText)
        column_name = features.feature_names_to_col_names_mapper()[featureText]
        lowValue = self.lowRangeWidgets.value()
        highValue = self.highRangeWidgets.value()
        self.selectButton.setToolTip(f'{column_name}')

class GopFeaturesAndThresholdsGroupbox(QGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setTitle('Features and thresholds for filtering true spots')
        # self.setCheckable(True)

        self._layout = QGridLayout()
        self._layout.setVerticalSpacing(0)

        firstSelector = FeatureRangeSelector()
        self.addButton = acdc_widgets.addPushButton('  Add feature    ')
        self.addButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        for col, widget in enumerate(firstSelector.widgets):
            row, col = widget['pos']
            self._layout.addWidget(widget['widget'], row, col)
        lastCol = self._layout.columnCount()
        self._layout.addWidget(self.addButton, 0, lastCol+1, 2, 1)
        self.lastCol = lastCol+1
        self.selectors = [firstSelector]

        self._layout.setColumnStretch(0, 1)
        self._layout.setColumnStretch(1, 0)
        self.setLayout(self._layout)

        self.setFont(font)

        self.addButton.clicked.connect(self.addFeatureField)

    def addFeatureField(self):
        row = self._layout.rowCount()
        selector = FeatureRangeSelector()
        delButton = acdc_widgets.delPushButton('Remove feature')
        delButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        delButton.selector = selector
        for col, widget in enumerate(selector.widgets):
            relRow, col = widget['pos']
            self._layout.addWidget(widget['widget'], relRow+row, col)
        self._layout.addWidget(delButton, row, self.lastCol, 2, 1)
        self.selectors.append(selector)
        delButton.clicked.connect(self.removeFeatureField)
    
    def removeFeatureField(self):
        delButton = self.sender()
        for widget in delButton.selector.widgets:
            self._layout.removeWidget(widget['widget'])
        self._layout.removeWidget(delButton)
        self.selectors.remove(delButton.selector)
    
    def setValue(self, value):
        pass
            

class _GopFeaturesAndThresholdsButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        super().setText(' Set features or view the selected ones... ')
        self.selectedFeaturesWindow = dialogs.GopFeaturesAndThresholdsDialog(
            parent=self
        )
        self.clicked.connect(self.setFeatures)
        self.col_to_feature_mapper = {
            value:key for key, value 
            in features.feature_names_to_col_names_mapper().items()
        }
        self.selectedFeaturesWindow.hide()
    
    def setParent(self, parent):
        super().setParent(parent)
        self.selectedFeaturesWindow.setParent(self)
    
    def setFeatures(self):
        self.selectedFeaturesWindow.exec_()
        if self.selectedFeaturesWindow.cancel:
            return
        
        tooltip = self.selectedFeaturesWindow.configIniParam()
        self.setToolTip(tooltip)
    
    def text(self):
        tooltip = self.toolTip()
        start_idx = len('Features and ranges set:\n')
        text = tooltip[start_idx:]
        return text.replace('  * ', '')
    
    def value(self):
        return self.text()
    
    def setText(self, text):
        text = text.lstrip('\n')
        if not text:
            super().setText(' Set features or view the selected ones... ')
            return
        paramsText = ''
        gop_thresholds = config.get_gop_thresholds(text)
        featuresGroupBox = self.selectedFeaturesWindow.setFeaturesGroupbox
        for i, (col_name, values) in enumerate(gop_thresholds.items()):
            low_val, high_val = values
            paramsText = f'{paramsText}  * {col_name}, {low_val}, {high_val}\n'
            if i > 0:
                featuresGroupBox.addFeatureField()
            selector = featuresGroupBox.selectors[i]
            if low_val is not None:
                selector.lowRangeWidgets.checkbox.setChecked(True)
                selector.lowRangeWidgets.spinbox.setValue(low_val)
            if high_val is not None:
                selector.highRangeWidgets.checkbox.setChecked(True)
                selector.highRangeWidgets.spinbox.setValue(high_val)
            feature_name = self.col_to_feature_mapper[col_name]
            selector.selectButton.setFlat(True)
            selector.selectButton.setFeatureText(feature_name)
            selector.selectButton.setToolTip(col_name)
        
        text = f'Features and ranges set:\n\n{paramsText}'
        self.setToolTip(text)

class _CenteredLineEdit(tooltipLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)

def _refChThresholdFuncWidget():
    widget = myQComboBox()
    items = config.skimageAutoThresholdMethods()
    widget.addItems(items)
    return widget

def _dfSpotsFileExtensionsWidget(parent=None):
    widget = myQComboBox(parent)
    items = ['.h5', '.csv']
    widget.addItems(items)
    return widget

def _spotThresholdFunc():
    widget = myQComboBox()
    items = config.skimageAutoThresholdMethods()
    widget.addItems(items)
    return widget

class _spotDetectionMethod(myQComboBox):
    def __init__(self, checkBox=None, parent=None):
        super().__init__(checkBox=checkBox, parent=parent)
        items = ['Detect local peaks', 'Label prediction mask']
        self.addItems(items)
    
    def currentText(self):
        text = super().currentText()
        if text == 'Detect local peaks':
            return 'peak_local_max'
        elif text == 'Label prediction mask':
            return 'label_prediction_mask'
    
    def setValue(self, value):
        if value == 'peak_local_max':
            self.setCurrentText('Detect local peaks')
            return True
        elif value == 'label_prediction_mask':
            self.setCurrentText('Label prediction mask')
            return True
        return False
    
    def setCurrentText(self, text: str) -> None:
        success = self.setValue(text)
        if success:
            return
        super().setCurrentText(text)
    
    def value(self):
        return self.currentText()

    def text(self):
        return self.text()

def _spotPredictionMethod():
    widget = myQComboBox()
    items = ['Thresholding', 'Neural network']
    widget.addItems(items)
    return widget

class SpotPredictionMethodWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.posData = None
        self.metadata_df = None
        self.nnetParams = None
        self.nnetModel = None
        
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        self.combobox = myQComboBox()
        items = ['Thresholding', 'Neural network']
        self.combobox.addItems(items)
        
        self.configButton = acdc_widgets.setPushButton()
        self.configButton.setDisabled(True)
        
        self.configButton.clicked.connect(self.promptConfigModel)
        self.combobox.currentTextChanged.connect(self.onTextChanged)
        
        layout.addWidget(self.combobox)
        layout.addWidget(self.configButton)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setContentsMargins(0, 0, 0, 0)
    
    def setCurrentText(self, text):
        self.setValue(text)
        
    def currentText(self):
        return self.value()
    
    def onTextChanged(self, text):
        self.configButton.setDisabled(text == 'Thresholding')
        if not self.configButton.isEnabled():
            return
        
        self.blinkFlag = False
        self.buttonBasePalette = self.configButton.palette()
        self.buttonBlinkPalette = self.configButton.palette()
        self.buttonBlinkPalette.setColor(QPalette.Button, QColor('#F38701'))
        self.blinkingTimer = QTimer(self)
        self.blinkingTimer.timeout.connect(self.blinkConfigButton)
        self.blinkingTimer.start(150)
        self.stopBlinkingTimer = QTimer(self)
        self.stopBlinkingTimer.timeout.connect(self.stopBlinkConfigButton)
        self.stopBlinkingTimer.start(2000)
    
    def stopBlinkConfigButton(self):
        self.blinkingTimer.stop()
        self.configButton.setPalette(self.buttonBasePalette)
    
    def blinkConfigButton(self):
        if self.blinkFlag:
            self.configButton.setPalette(self.buttonBlinkPalette)
        else:
            self.configButton.setPalette(self.buttonBasePalette)
        self.blinkFlag = not self.blinkFlag
        
    def value(self):
        return self.combobox.currentText()
    
    def setValue(self, value):
        return self.combobox.setCurrentText(str(value))
    
    def setDefaultParams(self, remove_hot_pixels, PhysicalSizeX, use_gpu):
        if self.nnetParams is None:
            self.nnetParams = { 'init': {}, 'segment': {}}
        self.nnetParams['init']['PhysicalSizeX'] = PhysicalSizeX
        self.nnetParams['init']['remove_hot_pixels'] = remove_hot_pixels
        self.nnetParams['init']['use_gpu'] = use_gpu
    
    def setDefaultPixelWidth(self, pixelWidth):
        if self.nnetParams is None:
            self.nnetParams = { 'init': {}, 'segment': {}}
        self.nnetParams['init']['PhysicalSizeX'] = pixelWidth
    
    def setDefaultRemoveHotPixels(self, remove_hot_pixels):
        if self.nnetParams is None:
            self.nnetParams = { 'init': {}, 'segment': {}}
        self.nnetParams['init']['remove_hot_pixels'] = remove_hot_pixels
    
    def setDefaultUseGpu(self, use_gpu):
        if self.nnetParams is None:
            self.nnetParams = { 'init': {}, 'segment': {}}
        self.nnetParams['init']['use_gpu'] = use_gpu
    
    def setPosData(self, posData):
        self.posData = posData
        self.metadata_df = posData.metadata_df
    
    def _importModel(self):
        try:
            paramsGroupBox = self.parent().parent()
            paramsGroupBox.logging_func('Importing neural network model...')
        except Exception as e:
            printl('Importing neural network model...')
        from .nnet import model
        return model
    
    def _promptConfigNeuralNet(self):
        model = self._importModel()
        init_params, segment_params = acdc_myutils.getModelArgSpec(model)
        url = model.url_help()
        win = acdc_apps.QDialogModelParams(
            init_params,
            segment_params,
            'spotMAX-UNet', 
            parent=self,
            url=url, 
            initLastParams=True, 
            posData=self.posData,
            df_metadata=self.metadata_df,
            force_postprocess_2D=False,
            is_tracker=True,
            model_module=model
        )
        if self.nnetParams is not None:
            win.setValuesFromParams(
                self.nnetParams['init'], self.nnetParams['segment']
            )
        win.exec_()
        if win.cancel:
            return
        
        self.nnetModel = model.Model(**win.init_kwargs)
        self.nnetParams = {
            'init': win.init_kwargs, 'segment': win.model_kwargs
        }
        self.configButton.confirmAction()
    
    def promptConfigModel(self):
        if self.value() == 'Neural network':
            self._promptConfigNeuralNet()
    
    def nnet_params_to_ini_sections(self):
        if self.nnetParams is None:
            return

        if self.value() != 'Neural network':
            return 
        
        init_model_params = {
            key:str(value) for key, value in self.nnetParams['init'].items()
        }
        segment_model_params = {
            key:str(value) for key, value in self.nnetParams['segment'].items()
        }
        return init_model_params, segment_model_params
    
    def nnet_params_from_ini_sections(self, ini_params):
        from spotmax.nnet.model import get_nnet_params_from_ini_params
        self.nnetParams = get_nnet_params_from_ini_params(
            ini_params, use_default_for_missing=True
        )

class _spotMinSizeLabels(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        font = config.font()
        layout = QVBoxLayout()
        self.umLabel = QLabel()
        self.umLabel.setFont(font)
        self.umLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.pixelLabel = QLabel()
        self.pixelLabel.setFont(font)
        self.pixelLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.umLabel)
        layout.addWidget(self.pixelLabel)
        self.setLayout(layout)

    def setText(self, text):
        self.umLabel.setText(text)
        self.pixelLabel.setText(text)
    
    def text(self):
        return ''
    
    def pixelValues(self):
        text = self.pixelLabel.text()
        all_floats_re = re.findall(float_regex(), text)
        return [float(val) for val in all_floats_re]

    def umValues(self):
        text = self.umLabel.text()
        all_floats_re = re.findall(float_regex(), text)
        return [float(val) for val in all_floats_re]

class formWidget(QWidget):
    sigApplyButtonClicked = Signal(object)
    sigComputeButtonClicked = Signal(object)
    sigBrowseButtonClicked = Signal(object)
    sigAutoButtonClicked = Signal(object)
    sigLinkClicked = Signal(str)
    sigEditClicked = Signal(object)

    def __init__(
            self, widget,
            anchor='',
            initialVal=None,
            stretchWidget=True,
            labelTextLeft='',
            labelTextRight='',
            font=None,
            addInfoButton=False,
            addApplyButton=False,
            addComputeButton=False,
            addBrowseButton=False,
            addAutoButton=False,
            addEditButton=False,
            addLabel=True,
            disableComputeButtons=False,
            key='',
            parent=None,
            valueSetter=None,
        ):
        super().__init__(parent)
        self.widget = widget
        self.anchor = anchor
        self.key = key
        self.addLabel = addLabel
        self.labelTextLeft = labelTextLeft
        self._isComputeButtonConnected = False

        widget.setParent(self)
        widget.parentFormWidget = self

        self.setValue(initialVal, valueSetter=valueSetter)
        
        self.items = []

        if font is None:
            font = QFont()
            font.setPixelSize(11)

        if addLabel:
            self.labelLeft = acdc_widgets.QClickableLabel(widget)
            self.labelLeft.setText(labelTextLeft)
            self.labelLeft.setFont(font)
            self.items.append(self.labelLeft)
        else:
            self.items.append(None)

        if not stretchWidget:
            widgetLayout = QHBoxLayout()
            widgetLayout.addStretch(1)
            widgetLayout.addWidget(widget)
            widgetLayout.addStretch(1)
            self.items.append(widgetLayout)
        else:
            self.items.append(widget)

        self.labelRight = acdc_widgets.QClickableLabel(widget)
        self.labelRight.setText(labelTextRight)
        self.labelRight.setFont(font)
        self.items.append(self.labelRight)

        if addInfoButton:
            infoButton = acdc_widgets.infoPushButton(self)
            infoButton.setCursor(Qt.WhatsThisCursor)
            if labelTextLeft:
                infoButton.setToolTip(
                    f'Info about "{labelTextLeft}" parameter'
                )
            elif labelTextRight:
                infoButton.setToolTip(
                    f'Info about "{labelTextRight}" parameter'
                )
            infoButton.clicked.connect(self.showInfo)
            self.items.append(infoButton)
        
        if addBrowseButton:
            browseButton = acdc_widgets.showInFileManagerButton(self)
            browseButton.setToolTip('Browse')
            browseButton.clicked.connect(self.browseButtonClicked)
            self.browseButton = browseButton
            self.items.append(browseButton)
        
        if addEditButton:
            editButton = acdc_widgets.editPushButton(self)
            editButton.setToolTip('Edit field')
            editButton.clicked.connect(self.editButtonClicked)
            self.editButton = editButton
            self.items.append(editButton)

        self.computeButtons = []
        
        if addApplyButton:
            applyButton = applyPushButton(self)
            applyButton.setCursor(Qt.PointingHandCursor)
            applyButton.setCheckable(True)
            applyButton.setToolTip('Apply this step and visualize results')
            applyButton.clicked.connect(self.applyButtonClicked)
            self.applyButton = applyButton
            self.items.append(applyButton)
            self.computeButtons.append(applyButton)

        if addAutoButton:
            autoButton = acdc_widgets.autoPushButton(self)
            autoButton.setCheckable(True)
            autoButton.setToolTip('Automatically infer this parameter')
            autoButton.clicked.connect(self.autoButtonClicked)
            self.autoButton = autoButton
            self.items.append(autoButton)
            self.computeButtons.append(autoButton)

        if addComputeButton:
            computeButton = computePushButton(self)
            computeButton.setToolTip('Compute this step and visualize results')
            computeButton.clicked.connect(self.computeButtonClicked)
            self.computeButton = computeButton
            self.items.append(computeButton)
            self.computeButtons.append(computeButton)

        if addLabel:
            self.labelLeft.clicked.connect(self.tryChecking)
        self.labelRight.clicked.connect(self.tryChecking)
    
    def setComputeButtonConnected(self, connected):
        self._isComputeButtonConnected = connected
    
    def text(self):
        return self.labelTextLeft
    
    def setValue(self, value, valueSetter=None):
        if value is None:
            return
        
        if valueSetter is not None:
            if isinstance(valueSetter, str):
                getattr(self.widget, valueSetter)(value)
            else:
                valueSetter(value)
            return 
        
        if isinstance(value, bool):
            self.widget.setChecked(value)
        elif isinstance(value, str):
            try:
                self.widget.setCurrentText(value)
            except AttributeError:
                self.widget.setText(value)
        elif isinstance(value, float) or isinstance(value, int):
            self.widget.setValue(value)

    def tryChecking(self, label):
        try:
            self.widget.setChecked(not self.widget.isChecked())
        except AttributeError as e:
            pass

    def browseButtonClicked(self):
        file_path = getOpenImageFileName(
            parent=self, mostRecentPath=acdc_myutils.getMostRecentPath()
        )
        if file_path == '':
            return

        self.widget.setText(file_path)
        self.sigBrowseButtonClicked.emit(self)
    
    def editButtonClicked(self):
        self.sigEditClicked.emit(self)

    def autoButtonClicked(self):
        self.sigAutoButtonClicked.emit(self)

    def applyButtonClicked(self):
        self.sigApplyButtonClicked.emit(self)

    def computeButtonClicked(self):
        if not self._isComputeButtonConnected:
            self.warnComputeButtonNotConnected()
        self.sigComputeButtonClicked.emit(self)
    
    def warnComputeButtonNotConnected(self):
        txt = html_func.paragraph("""
            Before computing any of the analysis steps you need to <b>load some 
            image data</b>.<br><br>
            To do so, click on the <code>Open folder</code> button on the left of 
            the top toolbar (Ctrl+O) and choose an experiment folder to load. 
        """)
        msg = acdc_widgets.myMessageBox()
        msg.warning(self, 'Data not loaded', txt)

    def linkActivatedCallBack(self, link):
        if utils.is_valid_url(link):
            webbrowser.open(link)
        else:
            self.sigLinkClicked.emit(link)

    def showInfo(self):
        anchor = self.anchor
        txt = html_func.paragraph(
            _docs.paramsInfoText().get(anchor, _docs.notDocumentedYetText())
        )
        if not txt:
            return
        msg = acdc_widgets.myMessageBox(parent=self, showCentered=False)
        msg.setIcon(iconName='SP_MessageBoxInformation')
        msg.setWindowTitle(f'{self.labelLeft.text()} info')
        msg.addText(txt)
        msg.setWidth(600)
        msg.addButton('  Ok  ')
        for label in msg.labels:
            label.setOpenExternalLinks(False)
            label.linkActivated.connect(self.linkActivatedCallBack)
        msg.exec_()
        # Here show user manual already scrolled at anchor
        # see https://stackoverflow.com/questions/20678610/qtextedit-set-anchor-and-scroll-to-it

class FormLayout(QGridLayout):
    def __init__(self):
        QGridLayout.__init__(self)

    def addFormWidget(self, formWidget, row=0):
        for col, item in enumerate(formWidget.items):
            if item is None:
                continue
            
            if col == 1 and not formWidget.addLabel:
                col = 0
                colspan = 2
            else:
                colspan = 1
            
            if col==0:
                alignment = Qt.AlignRight
            elif col==2:
                alignment = Qt.AlignLeft
            else:
                alignment = None
            try:
                if alignment is None:
                    self.addWidget(item, row, col, 1, colspan)
                else:
                    self.addWidget(item, row, col, 1, colspan, alignment=alignment)
            except TypeError:
                self.addLayout(item, row, col, 1, colspan)
        formWidget.row = row

class ReadOnlyElidingLineEdit(acdc_widgets.ElidingLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

class myQScrollBar(QScrollBar):
    sigActionTriggered = Signal(int)

    def __init__(self, *args, checkBox=None, label=None):
        QScrollBar.__init__(self, *args)
        # checkBox that controls if ComboBox can be enabled or not
        self.checkBox = checkBox
        self.label = label
        self.actionTriggered.connect(self.onActionTriggered)

    def onActionTriggered(self, action):
        # Disable SliderPageStepAdd and SliderPageStepSub
        if action == self.SliderPageStepAdd:
            self.setSliderPosition(self.value())
        elif action == self.SliderPageStepSub:
            self.setSliderPosition(self.value())
        else:
            self.sigActionTriggered.emit(action)

    def setEnabled(self, enabled, applyToCheckbox=True):
        enforceDisabled = False
        if self.checkBox is None or self.checkBox.isChecked():
            QScrollBar.setEnabled(self, enabled)
        else:
            QScrollBar.setEnabled(self, False)
            enforceDisabled = True

        if applyToCheckbox and self.checkBox is not None:
            self.checkBox.setEnabled(enabled)

        if enforceDisabled:
            self.label.setStyleSheet('color: gray')
        elif enabled and self.label is not None:
            self.label.setStyleSheet('color: black')
        elif self.label is not None:
            self.label.setStyleSheet('color: gray')

    def setDisabled(self, disabled, applyToCheckbox=True):
        enforceDisabled = False
        if self.checkBox is None or self.checkBox.isChecked():
            QScrollBar.setDisabled(self, disabled)
        else:
            QScrollBar.setDisabled(self, True)
            enforceDisabled = True

        if applyToCheckbox and self.checkBox is not None:
            self.checkBox.setDisabled(disabled)

        if enforceDisabled:
            self.label.setStyleSheet('color: gray')
        elif disabled and self.label is not None:
            self.label.setStyleSheet('color: gray')
        elif self.label is not None:
            self.label.setStyleSheet('color: black')

class ReadOnlyLineEdit(QLineEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setReadOnly(True)
        self.setAlignment(Qt.AlignCenter)
    
    def setValue(self, value):
        super().setText(str(value))

class ReadOnlySpinBox(acdc_widgets.SpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)

class ReadOnlyDoubleSpinBox(acdc_widgets.DoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)

class FloatLineEdit(QLineEdit):
    valueChanged = Signal(float)

    def __init__(
            self, *args, notAllowed=None, allowNegative=True, initial=None,
        ):
        QLineEdit.__init__(self, *args)
        self.notAllowed = notAllowed

        self.isNumericRegExp = rf'^{float_regex(allow_negative=allowNegative)}$'

        regExp = QRegularExpression(self.isNumericRegExp)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)
        if initial is None:
            self.setText('0.0')

    def setValue(self, value: float):
        self.setText(str(value))

    def value(self):
        m = re.match(self.isNumericRegExp, self.text())
        if m is not None:
            text = m.group(0)
            try:
                val = float(text)
            except ValueError:
                val = 0.0
            return val
        else:
            return 0.0

    def emitValueChanged(self, text):
        val = self.value()
        if self.notAllowed is not None and val in self.notAllowed:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
        else:
            self.setStyleSheet('')
            self.valueChanged.emit(self.value())

class FloatLineEditWithStepButtons(QWidget):
    valueChanged = Signal(float)
    
    def __init__(self, parent=None, **kwargs) -> None:
        super().__init__(parent)
        
        self.setStep(kwargs.get('step', 0.1))
        
        layout = QHBoxLayout()
        
        self._lineEdit = FloatLineEdit(**kwargs)
        self._stepUpButton = acdc_widgets.addPushButton()
        self._stepDownButton = acdc_widgets.subtractPushButton()
        
        layout.addWidget(self._lineEdit)
        layout.addWidget(self._stepDownButton)
        layout.addWidget(self._stepUpButton)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)
        
        self.setLayout(layout)
        
        self._stepUpButton.clicked.connect(self.stepUp)
        self._stepDownButton.clicked.connect(self.stepDown)
        
        self._lineEdit.textChanged.connect(self.emitValueChanged)
    
    def emitValueChanged(self, text):
        val = self.value()
        notAllowed = self._lineEdit.notAllowed
        if notAllowed is not None and val in notAllowed:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
        else:
            self.setStyleSheet('')
            self.valueChanged.emit(self.value())
        # self._lineEdit.emitValueChanged(text)
    
    def stepUp(self):
        newValue = self.value() + self.step()
        self.setValue(round(newValue, self._decimals))
    
    def stepDown(self):
        newValue = self.value() - self.step()
        self.setValue(round(newValue, self._decimals))
    
    def setStep(self, step: float):
        self._step = step
        decimals_str = str(step).split('.')[1]
        self._decimals = len(decimals_str)
    
    def step(self):
        return self._step
    
    def setValue(self, value: float):
        self._lineEdit.setText(str(value))

    def value(self):
        return self._lineEdit.value()

class YXresolutMultiplierAutoTuneWidget(FloatLineEditWithStepButtons):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStep(0.5)
        self._stepUpButton.setShortcut('Up')
        self._stepDownButton.setShortcut('Down')

class VectorLineEdit(QLineEdit):
    valueChanged = Signal(object)
    
    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        
        float_re = float_regex()
        vector_regex = fr'\(?\[?{float_re}(,\s?{float_re})+\)?\]?'
        regex = fr'^{vector_regex}$|^{float_re}$'
        self.validRegex = regex
        
        regExp = QRegularExpression(regex)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)
        
        self.textChanged.connect(self.emitValueChanged)
        if initial is None:
            self.setText('0.0')
        
        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)
    
    def emitValueChanged(self, text):
        val = self.value()
        m = re.match(self.validRegex, self.text())
        if m is None:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
        else:
            self.setStyleSheet('')
            self.valueChanged.emit(self.value())
    
    def setValue(self, value):
        self.setText(value)
    
    def setText(self, text):
        super().setText(str(text))
    
    def value(self):
        m = re.match(self.validRegex, self.text())
        if m is None:
            return 0.0
        else:
            try: 
                value = float(self.text())
                return value
            except Exception as e:
                text = self.text()
                text = text.replace('(', '')
                text = text.replace(')', '')
                text = text.replace('[', '')
                text = text.replace(']', '')
                values = text.split(',')
                return [float(value) for value in values]

class Gaussian3SigmasLineEdit(VectorLineEdit):
    def __init__(self, parent=None, initial=None):
        super().__init__(parent=parent, initial=initial)
        
        float_re = float_regex()
        vector_regex = fr'\(?\[?{float_re},\s?{float_re},\s?{float_re}\)?\]?'
        regex = fr'^{vector_regex}$|^{float_re}$'
        self.validRegex = regex
        
        regExp = QRegularExpression(regex)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

def getOpenImageFileName(parent=None, mostRecentPath=''):
    file_path = QFileDialog.getOpenFileName(
        parent, 'Select image file', mostRecentPath,
        "Images/Videos (*.npy *.npz *.h5, *.png *.tif *.tiff *.jpg *.jpeg "
        "*.mov *.avi *.mp4)"
        ";;All Files (*)"
    )[0]
    return file_path

class DblClickQToolButton(QToolButton):
    sigDoubleClickEvent = Signal(object, object)
    sigClickEvent = Signal(object, object)

    def __init__(self, *args, **kwargs):
        QToolButton.__init__(self, *args, **kwargs)
        self.countClicks = 0

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        self.event = event
        self.countClicks += 1
        if self.countClicks == 1:
            QTimer.singleShot(250, self.checkDoubleClick)

    def checkDoubleClick(self):
        if self.countClicks == 2:
            self.countClicks = 0
            self.isDoubleClick = True
            # dblclick checks button only if checkable and originally unchecked
            if self.isCheckable() and not self.isChecked():
                self.setChecked(True)
            self.sigDoubleClickEvent.emit(self, self.event)
        else:
            self.countClicks = 0
            self.isDoubleClick = False
            if self.isCheckable():
                self.setChecked(not self.isChecked())
            self.sigClickEvent.emit(self, self.event)

class ImageItem(pg.ImageItem):
    sigHoverEvent = Signal(object, object)

    def __init__(self, *args, **kwargs):
        pg.ImageItem.__init__(self, *args, **kwargs)

    def hoverEvent(self, event):
        self.sigHoverEvent.emit(self, event)

class ScatterPlotItem(pg.ScatterPlotItem):
    sigClicked = Signal(object, object, object)

    def __init__(
            self, guiWin, side, what, clickedFunc,
            **kwargs
        ):

        self.guiWin = guiWin
        self.what = what
        self.side = side
        self.df_settings = guiWin.df_settings
        self.colorItems = guiWin.colorItems
        self.clickedFunc = clickedFunc
        self.sideToolbar = guiWin.sideToolbar

        self.createBrushesPens()

        pg.ScatterPlotItem.__init__(
            self, **kwargs
        )
        self.clickedSpot = (-1, -1)

    def createBrushesPens(self):
        what = self.what
        alpha = float(self.df_settings.at[f'{what}_opacity', 'value'])
        penWidth = float(self.df_settings.at[f'{what}_pen_width', 'value'])
        self.pens = {'left': {}, 'right': {}}
        self.brushes = {'left': {}, 'right': {}}
        for side, colors in self.colorItems.items():
            for key, color in colors.items():
                if key.lower().find(f'{self.what}') == -1:
                    continue
                penColor = color.copy()
                penColor[-1] = 255
                self.pens[side][key] = pg.mkPen(penColor, width=penWidth)
                brushColor = penColor.copy()
                brushColor[-1] = int(color[-1]*alpha)
                self.brushes[side][key] = (
                    pg.mkBrush(brushColor), pg.mkBrush(color)
                )

    def selectColor(self):
        """Callback of the actions from spotsClicked right-click QMenu"""
        side = self.side
        key = self.sender().text()
        viewToolbar = self.sideToolbar[side]['viewToolbar']
        currentQColor = self.clickedSpotItem.brush().color()

        # Trigger color button on the side toolbar which is connected to
        # gui_setColor
        colorButton = viewToolbar['colorButton']
        colorButton.side = side
        colorButton.key = key
        colorButton.scatterItem = self
        colorButton.setColor(currentQColor)
        colorButton.selectColor()

    def selectStyle(self):
        """Callback of the spotStyleAction from spotsClicked right-click QMenu"""
        side = self.sender().parent().side

        what = self.what
        alpha = float(self.df_settings.at[f'{what}_opacity', 'value'])
        penWidth = float(self.df_settings.at[f'{what}_pen_width', 'value'])
        size = int(self.df_settings.at[f'{what}_size', 'value'])

        opacityVal = int(alpha*100)
        penWidthVal = int(penWidth*2)

        self.origAlpha = alpha
        self.origWidth = penWidth
        self.origSize = size

        self.styleWin = dialogs.spotStyleDock(
            'Spots style', parent=self.guiWin
        )
        self.styleWin.side = side

        self.styleWin.transpSlider.setValue(opacityVal)
        self.styleWin.transpSlider.sigValueChange.connect(
            self.setOpacity
        )

        self.styleWin.penWidthSlider.setValue(penWidthVal)
        self.styleWin.penWidthSlider.sigValueChange.connect(
            self.setPenWidth
        )

        self.styleWin.sizeSlider.setValue(size)
        self.styleWin.sizeSlider.sigValueChange.connect(
            self.setSize
        )

        self.styleWin.sigCancel.connect(self.styleCanceled)

        self.styleWin.show()

    def styleCanceled(self):
        what = self.what

        self.df_settings.at[f'{what}_opacity', 'value'] = self.origAlpha
        self.df_settings.at[f'{what}_pen_width', 'value'] = self.origWidth
        self.df_settings.at[f'{what}_size', 'value'] = self.origSize

        self.createBrushesPens()
        self.clickedFunc(self.styleWin.side)

    def setOpacity(self, opacityVal):
        what = self.what

        alpha = opacityVal/100
        self.df_settings.at[f'{what}_opacity', 'value'] = alpha

        self.createBrushesPens()
        self.clickedFunc(self.styleWin.side)

    def setPenWidth(self, penWidth):
        what = self.what

        penWidthVal = penWidth/2
        self.df_settings.at[f'{what}_pen_width', 'value'] = penWidthVal

        self.createBrushesPens()
        self.clickedFunc(self.styleWin.side)

    def setScatterSize(self, size):
        what = self.what
        self.df_settings.at[f'{what}_size', 'value'] = size

        self.createBrushesPens()
        self.clickedFunc(self.styleWin.side)

    def mousePressEvent(self, ev):
        pts = self.pointsAt(ev.pos())
        if len(pts) > 0:
            self.ptsClicked = pts
            ev.accept()
            self.sigClicked.emit(self, self.ptsClicked, ev)
        else:
            ev.ignore()

    def mouseClickEvent(self, ev):
        pass

class colorToolButton(QToolButton):
    sigClicked = Signal()

    def __init__(self, parent=None, color=(0,255,255)):
        super().__init__(parent)
        self.setColor(color)

    def setColor(self, color):
        self.penColor = color
        self.brushColor = [0, 0, 0, 150]
        self.brushColor[:3] = color[:3]
        self.update()

    def mousePressEvent(self, event):
        self.sigClicked.emit()

    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)
        p = QPainter(self)
        w, h = self.width(), self.height()
        sf = 0.6
        p.scale(w*sf, h*sf)
        p.translate(0.5/sf, 0.5/sf)
        symbol = pg.graphicsItems.ScatterPlotItem.Symbols['s']
        pen = pg.mkPen(color=self.penColor, width=2)
        brush = pg.mkBrush(color=self.brushColor)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(pen)
            p.setBrush(brush)
            p.drawPath(symbol)
        except Exception as e:
            traceback.print_exc()
        finally:
            p.end()

class QLogConsole(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont()
        font.setPixelSize(12)
        self.setFont(font)

    def write(self, message):
        # Method required by tqdm pbar
        message = message.replace('\r ', '')
        if message:
            self.apppendText(message)

class mathTeXLabel(QWidget):
    def __init__(self, mathTeXtext, parent=None, font_size=15):
        super(QWidget, self).__init__(parent)

        l=QVBoxLayout(self)
        l.setContentsMargins(0,0,0,0)

        r,g,b,a = self.palette().color(self.backgroundRole()).getRgbF()

        self._figure=Figure(edgecolor=(r,g,b), facecolor=(r,g,b,a))
        self._canvas=FigureCanvasQTAgg(self._figure)
        l.addWidget(self._canvas)
        self._figure.clear()
        text=self._figure.suptitle(
            mathTeXtext,
            x=0.0,
            y=1.0,
            horizontalalignment='left',
            verticalalignment='top',
            size=15
        )
        self._canvas.draw()

        (x0,y0),(x1,y1)=text.get_window_extent().get_points()
        w=x1-x0; h=y1-y0

        self._figure.set_size_inches(w/80, h/80)
        self.setFixedSize(w,h)

class SpotsItemToolButton(acdc_widgets.PointsLayerToolButton):
    sigToggled = Signal(object, bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toggled.connect(self.emitToggled)
    
    def emitToggled(self, checked):
        self.sigToggled.emit(self, checked)

class SpotsItems:
    sigButtonToggled = Signal(object, bool)

    def __init__(self):
        self.buttons = []

    def addLayer(self, h5files):
        win = dialogs.SpotsItemPropertiesDialog(
            h5files, 
            spotmax_out_path=self.spotmax_out_path
        )
        win.exec_()
        if win.cancel:
            return

        toolbutton = self.addToolbarButton(win.state)
        toolbutton.h5files = h5files
        self.buttons.append(toolbutton)
        self.createSpotItem(win.state, toolbutton)
        self.loadSpotsTables(toolbutton)
        toolbutton.setChecked(True)
        return toolbutton

    def addToolbarButton(self, state):
        symbol = state['pg_symbol']
        color = state['symbolColor']
        toolbutton = SpotsItemToolButton(symbol, color=color)
        toolbutton.state = state
        toolbutton.setCheckable(True)
        toolbutton.sigToggled.connect(self.buttonToggled)
        toolbutton.sigEditAppearance.connect(self.editAppearance)
        toolbutton.filename = state['h5_filename']
        return toolbutton
    
    def buttonToggled(self, button, checked):
        button.item.setVisible(checked)
    
    def editAppearance(self, button):
        win = dialogs.SpotsItemPropertiesDialog(
            button.h5files, state=button.state, 
            spotmax_out_path=self.spotmax_out_path
        )
        win.exec_()
        if win.cancel:
            return
        
        button.state = win.state
        state = win.state
        symbol = state['pg_symbol']
        color = state['symbolColor']
        button.updateIcon(symbol, color)

        alpha = self.getAlpha(state)
        
        pen = self.getPen(state)
        brush = self.getBrush(state, alpha)
        hoverBrush = self.getBrush(state)
        symbol = state['pg_symbol']
        size = state['size']
        xx, yy = button.item.getData()
        button.item.setData(
            xx, yy, size=size, pen=pen, brush=brush, hoverBrush=hoverBrush, 
            symbol=symbol
        )
    
    def getHoveredPoints(self, frame_i, z, y, x):
        hoveredPoints = []
        item = None
        for toolbutton in self.buttons:
            if not toolbutton.isChecked():
                continue
            df = toolbutton.df
            if df is None:
                continue
            item = toolbutton.item
            hoveredMask = item._maskAt(QPointF(x, y))
            points = item.points()[hoveredMask][::-1]
            if frame_i != item.frame_i:
                continue
            if z != item.z:
                continue
            if len(points) == 0:
                continue
            hoveredPoints.extend(points)
            break
        return hoveredPoints, item
    
    def getHoveredPointData(self, frame_i, z, y, x):
        for toolbutton in self.buttons:
            if not toolbutton.isChecked():
                continue
            df = toolbutton.df
            if df is None:
                continue
            item = toolbutton.item
            hoveredMask = item._maskAt(QPointF(x, y))
            points = item.points()[hoveredMask][::-1]
            if frame_i != item.frame_i:
                continue
            if z != item.z:
                continue
            if len(points) == 0:
                continue
            point = points[0]
            pos = point.pos()
            x, y = int(pos.x()-0.5), int(pos.y()-0.5)
            df = df.loc[[(frame_i, z)]].reset_index().set_index(['x', 'y'])
            point_df = df.loc[[(x, y)]].reset_index()
            point_features = point_df.set_index(['frame_i', 'z', 'y', 'x']).iloc[0]
            return point_features
    
    def getBrush(self, state, alpha=255):
        r,g,b,a = state['symbolColor'].getRgb()
        brush = pg.mkBrush(color=(r,g,b,alpha))
        return brush
    
    def getPen(self, state):
        r,g,b,a = state['symbolColor'].getRgb()
        pen = pg.mkPen(width=2, color=(r,g,b))
        return pen

    def getAlpha(self, state):
        return round(state['opacity']*255)
    
    def createSpotItem(self, state, toolbutton):
        alpha = self.getAlpha(state)
        pen = self.getPen(state)
        brush = self.getBrush(state, alpha)
        hoverBrush = self.getBrush(state)
        symbol = state['pg_symbol']
        size = state['size']
        scatterItem = pg.ScatterPlotItem(
            [], [], symbol=symbol, pxMode=False, size=size,
            brush=brush, pen=pen, hoverable=True, hoverBrush=hoverBrush, 
            tip=None
        )
        scatterItem.frame_i = -1
        scatterItem.z = -1
        toolbutton.item = scatterItem
    
    def setPosition(self, spotmax_out_path):
        self.spotmax_out_path = spotmax_out_path
    
    def _loadSpotsTable(self, toolbutton):
        spotmax_out_path = self.spotmax_out_path
        filename = toolbutton.filename
        df = io.load_spots_table(spotmax_out_path, filename)
        if df is None:
            toolbutton.df = None
        else:
            toolbutton.df = df.reset_index().set_index(['frame_i', 'z'])
    
    def loadSpotsTables(self, toolbutton=None):
        if toolbutton is None:
            for toolbutton in self.buttons:
                self._loadSpotsTable(toolbutton)
        else:
            self._loadSpotsTable(toolbutton)
    
    def _setDataButton(self, toolbutton, frame_i, z=None):
        scatterItem = toolbutton.item
        if frame_i == scatterItem.frame_i and z == scatterItem.z:
            return
        if toolbutton.df is None:
            return
        
        data = toolbutton.df.loc[frame_i]
        if z is not None:
            try:
                data_z = data.loc[[z]]
                yy, xx = data_z['y'].values + 0.5, data_z['x'].values + 0.5
            except Exception as e:
                yy, xx = [], []
        else:
            data_z = data
            yy, xx = data_z['y'].values + 0.5, data_z['x'].values + 0.5
        
        scatterItem.setData(xx, yy)
        scatterItem.z = z
        scatterItem.frame_i = frame_i

    def setData(self, frame_i, toolbutton=None, z=None):
        if toolbutton is None:
            for toolbutton in self.buttons:
                self._setDataButton(toolbutton, frame_i, z=z)
        else:
            self._setDataButton(toolbutton, frame_i, z=z)

def ParamFormWidget(anchor, param, parent, use_tune_widget=False):
    if use_tune_widget:
        widgetName = param['autoTuneWidget']
    else:
        widgetName = param['formWidgetFunc']
    
    module_name, attr = widgetName.split('.')
    try:
        widgets_module = globals()[module_name]
        widgetFunc = getattr(widgets_module, attr)
    except KeyError as e:
        widgetFunc = globals()[attr]
    
    return formWidget(
        widgetFunc(),
        anchor=anchor,
        labelTextLeft=param.get('desc', ''),
        initialVal=param.get('initialVal', None),
        stretchWidget=param.get('stretchWidget', True),
        addInfoButton=param.get('addInfoButton', True),
        addComputeButton=param.get('addComputeButton', False),
        addApplyButton=param.get('addApplyButton', False),
        addBrowseButton=param.get('addBrowseButton', False),
        addAutoButton=param.get('addAutoButton', False),
        addEditButton=param.get('addEditButton', False),
        addLabel=param.get('addLabel', True),
        valueSetter=param.get('valueSetter'),
        disableComputeButtons=True,
        parent=parent
    )

class SelectFeatureAutoTuneButton(acdc_widgets.editPushButton):
    sigFeatureSelected = Signal(object, str, str)

    def __init__(self, featureGroupbox, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clicked.connect(self.selectFeature)
        self.featureGroupbox = featureGroupbox
    
    def getFeatureGroup(self):
        if self.featureGroupbox.title().find('Click') != -1:
            return ''

        title = self.featureGroupbox.title()
        topLevelText, childText = title.split(', ')
        return {topLevelText: childText}

    def clearSelectedFeature(self):
        self.featureGroupbox.clear()
    
    def selectFeature(self):
        self.selectFeatureDialog = FeatureSelectorDialog(
            parent=self, multiSelection=False, 
            expandOnDoubleClick=True, isTopLevelSelectable=False, 
            infoTxt='Select feature to tune', allItemsExpanded=False,
            title='Select feature'
        )
        self.selectFeatureDialog.setCurrentItem(self.getFeatureGroup())
        # self.selectFeatureDialog.resizeVertical()
        self.selectFeatureDialog.sigClose.connect(self.setFeatureText)
        self.selectFeatureDialog.show()
    
    def setFeatureText(self):
        if self.selectFeatureDialog.cancel:
            return
        
        selection = self.selectFeatureDialog.selectedItems()
        group_name = list(selection.keys())[0]
        feature_name = selection[group_name][0]
        featureText = f'{group_name}, {feature_name}'

        column_name = features.feature_names_to_col_names_mapper()[featureText]
        self.featureGroupbox.setTitle(featureText)
        self.featureGroupbox.column_name = column_name
        self.sigFeatureSelected.emit(self, featureText, column_name)

class ReadOnlySelectedFeatureLabel(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        txt = ' Click on edit button to select feature. '
        txt = html_func.span(f'<i>{txt}</i>', font_color='rgb(100,100,100)')
        self.setText(txt)
        # self.setFrameShape(QFrame.Shape.StyledPanel)
        # self.setFrameShadow(QFrame.Shadow.Plain)
    
    def setText(self, text):
        super().setText(text)

class SelectedFeatureAutoTuneGroupbox(QGroupBox):    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._txt = ' Click on edit button to select feature to tune. '
        
        layout = QFormLayout()
        
        self.minLineEdit = QLineEdit()
        self.minLineEdit.setAlignment(Qt.AlignCenter)
        self.minLineEdit.setReadOnly(True)
        layout.addRow('Minimum: ', self.minLineEdit)
        
        self.maxLineEdit = QLineEdit()
        self.maxLineEdit.setAlignment(Qt.AlignCenter)
        self.maxLineEdit.setReadOnly(True)
        layout.addRow('Maximum: ', self.maxLineEdit)
        
        self.setLayout(layout)
        
        self.setFont(font)
        self.clear()
        
    def clear(self):
        self.minLineEdit.setDisabled(True)
        self.layout().labelForField(self.minLineEdit).setDisabled(True)
        self.maxLineEdit.setDisabled(True)
        self.layout().labelForField(self.maxLineEdit).setDisabled(True)
        super().setTitle(self._txt)
    
    def setTitle(self, title):
        self.minLineEdit.setDisabled(False)
        self.layout().labelForField(self.minLineEdit).setDisabled(False)
        self.maxLineEdit.setDisabled(False)
        self.layout().labelForField(self.maxLineEdit).setDisabled(False)
        super().setTitle(title)
    
    def range(self):
        minimum = self.minLineEdit.text()
        if not minimum or minimum == 'None':
            minimum = None
        else:
            minimum = float(minimum)
        maximum = self.maxLineEdit.text()
        if not maximum or maximum == 'None':
            maximum = None
        else:
            minimum = float(minimum)
        return minimum, maximum

    def setRange(self, minimum, maximum):
        self.minLineEdit.setText(str(minimum))
        self.maxLineEdit.setText(str(maximum))
        
class SelectFeaturesAutoTune(QWidget):
    sigFeatureSelected = Signal(object, str, str)
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QGridLayout()
        self.featureGroupboxes = {}
        
        featureGroupbox = SelectedFeatureAutoTuneGroupbox()
        layout.addWidget(featureGroupbox, 0, 0)
        
        self.featureGroupboxes[0] = featureGroupbox
        
        buttonsLayout = QVBoxLayout()
        selectFeatureButton = SelectFeatureAutoTuneButton(featureGroupbox)    
        addFeatureButton = acdc_widgets.addPushButton()   
        clearPushButton = acdc_widgets.delPushButton()
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(selectFeatureButton)
        buttonsLayout.addWidget(addFeatureButton)
        buttonsLayout.addWidget(clearPushButton)
        buttonsLayout.addStretch(1)
        
        layout.addLayout(buttonsLayout, 0, 1)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)

        selectFeatureButton.sigFeatureSelected.connect(self.emitFeatureSelected)
        addFeatureButton.clicked.connect(self.addFeatureField)
        clearPushButton.clicked.connect(self.clearTopFeatureField)

        self.setLayout(layout)
        self._layout = layout
    
    def emitFeatureSelected(self, button, featureText, colName):
        self.sigFeatureSelected.emit(button, featureText, colName)
    
    def clearTopFeatureField(self):
        self.featureGroupboxes[0].clear()
    
    def addFeatureField(self):
        parentFormWidget = self.parentFormWidget
        parentFormLayout = self.parent().layout()

        layout = self.layout()
        row = layout.rowCount()
        
        featureGroupbox = SelectedFeatureAutoTuneGroupbox()
        
        buttonsLayout = QVBoxLayout()
        selectFeatureButton = SelectFeatureAutoTuneButton(featureGroupbox)
        delButton = acdc_widgets.delPushButton()
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(selectFeatureButton)
        buttonsLayout.addWidget(delButton)
        
        layout.addWidget(featureGroupbox, row, 0)
        layout.addLayout(buttonsLayout, row, 1)

        delButton._widgets = (featureGroupbox, selectFeatureButton)
        delButton._buttonsLayout = buttonsLayout
        delButton._row = row
        delButton.clicked.connect(self.removeFeatureField)

        self.featureGroupboxes[row] = featureGroupbox
    
    def removeFeatureField(self):
        delButton = self.sender()
        row = delButton._row
        for widget in delButton._widgets:
            widget.hide()
            self._layout.removeWidget(widget)
        delButton.hide()
        self._layout.removeItem(delButton._buttonsLayout)
        self._layout.removeWidget(delButton)
        del self.featureGroupboxes[row]

    
class SpinBox(acdc_widgets.SpinBox):
    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent, disableKeyPress=disableKeyPress)
    
    def setValue(self, value):
        if isinstance(value, str):
            value = int(value)
        super().setValue(value)
    
    def setText(self, text):
        value = int(text)
        super().setValue(value)

class RunNumberSpinbox(SpinBox):
    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent, disableKeyPress=disableKeyPress)
        self.installEventFilter(self)
        self.setMinimum(1)
    
    def eventFilter(self, object, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            return True
        return False
        