import os

import sys
import time
import re
import traceback
from natsort import natsorted
import webbrowser

from functools import partial

import math
import numpy as np
import pandas as pd

import skimage.draw
import skimage.measure

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from qtpy.QtCore import (
    Signal, QTimer, Qt, QRegularExpression, QEvent, QPropertyAnimation,
    QPointF, QUrl
)
from qtpy.QtGui import (
    QFont,  QPainter, QRegularExpressionValidator, QIcon, QColor, QPalette,
    QDesktopServices
)
from qtpy.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QFormLayout, QWidget, QVBoxLayout, QMainWindow, QStyleFactory,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QDockWidget,
    QScrollArea, QSizePolicy, QComboBox, QPushButton, QScrollBar,
    QGroupBox, QAbstractSlider, QDialog, QStyle, QSpacerItem,
    QAction, QWidgetAction, QMenu, QActionGroup, QFileDialog, QFrame,
    QListWidget, QApplication, QDoubleSpinBox
)

import pyqtgraph as pg

from cellacdc import apps as acdc_apps
from cellacdc import widgets as acdc_widgets
from cellacdc import load as acdc_load
from cellacdc._palettes import lineedit_invalid_entry_stylesheet
from cellacdc import myutils as acdc_myutils

try:
    from cellacdc.regex import float_regex
except Exception as err:
    from cellacdc.acdc_regex import float_regex

from . import is_mac, is_win, printl, font, font_small
from . import dialogs, config, html_func, docs
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

class showPushButton(acdc_widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':magnGlass.svg'))

class TunePushButton(acdc_widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':tune.svg'))

class applyPushButton(acdc_widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':magnGlass.svg'))

class computePushButton(acdc_widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':compute.svg'))

class AcdcLogoPushButton(acdc_widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':logo.svg'))

class LoadFromFilePushButton(acdc_widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':load_annotation.svg'))

class lessThanPushButton(acdc_widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':less_than.svg'))
        flat = kwargs.get('flat')
        if flat is not None:
            self.setFlat(True)

class RunSpotMaxButton(acdc_widgets.PushButton):
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

class TrainSpotmaxAIButton(acdc_widgets.TrainPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setToolTip(
            'Setup workflow to train SpotMAX AI from ground-truth annotations'
        )
        self.clicked.connect(self.onClicked)
    
    def onClicked(self):
        dialogs.setupSpotmaxAiTraining()

class IsFieldSetButton(acdc_widgets.PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flat = kwargs.get('flat', True)
        self.setFlat(flat)
        self.setSelected(False)
        self.setToolTip('This value/parameter is not set yet')
    
    def setSelected(self, selected):
        if selected:
            self.setIcon(QIcon(':greenTick.svg'))
            self.setToolTip('This value/parameter is set')
        else:
            self.setIcon(QIcon(':orange_question_mark.svg'))
            self.setToolTip('This value/parameter is not set yet')

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
        self._initText = text
    
    def reset(self):
        self._isFeatureSet = False
        self.setFlat(False)
        self.setText(self._initText)
        self.setToolTip('')
    
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
    sigValueSelected = Signal(str)

    def __init__(self, category='spots', title='Feature selector', **kwargs) -> None:
        super().__init__(title=title, **kwargs)

        features_groups = features.get_features_groups(category=category)
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
    def __init__(self, category='spots', withParenthesis=False) -> None:
        self.category = category
        
        self.lowRangeWidgets = CheckableSpinBoxWidgets()
        self.highRangeWidgets = CheckableSpinBoxWidgets()        
        
        features_groups = features.get_features_groups(category=category)
        texts = [
            f'{group}, {name}' for group, names in features_groups.items() 
            for name in names 
        ]
        lengths = [len(text) for text in texts]
        max_len_idx = lengths.index(max(lengths))
        longestText = texts[max_len_idx]
        
        self.selectButton = FeatureSelectorButton('Click to select feature...')
        self.selectButton.setSizeLongestText(longestText)
        self.selectButton.clicked.connect(self.selectFeature)
        self.selectButton.setCursor(Qt.PointingHandCursor)
        
        startCol = 0 if not withParenthesis else 1

        self.widgets = [
            {'pos': (0, startCol), 'widget': self.lowRangeWidgets.checkbox}, 
            {'pos': (1, startCol), 'widget': self.lowRangeWidgets.spinbox}, 
            {'pos': (1, startCol+1), 'widget': lessThanPushButton(flat=True)},
            {'pos': (1, startCol+2), 'widget': self.selectButton},
            {'pos': (1, startCol+3), 'widget': lessThanPushButton(flat=True)},
            {'pos': (0, startCol+4), 'widget': self.highRangeWidgets.checkbox},
            {'pos': (1, startCol+4), 'widget': self.highRangeWidgets.spinbox}, 
            {'pos': (2, startCol), 'widget': VerticalSpacerEmptyWidget(height=10)}
        ]
        if withParenthesis:
            self.openParenthesisCombobox = QComboBox()
            self.openParenthesisCombobox.addItems(['', '('])
            widget = {
                'pos': (1, 0), 'widget': self.openParenthesisCombobox
            }
            self.widgets.insert(0, widget)
            
            self.closeParenthesisCombobox = QComboBox()
            self.closeParenthesisCombobox.addItems(['', ')'])
            widget = {
                'pos': (1, startCol+5), 'widget': self.closeParenthesisCombobox
            }
            self.widgets.append(widget)
        
        self.selectFeatureDialog = FeatureSelectorDialog(
            category=self.category,
            parent=self.selectButton, 
            multiSelection=False, 
            expandOnDoubleClick=True, 
            isTopLevelSelectable=False, 
            infoTxt='Select feature', 
            allItemsExpanded=False,
            title='Select feature', 
            allowNoSelection=False,
        )
    
    def setText(self, text):
        self.selectButton.setText(text)
    
    def getFeatureGroup(self):
        if self.selectButton.text().find('Click') != -1:
            return ''

        text = self.selectButton.text()
        topLevelText, childText = text.split(', ')
        return {topLevelText: childText}
    
    def selectFeature(self): 
        self.selectFeatureDialog.setCurrentItem(self.getFeatureGroup())
        # self.selectFeatureDialog.resizeVertical()
        self.selectFeatureDialog.sigClose.connect(self.setFeatureText)
        self.selectFeatureDialog.show()
    
    def reset(self):
        self.selectButton.reset()
        self.lowRangeWidgets.checkbox.setChecked(False)
        self.lowRangeWidgets.spinbox.setValue(0.0)
        self.highRangeWidgets.checkbox.setChecked(False)
        self.highRangeWidgets.spinbox.setValue(0.0)
        try:
            self.openParenthesisCombobox.setCurrentIndex(0)
            self.closeParenthesisCombobox.setCurrentIndex(0)
        except Exception as err:
            pass
    
    def setFeatureText(self):
        if self.selectFeatureDialog.cancel:
            return
        self.selectButton.setFlat(True)
        selection = self.selectFeatureDialog.selectedItems()
        group_name = list(selection.keys())[0]
        feature_name = selection[group_name][0]
        featureText = f'{group_name}, {feature_name}'
        self.selectButton.setFeatureText(featureText)
        col_names_mapper = features.feature_names_to_col_names_mapper(
            category=self.category
        )
        column_name = col_names_mapper[featureText]
        lowValue = self.lowRangeWidgets.value()
        highValue = self.highRangeWidgets.value()
        self.selectButton.setToolTip(f'{column_name}')
        self.selectFeatureDialog.sigValueSelected.emit(column_name)

class GopFeaturesAndThresholdsGroupbox(QGroupBox):
    sigValueChanged = Signal()
    
    def __init__(self, parent=None, category='spots') -> None:
        super().__init__(parent)

        self.setTitle(f'Features and thresholds for filtering true {category}')
        self._category = category

        self._layout = QGridLayout()
        self._layout.setVerticalSpacing(0)
        
        logicStatementCombobox = QComboBox()
        logicStatementCombobox.addItems(['AND', 'OR'])
        sp = logicStatementCombobox.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        logicStatementCombobox.setSizePolicy(sp)
        logicStatementCombobox.hide()
        
        self._layout.addWidget(logicStatementCombobox, 1, 0)

        firstSelector = FeatureRangeSelector(
            category=category, withParenthesis=True
        )
        firstSelector.logicStatementCombobox = logicStatementCombobox
        self.addButton = acdc_widgets.addPushButton('  Add feature    ')
        self.addButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        for col, widget in enumerate(firstSelector.widgets):
            row, col = widget['pos']
            self._layout.addWidget(widget['widget'], row, col+1)
        lastCol = self._layout.columnCount()
        self._layout.addWidget(self.addButton, 0, lastCol+1, 2, 1)
        self.lastCol = lastCol+1
        self.selectors = [firstSelector]
        self.delButtons = []

        self.setLayout(self._layout)

        self.setFont(font)

        self.addButton.clicked.connect(self.addFeatureField)
        self.connectSelector(firstSelector)
    
    def connectSelector(self, selector):
        selector.logicStatementCombobox.currentTextChanged.connect(
            self.emitValueChanged
        )
        selector.selectFeatureDialog.sigValueSelected.connect(
            self.emitValueChanged
        )
        selector.lowRangeWidgets.spinbox.valueChanged.connect(
            self.emitValueChanged
        )
        selector.lowRangeWidgets.checkbox.toggled.connect(
            self.emitValueChanged
        )
        selector.highRangeWidgets.spinbox.valueChanged.connect(
            self.emitValueChanged
        )
        selector.highRangeWidgets.checkbox.toggled.connect(
            self.emitValueChanged
        )
        selector.openParenthesisCombobox.currentTextChanged.connect(
            self.emitValueChanged
        )
        selector.closeParenthesisCombobox.currentTextChanged.connect(
            self.emitValueChanged
        )
    
    def emitValueChanged(self):
        self.sigValueChanged.emit()

    def addFeatureField(self):
        row = self._layout.rowCount()
        
        logicStatementCombobox = QComboBox()
        logicStatementCombobox.addItems(['AND', 'OR'])
        self._layout.addWidget(logicStatementCombobox, row+1, 0)
        
        selector = FeatureRangeSelector(
            category=self._category, withParenthesis=True
        )
        selector.logicStatementCombobox = logicStatementCombobox
        delButton = acdc_widgets.delPushButton('Remove feature')
        delButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        delButton.selector = selector
        for col, widget in enumerate(selector.widgets):
            relRow, col = widget['pos']
            self._layout.addWidget(widget['widget'], relRow+row, col+1)
        self._layout.addWidget(delButton, row, self.lastCol, 2, 1)
        self.selectors.append(selector)
        delButton.clicked.connect(self.removeFeatureField)
        self.delButtons.append(delButton)
        self.connectSelector(selector)
    
    def clearAll(self):
        self.selectors[0].reset()
        
        for delButton in self.delButtons:
            self.removeFeatureField(delButton=delButton, removeDelButton=False)
        
        self.delButtons = []
        self.emitValueChanged()
    
    def removeFeatureField(
            self, checked=True, delButton=None, removeDelButton=True
        ):
        if delButton is None:
            delButton = self.sender()
        for widget in delButton.selector.widgets:
            self._layout.removeWidget(widget['widget'])
        self._layout.removeWidget(delButton.selector.logicStatementCombobox)
        self._layout.removeWidget(delButton)
        self.selectors.remove(delButton.selector)
        if removeDelButton:
            self.delButtons.remove(delButton)
        self.emitValueChanged()
    
    def setValue(self, value):
        pass
            
class _GopFeaturesAndThresholdsButton(QPushButton):
    def __init__(self, parent=None, category='spots'):
        super().__init__(parent)
        super().setText(' Set features or view the selected ones... ')
        self.selectedFeaturesWindow = dialogs.GopFeaturesAndThresholdsDialog(
            parent=self, category=category
        )
        self.clicked.connect(self.setFeatures)
        col_names_mapper = features.feature_names_to_col_names_mapper(
            category=category
        )
        self.col_to_feature_mapper = {
            value:key for key, value 
            in col_names_mapper.items()
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
        # self.setStyleSheet("background-color : yellow")
    
    def text(self):
        tooltip = self.toolTip()
        start_idx = len('Features and ranges set:\n')
        text = tooltip[start_idx:]
        return text
    
    def value(self):
        return self.text()
    
    def setText(self, text):
        text = text.lstrip('\n')
        if not text:
            super().setText(' Set features or view the selected ones... ')
            return
        
        tooltip = f'Features and ranges set:\n\n{text}'
        self.setToolTip(tooltip)
        
        features_thresholds = config.get_features_thresholds_filter(text)
        featuresGroupBox = self.selectedFeaturesWindow.setFeaturesGroupbox
        for f, (col_name, thresholds) in enumerate(features_thresholds.items()):
            if f > 0:
                featuresGroupBox.addFeatureField()
            
            selector = featuresGroupBox.selectors[f]
            
            close_parenthesis = False            
            if col_name.startswith('| '):
                col_name = col_name[2:]
                selector.logicStatementCombobox.setCurrentText('OR')
            elif col_name.startswith('& '):
                col_name = col_name[2:]
                selector.logicStatementCombobox.setCurrentText('AND')
            elif f > 0:
                selector.logicStatementCombobox.setCurrentText('AND')
                
            if col_name.startswith('('):
                col_name = col_name[1:]
                selector.openParenthesisCombobox.setCurrentText('(')
            
            if col_name.endswith(')'):
                col_name = col_name[:-1]
                selector.closeParenthesisCombobox.setCurrentText(')')
            
            low_val, high_val = thresholds
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
        
        self.selectedFeaturesWindow.updateExpression()       
        

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
        return self.currentText()

class SpotPredictionMethodWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.posData = None
        self.metadata_df = None
        self.nnetParams = None
        self.nnetModel = None
        self.bioImageIOModel = None
        self.bioImageIOParams = None
        self.SpotiflowParams = None
        self.SpotiflowModel = None
        
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        self.combobox = myQComboBox()
        items = ['Thresholding', 'spotMAX AI', 'BioImage.IO model', 'Spotiflow']
        self.combobox.addItems(items)
        
        self.configButton = acdc_widgets.setPushButton()
        self.configButton.setDisabled(True)
        
        self.trainSmaxAiButton = TrainSpotmaxAIButton()
        self.trainSmaxAiButton.setDisabled(True)
        
        self.configButton.clicked.connect(self.promptConfigModel)
        self.combobox.currentTextChanged.connect(self.onTextChanged)

        self.configButton.setToolTip(
            'Set/view neural network model parameters'
        )
        
        layout.addWidget(self.combobox)
        layout.addWidget(self.configButton)
        layout.addWidget(self.trainSmaxAiButton)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setContentsMargins(0, 0, 0, 0)
    
    def setCurrentText(self, text):
        self.setValue(text)
        
    def currentText(self):
        return self.value()
    
    def onTextChanged(self, text):
        self.trainSmaxAiButton.setEnabled(text == 'spotMAX AI')
        
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
    
    def setExpectedYXSpotRadius(self, spot_yx_radius_pixel):
        if self.SpotiflowParams is None:
            self.SpotiflowParams = { 'init': {}, 'segment': {}}
        self.SpotiflowParams['segment']['expected_spot_radius'] = (
            math.ceil(spot_yx_radius_pixel)
        )
    
    def setDefaultRemoveHotPixels(self, remove_hot_pixels):
        if self.nnetParams is None:
            self.nnetParams = { 'init': {}, 'segment': {}}
        self.nnetParams['init']['remove_hot_pixels'] = remove_hot_pixels
    
    def setDefaultUseGpu(self, use_gpu):
        if self.nnetParams is None:
            self.nnetParams = { 'init': {}, 'segment': {}}
        self.nnetParams['init']['use_gpu'] = use_gpu
    
    def setDefaultGaussianSigma(self, sigma):
        if self.nnetParams is None:
            self.nnetParams = { 'init': {}, 'segment': {}}
        self.nnetParams['init']['gaussian_filter_sigma'] = sigma
    
    def setDefaultResolutionMultiplier(self, value):
        if self.nnetParams is None:
            self.nnetParams = { 'init': {}, 'segment': {}}
        self.nnetParams['init']['resolution_multiplier_yx'] = value
    
    def setPosData(self, posData):
        self.posData = posData
        self.metadata_df = posData.metadata_df
    
    def _importModel(self):
        try:
            paramsGroupBox = self.parent().parent()
            paramsGroupBox.logging_func('Importing neural network model...')
        except Exception as e:
            print('Importing neural network model...')
        from .nnet import model
        return model
    
    def log(self, txt):
        try:
            paramsGroupBox = self.parent().parent()
            paramsGroupBox.logging_func(txt)
        except Exception as e:
            print(txt)
    
    def _promptConfigNeuralNet(self):
        model = self._importModel()
        init_params, segment_params = acdc_myutils.getModelArgSpec(model)
        url = model.url_help()
        win = acdc_apps.QDialogModelParams(
            init_params,
            segment_params,
            'SpotMAX AI', 
            parent=self,
            url=url, 
            initLastParams=True, 
            posData=self.posData,
            df_metadata=self.metadata_df,
            force_postprocess_2D=False,
            addPostProcessParams=False,
            addPreProcessParams=False,
            model_module=model
        )
        if self.nnetParams is not None:
            win.setValuesFromParams(
                self.nnetParams['init'], self.nnetParams['segment']
            )
        win.exec_()
        if win.cancel:
            return
        
        
        self.log(
            'Initializing neural network model '
            '(GUI will be unresponsive, no panic)...'
        )
        
        self.nnetModel = model.Model(**win.init_kwargs)
        self.nnetParams = {
            'init': win.init_kwargs, 'segment': win.model_kwargs
        }
        self.configButton.confirmAction()
        self.log('SpotMAX AI initialized' )
    
    def _promptConfigBioImageIOModel(self):
        from spotmax.BioImageIO import model
        win = dialogs.QDialogBioimageIOModelParams(
            self.posData, self.metadata_df, parent=self
        )
        if self.bioImageIOParams is not None:
            win.setValuesFromParams(
                self.bioImageIOParams['init'], 
                self.bioImageIOParams['segment'],
                self.bioImageIOParams['kwargs'],
            )
        win.exec_()
        if win.cancel:
            return
        
        self.log(
            'Initializing BioImage.IO model '
            '(GUI will be unresponsive, no panic)...'
        )
        self.bioImageIOModel = model.Model(**win.init_kwargs)
        self.bioImageIOModel.set_kwargs(win.additionalKwargs)
        self.bioImageIOParams = {
            'init': win.init_kwargs, 
            'segment': win.model_kwargs,
            'kwargs': win.additionalKwargs
        }
        self.configButton.confirmAction()
        self.log('Model initialized' )
    
    def _promptConfigSpotiflowModel(self):
        from spotmax.Spotiflow import spotiflow_smax_model as model
        init_params, segment_params = acdc_myutils.getModelArgSpec(model)
        url = model.url_help()
        win = acdc_apps.QDialogModelParams(
            init_params,
            segment_params,
            'Spotiflow', 
            parent=self,
            url=url, 
            initLastParams=True, 
            posData=self.posData,
            df_metadata=self.metadata_df,
            force_postprocess_2D=False,
            addPostProcessParams=False,
            addPreProcessParams=False,
            model_module=model
        )
        if self.SpotiflowParams is not None:
            win.setValuesFromParams(
                self.SpotiflowParams['init'], self.SpotiflowParams['segment']
            )
        win.exec_()
        if win.cancel:
            return
        
        self.log(
            'Initializing Spotiflow '
            '(GUI will be unresponsive, no panic)...'
        )
        self.SpotiflowModel = model.Model(**win.init_kwargs)
        self.SpotiflowParams = {
            'init': win.init_kwargs, 'segment': win.model_kwargs
        }
        self.configButton.confirmAction()
        self.log('Spotiflow initialized' )
    
    def promptConfigModel(self):
        if self.value() == 'spotMAX AI':
            self._promptConfigNeuralNet()
        elif self.value() == 'BioImage.IO model':
            self._promptConfigBioImageIOModel()
        elif self.value() == 'Spotiflow':
            self._promptConfigSpotiflowModel()
    
    def nnet_params_to_ini_sections(self):
        if self.nnetParams is None:
            return

        if self.value() != 'spotMAX AI':
            return 
        
        init_model_params = {
            key:str(value) for key, value in self.nnetParams['init'].items()
        }
        segment_model_params = {
            key:str(value) for key, value in self.nnetParams['segment'].items()
        }
        return init_model_params, segment_model_params
    
    def bioimageio_model_params_to_ini_sections(self):
        if self.bioImageIOParams is None:
            return

        if self.value() != 'BioImage.IO model':
            return 
        
        init_model_params = {
            key:str(value) 
            for key, value in self.bioImageIOParams['init'].items()
        }
        segment_model_params = {
            key:str(value) 
            for key, value in self.bioImageIOParams['segment'].items()
        }
        kwargs_model_params = {
            key:str(value) 
            for key, value in self.bioImageIOParams['kwargs'].items()
        }
        return init_model_params, segment_model_params, kwargs_model_params

    def spotiflow_model_params_to_ini_sections(self):
        if self.SpotiflowParams is None:
            return

        if self.value() != 'Spotiflow':
            return 
        
        init_model_params = {
            key:str(value) 
            for key, value in self.SpotiflowParams['init'].items()
        }
        segment_model_params = {
            key:str(value) 
            for key, value in self.SpotiflowParams['segment'].items()
        }
        return init_model_params, segment_model_params
    
    def nnet_params_from_ini_sections(self, ini_params):
        from spotmax.nnet.model import get_model_params_from_ini_params
        nnetParams = get_model_params_from_ini_params(
            ini_params, use_default_for_missing=True
        )
        if nnetParams is None:
            return
        
        self.nnetParams = nnetParams
    
    def bioimageio_params_from_ini_sections(self, ini_params):
        from spotmax.BioImageIO.model import get_model_params_from_ini_params
        bioImageIOParams = get_model_params_from_ini_params(
            ini_params, use_default_for_missing=True
        )
        if bioImageIOParams is None:
            return
        
        self.bioImageIOParams = bioImageIOParams
        init_params = self.bioImageIOParams['init']
        segment_params = self.bioImageIOParams['segment']
        kwargs = self.bioImageIOParams['kwargs']
        return init_params, segment_params, kwargs
    
    def spotiflow_params_from_ini_sections(self, ini_params):
        from spotmax.Spotiflow.spotiflow_smax_model import (
            get_model_params_from_ini_params
        )
        SpotiflowParams = get_model_params_from_ini_params(
            ini_params, use_default_for_missing=True
        )
        if SpotiflowParams is None:
            return
        
        self.SpotiflowParams = SpotiflowParams

class SpotMinSizeLabels(QWidget):
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
        self._isZstack = False

    def setIsZstack(self, isZstack):
        self._isZstack = isZstack
    
    def setText(self, text):
        self.umLabel.setText(text)
        self.pixelLabel.setText(text)
    
    def text(self):
        roundPixels = [str(round(val, 2)) for val in self.pixelValues()]
        if not self._isZstack:
            roundPixels[0] = 'nan'
        textPixel = ', '.join(roundPixels)
        roundUm = [str(round(val, 3)) for val in self.umValues()]
        if not self._isZstack:
            roundUm[0] = 'nan'
        textUm = ', '.join(roundUm)
        indent = ' '*41
        text = f'({textPixel}) pixel\n{indent}({textUm}) micrometer'
        return text
    
    def pixelValues(self):
        text = self.pixelLabel.text()
        all_floats_re = re.findall(float_regex(), text)
        values = [float(val) for val in all_floats_re]
        if len(values) == 2:
            values.insert(0, 1)
        return values

    def umValues(self):
        text = self.umLabel.text()
        all_floats_re = re.findall(float_regex(), text)
        values = [float(val) for val in all_floats_re]
        if len(values) == 2:
            values.insert(0, 1)
        return values

class formWidget(QWidget):
    sigApplyButtonClicked = Signal(object)
    sigComputeButtonClicked = Signal(object)
    sigBrowseButtonClicked = Signal(object)
    sigAutoButtonClicked = Signal(object)
    sigWarningButtonClicked = Signal(object)
    sigLinkClicked = Signal(str)
    sigEditClicked = Signal(object)
    sigAddField = Signal(object)
    sigRemoveField = Signal(str, str, int)
    sigToggled = Signal(object)

    def __init__(
            self, widget,
            anchor='',
            initialVal=None,
            stretchWidget=True,
            labelTextLeft='',
            labelTextRight='',
            labelTextMiddle='',
            confvalText='',
            useEditableLabel=False,
            font=None,
            addInfoButton=False,
            addApplyButton=False,
            addComputeButton=False,
            addBrowseButton=False,
            addWarningButton=False,
            addAutoButton=False,
            addEditButton=False,
            addLabel=True,
            addAddFieldButton=False,
            stretchFactors=None,
            disableComputeButtons=False,
            isFolderBrowse=False,
            browseExtensions=None,
            key='',
            parent=None,
            valueSetter=None,
            infoHtmlText=''
        ):
        super().__init__(parent)
        self.widget = widget
        self.anchor = anchor
        self.section = config.get_section_from_anchor(anchor)
        self.key = key
        self.addLabel = addLabel
        self.labelTextLeft = labelTextLeft
        self.confvalText = confvalText
        self.labelTextMiddle = labelTextMiddle
        self.labelTextRight = labelTextRight
        self._isComputeButtonConnected = False
        self.infoHtmlText = infoHtmlText
        self.stretchFactors = stretchFactors
        self.useEditableLabel = useEditableLabel
        self.browseExtensions = browseExtensions

        if widget is not None:
            widget.setParent(self)
            widget.parentFormWidget = self

        try:
            widget.toggled.connect(self.emitToggled)
        except Exception as err:
            pass
        
        self.initialVal = initialVal
        self.valueSetter = valueSetter
        self.setValue(initialVal, valueSetter=valueSetter)
        
        self.items = []

        if font is None:
            font = QFont()
            font.setPixelSize(11)

        self.labelLeft = None
        if addLabel:
            if useEditableLabel:
                self.labelLeft = EditableLabel(labelTextLeft)
            else:
                self.labelLeft = acdc_widgets.QClickableLabel(widget)
                self.labelLeft.setText(labelTextLeft)
            self.labelLeft.setFont(font)
            self.items.append(self.labelLeft)
        else:
            self.items.append(None)

        self.labelMiddle = None
        if labelTextMiddle:
            self.labelMiddle = acdc_widgets.QClickableLabel(widget)
            self.labelMiddle.setText(labelTextMiddle)
            self.labelMiddle.setFont(font)
            self.items.append(self.labelMiddle)
        
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
        
        self.addFieldButton = None
        if addAddFieldButton:
            self.fieldIdx = 0
            addFieldButton = acdc_widgets.addPushButton()
            addFieldButton.setToolTip('Add new entry')
            addFieldButton.clicked.connect(self.addField)
            self.addFieldButton = addFieldButton
            self.items.append(addFieldButton)

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
            self.isFolderBrowse = isFolderBrowse
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

        if addWarningButton:
            warningButton = acdc_widgets.WarningButton(self)
            warningButton.setRetainSizeWhenHidden(True)
            warningButton.setToolTip('WARNING! Click for more details')
            warningButton.clicked.connect(self.warningButtonClicked)
            warningButton.hide()
            self.warningButton = warningButton
            self.items.append(warningButton)
        
        if addLabel:
            self.labelLeft.clicked.connect(self.tryChecking)
        self.labelRight.clicked.connect(self.tryChecking)
    
    def setDisabled(self, disabled: bool) -> None:
        for item in self.items:
            try:
                item.setDisabled(disabled)
            except Exception as err:
                pass
    
    def setToolTip(self, tooltip):
        self.labelLeft.setToolTip(tooltip)
        self.widget.setToolTip(tooltip)
    
    def emitToggled(self):
        self.sigToggled.emit(self)
    
    def addField(self):
        items = [None]*len(self.items)
        
        i = 0
        if self.labelLeft is not None:
            labelLeft = self.labelLeft.__class__(self.labelTextLeft)
            labelLeft.setFont(font)
            items[i] = labelLeft
            i += 1
        
        if self.labelMiddle is not None:
            label = self.labelMiddle.__class__(self.labelTextMiddle)
            label.setFont(font)
            items[i] = label
            i += 1
        
        widget = self.widget.__class__()
        self.setValue(
            self.initialVal, 
            valueSetter=self.valueSetter,
            widget=widget
        )
        items[i] = widget
        i += 1
        
        label = self.labelRight.__class__(self.labelTextRight)
        label.setFont(font)
        items[i] = label
        i += 1
        
        delButton = acdc_widgets.delPushButton()
        items[i] = delButton
        delButton.clicked.connect(self.removeField)
        
        # Replace None items with buttons spacers (hidden with retained size)
        missing_items_idxs = [i for i, item in enumerate(items) if item is None]
        for missing_idx in missing_items_idxs:
            button = acdc_widgets.infoPushButton(self)
            button.setRetainSizeWhenHidden(True)
            button.hide()
            items[missing_idx] = button
        
        delButton.items = items
        
        self.row += 1
        # adderFunc and row are defined in FormLayout.addFormWidget
        self.adderFunc(self, self.row, items=items)
        
        newFormWidget = formWidget(
            None,
            labelTextLeft=self.labelTextLeft,
            labelTextMiddle=self.labelTextMiddle,
            labelTextRight=self.labelTextRight,
        )
        newFormWidget.widget = widget
        newFormWidget.items = self.items
        newFormWidget.section = self.section
        newFormWidget.anchor = self.anchor
        newFormWidget.useEditableLabel = self.useEditableLabel
        newFormWidget.addFieldButton = self.addFieldButton
        newFormWidget.delFieldButton = delButton
        newFormWidget.labelLeft = labelLeft
        self.fieldIdx += 1
        newFormWidget.fieldIdx = self.fieldIdx
        delButton.newFormWidget = newFormWidget
        
        self.sigAddField.emit(newFormWidget)
    
    def removeField(self):
        delButton = self.sender()
        for item in delButton.items:
            item.hide()
            # _layout is defined in FormLayout.addFormWidget
            self._layout.removeWidget(item)
        self.row -= 1
        
        self.sigRemoveField.emit(self.section, self.anchor, self.fieldIdx)
        self.fieldIdx -= 1
        
        delButton.newFormWidget.hide()
        
        del delButton.newFormWidget
        del delButton.items
        del delButton
    
    def setComputeButtonConnected(self, connected):
        self._isComputeButtonConnected = connected
    
    def text(self):
        return self.labelTextLeft
    
    def setValue(self, value, valueSetter=None, widget=None):
        if value is None:
            return
        
        if widget is None:
            widget = self.widget
        
        if valueSetter is not None:
            if isinstance(valueSetter, str):
                getattr(widget, valueSetter)(value)
            else:
                valueSetter(value)
            return 
        
        if isinstance(value, bool):
            widget.setChecked(value)
        elif isinstance(value, str):
            try:
                widget.setCurrentText(value)
            except AttributeError:
                widget.setText(value)
        elif isinstance(value, float) or isinstance(value, int):
            widget.setValue(value)

    def tryChecking(self, label):
        try:
            self.widget.setChecked(not self.widget.isChecked())
        except AttributeError as e:
            pass

    def getQtOpenFileNameExtensions(self):
        if self.browseExtensions is None:
            return
        
        s = ''
        s_li = []
        for name, extensions in self.browseExtensions.items():
            _s = ''
            if isinstance(extensions, str):
                extensions = [extensions]
            for ext in extensions:
                _s = f'{_s}*{ext} '
            s_li.append(f'{name} {_s.strip()}')

        fileTypes = ';;'.join(s_li)
        fileTypes = f'{fileTypes};;All Files (*)'
        return fileTypes
    
    def browseButtonClicked(self):
        if self.isFolderBrowse:
            folderpath = QFileDialog.getExistingDirectory(
                self, 'Select folder', acdc_myutils.getMostRecentPath()
            )
            if not folderpath:
                return
            self.widget.setText(folderpath)
        else:
            fileTypes = self.getQtOpenFileNameExtensions()
            file_path = getOpenImageFileName(
                parent=self, 
                mostRecentPath=acdc_myutils.getMostRecentPath(), 
                fileTypes=fileTypes
            )
            if not file_path:
                return

            if self.labelLeft.text().endswith('end name'):
                value = acdc_load.get_endname_from_filepath(file_path)
            else:
                value = file_path
            
            self.widget.setText(value)
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
    
    def warningButtonClicked(self):
        self.sigWarningButtonClicked.emit(self)
    
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

    def _try_open_url(self, url):
        try:
            webbrowser.open(url)
            return True
        except Exception as err:
            return False
    
    def showInfo(self):
        if self.confvalText:
            url_key = self.confvalText
        else:
            url_key = self.text()
        url = docs.param_name_to_url(url_key)
        
        msg = acdc_widgets.myMessageBox(wrapText=False)
        txt = f'{html_func.paragraph(self.infoHtmlText)}<br>'
        buttons = (
            acdc_widgets.OpenUrlButton(url, 'Browse documentation...'), 
            'Ok'
        )
        msg.information(
            self, f'`{self.text()}` description', txt, 
            buttonsTexts=buttons
        )

class FormLayout(QGridLayout):
    def __init__(self):
        QGridLayout.__init__(self)

    def _addItems(self, formWidget, row, items=None):
        if items is None:
            items = formWidget.items
        
        for col, item in enumerate(items):
            if item is None:
                continue
            
            if col == 1 and not formWidget.addLabel:
                col = 0
                colspan = 2
            else:
                colspan = 1
            
            if col==0:
                alignment = Qt.AlignRight
            elif col==len(formWidget.items)-1:
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
    
    def _addItemsAsLayout(self, formWidget, row, items=None):
        _layout = QHBoxLayout()
        if items is None:
            items = formWidget.items
        colspan = len(items)
        for col, item in enumerate(items):
            if item is None:
                continue
            _layout.addWidget(item)
            try:
                _layout.setStretch(col, formWidget.stretchFactors[col])
            except IndexError:
                _layout.setStretch(col, 0)
        
        self.addLayout(_layout, row, 0, 1, colspan)
    
    def addFormWidget(self, formWidget, row=0):
        if formWidget.stretchFactors is not None:
            self._addItemsAsLayout(formWidget, row)
            formWidget.adderFunc = self._addItemsAsLayout
        else:
            self._addItems(formWidget, row)
            formWidget.adderFunc = self._addItems
            
        formWidget.row = row
        formWidget._layout = self
        

class ReadOnlyElidingLineEdit(acdc_widgets.ElidingLineEdit):
    def __init__(self, parent=None, transparent=False):
        super().__init__(parent)
        self.setReadOnly(True)
        if transparent:
            self.setFrame(False)
            palette = self.palette()
            palette.setColor(QPalette.Base, Qt.transparent)
            self.setPalette(palette)

    def setInvalidEntry(self, invalid):
        if invalid:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
        else:
            self.setStyleSheet('')
    
    def isTextElided(self):
        return QLineEdit.text(self).startswith(b'\xe2\x80\xa6'.decode())
    
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

        self.setAlignment(Qt.AlignCenter)
        
        pattern = rf'^{float_regex(allow_negative=allowNegative)}$'
        self.setRegexValidator(pattern)

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)
        if initial is None:
            self.setText('0.0')
    
    def setRegexValidator(self, pattern):
        self.isNumericRegExp = pattern
        regExp = QRegularExpression(self.isNumericRegExp)
        self.setValidator(QRegularExpressionValidator(regExp))

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

class ParentLinePlotItem(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        self._childrenItem = None
        super().__init__(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs
    
    def addChildrenItem(self):
        self._childrenItem = pg.PlotDataItem(*self._args, **self._kwargs)
        return self._childrenItem
    
    def setData(self, *args, **kwargs):
        super().setData(*args, **kwargs)
        if self._childrenItem is None:
            return
        self._childrenItem.setData(*args, **kwargs)
    
    def clearData(self):
        self.setData([], [])

class FloatLineEditWithStepButtons(QWidget):
    valueChanged = Signal(float)
    sigActivated = Signal(bool)
    
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
        
        self.setContentsMargins(0, 0, 0, 0)
        
        self.setLayout(layout)
        
        self._stepUpButton.clicked.connect(self.stepUp)
        self._stepDownButton.clicked.connect(self.stepDown)
        
        self._lineEdit.textChanged.connect(self.emitValueChanged)
    
    def addActivateCheckbox(self):
        self.activateCheckbox = QCheckBox('Activate')
        self.layout().addWidget(self.activateCheckbox)
        self.activateCheckbox.clicked.connect(self.activateCheckboxToggled)
    
    def setDisabled(self, disabled):
        self._lineEdit.setDisabled(disabled)
        self._stepDownButton.setDisabled(disabled)
        self._stepUpButton.setDisabled(disabled)
    
    def activateCheckboxToggled(self):
        checked = self.activateCheckbox.isChecked()
        self.setDisabled(not checked)
        self.sigActivated.emit(checked)
    
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

class ResolutMultiplierAutoTuneWidget(FloatLineEditWithStepButtons):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStep(0.5)
        self.addActivateCheckbox()
        self.sigActivated.connect(self.onSigActivated)
    
    def setDisabled(self, disabled):
        super().setDisabled(disabled)
        if not disabled:
            self._stepUpButton.setShortcut('Up')
            self._stepDownButton.setShortcut('Down')
        else:
            self._stepUpButton.setShortcut('')
            self._stepDownButton.setShortcut('')
    
    def onSigActivated(self, checked):
        self.setDisabled(not checked)
        self._stepUpButton.clearFocus()
        self._stepDownButton.clearFocus()
        self.activateCheckbox.clearFocus()

class ResolutMultiplierAutoTuneWidget(FloatLineEditWithStepButtons):        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStep(0.5)
        self.addActivateCheckbox()
        self.sigActivated.connect(self.onSigActivated)
        
    def onSigActivated(self, checked):
        if checked:
            self._stepUpButton.setShortcut('Up')
            self._stepDownButton.setShortcut('Down')
        else:
            self._stepUpButton.setShortcut('')
            self._stepDownButton.setShortcut('')

class Gaussian3SigmasLineEdit(acdc_widgets.VectorLineEdit):
    def __init__(self, parent=None, initial=None):
        super().__init__(parent=parent, initial=initial)
        
        float_re = float_regex()
        vector_regex = fr'\(?\[?{float_re}?,?\s?{float_re},\s?{float_re}\)?\]?'
        regex = fr'^{vector_regex}$|^{float_re}$'
        self.validRegex = regex
        
        regExp = QRegularExpression(regex)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

def getOpenImageFileName(parent=None, mostRecentPath='', fileTypes=None):
    if fileTypes is None:
        fileTypes = (
            "Images/Videos (*.npy *.npz *.h5, *.png *.tif *.tiff *.jpg *.jpeg "
            "*.mov *.avi *.mp4)"
            ";;All Files (*)"
        )
    file_path = QFileDialog.getOpenFileName(
        parent, 'Select image file', mostRecentPath, fileTypes
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
    sigRemove = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toggled.connect(self.emitToggled)
    
    def emitToggled(self, checked):
        self.sigToggled.emit(self, checked)
    
    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()

        editAction = QAction('Edit points appearance...')
        editAction.triggered.connect(self.editAppearance)
        contextMenu.addAction(editAction)
        
        removeAction = QAction('Remove points')
        removeAction.triggered.connect(self.emitRemove)
        contextMenu.addAction(removeAction)

        contextMenu.exec(event.globalPos())
    
    def emitRemove(self):
        self.sigRemove.emit(self)

class SpotsItems:
    sigButtonToggled = Signal(object, bool)

    def __init__(self, parent):
        self.buttons = []
        self.parent = parent
        self.currentPointSize = None
        self.loadedDfs = {}

    def addLayer(self, df_spots_files: dict, selected_file=None):
        all_df_spots_files = set()
        for files in df_spots_files.values():
            all_df_spots_files.update(files)
            
        win = dialogs.SpotsItemPropertiesDialog(
            natsorted(all_df_spots_files), 
            spotmax_out_path=self.spotmax_out_path,
            parent=self.parent, 
            color_idx=len(self.buttons), 
            selected_file=selected_file
        )
        win.exec_()
        if win.cancel:
            return

        toolbutton = self.addToolbarButton(win.state)
        toolbutton.df_spots_files = all_df_spots_files
        self.buttons.append(toolbutton)
        self.createSpotItem(win.state, toolbutton)
        self.loadSpotsTables(toolbutton)
        toolbutton.setToolTip(toolbutton.filename)
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
        toolbutton.sigRemove.connect(self.removeButton)
        toolbutton.filename = state['selected_file']
        return toolbutton
    
    def removeButton(self, button):
        button.item.setData([], [])
        self.parent.ax1.removeItem(button.item)
        filename = button.filename
        key = (self.posFoldername(), filename)
        try:
            self.loadedDfs.pop(key)
        except Exception as err:
            pass
        toolbar = self.parent.spotmaxToolbar.removeAction(button.action)
        self.buttons.remove(button)
    
    def buttonToggled(self, button, checked):
        button.item.setVisible(checked)
    
    def editAppearance(self, button):
        win = dialogs.SpotsItemPropertiesDialog(
            button.df_spots_files, 
            spotmax_out_path=self.spotmax_out_path,
            state=button.state, 
            parent=self.parent
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
    
    def getHoveredPoints(self, frame_i, z, y, x, return_button=False):
        hoveredPoints = []
        item = None
        toolbutton = None
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
        
        if return_button:
            return hoveredPoints, item, toolbutton
        else:
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
            try:
                df_xy = df.loc[[(frame_i, z)]].reset_index().set_index(['x', 'y'])
            except Exception as err:
                # This happens when hovering points in projections where they 
                # are all visibile and the z is unknown
                df_xy = df.loc[[frame_i]].reset_index().set_index(['x', 'y'])
            point_df = df_xy.loc[[(x, y)]].reset_index()
            point_df['Position_n'] = self.posFoldername()
            point_features = point_df.set_index(
                ['Position_n', 'frame_i', 'z', 'y', 'x']).iloc[0]
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
    
    def setCurrentPointSize(self):
        self.currentPointSize = self.getPointSize(force=True)
    
    def setCurrentPointMask(self, img_data):
        Y, X = img_data.shape[-2:]
        self._pointMask = np.zeros((Y, X), dtype=np.uint8)
    
    def createSpotItem(self, state, toolbutton):
        alpha = self.getAlpha(state)
        pen = self.getPen(state)
        brush = self.getBrush(state, alpha)
        hoverBrush = self.getBrush(state)
        symbol = state['pg_symbol']
        size = state['size']
        scatterItem = acdc_widgets.ScatterPlotItem(
            [], [], symbol=symbol, pxMode=False, size=size,
            brush=brush, pen=pen, hoverable=True, hoverBrush=hoverBrush, 
            tip=None
        )
        scatterItem._size = size
        scatterItem.frame_i = -1
        scatterItem.z = -1
        toolbutton.item = scatterItem
    
    def setPosition(self, posData):
        self.spotmax_out_path = posData.spotmax_out_path
        self.posData = posData
        self.posChanged = True
    
    def posFoldername(self):
        return self.posData.pos_foldername
    
    def _loadSpotsTable(self, toolbutton):
        spotmax_out_path = self.spotmax_out_path
        filename = toolbutton.filename
        key = (self.posFoldername(), filename)
        df = self.loadedDfs.get(key)
        if df is None:
            df = io.load_spots_table(spotmax_out_path, filename)
            self.loadedDfs[key] = df

        if df is None:
            toolbutton.df = None
        else:
            toolbutton.df = df.reset_index().set_index(['frame_i', 'z'])
    
    def setActiveButtonDf(self, df):
        toolbutton = self.getActiveButton()
        toolbutton.df = df.reset_index().set_index(['frame_i', 'z'])
        filename = toolbutton.filename
        key = (self.posFoldername(), filename)
        self.loadedDfs[key] = toolbutton.df  
    
    def loadSpotsTables(self, toolbutton=None):
        if toolbutton is None:
            for toolbutton in self.buttons:
                self._loadSpotsTable(toolbutton)
        else:
            self._loadSpotsTable(toolbutton)
    
    def getAnalysisParamsIniFilepath(self, toolbutton=None):
        if toolbutton is None:
            toolbutton = self.getActiveButton()
            
        df_spots_filename = toolbutton.filename
        ini_filepath = io.get_analysis_params_filepath_from_df_spots_filename(
            self.spotmax_out_path, df_spots_filename
        )
        return ini_filepath
    
    def loadAnalysisParams(self, toolbutton=None):
        if toolbutton is None:
            toolbutton = self.getActiveButton()
            
        ini_filepath = self.getAnalysisParamsIniFilepath(toolbutton=toolbutton)
        params = config.analysisInputsParams(params_path=ini_filepath)
        return params        
    
    def getLoadedSegmAndAnalysisSegm(self):
        toolbutton = self.getActiveButton()
        df_spots_filename = toolbutton.filename
        cp_params = io.load_analysis_params_from_df_spots_filename(
            self.spotmax_out_path, df_spots_filename
        )
        analysisSegmEndname = (
            cp_params['File paths and channels']
            ['Cells segmentation end name']
        )
        analysisSegmEndname = analysisSegmEndname.split('.npy')[0]
        analysisSegmEndname = analysisSegmEndname.split('.npz')[0]
        loadedSegmEndname = self.posData.getSegmEndname()
        return loadedSegmEndname, analysisSegmEndname
        
    def _setDataButton(self, toolbutton, frame_i, z=None):
        scatterItem = toolbutton.item
        if toolbutton.df is None:
            scatterItem.setData([], [])
            return
        
        noNeedToUpdate = (
            frame_i == scatterItem.frame_i
            and z == scatterItem.z
            and not self.posChanged
        )
        if noNeedToUpdate:
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
        self.posChanged = False
    
    def getActiveButton(self):
        for toolbutton in self.buttons:
            if toolbutton.isChecked():
                return toolbutton
    
    def getRefChannelSegmEndname(self, ref_ch_name):
        toolbutton = self.getActiveButton()
        if toolbutton is None:
            return ''
        
        if not ref_ch_name:
            return ''
        
        parts = io.df_spots_filename_parts(toolbutton.filename)
        run_num, df_id, df_text, desc, ext = parts
        ref_ch_segm_endname = (
            f'run_num{run_num}_{ref_ch_name}_ref_ch_segm_mask{desc}.npz'
        )
        pos_folderpath = os.path.dirname(self.spotmax_out_path)
        images_path = os.path.join(pos_folderpath, 'Images')
        for file in utils.listdir(images_path):
            if file.endswith(ref_ch_segm_endname):
                return ref_ch_segm_endname
            
        return ''
    
    def getActiveItemPointSize(self):
        activeButton = self.getActiveButton()
        if activeButton is None:
            return
        return activeButton.item._size
    
    def getPointSize(self, force=False):
        if force or self.currentPointSize is None:
            size = self.getActiveItemPointSize()
        else:
            size = self.currentPointSize
        return size
    
    def removePoint(self, hoveredPoints, item, button, frame_i, z):
        df = button.df
        ordered_columns = df.columns.to_list()
        try:
            df_xy = df.loc[[(frame_i, z)]].reset_index().set_index(['x', 'y'])
        except Exception as err:
            # This happens when hovering points in projections where they 
            # are all visibile and the z is unknown (z is None in proj)
            df_xy = df.loc[[frame_i]].reset_index().set_index(['x', 'y'])
        
        idx_to_drop = []
        for point in hoveredPoints:
            item.removePoint(point._index)
            pos = point.pos()
            xdata, ydata = int(pos.x()-0.5), int(pos.y()-0.5)
            zz_data = df_xy.loc[[(xdata, ydata)], 'z']
            for zdata in zz_data.values:
                idx_to_drop.append((frame_i, zdata, ydata, xdata))
        
        df_tzyx = (
            df.reset_index()
            .set_index(['frame_i', 'z', 'y', 'x'])
        )
        
        clickedIDs = df_tzyx.loc[idx_to_drop, 'Cell_ID']
        if 0 in clickedIDs.values:
            self.parent.warnRemovingPointCellIDzero()
        
        button.df = (
            df_tzyx.drop(index=idx_to_drop)
            .reset_index()
            .set_index(['frame_i', 'z'])
            [ordered_columns]
        )
        button.df['edited'] = 1
        button.df['do_not_drop'] = 1
        
        key = (self.posData.pos_foldername, button.filename)
        self.loadedDfs[key] = button.df        
    
    def initEdits(self, img_data, segm_data):
        self.setEditsEnabled(True)
        self.setCurrentPointSize()
        self.setCurrentPointMask(img_data)
        self._segm_data = segm_data
    
    def addPoint(self, item, img, frame_i, z, y, x, button):
        if z is None:
            self.parent.logger.info(
                '[WARNING]: Spots cannot be added on a z-projection'
            )
            return
        
        size = item._size
        radius = round(size/2)
        rr, cc = skimage.draw.disk((round(y), round(x)), radius)
        idx_max = (img[rr, cc]).argmax()
        xdata, ydata = cc[idx_max], rr[idx_max]
        
        lab = self._segm_data[frame_i]
        if lab.ndim == 3:
            lab = lab[z]
        
        ordered_columns = button.df.columns.to_list()
        
        ID = lab[ydata, xdata]
        
        spot_id = button.df['spot_id'].max() + 1
        
        df_tzid = (
            button.df.reset_index()
            .set_index(['frame_i', 'z', 'Cell_ID', 'spot_id'])
        )
        new_idx = (frame_i, z, ID, spot_id)
        empty_vals = features.get_df_row_empty_vals(df_tzid, index=new_idx)
        df_tzid = pd.concat([df_tzid, empty_vals])
        df_tzid.loc[new_idx, ['x', 'y']] = xdata, ydata
        button.df = (
            df_tzid.reset_index()
            .set_index(['frame_i', 'z'])
            .sort_index()
            [ordered_columns]
        )
        button.df['edited'] = 1
        button.df['do_not_drop'] = 1
        
        key = (self.posData.pos_foldername, button.filename)
        self.loadedDfs[key] = button.df
        
        xpoint, ypoint = xdata+0.5, ydata+0.5
        item.addPoints([xpoint], [ypoint])
    
    def setEditsEnabled(self, enabled):
        self._editsEnabled = enabled
    
    def editPoint(self, frame_i, z, y, x, img):
        if not self._editsEnabled:
            return
        
        hoveredPoints, item, button = self.getHoveredPoints(
            frame_i, z, y, x, return_button=True
        )
        if hoveredPoints:
            self.removePoint(hoveredPoints, item, button, frame_i, z)  
        else:
            self.addPoint(item, img, frame_i, z, y, x, button)
    
    def hideAllItems(self):
        for toolbutton in self.buttons:
            toolbutton.item.setVisible(False)
    
    def showItems(self):
        for toolbutton in self.buttons:
            toolbutton.item.setVisible(toolbutton.isChecked())

def ParamFormWidget(
        anchor, param, parent, use_tune_widget=False,
        section_option_to_desc_mapper=None
    ):
    if use_tune_widget:
        widgetName = param['autoTuneWidget']
    else:
        widgetName = param['formWidgetFunc']
    
    if section_option_to_desc_mapper is None:
        section_option_to_desc_mapper = {}
    
    module_name, attr = widgetName.split('.')
    try:
        widgets_module = globals()[module_name]
        widgetFunc = getattr(widgets_module, attr)
    except KeyError as e:
        widgetFunc = globals()[attr]
    
    section = config.get_section_from_anchor(anchor)
    confvalText = param.get('confvalText', param.get('desc', ''))
    key = (section, confvalText)
    infoHtmlText = section_option_to_desc_mapper.get(key, '')
    
    if use_tune_widget:
        addComputeButton = False
    else:
        addComputeButton = param.get('addComputeButton', False)
    
    return formWidget(
        widgetFunc(),
        anchor=anchor,
        labelTextLeft=param.get('desc', ''),
        confvalText=param.get('confvalText', ''),
        labelTextMiddle=param.get('labelTextMiddle', ''),
        useEditableLabel=param.get('useEditableLabel', ''),
        initialVal=param.get('initialVal', None),
        stretchWidget=param.get('stretchWidget', True),
        addInfoButton=param.get('addInfoButton', True),
        addAddFieldButton=param.get('addAddFieldButton', False),
        addComputeButton=addComputeButton,
        addWarningButton=param.get('addWarningButton', False),
        addApplyButton=param.get('addApplyButton', False),
        addBrowseButton=param.get('addBrowseButton', False),
        isFolderBrowse=param.get('isFolderBrowse', False),
        browseExtensions=param.get('browseExtensions'),
        addAutoButton=param.get('addAutoButton', False),
        addEditButton=param.get('addEditButton', False),
        stretchFactors=param.get('stretchFactors'),
        addLabel=param.get('addLabel', True),
        valueSetter=param.get('valueSetter'),
        disableComputeButtons=True,
        infoHtmlText=infoHtmlText,
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

class RefChPredictionMethodWidget(SpotPredictionMethodWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        # Remove 'spotMAX AI' item since we do not have a neural net for 
        # reference channel yet
        self.combobox.removeItem(1)
    
class SpinBox(acdc_widgets.SpinBox):
    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent, disableKeyPress=disableKeyPress)
        self.installEventFilter(self)
    
    def setValue(self, value):
        if isinstance(value, str):
            value = int(value)
        super().setValue(value)
    
    def setText(self, text):
        value = int(text)
        super().setValue(value)
    
    def eventFilter(self, object, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            return True
        return False

class DoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.installEventFilter(self)
    
    def eventFilter(self, object, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            return True
        return False

class RunNumberSpinbox(SpinBox):
    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent, disableKeyPress=disableKeyPress)
        self.installEventFilter(self)
        self.setMinimum(1)
    
    def eventFilter(self, object, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            return True
        return False

class SetCustomCombinedMeasurement(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        _layout = QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        
        self._nameLabel = QLabel('Expression')
        self._nameLabel.setFont(font)
        
        self._entryWidget = SetValueFromFeaturesWidget()
        self._entryWidget.calculatorWindow.setExpandedAll(False)
        
        _layout.addWidget(self._nameLabel)
        _layout.addWidget(self._entryWidget)
        
        self.setLayout(_layout)
    
    def setValue(self, value):
        self._entryWidget.setText(str(value))
    
    def value(self):
        return self._entryWidget.text()

    def text(self):
        return self.value()

    def setText(self, text):
        self.setValue(text)

class SetValueFromFeaturesWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        layout = QHBoxLayout()
        
        readOnlyLineEdit = ReadOnlyElidingLineEdit()
        self.readOnlyLineEdit = readOnlyLineEdit
        editButton = acdc_widgets.editPushButton()
        
        layout.addWidget(readOnlyLineEdit)
        layout.addWidget(editButton)
        
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        
        layout.setContentsMargins(0, 0, 0, 0)
        
        editButton.clicked.connect(self.editButtonClicked)
        
        self.initCalculatorWindow()
        
        self.setLayout(layout)
    
    def initCalculatorWindow(self):
        features_groups = features.get_features_groups()
        spotsize_features = features_groups.pop('SpotSIZE metrics')
        features_groups = {
            'SpotSIZE metrics': spotsize_features, **features_groups
        }
        all_features_to_col_mapper = (
            features.feature_names_to_col_names_mapper()
        )
        group_name_to_col_mapper = {'SpotSIZE metrics': {}}
        for group_feat_name, column_name in all_features_to_col_mapper.items():
            group_name, feat_name = group_feat_name.split(', ')
            if group_name != 'SpotSIZE metrics':
                continue
            group_name_to_col_mapper[group_name][feat_name] = column_name
        
        self.calculatorWindow = acdc_apps.CombineFeaturesCalculator(
            features_groups, 
            group_name_to_col_mapper=group_name_to_col_mapper,
            parent=self
        )
        self.calculatorWindow.expandAll()
        self.calculatorWindow.sigOk.connect(self.equationConfirmed)
    
    def setLabel(self, text):
        self.calculatorWindow.newFeatureNameLineEdit.setReadOnly(False)
        self.calculatorWindow.newFeatureNameLineEdit.setText(text)
        self.calculatorWindow.newFeatureNameLineEdit.setReadOnly(True)
    
    def setValue(self, text):
        self.readOnlyLineEdit.setText(text)
    
    def setText(self, text):
        self.setValue(text)
    
    def value(self):
        return self.readOnlyLineEdit.text()

    def text(self):
        return self.value()
    
    def equationConfirmed(self):
        self.readOnlyLineEdit.setText(self.calculatorWindow.equation)
    
    def equation(self):
        return self.readOnlyLineEdit.text()
    
    def editButtonClicked(self):
        self.calculatorWindow.show()
    
    def closeEvent(self, event):
        self.calculatorWindow.close()
    
class SetBoundsFromFeaturesGroupBox(QGroupBox):
    def __init__(self, title='', checkable=False, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setCheckable(checkable)
        
        layout = QFormLayout()
        
        self.minimumWidget = SetValueFromFeaturesWidget()
        layout.addRow('Minimum', self.minimumWidget)
        
        self.maximumWidget = SetValueFromFeaturesWidget()
        layout.addRow('Maximum', self.maximumWidget)
        
        self.setLayout(layout)
    
    def setLabel(self, text):
        self.minimumWidget.setLabel(f'Minimum `{text}`')
        self.maximumWidget.setLabel(f'Maximum `{text}`')
    
    def setText(self, text):
        self.setValue(text)
    
    def setValue(self, value):
        if value is None:
            return
        
        if isinstance(value, str):
            try:
                min_val, max_val = value.replace(' ', '').split(',')
            except Exception as err:
                min_val = value
                max_val = value
            self.minimumWidget.setValue(min_val)
            self.maximumWidget.setValue(max_val)
            return
        
        min_val, max_val = value
        self.minimumWidget.setValue(min_val)
        self.maximumWidget.setValue(max_val)
    
    def value(self):
        return f'{self.minimumWidget.value()}, {self.maximumWidget.value()}'
    
    def text(self):
        return self.value()

class NumericWidgetWithLabel(QWidget):
    def __init__(
            self, parent=None, is_float=False, text='', text_loc='top'
        ) -> None:
        super().__init__(parent)
        
        if text_loc == 'top' or text_loc == 'bottom':
            layout = QVBoxLayout()
        else:
            layout = QHBoxLayout()
        
        if is_float:
            widgetFunc = FloatLineEdit
        else:
            widgetFunc = acdc_widgets.SpinBox
        
        self.widget = widgetFunc()
        self.label = QLabel(text, self.widget)
        
        if text_loc == 'top' or text_loc == 'left':
            layout.addWidget(self.label)
            layout.addWidget(self.widget)
        else:
            layout.addWidget(self.widget)
            layout.addWidget(self.label)
        
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
    
    def setText(self, text):
        self.label.setText(text)
    
    def setValue(self, value):
        self.widget.setValue(value)
    
    def value(self):
        return self.widget.value()

    def setMinimum(self, minimum):
        self.widget.setMinimum(minimum)
    
    def setMinimum(self, maximum):
        self.widget.setMinimum(maximum)

class LowHighRangeWidget(QWidget):
    def __init__(self, parent=None, is_float=False) -> None:
        super().__init__(parent)
        
        layout = QHBoxLayout()         
        
        self.lowValueWidget = NumericWidgetWithLabel(is_float=is_float)
        self.lowValueWidget.setText('Low value')
        
        self.highValueWidget = NumericWidgetWithLabel(is_float=is_float)
        self.highValueWidget.setText('High value')
        
        layout.addWidget(self.lowValueWidget)
        layout.addWidget(self.highValueWidget)
        
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setFont(font)
    
    def setValue(self, low_high):
        if low_high is None or not low_high:
            low_high = 0, 0
        
        if isinstance(low_high, str):
            low_high = config.get_stack_3d_segm_range(low_high)
        
        low, high = low_high
        self.lowValueWidget.setValue(low)
        self.highValueWidget.setValue(high)
    
    def value(self):
        low = self.lowValueWidget.value()
        high = self.highValueWidget.value()
        return low, high        
    
    def text(self):
        return str(self.value())

class Extend3DsegmRangeWidget(LowHighRangeWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        
        self.lowValueWidget.setText('Below bottom z-slice')
        self.highValueWidget.setText('Above top z-slice')
        
        self.lowValueWidget.setMinimum(0)
        self.highValueWidget.setMinimum(0)
        
        self.layout().setContentsMargins(0, 5, 0, 0)

class sigmaXBoundsWidget(SetBoundsFromFeaturesGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLabel('sigma_x')
    
    def setValue(self, value):
        if value == 'Default' or value is None or not value:
            value = config.get_sigma_xy_bounds('Default')
        
        super().setValue(value)

class sigmaYBoundsWidget(SetBoundsFromFeaturesGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLabel('sigma_y')
    
    def setValue(self, value):
        if value == 'Default' or value is None or not value:
            value = config.get_sigma_xy_bounds('Default')
        
        super().setValue(value)

class sigmaZBoundsWidget(SetBoundsFromFeaturesGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLabel('sigma_z')
    
    def setValue(self, value):
        if value == 'Default' or value is None or not value:
            value = config.get_sigma_z_bounds('Default')
        
        super().setValue(value)

class AfitBoundsWidget(SetBoundsFromFeaturesGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLabel('amplitude_peak')
    
    def setValue(self, value):
        if value == 'Default' or value is None or not value:
            value = config.get_A_fit_bounds('Default')
        
        super().setValue(value)

class BfitBoundsWidget(SetBoundsFromFeaturesGroupBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLabel('background_peak')
    
    def setValue(self, value):
        if value == 'Default' or value is None or not value:
            value = config.get_B_fit_bounds('Default')
        
        super().setValue(value)

class PlusMinusFloatLineEdit(FloatLineEdit):
    def __init__(self, *args, **kwargs):
        self.plusminus_text = '± '
        super().__init__(*args, **kwargs)
        
        float_re = float_regex(
            allow_negative=False, left_chars=self.plusminus_text
        )
        pattern = fr'^{float_re}$'
        self.setRegexValidator(pattern)
        
        self.setText(self.plusminus_text)
    
    def emitValueChanged(self, text):
        if not self.text().startswith(self.plusminus_text):
            text = text.replace('±', '').lstrip()
            text = f'{self.plusminus_text}{text}'
            self.setText(text)
        
        super().emitValueChanged(text)
    
    def value(self):
        text = self.text().replace('±', '').lstrip()
        m = re.match(float_regex(), text)
        if m is not None:
            text = m.group(0)
            try:
                val = float(text)
            except ValueError:
                val = 0.0
            return val
        else:
            return 0.0

class ExpandableGroupbox(QGroupBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expanded = True
        self._scheduled_expansion = None
        self._layout = None
    
    def setLayout(self, layout):
        buttonLayout = QHBoxLayout()
        self.expandButton = acdc_widgets.showDetailsButton(txt='Show parameters')
        self.expandButton.setChecked(True)
        self.expandButton.sigToggled.connect(self.setExpanded)
        buttonLayout.addWidget(self.expandButton)
        buttonLayout.setStretch(0, 1)
        buttonLayout.addStretch(2)
        self._layout = layout
        _mainLayout = QVBoxLayout()
        _mainLayout.addLayout(buttonLayout)    
        _mainLayout.addLayout(layout)     
        super().setLayout(_mainLayout)
        if self._scheduled_expansion is not None:
            self.expandButton.setChecked(self._scheduled_expansion)
        
        if self.isCheckable():
            self.toggled.connect(self.expandButton.setChecked)
        
    def layout(self):
        return self._layout
    
    def setExpanded(self, expanded):
        if self._layout is None:
            self._scheduled_expansion = expanded
            return
        
        self._expanded = expanded            
        
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            widget = item.widget()
            if widget is None:
                continue
            widget.setVisible(expanded)       

class EditableLabel(QWidget):
    clicked = Signal(object)
    
    def __init__(self, name, parent=None):
        super().__init__(parent)
        
        _layout = QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        
        self._nameLabel = QLabel(name)
        
        self._lineEdit = acdc_widgets.alphaNumericLineEdit()
        self._lineEdit.setAlignment(Qt.AlignCenter)
        
        _layout.addWidget(self._nameLabel)
        _layout.addWidget(self._lineEdit)
        
        self.setLayout(_layout)
    
    def setValue(self, value):
        self._lineEdit.setText(str(value))
    
    def value(self):
        return self._lineEdit.text()

    def text(self):
        return self.value()

class TuneSpotPredictionMethodWidget(QWidget):
    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        mainLayout = QVBoxLayout()
        
        self.spotPredictioMethodWidget = SpotPredictionMethodWidget(
            parent=parent
        )
        self.spotPredictioMethodWidget.combobox.removeItem(2)        
        self.spotPredictioMethodWidget.combobox.currentIndexChanged.connect(
            self.onMethodIndexChanged
        )
        
        threshValLayout = QFormLayout()
        self.threshValLineEdit = QLineEdit()
        self.threshValLineEdit.setReadOnly(True)
        self.threshValLineEdit.setAlignment(Qt.AlignCenter)
        label = QLabel(' | Threshold value')
        label.setFont(config.font(pixelSizeDelta=-2))
        self.threshValLineEdit.label = label
        threshValLayout.addRow(label, self.threshValLineEdit)
        label.setDisabled(True)
        self.threshValLineEdit.setDisabled(True)
        
        mainLayout.addWidget(self.spotPredictioMethodWidget)
        mainLayout.addLayout(threshValLayout)
        
        mainLayout.setContentsMargins(0, 5, 0, 5)
        self.setLayout(mainLayout)
    
    def onMethodIndexChanged(self, idx):
        self.threshValLineEdit.label.setDisabled(idx==0)
        self.threshValLineEdit.setDisabled(idx==0)
    
    def setText(self, text):
        try:
            val = float(text)
            self.threshValLineEdit.setText(val)
        except Exception as err:
            pass
        

class CenteredAlphaNumericLineEdit(acdc_widgets.alphaNumericLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)

def toClipboard(text):
    cb = QApplication.clipboard()
    cb.clear(mode=cb.Clipboard)
    cb.setText(text, mode=cb.Clipboard)

class RefChannelFeaturesThresholdsButton(_GopFeaturesAndThresholdsButton):
    def __init__(self, parent=None):
        super().__init__(parent, category='ref. channel objects')

class TuneScatterPlotItem(acdc_widgets.ScatterPlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def coordsToDf(self, includeData=True, **kwargs):
        if len(self.points()) == 0:
            return []
        
        columns = ['Position_n', 'frame_i', 'neigh_z', 'z', 'y', 'x']
        df = {col:[] for col in columns}
        for p, point in enumerate(self.points()):
            pos = point.pos()
            x, y = pos.x(), pos.y()
            df['x'].append(round(x))
            df['y'].append(round(y))
            pos_foldername, frame_i, neigh_z, z = point.data()
            df['Position_n'].append(pos_foldername)
            df['frame_i'].append(frame_i)
            df['neigh_z'].append(neigh_z)
            df['z'].append(z)

        df = pd.DataFrame(df)
        df = df[df['neigh_z'] == df['z']].drop(columns='neigh_z')
        
        return df

class LocalBackgroundRingWidthWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        mainLayout = QGridLayout()
        
        controlWidget = DoubleSpinBox()
        controlWidget.setDecimals(0)
        controlWidget.setMinimum(1)
        controlWidget.setSingleStep(1)
        controlWidget.setAlignment(Qt.AlignCenter)
        self.controlWidget = controlWidget
        
        unitCombobox = QComboBox()
        unitCombobox.addItems(['pixel', 'micrometre'])
        self.unitCombobox = unitCombobox
        
        mainLayout.addWidget(controlWidget, 0, 0)
        mainLayout.addWidget(unitCombobox, 0, 1)
        
        indicatorWidget = QLineEdit()
        indicatorWidget.setAlignment(Qt.AlignCenter)
        indicatorWidget.setReadOnly(True)
        otherUnitLabel = QLabel('micrometre')
        self.indicatorWidget = indicatorWidget
        self.otherUnitLabel = otherUnitLabel
        
        mainLayout.addWidget(indicatorWidget, 1, 0)
        mainLayout.addWidget(otherUnitLabel, 1, 1)
        
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 0)
        
        mainLayout.setContentsMargins(0,0,0,0)
        self.setLayout(mainLayout)
        
        self.setPixelSize(None)
        self.valueUpdated(None)
        
        unitCombobox.currentTextChanged.connect(self.unitUpdated)
        controlWidget.valueChanged.connect(self.valueUpdated)
        
        self.installEventFilter(controlWidget)
    
    def valueUpdated(self, value):
        if self._pixelSize is None:
            self.indicatorWidget.setText('n.a.')
            return
        
        if self.unit() == 'pixel':
            multiplier = self.pixelSize()
            decimals = 3
        else:
            multiplier = 1/self.pixelSize()
            decimals = 0
    
        indicatorValue = round(value*multiplier, decimals)
        self.indicatorWidget.setText(str(indicatorValue))
    
    def unitUpdated(self, unit):
        if unit == 'pixel':
            self.controlWidget.setDecimals(0)
            self.controlWidget.setMinimum(1)
            self.otherUnitLabel.setText('micrometre')
        else:
            self.controlWidget.setDecimals(3)
            self.controlWidget.setMinimum(self.pixelSize())
            self.otherUnitLabel.setText('pixel')
        self.valueUpdated(self.value())
    
    def updateStep(self):
        if self.unit() == 'pixel':
            self.controlWidget.setSingleStep(1)
        else:
            self.controlWidget.setSingleStep(self._pixelSize)
    
    def eventFilter(self, object, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            return True
        return False 
    
    def setPixelSize(self, pixelSize):
        self._pixelSize = pixelSize
        self.updateStep()
        self.valueUpdated(self.value())
    
    def pixelSize(self):
        if self._pixelSize is None:
            return 0
        
        return self._pixelSize

    def setText(self, text: str):
        value, unit = text.split()
        self.setValue(float(value))
        self.setUnit(unit)
    
    def text(self):
        return f'{self.value()} {self.unit()}'
    
    def setValue(self, value):
        self.controlWidget.setValue(value)
    
    def setUnit(self, unit: str):
        return self.unitCombobox.setCurrentText(unit)
    
    def value(self):
        return self.controlWidget.value()
    
    def unit(self):
        return self.unitCombobox.currentText()

class InvisibleScrollArea(QScrollArea):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(
            'QScrollArea {background: transparent;}'
            'QScrollArea > QWidget > QWidget { background: transparent;}'
        )

class DockWidget(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.installEventFilter(self)

class VoxelSizeWidget(QWidget):
    def __init__(self, parent=None, um_to_pixel=1.0, unit='pixel'):
        super().__init__(parent)
        
        self.um_to_pixel = um_to_pixel
        
        try:
            len(um_to_pixel)
            self.um_to_pixel = np.array(um_to_pixel, dtype=float)
        except Exception as err:
            pass
        
        layout = QGridLayout()
        
        controlWidget = acdc_widgets.VectorLineEdit()
        self.controlWidget = controlWidget
        
        displayLabel = QLabel('0.0')
        displayLabel.setAlignment(Qt.AlignCenter)
        self.displayLabel = displayLabel
        
        unitCombobox = QComboBox()
        unitCombobox.addItems(['pixel', 'micrometre'])
        self.unitCombobox = unitCombobox
        
        unitLabel = QLabel('micrometre')
        
        layout.addWidget(controlWidget, 0, 0)
        layout.addWidget(unitCombobox, 0, 1, alignment=Qt.AlignLeft)
        
        layout.addWidget(displayLabel, 1, 0)
        layout.addWidget(unitLabel, 1, 1, alignment=Qt.AlignLeft)
        
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.controlWidget.valueChanged.connect(self.onValueChanged)
        self.unitCombobox.currentTextChanged.connect(self.onValueChanged)
        
        self.setToolTip()
        
        self.setLayout(layout)
    
    def setValue(self, value):
        self.controlWidget.setValue(value)
    
    def setToolTip(self):
        unit = self.unitCombobox.currentText()
        try:
            len(self.um_to_pixel)
            um_to_pixel = tuple(self.um_to_pixel)
        except Exception as err:
            um_to_pixel = self.um_to_pixel
            
        kwargs = {
            'um_to_pixel': um_to_pixel, 
            'unit': unit
        }     
        super().setToolTip(str(kwargs))   
    
    def onValueChanged(self, placeholder):
        value = self.controlWidget.value()
        
        if self.unitCombobox.currentText() == 'pixel':
            mult_factor = self.um_to_pixel
            decimals = 4
        else:
            mult_factor = 1/self.um_to_pixel
            decimals = 1
        
        self.setToolTip()
        
        try:
            len(value)
            value = np.round(np.array(value, dtype=float), decimals)
            conv_value = value * mult_factor
            text = str(tuple(conv_value))
            self.displayLabel.setText(text)
            return
        except Exception as err:
            pass
        
        try:
            conv_value = round(value * mult_factor, decimals)
            text = str(conv_value)
            self.displayLabel.setText(text)
            return
        except Exception as err:
            pass
        
        self.displayLabel.setText(html_func.span('ERROR', font_color='red'))
    
    def value(self):
        if self.unitCombobox.currentText() == 'pixel':
            try:
                value = tuple(self.controlWidget.value())
                return value
            except Exception as err:
                return self.controlWidget.value()
        else:
            return eval(self.displayLabel.text())
        

class LineEdit(QLineEdit):
    def __init__(self, *args, centered=True, **kwargs):
        super().__init__(*args, **kwargs)
        if centered:
            self.setAlignment(Qt.AlignCenter)
    
    def setValue(self, value):
        super().setText(str(value))
    
    def value(self):
        return self.text()

class EndnameLineEdit(LineEdit):
    sigValueChanged = Signal(object)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setValue(self, value):
        value_str = str(value)
        folderpath = os.path.dirname(str(value))
        is_images_folder = os.path.basename(folderpath) == 'Images'
        is_spotmax_out_folder = os.path.basename(folderpath) == 'spotMAX_output'
        if is_images_folder:
            filepath = value_str
            filename = os.path.basename(filepath)
            posData = acdc_load.loadData(filepath, '')
            posData.getBasenameAndChNames()
            endname = filename[len(posData.basename):]
            text = endname
            self.setToolTip('Images')
        elif is_spotmax_out_folder:
            filepath = value_str
            filename = os.path.basename(filepath)
            text = filename
            self.setToolTip('spotMAX_output')
        else:
            text = value_str
            
        self.setText(text)
        self.sigValueChanged.emit(text)     

class SelectPosFoldernamesButton(acdc_widgets.editPushButton):
    def __init__(self, *args, exp_path='', **kwargs):
        if not args:
            args = ['Select/view Positions']
            
        super().__init__(*args, **kwargs)
        
        self._exp_path = exp_path
        self._value = []
        self.setToolTip()
        
        self.clicked.connect(self.selectPositions)
    
    def setToolTip(self):
        super().setToolTip(str({'exp_path': self._exp_path}))
    
    def selectPositions(self):
        pos_foldernames = acdc_myutils.get_pos_foldernames(self._exp_path)
        win = acdc_widgets.QDialogListbox(
            'Select Positions', 
            'Select Positions', 
            pos_foldernames,
            parent=self, 
            preSelectedItems=self.value()
        )
        win.exec_()
        if win.cancel:
            return
        self.setValue(win.selectedItemsText)
    
    def setValue(self, value):
        self._value = value
    
    def value(self):
        return self._value

class ArrowButtons(QWidget):
    sigButtonClicked = Signal(str)
    
    def __init__(
            self, order=('left', 'right', 'down', 'up'), 
            orientation='horizontal',
            tooltips=None,
            parent=None
        ):
        super().__init__(parent)
        
        if orientation == 'horizontal':
            layout = QHBoxLayout()
        elif orientation == 'vertical':
            layout = QVBoxLayout()
        else:
            raise ValueError(
                'Only orientations allowd are "horizontal" and "vertical"'
            )
        
        for d, direction in enumerate(order):
            buttonName = f'arrow{direction.title()}PushButton'
            button = getattr(acdc_widgets, buttonName)()
            layout.addWidget(button)
            button.clicked.connect(self.buttonClicked)
            button.direction = direction
            if tooltips is None:
                continue
            button.setToolTip(tooltips[d])
        
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
    
    def buttonClicked(self):
        self.sigButtonClicked.emit(self.sender().direction)