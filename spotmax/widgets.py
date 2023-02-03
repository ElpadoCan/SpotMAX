import sys
import time
import re
import traceback
import webbrowser
from pprint import pprint
from functools import partial

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from PyQt5.QtCore import (
    pyqtSignal, QTimer, Qt, QPoint, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
    QSize, QRectF, QPointF, QRect, QPoint, QEasingCurve, QRegExp,
    QEvent, QEventLoop
)
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QPen, QPaintEvent, QBrush, QPainter,
    QRegExpValidator, QIcon, QFontMetrics, QFocusEvent
)
from PyQt5.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QApplication, QWidget, QVBoxLayout, QMainWindow, QStyleFactory,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QDockWidget,
    QScrollArea, QSizePolicy, QComboBox, QPushButton, QScrollBar,
    QGroupBox, QAbstractSlider, QDialog, QStyle, QSpacerItem,
    QAction, QWidgetAction, QMenu, QActionGroup, QFileDialog, QFrame,
    QListWidget, QPlainTextEdit
)

import pyqtgraph as pg

from cellacdc import widgets as acdc_widgets

from . import is_mac, is_win
from . import utils, dialogs, config, html_func, docs

# NOTE: Enable icons
from . import qrc_resources

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

class applyPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':magnGlass.svg'))

class computePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':compute.svg'))

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

class measurementsQGroupBox(QGroupBox):
    def __init__(self, names, parent=None):
        QGroupBox.__init__(self, 'Single cell measurements', parent)
        self.formWidgets = []

        self.setCheckable(True)
        layout = myFormLayout()

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

def _refChThresholdFuncWidget():
    widget = myQComboBox()
    items = config.skimageAutoThresholdMethods()
    widget.addItems(items)
    return widget

def _spotThresholdFunc():
    widget = myQComboBox()
    items = config.skimageAutoThresholdMethods()
    widget.addItems(items)
    return widget

def _filterSpotsVsRefChMethodWidget():
    widget = myQComboBox()
    items = config.filterSpotsVsRefChMethods()
    widget.addItems(items)
    return widget

def _spotDetectionMethod():
    widget = myQComboBox()
    items = ['Detect local peaks', 'Label prediction mask']
    widget.addItems(items)
    return widget

def _spotPredictionMethod():
    widget = myQComboBox()
    items = ['Thresholding', 'Neural network']
    widget.addItems(items)
    return widget

def _gopMethod():
    widget = myQComboBox()
    items = ['Effect size', 't-test (p-value)']
    widget.addItems(items)
    return widget

class _spotMinSizeLabels(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        font = config.font()
        layout = QVBoxLayout()
        self.umLabel = QLabel()
        self.umLabel.setFont(font)
        self.pixelLabel = QLabel()
        self.pixelLabel.setFont(font)
        layout.addWidget(self.umLabel)
        layout.addWidget(self.pixelLabel)
        self.setLayout(layout)

    def setText(self, text):
        self.umLabel.setText(text)
        self.pixelLabel.setText(text)

class formWidget(QWidget):
    sigApplyButtonClicked = pyqtSignal(object)
    sigComputeButtonClicked = pyqtSignal(object)
    sigBrowseButtonClicked = pyqtSignal(object)
    sigAutoButtonClicked = pyqtSignal(object)
    sigLinkClicked = pyqtSignal(str)
    sigEditClicked = pyqtSignal(object)

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
            key='',
            parent=None
        ):
        QWidget.__init__(self, parent)
        self.widget = widget
        self.anchor = anchor
        self.key = key

        widget.setParent(self)

        if isinstance(initialVal, bool):
            widget.setChecked(initialVal)
        elif isinstance(initialVal, str):
            try:
                widget.setCurrentText(initialVal)
            except AttributeError:
                widget.setText(initialVal)
        elif isinstance(initialVal, float) or isinstance(initialVal, int):
            widget.setValue(initialVal)

        self.items = []

        if font is None:
            font = QFont()
            font.setPixelSize(13)

        self.labelLeft = acdc_widgets.QClickableLabel(widget)
        self.labelLeft.setText(labelTextLeft)
        self.labelLeft.setFont(font)
        self.items.append(self.labelLeft)

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
                    f'Info about "{self.labelLeft.text()}" parameter'
                )
            else:
                infoButton.setToolTip(
                    f'Info about "{self.labelRight.text()}" measurement'
                )
            infoButton.clicked.connect(self.showInfo)
            self.items.append(infoButton)
        
        if addEditButton:
            editButton = acdc_widgets.editPushButton(self)
            editButton.setToolTip('Edit field')
            self.sigEditClicked.connect(self.editButtonClicked)
            self.items.append(editButton)

        if addBrowseButton:
            browseButton = acdc_widgets.showInFileManagerButton(self)
            browseButton.setToolTip('Browse')
            browseButton.clicked.connect(self.browseButtonClicked)
            self.items.append(browseButton)

        if addApplyButton:
            applyButton = applyPushButton(self)
            applyButton.setCursor(Qt.PointingHandCursor)
            applyButton.setCheckable(True)
            applyButton.setToolTip('Apply this step and visualize results')
            applyButton.clicked.connect(self.applyButtonClicked)
            self.items.append(applyButton)

        if addAutoButton:
            autoButton = acdc_widgets.autoPushButton(self)
            autoButton.setCheckable(True)
            autoButton.setToolTip('Automatically infer this parameter')
            autoButton.clicked.connect(self.autoButtonClicked)
            self.items.append(autoButton)

        if addComputeButton:
            computeButton = computePushButton(self)
            computeButton.setToolTip('Compute this step and visualize results')
            computeButton.clicked.connect(self.computeButtonClicked)
            self.items.append(computeButton)

        self.labelLeft.clicked.connect(self.tryChecking)
        self.labelRight.clicked.connect(self.tryChecking)

    def tryChecking(self, label):
        try:
            self.widget.setChecked(not self.widget.isChecked())
        except AttributeError as e:
            pass

    def browseButtonClicked(self):
        file_path = getOpenImageFileName(
            parent=self, mostRecentPath=utils.getMostRecentPath()
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
        self.sigComputeButtonClicked.emit(self)

    def linkActivatedCallBack(self, link):
        if utils.is_valid_url(link):
            webbrowser.open(link)
        else:
            self.sigLinkClicked.emit(link)

    def showInfo(self):
        anchor = self.anchor
        txt = html_func.paragraph(docs.paramsInfoText().get(anchor, ''))
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

class myFormLayout(QGridLayout):
    def __init__(self):
        QGridLayout.__init__(self)

    def addFormWidget(self, formWidget, row=0):
        for col, item in enumerate(formWidget.items):
            if col==0:
                alignment = Qt.AlignRight
            elif col==2:
                alignment = Qt.AlignLeft
            else:
                alignment = Qt.AlignmentFlag()
            try:
                self.addWidget(item, row, col, alignment=alignment)
            except TypeError:
                self.addLayout(item, row, col)

class myQScrollBar(QScrollBar):
    sigActionTriggered = pyqtSignal(int)

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

class myQComboBox(QComboBox):
    def __init__(self, checkBox=None, parent=None):
        QComboBox.__init__(self, parent)

        # checkBox that controls if ComboBox can be enabled or not
        self.checkBox = checkBox
        self.activated.connect(self.clearFocus)
        self.installEventFilter(self)

    def eventFilter(self, object, event):
        # Disable wheel scroll on widgets to allow scroll only on scrollarea
        if event.type() == QEvent.Wheel:
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

class intLineEdit(QLineEdit):
    valueChanged = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)

        regExp = QRegExp('\d+')
        self.setValidator(QRegExpValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(13)
        self.setFont(font)
        self.setText('0')

        self.textChanged.connect(self.emitValueChanged)

    def setValue(self, value: int):
        self.setText(str(value))

    def value(self):
        return int(self.text())

    def emitValueChanged(self, text):
        self.valueChanged.emit(self.value())

class floatLineEdit(QLineEdit):
    valueChanged = pyqtSignal(float)

    def __init__(
            self, *args, notAllowed=(0,), allowNegative=False, initial=None
        ):
        QLineEdit.__init__(self, *args)
        self.notAllowed = notAllowed

        self.isNumericRegExp = (
            r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
        )
        if not allowNegative:
            self.isNumericRegExp = self.isNumericRegExp.replace('[-+]?', '')

        regExp = QRegExp(self.isNumericRegExp)
        self.setValidator(QRegExpValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(13)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)
        if initial is None:
            self.setText('0.0')

    def setNotAllowedStyleSheet(self):
        self.setStyleSheet(
            'background: #FEF9C3;'
            'border-radius: 4px;'
            'border: 1.5px solid red;'
            'padding: 1px 0px 1px 0px'
        )

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
            self.setNotAllowedStyleSheet()
        else:
            self.setStyleSheet('background: #ffffff;')
            self.valueChanged.emit(self.value())

def getOpenImageFileName(parent=None, mostRecentPath=''):
    file_path = QFileDialog.getOpenFileName(
        parent, 'Select image file', mostRecentPath,
        "Images/Videos (*.npy *.npz *.h5, *.png *.tif *.tiff *.jpg *.jpeg "
        "*.mov *.avi *.mp4)"
        ";;All Files (*)"
    )[0]
    return file_path

class DblClickQToolButton(QToolButton):
    sigDoubleClickEvent = pyqtSignal(object, object)
    sigClickEvent = pyqtSignal(object, object)

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
    sigHoverEvent = pyqtSignal(object, object)

    def __init__(self, *args, **kwargs):
        pg.ImageItem.__init__(self, *args, **kwargs)

    def hoverEvent(self, event):
        self.sigHoverEvent.emit(self, event)

class ScatterPlotItem(pg.ScatterPlotItem):
    sigClicked = pyqtSignal(object, object, object)

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
    sigClicked = pyqtSignal()

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
            p.setRenderHint(QPainter.Antialiasing)
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
