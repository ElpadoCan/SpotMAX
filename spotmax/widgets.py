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
    QListWidget
)

import pyqtgraph as pg

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

class okPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':yesGray.svg'))
        self.setDefault(True)
        # QShortcut(Qt.Key_Return, self, self.click)
        # QShortcut(Qt.Key_Enter, self, self.click)

class reloadPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':reload.svg'))

class reloadPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':more.svg'))

class savePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':file-save.svg'))

class newFilePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':file-new.svg'))

class infoPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':info.svg'))

class addPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':add.svg'))

class subtractPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':subtract.svg'))

class continuePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':continue.svg'))

class calcPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':calc.svg'))

class showInFileManagerButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':folder-open.svg'))

class showDetailsButton(QPushButton):
    def __init__(self, *args, txt='Show details...'):
        super().__init__(*args)
        self.setText(txt)
        self.txt = txt
        self.checkedIcon = QIcon(':hideUp.svg')
        self.uncheckedIcon = QIcon(':showDown.svg')
        self.setIcon(self.uncheckedIcon)
        self.toggled.connect(self.onClicked)
        w = self.sizeHint().width()
        self.setFixedWidth(w)

    def onClicked(self, checked):
        if checked:
            self.setText(' Hide details   ')
            self.setIcon(self.checkedIcon)
        else:
            self.setText(self.txt)
            self.setIcon(self.uncheckedIcon)

class cancelPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':cancelButton.svg'))

class showPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':magnGlass.svg'))

class setPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':cog.svg'))

class noPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':no.svg'))

class editPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':edit-id.svg'))

class applyPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':apply.svg'))

class helpPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':help.svg'))

class computePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':compute.svg'))

class delPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':bin.svg'))

class browseFileButton(QPushButton):
    sigPathSelected = pyqtSignal(str)

    def __init__(self, *args, ext=None, title='Select file', start_dir=''):
        super().__init__(*args)
        self.setIcon(QIcon(':folder-open.svg'))
        self.clicked.connect(self.browse)
        self._file_types = 'All Files (*)'
        self._title = title
        self._start_dir = start_dir
        if ext is not None:
            s = ''
            s_li = []
            for name, extensions in ext.items():
                _s = ''
                for ext in extensions:
                    _s = f'{_s}*{ext} '
                s_li.append(f'{name} {_s.strip()}')

            self._file_types = ';;'.join(s_li)
            self._file_types = f'{self._file_types};;All Files (*)'

    def browse(self):
        print(self._start_dir)
        file_path = QFileDialog.getOpenFileName(
            self, self._title, self._start_dir, self._file_types
        )[0]
        if file_path:
            self.sigPathSelected.emit(file_path)

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class listWidget(QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QListWidget::item:hover {background-color:#E6E6E6;}
            QListWidget::item:selected {background-color:#CFEB9B;}
            QListWidget::item:selected {color:black;}
            QListView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
            }
        """)

class QClickableLabel(QLabel):
    clicked = pyqtSignal(object)

    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit(self)

class myMessageBox(QDialog):
    def __init__(
            self, parent=None, showCentered=True, wrapText=True,
            scrollableText=False, enlargeWidthFactor=0,
            resizeButtons=True
        ):
        super().__init__(parent)

        self.wrapText = wrapText
        self.enlargeWidthFactor = enlargeWidthFactor
        self.resizeButtons = resizeButtons

        self.cancel = True
        self.cancelButton = None
        self.okButton = None
        self.clickedButton = None

        self.showCentered = showCentered

        self.scrollableText = scrollableText

        self.layout = QGridLayout()
        self.layout.setHorizontalSpacing(20)
        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.setSpacing(2)
        self.buttons = []
        self.widgets = []
        self.layouts = []
        self.labels = []
        self.detailsTextWidget = None
        self.showInFileManagButton = None
        self.visibleDetails = False
        self.doNotShowAgainCheckbox = None

        self.currentRow = 0
        self._w = None

        self.layout.setColumnStretch(1, 1)
        self.setLayout(self.layout)

    def mousePressEvent(self, event):
        for label in self.labels:
            label.setTextInteractionFlags(
                Qt.TextBrowserInteraction | Qt.TextSelectableByKeyboard
            )

    def setIcon(self, iconName='SP_MessageBoxInformation'):
        label = QLabel(self)

        standardIcon = getattr(QStyle, iconName)
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)

        self.layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

    def addShowInFileManagerButton(self, path, txt=None):
        if txt is None:
            txt = 'Reveal in Finder...' if is_mac else 'Show in Explorer...'
        self.showInFileManagButton = showInFileManagerButton(txt)
        self.buttonsLayout.addWidget(self.showInFileManagButton)
        func = partial(myutils.showInExplorer, path)
        self.showInFileManagButton.clicked.connect(func)

    def addCancelButton(self, button=None, connect=False):
        if button is None:
            self.cancelButton = cancelPushButton('Cancel')
        else:
            self.cancelButton = button
            self.cancelButton.setIcon(QIcon(':cancelButton.svg'))

        self.buttonsLayout.insertWidget(0, self.cancelButton)
        self.buttonsLayout.insertSpacing(1, 20)
        if connect:
            self.cancelButton.clicked.connect(self.buttonCallBack)

    def addText(self, text):
        label = QLabel(self)
        label.setText(text)
        label.setWordWrap(self.wrapText)
        label.setOpenExternalLinks(True)
        self.labels.append(label)
        if self.scrollableText:
            textWidget = QScrollArea()
            textWidget.setFrameStyle(QFrame.NoFrame)
            textWidget.setWidget(label)
        else:
            textWidget = label

        self.layout.addWidget(textWidget, self.currentRow, 1)#, alignment=Qt.AlignTop)
        self.currentRow += 1
        return label

    def addButton(self, buttonText):
        if not isinstance(buttonText, str):
            # Passing button directly
            button = buttonText
            self.buttonsLayout.addWidget(button)
            button.clicked.connect(self.buttonCallBack)
            self.buttons.append(button)
            return button

        isCancelButton = (
            buttonText.lower().find('cancel') != -1
            or buttonText.lower().find('abort') != -1
        )
        isYesButton = (
            buttonText.lower().find('yes') != -1
            or buttonText.lower().find('ok') != -1
            or buttonText.lower().find('continue') != -1
            or buttonText.lower().find('recommended') != -1
        )
        isSettingsButton = buttonText.lower().find('set') != -1
        isNoButton = buttonText.replace(' ', '').lower() == 'no'
        isDelButton = buttonText.lower().find('delete') != -1

        if isCancelButton:
            button = cancelPushButton(buttonText, self)
            self.addCancelButton(button=button)
        elif isYesButton:
            button = okPushButton(buttonText, self)
            self.buttonsLayout.addWidget(button)
            self.okButton = button
        elif isSettingsButton:
            button = setPushButton(buttonText, self)
            self.buttonsLayout.addWidget(button)
        elif isNoButton:
            button = noPushButton(buttonText, self)
            self.buttonsLayout.addWidget(button)
        elif isDelButton:
            button = delPushButton(buttonText, self)
            self.buttonsLayout.addWidget(button)
        else:
            button = QPushButton(buttonText, self)
            self.buttonsLayout.addWidget(button)

        button.clicked.connect(self.buttonCallBack)
        self.buttons.append(button)
        return button

    def addDoNotShowAgainCheckbox(self, text='Do not show again'):
        self.doNotShowAgainCheckbox = QCheckBox('Do not show again')

    def addWidget(self, widget):
        self.layout.addWidget(widget, self.currentRow, 1)
        self.widgets.append(widget)
        self.currentRow += 1

    def addLayout(self, layout):
        self.layout.addLayout(layout, self.currentRow, 1)
        self.layouts.append(layout)
        self.currentRow += 1

    def setWidth(self, w):
        self._w = w

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        # spacer
        spacer = QSpacerItem(10, 10)
        self.layout.addItem(spacer, self.currentRow, 1)
        self.layout.setRowStretch(self.currentRow, 0)

        # buttons
        self.currentRow += 1

        if self.detailsTextWidget is not None:
            self.buttonsLayout.insertWidget(1, self.detailsButton)

        # Do not show again checkbox
        if self.doNotShowAgainCheckbox is not None:
            self.layout.addWidget(
                self.doNotShowAgainCheckbox, self.currentRow, 0, 1, 2
            )
            self.currentRow += 1

        self.layout.addLayout(
            self.buttonsLayout, self.currentRow, 0, 1, 2,
            alignment=Qt.AlignRight
        )

        # Details
        if self.detailsTextWidget is not None:
            self.currentRow += 1
            self.layout.addWidget(
                self.detailsTextWidget, self.currentRow, 0, 1, 2
            )

        # spacer
        self.currentRow += 1
        spacer = QSpacerItem(10, 10)
        self.layout.addItem(spacer, self.currentRow, 1)
        self.layout.setRowStretch(self.currentRow, 0)

        super().show()
        QTimer.singleShot(5, self._resize)

        if block:
            self._block()

    def setDetailedText(self, text, visible=False):
        self.detailsTextWidget = QPlainTextEdit(text)
        self.detailsTextWidget.setReadOnly(True)
        self.detailsButton = showDetailsButton()
        self.detailsButton.setCheckable(True)
        self.detailsButton.clicked.connect(self._showDetails)
        self.detailsTextWidget.hide()
        self.visibleDetails = visible

    def _showDetails(self, checked):
        if checked:
            self.origHeight = self.height()
            self.resize(self.width(), self.height()+300)
            self.detailsTextWidget.show()
        else:
            self.detailsTextWidget.hide()
            func = partial(self.resize, self.width(), self.origHeight)
            QTimer.singleShot(10, func)


    def _resize(self):
        if self.resizeButtons:
            widths = [button.width() for button in self.buttons]
            if widths:
                max_width = max(widths)
                for button in self.buttons:
                    button.setMinimumWidth(max_width)

        heights = [button.height() for button in self.buttons]
        if heights:
            max_h = max(heights)
            for button in self.buttons:
                button.setMinimumHeight(max_h)
            if self.detailsTextWidget is not None:
                self.detailsButton.setMinimumHeight(max_h)
            if self.showInFileManagButton is not None:
                self.showInFileManagButton.setMinimumHeight(max_h)

        if self._w is not None and self.width() < self._w:
            self.resize(self._w, self.height())

        if self.width() < 350:
            self.resize(350, self.height())

        if self.enlargeWidthFactor > 0:
            self.resize(int(self.width()*self.enlargeWidthFactor), self.height())

        if self.visibleDetails:
            self.detailsButton.click()

        if self.showCentered:
            screen = self.screen()
            screenWidth = screen.size().width()
            screenHeight = screen.size().height()
            screenLeft = screen.geometry().x()
            screenTop = screen.geometry().y()
            w, h = self.width(), self.height()
            left = int(screenLeft + screenWidth/2 - w/2)
            top = int(screenTop + screenHeight/2 - h/2)
            self.move(left, top)

        self._h = self.height()

        if self.okButton is not None:
            self.okButton.setFocus(True)

        if self.widgets:
            return

        if self.layouts:
            return

        # # Start resizing height every 1 ms
        # self.resizeCallsCount = 0
        # self.timer = QTimer()
        # from config import warningHandler
        # warningHandler.sigGeometryWarning.connect(self.timer.stop)
        # self.timer.timeout.connect(self._resizeHeight)
        # self.timer.start(1)

    def _resizeHeight(self):
        try:
            # Resize until a "Unable to set geometry" warning is captured
            # by copnfig.warningHandler._resizeWarningHandler or #
            # height doesn't change anymore
            self.resize(self.width(), self.height()-1)
            if self.height() == self._h or self.resizeCallsCount > 100:
                self.timer.stop()
                return

            self.resizeCallsCount += 1
            self._h = self.height()
        except Exception as e:
            # traceback.format_exc()
            self.timer.stop()

    def _template(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None
        ):
        if parent is not None:
            self.setParent(parent)
        self.setWindowTitle(title)
        self.addText(message)
        if layouts is not None:
            if myutils.is_iterable(layouts):
                for layout in layouts:
                    self.addLayout(layout)
            else:
                self.addLayout(layout)

        if widgets is not None:
            if myutils.is_iterable(widgets):
                for widget in widgets:
                    self.addWidget(widget)
            else:
                self.addWidget(widgets)

        buttons = []
        if buttonsTexts is None:
            okButton = self.addButton('  Ok  ')
            buttons.append(okButton)
        elif isinstance(buttonsTexts, str):
            button = self.addButton(buttonsTexts)
            buttons.append(button)
        else:
            for buttonText in buttonsTexts:
                button = self.addButton(buttonText)
                buttons.append(button)
        return buttons

    def critical(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None
        ):
        self.setIcon(iconName='SP_MessageBoxCritical')
        buttons = self._template(
            parent, title, message,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets
        )
        self.exec_()
        return buttons

    def information(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None
        ):
        self.setIcon(iconName='SP_MessageBoxInformation')
        buttons = self._template(
            parent, title, message,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets
        )
        self.exec_()
        return buttons

    def warning(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None
        ):
        self.setIcon(iconName='SP_MessageBoxWarning')
        buttons = self._template(
            parent, title, message,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets
        )
        self.exec_()
        return buttons

    def question(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None
        ):
        self.setIcon(iconName='SP_MessageBoxQuestion')
        buttons = self._template(
            parent, title, message,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets
        )
        self.exec_()
        return buttons

    def _block(self):
        self.loop = QEventLoop()
        self.loop.exec_()

    def exec_(self):
        self.show(block=True)

    def buttonCallBack(self, checked=True):
        self.clickedButton = self.sender()
        if self.clickedButton != self.cancelButton:
            self.cancel = False
        self.close()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

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
    sigLinkClicked = pyqtSignal(str)

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

        self.labelLeft = QClickableLabel(widget)
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

        self.labelRight = QClickableLabel(widget)
        self.labelRight.setText(labelTextRight)
        self.labelRight.setFont(font)
        self.items.append(self.labelRight)

        if addInfoButton:
            infoButton = infoPushButton(self)
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

        if addBrowseButton:
            browseButton = showInFileManagerButton(self)
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
        mostRecentPath = utils.getMostRecentPath()
        file_path = getOpenImageFileName()
        if file_path == '':
            return

        self.widget.setText(file_path)
        self.sigBrowseButtonClicked.emit(self)

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
        msg = myMessageBox(parent=self)
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

class expandCollapseButton(QPushButton):
    def __init__(self, parent=None):
        QPushButton.__init__(self, parent)
        self.setIcon(QIcon(":expand.svg"))
        self.setFlat(True)
        self.installEventFilter(self)
        self.isExpand = True
        self.clicked.connect(self.buttonClicked)

    def buttonClicked(self, checked=False):
        if self.isExpand:
            self.setIcon(QIcon(":collapse.svg"))
            self.isExpand = False
        else:
            self.setIcon(QIcon(":expand.svg"))
            self.isExpand = True

    def eventFilter(self, object, event):
        if event.type() == QEvent.HoverEnter:
            self.setFlat(False)
        elif event.type() == QEvent.HoverLeave:
            self.setFlat(True)
        return False

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

class sliderWithSpinBox(QWidget):
    sigValueChange = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args)

        layout = QGridLayout()

        title = kwargs.get('title')
        if title is not None:
            titleLabel = QLabel(self)
            titleLabel.setText(title)
            layout.addWidget(titleLabel, 0, 0, alignment=Qt.AlignLeft)

        self.slider = QSlider(Qt.Horizontal, self)
        layout.addWidget(self.slider, 1, 0)

        self.spinBox = QSpinBox(self)
        self.spinBox.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.spinBox, 1, 1)
        layout.setColumnStretch(0, 6)
        layout.setColumnStretch(1, 1)

        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.setLayout(layout)

    def setValue(self, value):
        self.slider.setValue(value)

    def setMaximum(self, max):
        self.slider.setMaximum(max)
        self.spinBox.setMaximum(max)

    def setMinimum(self, min):
        self.slider.setMinimum(min)
        self.spinBox.setMinimum(min)

    def sliderValueChanged(self, val):
        self.spinBox.valueChanged.disconnect()
        self.spinBox.setValue(val)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.sigValueChange.emit(val)

    def spinboxValueChanged(self, val):
        self.slider.valueChanged.disconnect()
        self.slider.setValue(val)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.sigValueChange.emit(val)

    def value(self):
        return self.slider.value()

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

def getOpenImageFileName():
    file_path = QFileDialog.getOpenFileName(
        self, 'Select image file', mostRecentPath,
        "Images/Videos (*.npy *.npz *.h5, *.png *.tif *.tiff *.jpg *.jpeg "
        "*.mov *.avi *.mp4)"
        ";;All Files (*)"
    )[0]
    return file_path

class Toggle(QCheckBox):
    def __init__(
        self,
        initial=None,
        width=80,
        bg_color='#b3b3b3',
        circle_color='#dddddd',
        active_color='#005ce6',
        animation_curve=QEasingCurve.InOutQuad
    ):
        QCheckBox.__init__(self)

        # self.setFixedSize(width, 28)
        self.setCursor(Qt.PointingHandCursor)

        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color
        self._original_active_color = self._active_color
        self._original_bg_color = self._bg_color
        self._disabled_active_color = utils.lighten_color(active_color)
        self._disabled_circle_color = utils.lighten_color(circle_color)
        self._disabled_bg_color = utils.lighten_color(bg_color, amount=0.5)
        self._circle_margin = 10

        self._circle_position = int(self._circle_margin/2)
        self.animation = QPropertyAnimation(self, b'circle_position', self)
        self.animation.setEasingCurve(animation_curve)
        self.animation.setDuration(200)

        self.stateChanged.connect(self.start_transition)
        self.requestedState = None

        self.installEventFilter(self)

        if initial is not None:
            self.setChecked(initial)

    def sizeHint(self):
        return QSize(45, 22)

    def eventFilter(self, object, event):
        # To get the actual position of the circle we need to wait that
        # the widget is visible before setting the state
        if event.type() == QEvent.Show and self.requestedState is not None:
            self.setChecked(self.requestedState)
        return False

    def setChecked(self, state):
        # To get the actual position of the circle we need to wait that
        # the widget is visible before setting the state
        if self.isVisible():
            self.requestedState = None
            QCheckBox.setChecked(self, state>0)
        else:
            self.requestedState = state

    def circlePos(self, state: bool):
        start = int(self._circle_margin/2)
        if state:
            if self.isVisible():
                height, width = self.height(), self.width()
            else:
                sizeHint = self.sizeHint()
                height, width = sizeHint.height(), sizeHint.width()
            circle_diameter = height-self._circle_margin
            pos = width-start-circle_diameter
        else:
            pos = start
        return pos

    @pyqtProperty(float)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def start_transition(self, state):
        self.animation.stop()
        pos = self.circlePos(state)
        self.animation.setEndValue(pos)
        self.animation.start()

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)

    def setDisabled(self, state):
        QCheckBox.setDisabled(self, state)
        self.update()

    def paintEvent(self, e):
        circle_color = (
            self._circle_color if self.isEnabled()
            else self._disabled_circle_color
        )
        active_color = (
            self._active_color if self.isEnabled()
            else self._disabled_active_color
        )
        unchecked_color = (
            self._bg_color if self.isEnabled()
            else self._disabled_bg_color
        )

        # set painter
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # set no pen
        p.setPen(Qt.NoPen)

        # draw rectangle
        rect = QRect(0, 0, self.width(), self.height())

        if not self.isChecked():
            # Draw background
            p.setBrush(QColor(unchecked_color))
            half_h = int(self.height()/2)
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), half_h, half_h
            )

            # Draw circle
            p.setBrush(QColor(circle_color))
            p.drawEllipse(
                int(self._circle_position), int(self._circle_margin/2),
                self.height()-self._circle_margin,
                self.height()-self._circle_margin
            )
        else:
            # Draw background
            p.setBrush(QColor(active_color))
            half_h = int(self.height()/2)
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), half_h, half_h
            )

            # Draw circle
            p.setBrush(QColor(circle_color))
            p.drawEllipse(
                int(self._circle_position), int(self._circle_margin/2),
                self.height()-self._circle_margin,
                self.height()-self._circle_margin
            )

        p.end()

    def setStyleSheet(self, styleSheet):
        if not self.isEnabled():
            return
        try:
            color = re.findall('color: (\w+)', styleSheet)[0]
            if color == 'none':
                self._active_color = self._original_active_color
                self._bg_color = self._original_bg_color
            else:
                self._active_color = color
                self._bg_color = color
            self.update()
        except Exception as e:
            pass

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

class HistogramLUTItem(pg.HistogramLUTItem):
    sigContextMenu = pyqtSignal(object, object)

    def __init__(self, *args, **kwargs):
        pg.HistogramLUTItem.__init__(self, *args, **kwargs)

        # Remove default ViewBox raiseContextMenu event
        self.vb.raiseContextMenu = lambda ev: ev.ignore()
        self.vb.contextMenuEvent = self.contextMenuEvent

    def contextMenuEvent(self, event):
        self.sigContextMenu.emit(self, event)

class myHistogramLUTitem(pg.HistogramLUTItem):
    sigGradientMenuEvent = pyqtSignal(object)

    def __init__(self, *args,**kwargs):
        self.cmaps = cmaps

        super().__init__(**kwargs)

        for action in self.gradient.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.gradient.menu.removeAction(HSV_action)
        self.gradient.menu.removeAction(RGB_ation)

        # Invert bw action
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        self.gradient.menu.addAction(self.invertBwAction)
        self.gradient.menu.addSeparator()

        # Contours color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Contours color: '))
        self.contoursColorButton = pg.ColorButton(color=(25,25,25))
        hbox.addWidget(self.contoursColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.gradient.menu.addAction(act)

        # Contours line weight
        contLineWeightMenu = QMenu('Contours line weight', self.gradient.menu)
        self.contLineWightActionGroup = QActionGroup(self)
        self.contLineWightActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        for w in range(1, 11):
            action = QAction(str(w))
            action.setCheckable(True)
            if w == 2:
                action.setChecked(True)
            action.lineWeight = w
            self.contLineWightActionGroup.addAction(action)
            action = contLineWeightMenu.addAction(action)
        self.gradient.menu.addMenu(contLineWeightMenu)

        self.labelsAlphaMenu = self.gradient.menu.addMenu(
            'Segm. masks overlay alpha...'
        )
        self.labelsAlphaMenu.setDisabled(True)
        hbox = QHBoxLayout()
        self.labelsAlphaSlider = sliderWithSpinBox(
            title='Alpha', title_loc='in_line', is_float=True,
            normalize=True
        )
        self.labelsAlphaSlider.setMaximum(100)
        self.labelsAlphaSlider.setValue(0.3)
        hbox.addWidget(self.labelsAlphaSlider)
        shortCutText = 'Command+Up/Down' if is_mac else 'Ctrl+Up/Down'
        hbox.addWidget(QLabel(f'({shortCutText})'))
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.labelsAlphaMenu.addSeparator()
        self.labelsAlphaMenu.addAction(act)

        # Default settings
        self.defaultSettingsAction = QAction('Restore default settings...', self)
        self.gradient.menu.addAction(self.defaultSettingsAction)

        # Select channels section
        self.gradient.menu.addSeparator()
        self.gradient.menu.addSection('Select channel: ')

        # hide histogram tool
        self.vb.hide()

    def uncheckContLineWeightActions(self):
        for act in self.contLineWightActionGroup.actions():
            act.toggled.disconnect()
            act.setChecked(False)

    def restoreState(self, df):
        if 'contLineColor' in df.index:
            rgba_str = df.at['contLineColor', 'value']
            rgb = utils.rgba_str_to_values(rgba_str)[:3]
            self.contoursColorButton.setColor(rgb)

        if 'contLineWeight' in df.index:
            w = df.at['contLineWeight', 'value']
            w = int(w)
            for action in self.contLineWightActionGroup.actions():
                if action.lineWeight == w:
                    action.setChecked(True)
                    break

        if 'overlaySegmMasksAlpha' in df.index:
            alpha = df.at['overlaySegmMasksAlpha', 'value']
            self.labelsAlphaSlider.setValue(float(alpha))

        checked = df.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

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

class QProgressBarWithETA(QProgressBar):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)

        palette = QPalette()
        palette.setColor(QPalette.Highlight, QColor(207, 235, 155))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)
        self.ETA_label = QLabel('NDh:NDm:NDs')
        self.last_time_update = time.perf_counter()

    def update(self, step):
        t = time.perf_counter()
        self.setValue(self.value()+step)
        elpased_seconds = (t - self.last_time_update)/step
        steps_left = self.maximum() - self.value()
        seconds_left = elpased_seconds*steps_left
        ETA = utils.seconds_to_ETA(seconds_left)
        self.ETA_label.setText(ETA)
        self.last_time_update = t
        return ETA

    def show(self):
        QProgressBar.show(self)
        self.ETA_label.show()

    def hide(self):
        QProgressBar.hide(self)
        self.ETA_label.hide()

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
