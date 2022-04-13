import sys
import time
import re

from pprint import pprint

from functools import partial

from PyQt5.QtCore import (
    pyqtSignal, QTimer, Qt, QPoint, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
    QSize, QRectF, QPointF, QRect, QPoint, QEasingCurve, QRegExp,
    QEvent
)
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QPen, QPaintEvent, QBrush, QPainter,
    QRegExpValidator, QIcon
)
from PyQt5.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QApplication, QWidget, QVBoxLayout, QMainWindow, QStyleFactory,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QDockWidget,
    QScrollArea, QSizePolicy, QComboBox, QPushButton, QScrollBar,
    QGroupBox, QAbstractSlider, QDialog, QStyle, QSpacerItem
)

import pyqtgraph as pg

from . import utils, dialogs, is_mac, is_win

# NOTE: Enable icons
from . import qrc_resources

class myMessageBox(QDialog):
    def __init__(self, parent=None, showCentered=True):
        super().__init__(parent)

        self.cancel = True
        self.cancelButton = None

        self.showCentered = showCentered

        self.detailsWidget = None

        self.layout = QGridLayout()
        self.layout.setHorizontalSpacing(20)
        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.setSpacing(2)
        self.buttons = []

        self.currentRow = 0
        self._w = None

        self.layout.setColumnStretch(1, 1)
        self.setLayout(self.layout)

    def setIcon(self, iconName='SP_MessageBoxInformation'):
        label = QLabel(self)

        standardIcon = getattr(QStyle, iconName)
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)

        self.layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

    def addShowInFileManagerButton(self, path, txt=None):
        if txt is None:
            txt = 'Reveal in Finder' if is_mac else 'Show in Explorer'
        self.showInFileManagButton = QPushButton(txt)
        self.buttonsLayout.addWidget(self.showInFileManagButton)
        func = partial(utils.showInExplorer, path)
        self.showInFileManagButton.clicked.connect(func)

    def addCancelButton(self):
        self.cancelButton = QPushButton('Cancel', self)
        self.buttonsLayout.insertWidget(0, self.cancelButton)
        self.buttonsLayout.insertSpacing(1, 20)

    def addText(self, text):
        label = QLabel(self)
        label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        )
        label.setText(text)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        self.layout.addWidget(label, self.currentRow, 1)#, alignment=Qt.AlignTop)
        self.currentRow += 1
        return label

    def showDetails(self, checked):
        if checked:
            self.showDetailsButton.setText('Hide details')
            self.detailsWidget.show()
        else:
            self.showDetailsButton.setText('Show details...')
            self.detailsWidget.hide()
            QTimer.singleShot(50, self._resize)

    def _resize(self):
        self.resize(self.width(), self._h)

    def setDetailedText(self, text):
        self.showDetailsButton = QPushButton('Show details...', self)
        self.showDetailsButton.setCheckable(True)
        self.showDetailsButton.clicked.connect(self.showDetails)
        self.buttonsLayout.addWidget(self.showDetailsButton)
        self.detailsWidget = QTextEdit()
        self.detailsWidget.setReadOnly(True)
        self.detailsWidget.setText(text)
        self.detailsWidget.hide()

    def addButton(self, buttonText):
        button = QPushButton(buttonText, self)
        if buttonText.find('Cancel') != -1:
            self.cancelButton = button
            self.buttonsLayout.insertWidget(0, button)
            self.buttonsLayout.insertSpacing(1, 20)
        else:
            self.buttonsLayout.addWidget(button)
        button.clicked.connect(self.close)
        self.buttons.append(button)
        return button

    def addWidget(self, widget):
        self.layout.addWidget(widget, self.currentRow, 1)
        self.currentRow += 1

    def addLayout(self, layout):
        self.layout.addLayout(layout, self.currentRow, 1)
        self.currentRow += 1

    def setWidth(self, w):
        self._w = w

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)

        # spacer
        self.currentRow += 1
        spacer = QSpacerItem(10, 10)
        self.layout.addItem(spacer, self.currentRow, 1)

        # buttons
        self.currentRow += 1
        self.layout.addLayout(
            self.buttonsLayout, self.currentRow, 0, 1, 2,
            alignment=Qt.AlignRight
        )

        # spacer
        self.currentRow += 1
        spacer = QSpacerItem(10, 10)
        self.layout.addItem(spacer, self.currentRow, 1)

        # Add stretch after buttons
        self.currentRow += 1
        self.layout.setRowStretch(self.currentRow, 1)

        if self.detailsWidget is not None:
            self.currentRow += 1
            self.layout.addWidget(
                self.detailsWidget, self.currentRow, 0, 1, 2
            )

        super().show()
        widths = [button.width() for button in self.buttons]
        if widths:
            max_width = max(widths)
            for button in self.buttons:
                button.setMinimumWidth(max_width)

        if self.width() < 350:
            self.resize(350, self.height())

        if self._w is not None:
            self.resize(self._w, self.height())

        if self.showCentered:
            screen = self.screen()
            screenWidth = screen.size().width()
            screenHeight = screen.size().height()
            w, h = self.width(), self.height()
            left = int(screenWidth/2 - w/2)
            top = int(screenHeight/2 - h/2)
            self.move(left, top)

        self._h = self.height()

        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    # def keyPressEvent(self, event):
    #     print(self.height(), self._h)

    def _template(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            detailedText=None, showPath=None
        ):
        if parent is not None:
            self.setParent(parent)
        self.setWindowTitle(title)
        self.addText(message)
        if layouts is not None:
            if utils.is_iterable(layouts):
                for layout in layouts:
                    self.addLayout(layout)
            else:
                self.addLayout(layout)

        if widgets is not None:
            if utils.is_iterable(widgets):
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

        if showPath is not None:
            path, txt = showPath
            self.addShowInFileManagerButton(path, txt=txt)

        if detailedText is not None:
            self.setDetailedText(detailedText)

        return buttons

    def critical(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            detailedText=None, showPath=None
        ):
        self.setIcon(iconName='SP_MessageBoxCritical')
        buttons = self._template(
            parent, title, message,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets,
            detailedText=detailedText, showPath=showPath
        )
        self.exec_()
        return buttons

    def information(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            detailedText=None, showPath=None
        ):
        self.setIcon(iconName='SP_MessageBoxInformation')
        buttons = self._template(
            parent, title, message,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets,
            detailedText=detailedText, showPath=showPath
        )
        self.exec_()
        return buttons

    def warning(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            detailedText=None, showPath=None
        ):
        self.setIcon(iconName='SP_MessageBoxWarning')
        buttons = self._template(
            parent, title, message,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets,
            detailedText=detailedText, showPath=showPath
        )
        self.exec_()
        return buttons

    def question(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            detailedText=None, showPath=None
        ):
        self.setIcon(iconName='SP_MessageBoxQuestion')
        buttons = self._template(
            parent, title, message,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets,
            detailedText=detailedText, showPath=showPath
        )
        self.exec_()
        return buttons

    def exec_(self):
        self.show()
        super().exec_()

    def close(self):
        self.clickedButton = self.sender()
        if self.clickedButton is not None:
            self.cancel = self.clickedButton == self.cancelButton
        super().close()
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

class formWidget(QWidget):
    sigApplyButtonClicked = pyqtSignal(object)
    sigComputeButtonClicked = pyqtSignal(object)

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
            key='',
            parent=None
        ):
        QWidget.__init__(self, parent)
        self.widget = widget
        self.anchor = anchor
        self.key = key

        widget.setParent(self)

        if anchor:
            self.kwargs = utils.analysisInputsParams()[anchor]
            labelTextLeft = self.kwargs['desc']
            if initialVal is None:
                initialVal = self.kwargs['initialVal']

        if isinstance(initialVal, bool):
            widget.setChecked(initialVal)
        elif isinstance(initialVal, str):
            widget.setCurrentText(initialVal)
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
            infoButton = QPushButton(self)
            infoButton.setCursor(Qt.WhatsThisCursor)
            infoButton.setIcon(QIcon(":info.svg"))
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

        if addApplyButton:
            applyButton = QPushButton(self)
            applyButton.setCursor(Qt.PointingHandCursor)
            applyButton.setCheckable(True)
            applyButton.setIcon(QIcon(":apply.svg"))
            applyButton.setToolTip(f'Apply this step and visualize results')
            applyButton.clicked.connect(self.applyButtonClicked)
            self.items.append(applyButton)

        if addComputeButton:
            computeButton = QPushButton(self)
            computeButton.setCursor(Qt.BusyCursor)
            computeButton.setIcon(QIcon(":compute.svg"))
            computeButton.setToolTip(f'Compute this step and visualize results')
            computeButton.clicked.connect(self.computeButtonClicked)
            self.items.append(computeButton)

        self.labelLeft.clicked.connect(self.tryChecking)
        self.labelRight.clicked.connect(self.tryChecking)

    def tryChecking(self, label):
        try:
            self.widget.setChecked(not self.widget.isChecked())
        except AttributeError as e:
            pass

    def applyButtonClicked(self):
        self.sigApplyButtonClicked.emit(self)

    def computeButtonClicked(self):
        self.sigComputeButtonClicked.emit(self)

    def showInfo(self):
        anchor = self.anchor
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
                alignment = Qt.AlignCenter
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
            '^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
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

class Toggle(QCheckBox):
    def __init__(
        self,
        initial=None,
        width=80,
        bg_color='#b3b3b3',
        circle_color='#DDD',
        active_color='#005ce6',
        animation_curve=QEasingCurve.InOutQuad
    ):
        QCheckBox.__init__(self)

        # self.setFixedSize(width, 28)
        self.setCursor(Qt.PointingHandCursor)

        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color
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

    def paintEvent(self, e):
        # set painter
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # set no pen
        p.setPen(Qt.NoPen)

        # draw rectangle
        rect = QRect(0, 0, self.width(), self.height())

        if not self.isChecked():
            # Draw background
            p.setBrush(QColor(self._bg_color))
            half_h = int(self.height()/2)
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), half_h, half_h
            )

            # Draw circle
            p.setBrush(QColor(self._circle_color))
            p.drawEllipse(
                int(self._circle_position), int(self._circle_margin/2),
                self.height()-self._circle_margin,
                self.height()-self._circle_margin
            )
        else:
            # Draw background
            p.setBrush(QColor(self._active_color))
            half_h = int(self.height()/2)
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), half_h, half_h
            )

            # Draw circle
            p.setBrush(QColor(self._circle_color))
            p.drawEllipse(
                int(self._circle_position), int(self._circle_margin/2),
                self.height()-self._circle_margin,
                self.height()-self._circle_margin
            )

        p.end()

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

class QClickableLabel(QLabel):
    clicked = pyqtSignal(object)

    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit(self)

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

if __name__ == '__main__':
    class Window(QMainWindow):
        def __init__(self):
            super().__init__()

            toggle_1 = Toggle()
            toggle_1.setChecked(True)

            container = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(toggle_1)

            le = intLineEdit()
            layout.addWidget(le)

            le.valueChanged.connect(self.lineEditValueChanged)

            le = floatLineEdit(notAllowed=[0])
            layout.addWidget(le)

            le.valueChanged.connect(self.lineEditValueChanged)

            slider = sliderWithSpinBox(title='test slider')
            layout.addWidget(slider)

            ComboBox = myQComboBox()
            ComboBox.addItems(['ciao', 'test'])
            layout.addWidget(ComboBox)

            plotSpotsCoordsButton = DblClickQToolButton(self)
            plotSpotsCoordsButton.setIcon(QIcon(":plotSpots.svg"))
            plotSpotsCoordsButton.setCheckable(True)
            layout.addWidget(plotSpotsCoordsButton)

            print(plotSpotsCoordsButton.isCheckable())

            layout.addWidget(QSpinBoxOdd())
            scrollbar = QScrollBar(Qt.Horizontal)
            scrollbar.actionTriggered.connect(self.scrollbarAction)
            layout.addWidget(scrollbar)
            self.scrollbar = scrollbar


            layout.addStretch(1)
            container.setLayout(layout)
            self.setCentralWidget(container)

            self.setFocus()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Right:
                pos = self.scrollbar.sliderPosition()
                self.scrollbar.setSliderPosition(pos+1)
            elif event.key() == Qt.Key_Left:
                pos = self.scrollbar.sliderPosition()
                self.scrollbar.setSliderPosition(pos-1)

        def scrollbarAction(self, action):
            print(action)
            print(action == QAbstractSlider.SliderMove)
            print(action == QAbstractSlider.SliderPageStepAdd)
            print(action == QAbstractSlider.SliderPageStepSub)

        def lineEditValueChanged(self, value):
            print(value)
            print(self.sender().value())

        def show(self):
            QMainWindow.show(self)

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    w = Window()
    w.show()
    app.exec_()
