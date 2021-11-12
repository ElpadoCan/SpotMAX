import sys
import time
import re

from PyQt5.QtCore import (
    pyqtSignal, QTimer, Qt, QPoint, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
    QSize, QRectF, QPointF, QRect, QPoint, QEasingCurve, QRegExp
)
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QPen, QPaintEvent, QBrush, QPainter,
    QRegExpValidator
)
from PyQt5.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QApplication, QWidget, QVBoxLayout, QMainWindow, QStyleFactory,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QDockWidget
)

import pyqtgraph as pg

import utils, dialogs

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
        font.setPointSize(10)
        self.setFont(font)
        self.setText('0')

        self.textChanged.connect(self.emitValueChanged)

    def value(self):
        return int(self.text())

    def emitValueChanged(self, text):
        self.valueChanged.emit(self.value())

class floatLineEdit(QLineEdit):
    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)

        self.isNumericRegExp = (
            '((?!0)|[-+]|(?=0+\.))(\d*\.)?\d+(e[-+]\d+)?'
        )

        regExp = QRegExp(self.isNumericRegExp)
        self.setValidator(QRegExpValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
        self.setText('0.0')

        self.textChanged.connect(self.emitValueChanged)

    def value(self):
        m = re.match(self.isNumericRegExp, self.text())
        if m is not None:
            text = m.group(0)
            return float(text)
        else:
            return 0.0

    def emitValueChanged(self, text):
        self.valueChanged.emit(self.value())

class Toggle(QCheckBox):
    def __init__(
        self,
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

    def sizeHint(self):
        return QSize(45, 22)

    @pyqtProperty(float)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def start_transition(self, state):
        self.animation.stop()
        start = int(self._circle_margin/2)
        if state:
            circle_diameter = self.height()-self._circle_margin
            w = self.width()
            end = w-start-circle_diameter
            self.animation.setEndValue(end)
        else:
            self.animation.setEndValue(start)
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
        self.isCheckable = False
        self.countClicks = 0

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        self.event = event
        self.countClicks += 1
        if self.countClicks == 1:
            QTimer.singleShot(250, self.checkDoubleClick)

    def setCheckable(self, isCheckable):
        self.isCheckable = isCheckable
        QToolButton.setCheckable(self, isCheckable)

    def checkDoubleClick(self):
        if self.countClicks == 2:
            self.countClicks = 0
            self.isDoubleClick = True
            # dblclick checks button only if checkable and originally unchecked
            if self.isCheckable and not self.isChecked():
                self.setChecked(True)
            self.sigDoubleClickEvent.emit(self, self.event)
        else:
            self.countClicks = 0
            self.isDoubleClick = False
            if self.isCheckable:
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

    def __init__(self, *args, **kwargs):
        pg.ScatterPlotItem.__init__(self, *args, **kwargs)
        self.clickedSpot = (-1, -1)

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


class QLogConsole(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont()
        font.setPointSize(9)
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

            container = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(toggle_1)

            le = intLineEdit()
            layout.addWidget(le)

            le.valueChanged.connect(self.lineEditValueChanged)

            le = floatLineEdit()
            layout.addWidget(le)

            le.valueChanged.connect(self.lineEditValueChanged)

            slider = sliderWithSpinBox(title='test slider')

            layout.addWidget(slider)


            layout.addStretch(1)
            container.setLayout(layout)

            computeDockWidget = QDockWidget('spotMAX analysis inputs', self)
            frame = dialogs.analysisInputsQFrame(computeDockWidget)
            computeDockWidget.setWidget(frame)
            self.addDockWidget(Qt.LeftDockWidgetArea, computeDockWidget)

            self.setCentralWidget(container)

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
