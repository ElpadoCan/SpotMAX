import pathlib
import os
import sys
import numpy as np
import traceback
from pprint import pprint

from natsort import natsorted
from natsort import natsort_keygen

from collections import OrderedDict

import pandas as pd

import h5py

import utils, html

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
    QApplication, QMessageBox, QStyleFactory, QScrollBar,
    QMainWindow, QWidget, QVBoxLayout, QAbstractSlider,
    QHBoxLayout, QLabel
)

from queue import Queue

def shiftWindow_axis0(
        dset, window_arr, windowSize, t0_window, current_t, axis1_range=None
    ):
    try:
        t1_window = t0_window + windowSize - 1
        halfWindowSize = int(windowSize/2)

        t0_chunk = t1_window + 1
        chunkSizeT = current_t + halfWindowSize - t0_chunk + 1

        print('=========================')
        print(current_t, t0_chunk, chunkSizeT)
        print(window_arr)

        rightBoundary = dset.shape[0]-halfWindowSize
        leftBoundary = halfWindowSize
        if current_t <= halfWindowSize:
            print('Left boundary')
            if leftBoundary < t0_window:
                direction = 'new'
                current_t = leftBoundary + 1
            else:
                return window_arr, t0_window
        elif current_t >= rightBoundary:
            print('Right boundary')
            if rightBoundary > t1_window:
                direction = 'new'
                current_t = rightBoundary
            else:
                return window_arr, t0_window

        if abs(chunkSizeT) >= windowSize:
            direction = 'new'
        elif chunkSizeT <= 0:
            direction = 'backward'
        else:
            direction = 'forward'

        if direction == 'new':
            t0_chunk = current_t - halfWindowSize - 1
            t1_chunk = t0_chunk + windowSize
            if axis1_range is None:
                window_arr = dset[t0_chunk:t1_chunk]
            else:
                axis1_c0, axis1_c1 = axis1_range
                window_arr = dset[t0_chunk:t1_chunk, axis1_c0:axis1_c1]
            t0_window = t0_chunk
            print('New window')
            print(t0_chunk)
            print(window_arr)

            # Test
            arr = np.arange(t0_chunk, t1_chunk)
            print(arr)
            print((window_arr == arr).all())
            print('=========================')
            return window_arr, t0_window

        print('-------------------------')
        window_arr = np.roll(window_arr, -chunkSizeT, axis=0)

        print(window_arr)

        if direction == 'forward':

            if axis1_range is None:
                chunk = dset[t0_chunk:t0_chunk+chunkSizeT]
            else:
                axis1_c0, axis1_c1 = axis1_range
                chunk = dset[t0_chunk:t0_chunk+chunkSizeT, axis1_c0:axis1_c1]

            window_arr[-chunkSizeT:] = chunk
            t0_window += chunkSizeT

        elif direction == 'backward':
            t0_chunk = t0_window + chunkSizeT
            chunkSizeT = abs(chunkSizeT)
            if axis1_range is None:
                chunk = dset[t0_chunk:t0_chunk+chunkSizeT]
            else:
                axis1_c0, axis1_c1 = axis1_range
                chunk = dset[t0_chunk:t0_chunk+chunkSizeT, axis1_c0:axis1_c1]
            window_arr[:chunkSizeT] = chunk
            t0_window = t0_chunk


        print(chunk)
        print(window_arr)

        # Test
        arr = np.arange(current_t-halfWindowSize, current_t+halfWindowSize+1)
        if axis1_range is not None:
            axis1_c0, axis1_c1 = axis1_range
            SizeZ = axis1_c1-axis1_c0
            arr = np.column_stack([arr]*SizeZ)
        print(arr)
        print((window_arr == arr).all())

        print('=========================')

    except Exception as e:
        traceback.print_exc()

    return window_arr, t0_window

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.windowSize = 11
        self.t0_window = 0

        container = QWidget()
        layout = QHBoxLayout()

        self.SizeZ = 8
        left_range, right_range = 0, 60
        dset = np.arange(left_range, right_range)
        self.dset = np.column_stack([dset]*self.SizeZ)
        self.SizeT = right_range
        scrollbar = QScrollBar(Qt.Horizontal)
        scrollbar.setMaximum(right_range-1)
        scrollbar.actionTriggered.connect(self.action_cb)
        scrollbar.sliderReleased.connect(self.released_cb)
        scrollbar.setSliderPosition(self.t0_window)
        self.scrollbar = scrollbar
        layout.addWidget(scrollbar)

        self.label = QLabel()
        layout.addWidget(self.label)
        self.label.setText(f'{self.t0_window}')

        self.window_arr = self.dset[self.t0_window:self.t0_window+self.windowSize]

        layout.setStretch(0,1)
        container.setLayout(layout)
        self.setCentralWidget(container)

    def released_cb(self):
        current_t = self.scrollbar.sliderPosition()
        self.label.setText(f'{current_t}')

        self.window_arr, self.t0_window = shiftWindow_axis0(
            self.dset, self.window_arr, self.windowSize, self.t0_window,
            current_t
        )

    def action_cb(self, action):
        current_t = self.scrollbar.sliderPosition()
        self.label.setText(f'{current_t}')
        if action == QAbstractSlider.SliderMove:
            t1_window = self.t0_window + self.windowSize - 1
            if current_t <= t1_window:
                print('Updating')
            else:
                print('Off-window')
            return

        self.window_arr, self.t0_window = shiftWindow_axis0(
            self.dset, self.window_arr, self.windowSize, self.t0_window,
            current_t, axis1_range=(0, self.SizeZ)
        )

app = QApplication([])
app.setStyle(QStyleFactory.create('Fusion'))
win = Window()
win.show()
win.resize(win.width()*3, win.height())
win.scrollbar.setFixedHeight(win.scrollbar.height()*3)
app.exec_()
