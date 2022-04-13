import pathlib
import os
import sys
import numpy as np
import traceback
from pprint import pprint
import time
from functools import wraps

from natsort import natsorted
from natsort import natsort_keygen

from collections import OrderedDict

import pandas as pd

import h5py

import utils, html

from PyQt5.QtCore import (
    Qt, pyqtSignal, QMutex, QWaitCondition, QObject,
    QThread
)

from PyQt5.QtWidgets import (
    QApplication, QMessageBox, QStyleFactory, QScrollBar,
    QMainWindow, QWidget, QVBoxLayout, QAbstractSlider,
    QHBoxLayout, QLabel, QPushButton
)

from queue import Queue

h5_path = r"G:\My Drive\1_MIA_Data\Maria-Elena\test0\TPR-GFP_live\exp2\Position_1\Images\Tpr-GFP-Emd-RFP_2021-10-08_14.19.21_F0_s1_GFP.h5"
h5f = h5py.File(h5_path, 'r')
dset = h5f['data']
T, Z, Y, X = dset.shape


def exception_handler(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                func(self, *args)
            else:
                func(self, *args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return inner_function
    return inner_function

class loadDataWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    sigLoadingDone = pyqtSignal()

    def __init__(self, mutex, waitCond):
        QObject.__init__(self)
        self.mutex = mutex
        self.waitCond = waitCond
        self.wait = True
        self.exit = False

    def setArgs(self, frame_i):
        self.abortLoading = False
        self.wait = False
        self.frame_i = frame_i

    def pause(self):
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()

    def readData(self):
        arr = np.empty((2, Z, Y, X), dtype=dset.dtype)
        i = 0
        for t0, t in enumerate(range(self.frame_i, self.frame_i+2)):
            for z in range(Z):
                if self.abortLoading:
                    self.progress.emit(
                        f'Loading frame {self.frame_i} ABORTED at i = {i}'
                    )
                    break
                else:
                    arr[t0, z] = dset[t, z]
                    i += 1
            if self.abortLoading:
                break
        return arr, self.abortLoading

    def run(self):
        while True:
            if self.exit:
                break
            elif self.wait:
                self.progress.emit('Thread paused')
                self.pause()
            else:
                self.progress.emit(f'Loading frame {self.frame_i}...')
                # loading data here
                arr, aborted = self.readData()
                if not aborted:
                    self.progress.emit(
                        f'Loading frame {self.frame_i} done. '
                        f'Loaded array memory size = {sys.getsizeof(arr)*1E-6} MB'
                    )

                self.sigLoadingDone.emit()
                self.wait = True

        self.progress.emit('Thread closed')
        self.finished.emit()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        container = QWidget()
        layout = QHBoxLayout()

        startButton = QPushButton('Start')
        stopButton = QPushButton('Stop')
        infoButton = QPushButton('Info')

        layout.addWidget(startButton)
        layout.addWidget(stopButton)
        layout.addWidget(infoButton)

        container.setLayout(layout)
        self.setCentralWidget(container)

        startButton.clicked.connect(self.startClicked)
        stopButton.clicked.connect(self.stopClicked)
        infoButton.clicked.connect(self.infoClicked)

        self.frame_i = 0

        self.setupWorker()

    def setupWorker(self):
        self.q = Queue()
        self.threadCount = 0
        self.thread = QThread()
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()

        self.worker = loadDataWorker(self.mutex, self.waitCond)
        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.workerProgress)
        self.worker.sigLoadingDone.connect(self.loadingDataDone)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def workerProgress(self, text):
        print(text)

    def startClicked(self):
        self.startWorker()

    def stopClicked(self):
        self.abortWorker()

    def infoClicked(self):
        print(f'Number of running threads = {self.threadCount}')

    def abortWorker(self):
        if self.threadCount > 0:
            self.worker.abortLoading = True

    @exception_handler
    def startWorker(self):
        if self.threadCount == 0:
            self.threadCount += 1
            self.worker.setArgs(self.frame_i)
            self.waitCond.wakeAll()
        else:
            print(f'Enqueing frame index {self.frame_i}')
            self.q.put((self.frame_i, ))
            self.abortWorker()
        self.frame_i += 1

    def loadingDataDone(self):
        self.threadCount -= 1
        if not self.q.empty():
            args = self.q.get()
            self.frame_i, = args
            print(f'Dequeing frame index {self.frame_i}')
            self.startWorker()

    def closeEvent(self, event):
        self.worker.exit = True
        if self.threadCount == 0:
            self.waitCond.wakeAll()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    win = Window()
    win.show()

    sys.exit(app.exec_())
