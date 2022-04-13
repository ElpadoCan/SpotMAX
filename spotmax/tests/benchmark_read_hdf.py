import pathlib
import os
import sys
import numpy as np
import traceback
from pprint import pprint
import time

from natsort import natsorted
from natsort import natsort_keygen

from collections import OrderedDict

import pandas as pd

import h5py

from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
    QApplication, QMessageBox, QStyleFactory, QScrollBar,
    QMainWindow, QWidget, QVBoxLayout, QAbstractSlider,
    QHBoxLayout, QLabel
)

from queue import Queue

h5_path = r"G:\My Drive\1_MIA_Data\Maria-Elena\test0\TPR-GFP_live\exp2\Position_1\Images\Tpr-GFP-Emd-RFP_2021-10-08_14.19.21_F0_s1_GFP.h5"

h5f = h5py.File(h5_path, 'r')

dset = h5f['data']

T, Z, Y, X = dset.shape

t0 = time.perf_counter()
arr = dset[0, 0]
t1 = time.perf_counter()

print(f'Time to read 2D = {(t1-t0)*1000:.3f} ms')

t0 = time.perf_counter()
arr = dset[1]
t1 = time.perf_counter()

print(f'Time to read 3D = {(t1-t0)*1000:.3f} ms')

t0 = time.perf_counter()
arr = dset[3:5]
t1 = time.perf_counter()

print(f'Time to read 4D = {(t1-t0)*1000:.3f} ms')

t_range = range(7,9)
t0 = time.perf_counter()
arr = np.empty((2, Z, Y, X), dtype=dset.dtype)
num_iter = 0
for t0, t in enumerate(t_range):
    for z in range(Z):
        arr[t0, z] = dset[t, z]
        num_iter += 1
t1 = time.perf_counter()

print(f'Time to read 4D in for loop ({num_iter} iterations) = {(t1-t0)*1000:.3f} ms')
