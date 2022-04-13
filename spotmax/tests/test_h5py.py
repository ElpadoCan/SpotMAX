import pathlib
import os
import sys
import numpy as np
from pprint import pprint

from natsort import natsorted
from natsort import natsort_keygen

from collections import OrderedDict

import pandas as pd

import h5py

Z, Y, X = 40,1024,1024
T = 40
shape = (T,Z, Y, X)

h5f = h5py.File('test.h5', 'w')

print(sys.getsizeof(h5f))


dset = h5f.create_dataset('test', shape, dtype='float')

print(dset.dtype)
print(sys.getsizeof(dset))

for i in range(shape[0]):
    z_stack = np.random.random(size=(Z, Y, X))
    dset[i] = z_stack
    print(sys.getsizeof(dset), sys.getsizeof(z_stack)*1E-6)

h5f.close()

print('Reading....')

h5f = h5py.File('test.h5', 'r')

print(sys.getsizeof(h5f))

dset = h5f['test']

for i in range(shape[0]):
    z_stack = dset[i]
    print(sys.getsizeof(dset), sys.getsizeof(z_stack)*1E-6)

h5f.close()
