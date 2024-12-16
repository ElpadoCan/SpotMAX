try:
    import pytest
    pytest.skip('skipping this test', allow_module_level=True)
except Exception as e:
    pass

import os

import numpy as np

import qtpy.compat

from cellacdc.plot import imshow
import cellacdc.myutils as acdc_myutils
from cellacdc._run import _setup_app

from spotmax import dialogs, io
from spotmax.BioImageIO.model import Model


def main():
    app, splashScreen = _setup_app(splashscreen=True)  
    splashScreen.close()
    
    image_filepath, _ = qtpy.compat.getopenfilename(
        basedir=acdc_myutils.getMostRecentPath(),
        filters=('Images (*.tif *.npz *.npy);;All Files (*.)')
    )
    image = io.load_image_data(image_filepath, to_float=True)
    
    win = dialogs.QDialogBioimageIOModelParams()
    win.exec_()
    if win.cancel:
        exit('Execution stopped by the user.')
    
    model = Model(**win.init_kwargs)
    model.set_kwargs(win.additionalKwargs)
    mask = model.segment(image, **win.model_kwargs)
    
    imshow(image, mask)

if __name__ == '__main__':
    main()