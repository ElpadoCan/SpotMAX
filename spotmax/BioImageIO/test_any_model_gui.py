import os

import numpy as np

from cellacdc.plot import imshow
from cellacdc._run import _setup_app
from spotmax.nnet.model import Model


def main():
    app, splashScreen = _setup_app(splashscreen=True)  
    splashScreen.close()
    
    

if __name__ == '__main__':
    main()