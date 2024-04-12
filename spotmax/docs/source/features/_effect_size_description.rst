.. important:: 

    Effect sizes in spotMAX are **calculated from the center z-slice** where 
    each spot was detected. The intensity data used is after gaussian filter 
    and sharpening. If sharpening is deactivated, then spotMAX will use the 
    gaussian filtered data and, if gaussian filter is deactivate as well, 
    the intensity data is the raw data.

    Additionally, if :confval:`Optimise detection for high spot density` is 
    ``True``