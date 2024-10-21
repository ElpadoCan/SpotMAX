.. important:: 

    Effect sizes in SpotMAX are **calculated from the center z-slice** where 
    each spot was detected. The intensity data used is after gaussian filter 
    and sharpening. If sharpening is deactivated, then SpotMAX will use the 
    gaussian filtered data and, if gaussian filter is deactivate as well, 
    the intensity data is the raw data.

    Additionally, if the parameter :confval:`Optimise detection for high spot density` 
    is ``True``, the spot intensities are normalized by the euclidean distance 
    transform.