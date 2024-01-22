.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/index.html

Preliminary steps
-----------------

The first step with every new dataset is to segment the objects. These typically 
are the single cells, but it can be any object where you want to detect spots. 

While you can detect spots in the entire image, it is highly recommended to 
identify region of interests (ROIs) and segment them.

The dataset provided with this tutorial already contains the segmentation files with 
the ROIs of the single cells, but if you need to segment ROIs we recommend using 
out other software called `Cell-ACDC`_.