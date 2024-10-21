.. _skimage.feature.peak_local_max: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max
.. _skimage.measure.label: _https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label

.. _spot_detect:

Spot detection
==============

To detect the centers of the spots, SpotMAX will start from the semantic 
segmentation mask from previous step (see :ref:`spot_pred`) and will run 
one of these two strategies:

1. Local peaks detection 
2. Connected-component labeling

For the "Local peak detection", SpotMAX will use the function from the 
Scikit-image library `skimage.feature.peak_local_max`_. 

For the "Connected-component labeling", SpotMAX will use the function from the 
Scikit-image library `skimage.measure.label`. The center of the spots will 
then be calculated as the centroid of the connected objects. 