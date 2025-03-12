.. _BioImage.IO Model Zoo: https://bioimage.io/#/
.. _Thresholding (scikit-image): https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html
.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/
.. _Spotiflow: https://github.com/weigertlab/spotiflow

.. _spot_pred:

Spot prediction
===============

This is the step performed before detecting the centers of the spots and the  
is a **semantic segmentation masks of the areas were SpotMAX will later 
look for spots**. 

This means that if an area in the image is marked as background in this step, 
spots will not be searched and detected there.

Segmentation of the spots image is performed in two steps:

1. :ref:`spots_ch_preproc`
2. :ref:`spots_sem_segm`

.. _spots_ch_preproc:

Image pre-processing
~~~~~~~~~~~~~~~~~~~~

SpotMAX provides two filters for image pre-processing of the spots signal: **blurring** 
(gaussian filter) and a **spot detection filter** 
(i.e., Difference of Gaussians). 

The gaussian filter can be applied with a different sigma for each dimension, 
which is very useful when working with anisotropic 3D z-stack data. 

The **spot detection filter** is a Difference of Gaussians filter whose parameters 
are automatically calculated from the expected spot size. See the 
description of the parameter :confval:`Sharpen spots signal prior detection` for 
more details.

.. _spots_sem_segm:

Semantic segmentation
~~~~~~~~~~~~~~~~~~~~~

The pre-processed image is used as input to the segmentation model of your 
choice. SpotMAX will segment the reference channel by using automatic 
thresholding, SpotMAX AI models, `Spotiflow`_, or any of the models available 
on the `BioImage.IO Model Zoo`_. 

For more details about the available methods for automatic thresholding see 
this guide `Thresholding (scikit-image)`. 

One crucial aspect of SpotMAX is that you can apply the segmentation model 
on each input segmented object (e.g., the single cells, a.k.a. "Local") or on
all the objects in the image (a.k.a. "Aggregated"). See the parameter 
:confval:`Aggregate cells prior analysis` to know how to toggle these two modes. 

We recommend testing with both modes, but as a rule of thumb if all the 
reference channel structures are present in all the cells but you have a 
large intensity variation between objects then using the "Local" mode could 
be beneficial. 

On the other hand, if some of the objects are completely devoid of any 
reference channel structure aggregating the objects might be the only 
option. 

.. note:: 

    If you are working with the *S. cerevisiase* model organism, most of the 
    times small buds do not have any structure. However, the "Local" will  
    still work if you annotate mother-bud relationship using our other software 
    `Cell-ACDC`_. This is because SpotMAX will consider the mother-object 
    as a single object while the bud is still attached to the mother (i.e., 
    before division is annotated). Make sure that you provide the annotations 
    to SpotMAX with the parameter :confval:`Table with lineage info end name`. 