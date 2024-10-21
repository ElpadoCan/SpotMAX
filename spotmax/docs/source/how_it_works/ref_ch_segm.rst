.. _BioImage.IO Model Zoo: https://bioimage.io/#/
.. _Thresholding (scikit-image): https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_thresholding.html
.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/

.. _ref_ch_segm:

Reference channel segmentation
==============================

The reference channel can be any additonal **intensity image that can aid with 
filtering valid spots**.

If you are not interested in spot detection, you can also just take advantage 
of this module to segment your structure of interest. In this case, do 
not provide any channel name for the parameter 
:confval:`Spots channel end name`. This will end the analysis after segmenting 
the reference channel and it will not proceed with the rest of the spot analysis.

Segmentation of the reference channel can be divided into two steps:

1. :ref:`ref_ch_preproc`
2. :ref:`ref_ch_sem_segm`

.. _ref_ch_preproc:

Image pre-processing
~~~~~~~~~~~~~~~~~~~~

SpotMAX provides two filters for image pre-processing: **blurring** 
(gaussian filter) and an **enhancer for network-like structures** 
(e.g., the mitochondria network). 

The gaussian filter can be applied with a different sigma for each dimension, 
which is very useful when working with anisotropic 3D z-stack data. 

.. _ref_ch_sem_segm:

Semantic segmentation
~~~~~~~~~~~~~~~~~~~~~

The pre-processed image is used as input to the segmentation model of your 
choice. SpotMAX will segment the reference channel by using automatic 
thresholding or any of the models available on the `BioImage.IO Model Zoo`_. 

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