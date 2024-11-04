.. _how_it_works:

How SpotMAX works
=================

SpotMAX has a **highly modular architecture** where each module can be run 
**independently or sequentially**. 

You can also skip some of these modules or you can **provide their result** 
generated outside of SpotMAX with a separate software. 

For example, you can provide the reference channel segmentation 
mask generated with Cellpose, skip its segmentation within SpotMAX and use 
it to filter valid spots (e.g., by keeping only spots that are on the reference 
channel mask).

The refence channel can also be used only with the intensities, e.g., to 
filter valid spots based on their relative intensities to the reference channel.

.. note:: 

    In this section we describe how SpotMAX works not how to run it or adjust 
    the parameters. For that, see the two sections :ref:`how-to-run` and 
    :ref:`params-desc`. 

.. toctree::
    :caption: Core modules
    :numbered:

    ref_ch_segm
    spot_pred
    spot_detect
    spot_quant
