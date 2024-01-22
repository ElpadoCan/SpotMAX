.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/index.html
.. _GitHub: https://github.com/ElpadoCan/spotMAX/issues
.. _BioImage Model Zoo: https://bioimage.io/#/
.. _Quasar 670: https://www.aatbio.com/fluorescence-excitation-emission-spectrum-graph-viewer/quasar_670
.. _seel-2023: https://www.nature.com/articles/s41594-023-01091-8

.. |load-folder| image:: ../images/folder-open.svg
    :width: 20

.. |compute| image:: ../images/compute.png
    :width: 20

.. |cog| image:: ../../../resources/icons/cog.svg
    :width: 20

.. |cog_play| image:: ../../../resources/icons/cog_play.svg
    :width: 20

.. _mtdna-yeast:

Count single mitochondrial DNA nucleoids and quantify mitochondrial network volume
==================================================================================

In this tutorial we will count the number of mitochondrial DNA nucleoids and we 
will segment the mitochondrial network in 3D as a reference channel. 

For details about the method used to visualize these structures see 
`this publication <seel-2023>`_. 

.. admonition:: Goals

    * Detect and separate highly connected spots
    * Segment network-like structures as reference channel (mitochondrial network)
    * Filter valid spots based on reference channel signal

.. include:: _preliminary_segment_cells.rst

Dataset
-------

To follow this tutorial, download the dataset from 
`here <https://hmgubox2.helmholtz-muenchen.de/index.php/s/yyq2XyYpjb8E7q6>`_.

This dataset was published in `this publication <seel-2023>`_. 



.. toctree:: 
    :hidden:

    _preliminary_segment_cells.rst
