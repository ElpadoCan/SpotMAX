.. role:: m(math)

.. _Edit button: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/main/cellacdc/resources/icons/edit-id.svg
.. _Create data structure: https://cell-acdc.readthedocs.io/en/latest/getting-started.html#creating-data-structures
.. _Cell-ACDC user manual: https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf

Description of the parameters
=============================

Description of all the parameters required to run spotMAX. The paramters can be 
set in the GUI and saved to a INI configuration file or edited directly 
in a INI template file. See `here <https://github.com/ElpadoCan/spotMAX/tree/main/examples/ini_config_files_templates>`_ 
for template INI files.

.. contents::

File paths and channels
-----------------------

* **Experiment folder path(s) to analyse**: one or more folder paths to analyse. To set up 
  this from the GUI click on the `Edit button`_ besides the parameter. An experiment 
  folder can be a folder containing the fluorescence channel separated into single TIFF files 
  or a folder containing multiple Position folders. We recommend structuring the data into the 
  same structure required by Cell-ACDC. Here you can find detailed instructions on how 
  to do that `_Create data structure`. You can also create the data structure 
  with Fiji/ImageJ macros or manually. See the section ``Create data structure 
  using Fiji Macros`` and ``Manually create data structure from microscopy file(s)`` 
  of the `_Cell-ACDC user manual`.

* **Spots channel end name or path**: Last part of the file name or full path 
  to the file containing the spots channel data. The data can be a single 2D image, 
  a 3D image (z-stack or 2D over time), or a 4D image (3D over time). 
  File formats supported: ``.tif``, ``.tiff``, ``.h5``, ``.npz``, or ``.npy``.

* **Cells segmentation end name or path**: Last part of the file name or full path 
  to the file containing the masks of the segmented obejcts (e.g., single cells). The data can be a single 2D image, 
  a 3D image (z-stack or 2D over time), or a 4D image (3D over time). 
  The segmentation data must have the same YX shape of the spots channel data. 
  However, when working with time-lapse data, it can have less time-points. 
  Additionally, with z-stack data, the segmentation data can be 2D. In this case, 
  spotMAX will stack the 2D segmentation masks into 3D data with the same number of 
  z-slices of the spots channel data. 
  File formats supported: ``.tif``, ``.tiff``, ``.h5``, ``.npz``, or ``.npy``.

* **Reference channel end name or path**: 
* **Spots channel segmentation end name or path**: 
* **Ref. channel segmentation end name or path**: 
* **Table with lineage info end name or path**: 
* **Run number**: 
* **Text to append at the end of the output files**: 
* **File extension of the output tables**: 

METADATA
--------

* **Number of frames (SizeT)**: 
* **Analyse until frame number**: 
* **Number of z-slices (SizeZ)**: 
* **Pixel width (μm)**: 
* **Pixel height (μm)**: 
* **Voxel depth (μm)**: 
* **Numerical aperture**: 
* **Spots reporter emission wavelength (nm)**: 
* **Spot minimum z-size (μm)**: 
* **Resolution multiplier in y- and x- direction**: 
* **Spot (z, y, x) minimum dimensions (radius)**: 


Pre-processing
--------------

* **Aggregate cells prior analysis**: 
* **Remove hot pixels**: 
* **Initial gaussian filter sigma**: 
* **Sharpen spots signal prior detection**: 


Reference channel
-----------------

* **Segment reference channel**: 
* **Keep only spots that are inside ref. channel mask**: 
* **Use the ref. channel mask to determine background**: 
* **Ref. channel is single object (e.g., nucleus)**: 
* **Ref. channel gaussian filter sigma**: 
* **Sigmas used to enhance network-like structures**: 
* **Ref. channel threshold function**: 
* **Calculate reference channel network length**: 
* **Save reference channel segmentation masks**: 


Spots channel
-------------

* **Spots detection method**: 
* **Spots segmentation method**: 
* **Spot detection threshold function**: 
* **Features and thresholds for filtering true spots**: 
* **Optimise detection for high spot density**: 
* **Compute spots size**: 
* **Save spots segmentation masks**: 


Configuration
-------------

* **Folder path of the log file**: 
* **Folder path of the final report**: 
* **Filename of final report**: 
* **Disable saving of the final report**: 
* **Use default values for missing parameters**: 
* **Stop analysis on critical error**: 
* **Use CUDA-compatible GPU**: 
* **Number of threads used by numba**: 
* **Reduce logging verbosity**: 

