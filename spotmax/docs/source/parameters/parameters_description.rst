.. _Create data structure: https://cell-acdc.readthedocs.io/en/latest/getting-started.html#creating-data-structures
.. _Cell-ACDC user manual: https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf
.. _Cell-ACDC: https://github.com/SchmollerLab/Cell_ACDC
.. _notebooks folder: https://github.com/ElpadoCan/SpotMAX/tree/main/examples/notebooks
.. _Sato filter: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato
.. _filters section: https://scikit-image.org/docs/stable/api/skimage.filters.html#
.. _GitHub page: https://github.com/ElpadoCan/SpotMAX
.. _BioImage Model Zoo: https://bioimage.io/#/
.. _INI configuration file templates: https://github.com/ElpadoCan/SpotMAX/tree/main/examples/ini_config_files_template
.. _pandas.eval: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html
.. _open: https://docs.python.org/3/library/functions.html#open
.. _pandas.read_hdf: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html

.. |edit-button| image:: ../../../resources/icons/cog.svg
    :width: 20

.. |compute| image:: ../images/compute.png
    :width: 20

.. |addIcon| image:: ../images/add.svg
    :width: 20
  
.. |infoIcon| image:: ../images/info.svg
    :width: 20
    
.. _params-desc:

Description of the parameters
=============================

Description of all the parameters required to run SpotMAX. The paramters can be 
set in the GUI and saved to a INI configuration file or edited directly 
in a INI template file. See `INI configuration file templates`_.

.. contents::

File paths and channels
-----------------------

.. confval:: Experiment folder path(s) to analyse

  One or more folder paths to analyse. To set up this from the GUI click on 
  the |edit-button| Edit button beside the parameter. An experiment  folder 
  can be a folder containing the fluorescence channel separated into single 
  TIFF files or a folder containing multiple Position folders. 
  We recommend structuring the data into the same structure required by 
  `Cell-ACDC`_. Here you can find detailed instructions on how to do that 
  `Create data structure`_.

  When writing multiple folder paths in the INI configuration file make sure 
  to write each folder path on a new line with some indentation, for example:

  .. code-block:: ini

    [File paths and channels]
    Experiment folder path(s) to analyse = 
        data/fist_folder
        data/second_folder

  If you provide relative paths, these must be **relative to the folder path in 
  the terminal where you will run the analysis**. 

  Alternatively, you can write the folder paths to analyse to any file that can 
  be opened with the Python function `open`_ (e.g., text file) and then provide 
  the path to this file in the INI configuration file as follows:
  
  .. code-block:: ini

    [File paths and channels]
    Experiment folder path(s) to analyse = file_with_paths_to_analyse.txt

  If you provide a relative path, the text file must be located in the same 
  folder as the INI file.

  .. note:: 

    You can also create the data structure with Fiji/ImageJ macros or manually. 
    See the sections ``Create data structure using Fiji Macros`` and 
    ``Manually create data structure from microscopy file(s)`` of the 
    `Cell-ACDC user manual`_.

  :type: string
  :required: True

.. confval:: Spots channel end name

  Last part of the file name or full path of the file containing the spots 
  channel data. The data can be a single 2D image, a 3D image (z-stack or 2D 
  over time), or a 4D image (3D over time). 
  
  File formats supported: ``.tif``, ``.tiff``, ``.h5``, ``.npz``, or ``.npy``.

  :type: string
  :default: ``''``

.. confval:: Cells segmentation end name

  Last part of the file name or full path of the file containing the masks of 
  the segmented obejcts (e.g., single cells). The data can be a single 2D image, 
  a 3D image (z-stack or 2D over time), or a 4D image (3D over time). 
  
  The segmentation data must have the same YX shape of the spots channel data. 
  However, when working with time-lapse data, it can have less time-points. 
  Additionally, with z-stack data, the segmentation data can be 2D. In this 
  case, SpotMAX will stack the 2D segmentation masks into 3D data with 
  the same number of z-slices of the spots channel data. Same applied when 
  working with 3D z-stacks over time. 
  
  Typical file formats: ``.npz``, or ``.npy``

  File formats supported: ``.npz``, ``.npy``, ``.tif``, ``.tiff``, ``.h5``.

  :type: string
  :default: ``''``

.. confval:: Reference channel end name

  Last part of the file name or full path of the file containing the reference 
  channel data. The reference channel is an additional fluorescence channel 
  that can aid with spot detection. For example, if the spots are located on a 
  specific sub-cellular structure, you can let SpotMAX segment the reference 
  channel and keep only those spots found on the reference channel. 
  
  Example of reference channels are the nucleus, or the mitochondrial network. 
  The data can be a single 2D image, a 3D image (z-stack or 2D over time),
  or a 4D image (3D over time).
  
  File formats supported: ``.tif``, ``.tiff``, ``.h5``, ``.npz``, or ``.npy``.

  :type: string
  :default: ``''``

.. confval:: Spots channel segmentation end name

  Last part of the file name or full path of the file containing the mask where 
  to search for spots. 
  
  If you use this parameter, SpotMAX will ignore the 
  :confval:`Spots segmentation method` and will move directly to spot detection.

  Use this parameter if you already have the spots segmented with an external 
  software.

  Typical file formats: ``.npz``, or ``.npy``
  
  File formats supported: ``.npz``, ``.npy``, ``.tif``, ``.tiff``, ``.h5``.

  :type: string
  :default: ``''``

.. confval:: Ref. channel segmentation end name

  Last part of the file name or full path of the file containing the 
  segmentation mask of the reference channel. 
  
  If you use this parameter, SpotMAX will ignore the parameters 
  :confval:`Ref. channel segmentation method` and 
  :confval:`Segment reference channel`.
  
  See the parameter :confval:`Reference channel end name` for more 
  details about the reference channel. 
  
  Use this parameter if you already have the reference channel segmented with 
  an external software.

  Typical file formats: ``.npz``, or ``.npy``
  
  File formats supported: ``.npz``, ``.npy``, ``.tif``, ``.tiff``, ``.h5``.

  :type: string
  :default: ``''``

.. confval:: Spots coordinates table end name

  Last part of the file name or full path of the file containing the columns 
  ``{'x', 'y', 'z'}`` with the coordinates of the spots to quantify. 

  When working with time-lapse data, make sure to also include the ``'frame_i'`` 
  column with the index of each timepoint (starting from 0 for first frame). 

  If the table has the ``.h5`` extension, it must have the frame index as 
  the group identifier (each frame will be loaded with 
  `pandas.read_hdf`_ ``(filepath, key='frame_0')``). 

  The output table with the quantified features will be saved in the 
  SpotMAX_output folder (see the section :ref:`output-files`) with the same 
  filename (or end name) of this parameter.

  This table can also be the same from a previous analysis where you simply 
  added new rows with the coordinates of the new spots whose features you want 
  to quantify. You can initialize missing values to ``NaN`` or ``-1``.

  If you need to quantify the features of the spots no matter if the spots are 
  considered valid or not, add a column called ``'do_not_drop'`` with the value 
  ``1`` at each row of the spot that must not be removed by SpotMAX filters.
  
  Use this parameter if you already have a table with spots coordinates 
  generated outside of SpotMAX.

  .. tip:: 

    In the SpotMAX GUI, you can edit the results of a previous analysis, 
    including adding new spots. To compute the features of these manually added 
    spots, SpotMAX will save the new table in each Position folder and it will 
    be used in this parameter. See the section :ref:`inspect-results-tab`, for 
    more details on how to manually edit the results.

  File formats supported: ``.csv``, or ``.h5`` (with single key).

  :type: string
  :default: ``''``

.. confval:: Table with lineage info end name

  Last part of the CSV file name or full path of the CSV file containing 
  parent-child relationship. The table must contain the following columns: 
  ``frame_i``, ``Cell_ID``, ``cell_cycle_stage``, ``relationship``, 
  and ``relative_ID``. 
  
  The ``frame_i`` is the time-point index (starting from 0).

  The ``Cell_ID`` is the ID of the segmented object (e.g., the single cells).

  The ``cell_cycle_stage`` must be either 'G1' or 'S' depending on whether the 
  cell is one or two objects (e.g., mother+bud in budding yeast when they are 
  segmented separately). 

  The ``relationship`` must be either 'mother' or 'bud' depending on whether 
  the cell is the mother or the daughter cell. 

  The ``relative_ID`` is the ID of the segmented object related to ``Cell_ID``. 
  When this information is provided, if the segmented object has 
  ``cell_cycle_stage = S`` it will be temporarily merged together with the 
  corresponding ``relative_ID`` for the prediction of the spots masks (i.e., the 
  areas where the spots are searched). 
  
  This is very useful when two related cells are segmented separately but must 
  be considered as a unique entity. 
  
  See the :confval:`Spots segmentation method` for more details on how the 
  spots masks are generated.
  
  .. note::
    
    We recommend using `Cell-ACDC`_ to generate the lineage table. 
  
  :type: string
  :default: ``''``

.. confval:: Run number

  An integer that will be prepended to SpotMAX output files that allows you to 
  identify a specific analysis run. You can have as many runs as you want. 
  Useful when trying out different parameters and you want to compare the 
  results of the different runs. 

  .. warning:: 

    The run number is the only discriminator to determine if files should be 
    overwritten or not. If you run the analysis with an already existing run 
    number, the older files will be overwritten regardless of the value of 
    :confval:`Text to append at the end of the output files`.

  :type: integer
  :default: ``1``

.. confval:: Text to append at the end of the output files

  A text to append at the end of the SpotMAX output files. In conjuction with 
  :confval:`Run number`, this parameter can be used to identify the output 
  files from a specific analysis run. 

  .. warning:: 

    Running the analysis with a different appended text but the same 
    :confval:`Run number` of an older analysis will result in overwriting the 
    files of the older analysis, regardless of the different text to append.

  :type: string
  :default: ``''``

.. confval:: File extension of the output tables

  Either ``.h5`` or ``.csv``. We recommend ``.h5`` when dealing with large 
  datasets. However, ``.h5`` files can be processed only with Python. 
  You can find example notebooks on how to process these files in the 
  `notebooks folder`_. 

  :type: string
  :default: ``.h5``

.. _metadata:

METADATA
--------

.. confval:: Number of frames (SizeT)

  The number of time-points in time-lapse data. This is the fourth to last 
  dimension of the image shape for 4D data (T, Z, Y, X) or the third to last 
  dimension for 3D data (T, Y, X).
  
  Write 1 if you load static data.

  :type: integer
  :default: ``1``

.. confval:: Analyse until frame number

  Leave at 1 if you load static data. Otherwise enter the frame number where 
  the analysis should stop.

  :type: integer
  :default: ``1``

.. confval:: Number of z-slices (SizeZ)

  Number of z-slices in the dataset. This is the third to last dimension 
  of the image shape. Leave at 1 if you don't have z-slices. 

  :type: integer
  :default: ``1``

.. confval:: Pixel width (μm)
  
  The pixel width in micrometers. This is typically given by the microscope 
  settings.

  :type: float
  :default: ``1.0``

.. confval:: Pixel height (μm)
  
  The pixel height in micrometers. This is typically given by the microscope 
  settings and it's usually the same as the pixel width.

  :type: float
  :default: ``1.0``

.. confval:: Voxel depth (μm)
  
  The voxel depth (in the z-direction) in micrometers. This is typically given 
  by the microscope settings. Leave at 1 if you don't have z-slices.

  :type: float
  :default: ``1.0``

.. confval:: Numerical aperture

  The numerical aperture of the microscope objective. This is typically given 
  by the microscope settings. This parameter will be used to determine the 
  diffraction limit (smallest spot size that can be resolved with 
  diffraction-limited microscope).
  
  .. note::
    
    For super-resolution data, you can modify the size of the PSF to a 
    smaller value than the diffraction limit by setting 
    :confval:`Resolution multiplier in y- and x- direction` parameter to a 
    value less than 1.
  
  :type: float
  :default: ``1.4``

.. confval:: Spots reporter emission wavelength (nm)

  The emission wavelength of the fluorescent reporter used. As with the 
  numerical aperture, this will be used to determine the diffraction limit 
  (smallest spot size that can be resolved with diffraction-limited microscope). 

  .. note::
    
    For super-resolution data, you can modify the size of the PSF to a 
    smaller value than the diffraction limit by setting 
    :confval:`Resolution multiplier in y- and x- direction` parameter to a 
    value less than 1.
  
  :type: float
  :default: ``500.0``

.. confval:: Spot minimum z-size (μm)
  
  Rough estimation of the smallest spot radius in z-direction. 
  
  As a rule of thumb you can use 2-3 times higher than the resolution limit 
  in X and Y. Another option is to visually measure this on a couple of spots. 
  The idea is that spots centers cannot be at a smaller distance than the 
  radius of the minimum size allowed. 
  
  In the GUI, you can see the estimated minimum spot 
  size at the :confval:`Spot (z, y, x) minimum dimensions (radius)` line. 

  :type: float
  :default: ``1.0``

.. confval:: Resolution multiplier in y- and x- direction

  This parameter allows you to modify the calculated minimum spots size. 
  The default value of 1 will result in the radius of the smallest spot being 
  the diffraction limit. 
  
  Enter 2 if for example your smallest spot is twice the diffraction limit. 
  You can visually tune this on the GUI in the :ref:`tune-parameters-tab`. 

  :type: float
  :default: ``1.0``

.. confval:: Spot (z, y, x) minimum dimensions (radius)

  This is not a parameter. On the GUI here you will see the result of minimum 
  spot radii estimation, both in pixels and micrometers. It will also be saved 
  in the INI configuration file.
  
  If :confval:`Number of z-slices (SizeZ)` is 1 the radius in the z-direction 
  will be equal to `Nan` (not a number).

  In the GUI, you will see a warning sign beside the parameter when any of the 
  values are lower than 2 pixels. This is because it would result in a spot 
  footprint with a radius of 1 pixel, effctively detecting spots one every 
  single pixel of the image (i.e., most likely this would be too small).

  In the :ref:`tune-parameters-tab` on the GUI you can visually set the 
  :confval:`Resolution multiplier in y- and x- direction` by adding points and 
  pressing up and down key arrows to adjust the size. 

  .. note::

    These values are the **radii** of the ellipsoid that determines the extent 
    of each spot (i.e., the pixels that belong to a spot). However, they are 
    also the **diameter** of the spot footprint, which is the minimum volume 
    where only one spot can be detected. This is because two spots can be 
    resolved as long as the distance between their centers is less or equal 
    than the radius of each spot (see the "Abbe diffraction limit").

.. _Pre-processing:

Pre-processing
--------------

.. confval:: Aggregate cells prior analysis

  If ``True``, SpotMAX will aggregate all the segmented objects together before 
  running the spot detection of the reference channel segmentation. 
  Activate this option if some of the objects do not have any spot. 
  Deactivate it if you have a large variation in signal's intensity across 
  objects. 
  
  .. note::

    Compared to automatic thresholding, the variation in intensity is less of 
    a problem when using the neural network. In any case, test with both 
    options.
  
  :type: boolean
  :default: ``True``

.. confval:: Threshold only inside segmented objects

  If ``True``, SpotMAX will use only the intensities from inside the segmented 
  objects to determine the threshold value. 

  The segmented objects are the one in the 
  :confval:`Cells segmentation end name` file.

  This parameter is ignored when using a :confval:`Spots segmentation method` 
  or :confval:`Ref. channel segmentation method` different from 
  ``Thresholding``

  .. tip:: 

    This parameter is useful when you have **bright artefacts close to the 
    segmented objects** that would otherwise skew towards a higher threshold 
    resulting in missed detections. 
  
  :type: boolean
  :default: ``True``

.. confval:: Remove hot pixels

  If ``True``, SpotMAX will run a morphological opening operation on the intensity 
  image. This will result in the removal of single bright pixels.

  :type: boolean
  :default: ``False``

.. confval:: Initial gaussian filter sigma

  If greater than 0, SpotMAX will apply a Gaussian blur before detection. 
  This is usually beneficial. Note that you can provide a single sigma value 
  or one for each axis (separated by a comma). 

  :type: float
  :default: ``0.75``

.. confval:: Sharpen spots signal prior detection

  If ``True``, SpotMAX will apply a Difference of Gaussians (DoG) filter that 
  result in enhancing the spots. This is usually beneficial. 
  
  A DoG filter works by subtracting two blurred versions of the image. 
  The subtracted image is with a larger sigma (more blurring). The sigmas for 
  the two blurred images is determined with the following formula:

  .. math::
    \sigma_1 = \frac{s_{zyx}}{1 + \sqrt{2}}
  
  .. math::
    \sigma_2 = \sigma_1\cdot\sqrt{2}
  
  where :math:`s_{zyx}` is the minimum spot size as calculated in the 
  :confval:`Spot (z, y, x) minimum dimensions (radius)` parameter. 
  
  The filtered image will be the result of subtracting the image blurred 
  with :math:`\sigma_2` from the image blurred with :math:`\sigma_1`.

  :type: boolean
  :default: ``True``

.. confval:: Extend 3D input segm. objects in Z
   
  Number of repetitions below and above the bottom and top z-slice of the input 
  segmentation objects (objects present in the :confval:`Cells segmentation end name`). 

  For example, if you provde the value ``(4, 6)``, SpotMAX will extend the 
  objects in Z by repeating the bottom z-slice 4 times below and the top z-slice 
  6 times above. 

  In the GUI, you can set the bottom and top values in the numeric controls 
  called 'Below bottom z-slice', and 'Above top z-slice', respectively. 

  .. tip:: 

    This parameter can be useful if you need to detect spots whose center might 
    be above or below the segmented object. For example, you could segment 
    only the center z-slice of the object and let spotmax search for spots in 
    a certain range above and below this center z-slice. 

    If you can afford this, segmenting only the center z-slice might be faster 
    than segmenting the entire object (e.g., single-cells).
   

Reference channel
-----------------

.. confval:: Segment reference channel

  If ``True`` and a reference channel name is provided in the parameter 
  :confval:`Reference channel end name`, SpotMAX will segment the 
  reference channel. The segmentation workflow is made of the following steps: 

  1. Gaussian filter (if :confval:`Ref. channel gaussian filter sigma` > 0)
  2. Ridge filter, to enhance network-like structures (if :confval:`Sigmas used to enhance network-like structures` > 0)
  3. Segmentation with one of the following methods (see :confval:`Ref. channel segmentation method`):
     
     a. Automatic thresholding using the method selected in the :confval:`Ref. channel threshold function` parameter.
     b. Any of the models available on the `BioImage Model Zoo`_ webpage.

  Note that the :confval:`Aggregate cells prior analysis` applies here too. 
  Do not aggregate if the signal's intensities varies widely between segmented 
  objects. 

  :type: boolean
  :default: ``False``

.. confval:: Keep only spots that are inside ref. channel mask

  If ``True``, spots whose center lies outside the reference channel mask 
  will be filtered out.

  :type: boolean
  :default: ``False``

.. confval:: Use the ref. channel mask to determine background

  If ``True``, the background value used to compute the :ref:`Effect size (vs. backgr.)` 
  feature is determined as the median of the pixels inside the reference channel 
  and outside of the spots. See the :ref:`Effect size (vs. backgr.)` section 
  for more details about how the spots masks are generated.

  :type: boolean
  :default: ``False``

.. confval:: Ref. channel is single object (e.g., nucleus)

  If ``True``, only the largest object in the reference channel mask per single 
  cell is kept. This is useful when segmenting the nucleus for example, 
  because artefacts that are not part of the nucleus can be easily removed.

  :type: boolean
  :default: ``False``

.. confval:: Keep external touching objects intact

  If ``True``, the masks that are partially external to the input objects 
  (e.g., the single cells, see :confval:`Cells segmentation end name`) 
  are maintained intact. 

  This is helpful when the reference channel structures can extend a 
  bit outside of the input objects.

  :type: boolean
  :default: ``False``

.. confval:: Ref. channel gaussian filter sigma

  If greater than 0, SpotMAX will appy a gaussian filter to the reference 
  channel before segmenting it. Note that you can provide a single sigma value 
  or one for each axis (separated by a comma). 

  :type: float
  :default: ``0.75``

.. confval:: Sigmas used to enhance network-like structures

  If greater than 0, SpotMAX will apply a ridge filter (more specifically, the 
  `Sato filter`_) that will enhance network-like structures. This parameter 
  will require some experimentation, but a good starting value is a single 
  sigma =  ``1.0``. If the reference channel mask should be smoother you can add a 
  second sigma = ``1.0, 2.0``. In the GUI, you can visualize the result of the 
  filter.

  :type: float or vector of (sigma_1, sigma_2, ..., sigma_n)
  :default: ``0.0``

.. confval:: Ref. channel segmentation method

  Method used to segment the reference channel. This can be either 
  ``Thresholding``, or ``BioImage.IO model``. 

  If you choose ``Thresholding``, you will also need to select which 
  thresholding  algorithm to use (parameter :confval:`Ref. channel threshold function`). 

  If you choose ``BioImage.IO model`` you will need either the DOI, the URL, or 
  the path to the downloaded zip file of the chosen model. You can choose any 
  of the models available on the `BioImage Model Zoo`_ webpage.

  :type: string
  :default: ``Thresholding``

.. confval:: Ref. channel threshold function

  The automatic thresholding algorithm to use when segmenting the reference 
  channel. In the GUI, you can visualize the result of all the algorithms 
  available. You can find more details about them on the scikit-image webpage 
  at the `filters section`_.

  :type: string
  :default: ``threshold_otsu``

.. confval:: Features for filtering ref. channel objects

  List of reference channel features with their threshold values 
  (minimum and maximum allowed) that will be used to filter valid reference 
  channel objects. 
  
  In the GUI you can set these by clicking on the ``Set features or view the selected ones...`` 
  button. 
  
  In the INI configuration file you could write
  
  .. code-block:: ini
    
    Features for filtering ref. channel objects =
      sub_obj_vol_vox, 10, None
  
  This example uses one feature: the ``sub_obj_vol_vox`` which is the 
  volume in voxels of each separated sub-object of the reference channel in each 
  single object (e.g., single cells). 
  
  The thresholds are written as ``min, max``, where ``None`` means "no threshold". 
  Therefore, non touching obejcts of the reference channel masks 
  (inside each segmented objects, e.g., the single cell) whose volume is less 
  than 10 voxels will be removed.

  You can also use ``OR`` statements and combine it with ``AND`` for more 
  complex filtering. For example you could write the following:

  .. code-block:: ini
    
    Features and thresholds for filtering true spots =
      (ref_ch_mean_intensity, None, 0.5
      OR ref_ch_median_intensity, None, 0.5)
      AND sub_obj_vol_vox, 10, None

  .. warning::

    When using ``OR`` and ``AND`` statements, make sure to have them at 
    the beginning of each new line. If the statement is missing, defaul is 
    ``AND``. Also, make sure to **open and close the parenthesis correctly**.

  See the section :ref:`ref_ch_features` for more details about the features 
  you can use for filtering.

.. confval:: Save reference channel features

  If ``True``, SpotMAX will save an additional file with tabular data containing 
  features based on the reference channel masks and intensities. 

  The file will be saved in the ``SpotMAX_output`` folder with the the name's 
  pattern ``<run_number>_3_ref_channel_features_<text_to_append>.csv``, where 
  ``<run_number>`` is the run number defined at :confval:`Run number`, and 
  ``<text_to_append>`` is the text provided at the 
  :confval:`Text to append at the end of the output files` 
  parameter. 

  See the section :ref:`ref_ch_features` for more details about the features 
  saved in this table.

.. confval:: Save reference channel segmentation masks

  If ``True``, SpotMAX will save the segmentation masks of the reference channel in 
  the same folder where the reference channel's data is located. 
  The file will be named with the pattern ``<basename>_run_num<run_number>_<ref_ch_name>_ref_ch_segm_mask_<text_to_append>.npz`` 
  where ``<basename>`` is the common part of all the file names in the Position 
  folder, ``<run_number>`` is the run number defined at :confval:`Run number`, 
  the ``<ref_ch_name>`` is the text provided at the :confval:`Reference channel end name` 
  parameter, and ``<text_to_append>`` is the text provided at the 
  :confval:`Text to append at the end of the output files` 
  parameter.

  :type: boolean
  :default: ``False``

.. confval:: Save pre-processed reference channel image

  If ``True``, SpotMAX will save the pre-processed reference channel's signal 
  image. This is the image used by SpotMAX to segment the reference channel.
  
  The file will be named with the pattern 
  ``<basename>_run_num<run_number>_<ref_ch_name>_preprocessed_<text_to_append>.<ext>`` 
  where ``<basename>`` is the common part of all the file names in the Position 
  folder, ``<run_number>`` is the run number defined at :confval:`Run number`, 
  the ``<ref_ch_name>`` is the text provided at the :confval:`Reference channel end name` 
  parameter, and ``<text_to_append>`` is the text provided at the 
  :confval:`Text to append at the end of the output files` 
  parameter.

  :type: boolean
  :default: ``False``

.. _spots-channel:

Spots channel
-------------

.. confval:: Spots segmentation method

  Method used to segment the spots. This can be either ``Thresholding``, 
  ``SpotMAX AI``, or ``BioImage.IO model``.

  If you choose ``SpotMAX AI`` you will need to setup additional parameters for 
  the model. In the GUI you can do so by clicking on the cog button just 
  beside the method selector. For more details about the AI parameters see 
  this section :ref:`ai_params`. 
  
  If you choose ``Thresholding``, you will also need to select which 
  thresholding  algorithm to use (parameter :confval:`Spot detection threshold function`). 
  
  If you choose ``BioImage.IO model`` you will need either the DOI, the URL, or 
  the path to the downloaded zip file of the chosen model. You can choose any 
  of the models available on the `BioImage Model Zoo`_ webpage.

  During the segmentation step SpotMAX will generate a binary mask from the 
  spots' intensity image with potential areas where to detect spots. 
  
  After this step, SpotMAX  will separate the spots by detecting local peaks 
  or labelling the prediction mask (separate by connected component labelling) 
  depending on the :confval:`Spots detection method` parameter. 
  
  In the GUI, you can visualize the output of all the thresholding algoritms 
  or of the neural networks vs a specific thresholding method by clicking 
  on the |compute| compute button beside the method selector. 

  :type: string
  :default: ``Thresholding``

.. confval:: Minimum size of spot segmentation mask

  Minimum size (area for 2D or volume for 3D) of the spots segmentation mask 
  generated by the method set in the :confval:`Spots segmentation method` 
  parameter.

  The unit is pixels (or voxels for 3D data). This value is useful to avoid 
  detecting spots on single-pixel spot mask.

  :type: integer
  :default: ``5``

.. confval:: Spot detection threshold function

  Automatic thresholding algorithm to use in case the :confval:`Spots segmentation method`  
  is ``Thresholding``. If instead it is ``SpotMAX AI`` or ``BioImage.IO model`` 
  here you can select which thresholding algorithm to compare to the neural 
  network output.
  
  .. note::

     More details about the available algorithms are available on the 
     scikit-image webpage at the `filters section`_. 
    
  :type: string
  :default: ``threshold_li``

.. confval:: Spots detection method

  Method used to detect the spots. This can be either ``peak_local_max`` (choose 
  ``Detect local peaks`` in the GUI) or ``label_prediction_mask`` ( choose 
  ``Label prediction mask`` in the GUI). 

  Choose ``label_prediction_mask`` when the masks of the spots after segmentation 
  are all separated. If some spots are merged, the only way to separate them is 
  to detect the local peaks. See :confval:`Spots segmentation method` for more 
  information. 

  :type: string
  :default: ``peak_local_max`` (``Detect local peaks`` in the GUI)  

.. confval:: Features and thresholds for filtering true spots

  List of single-spot features with their threshold values (minimum and maximum 
  allowed) that will be used to filter valid spots. 
  
  In the GUI you can set these by clicking on the ``Set features or view the selected ones...`` 
  button. 
  
  In the INI configuration file you could write
  
  .. code-block:: ini
    
    Features and thresholds for filtering true spots =
      spot_vs_ref_ch_ttest_pvalue, None, 0.025
      spot_vs_ref_ch_ttest_tstat, 0.0, None

  This example uses two features: the ``spot_vs_ref_ch_ttest_pvalue``, and the 
  ``spot_vs_ref_ch_ttest_tstat`` features (see :ref:`stat-test-vs-ref-ch`) 
  for details about these features). The thresholds, are written as ``min, max`` 
  after the feature name. Therefore, with the line ``spot_vs_ref_ch_ttest_pvalue, None, 0.025`` 
  SpotMAX will keep only those spots whose p-value of the t-test against the 
  reference channel is below 0.025. Equally, with the ``spot_vs_ref_ch_ttest_tstat, 0.0, None`` 
  SpotMAX will keep only those spots whose t-statistic of the t-test against the 
  reference channel is above 0.0. Using this syntax, you can filter using an 
  arbitrary number of single-spot features described in the :ref:`single-spot-features` 
  section.

  You can also use ``OR`` statements and combine it with ``AND`` for more 
  complex filtering. For example you could write the following:

  .. code-block:: ini
    
    Features and thresholds for filtering true spots =
      (spot_vs_ref_ch_ttest_pvalue, None, 0.025
      OR spot_vs_local_backgr_effect_size_glass, 1.6, None)
      AND spot_vs_ref_ch_ttest_tstat, 0.0, None

  .. warning::

    When using ``OR`` and ``AND`` statements, make sure to have them at 
    the beginning of each new line. If the statement is missing, defaul is 
    ``AND``. Also, make sure to **open and close the parenthesis correctly**. 

  :type: dictionary of {feature_name: (min_threshold, max_threshold)} or None
  :default: ``None``  

.. confval:: Local background ring width

  Width of the ring around each spot used to determine the local effect sizes 
  (see the section :ref:`Effect size (vs. backgr.)`). 

  You can specify this in ``pixel`` or ``micrometre``. The unit must be written 
  after the value in the INI configuration file separated by a space. The default 
  unit is ``pixel`` while the default value is ``5``. 
  
  Note that if the unit is ``micrometre`` the value will be converted to 
  ``pixel`` using the parameter :confval:`Pixel width (μm)`.

  The value in ``pixel`` is rounded to the nearest integer. 

  Example:

  .. code-block:: ini
    
    [Spots channel]
    Local background ring width = 5.0 pixel

  :type: string
  :default: ``5 pixel`` 

.. confval:: Optimise detection for high spot density

  If ``True``, SpotMAX will normalise the intensities within each single spot mask 
  by the euclidean distance transform of the spheroid mask. 
  
  More specifically, the further away from the center a pixel is, the more its 
  intensity will be reduced before computing the mean intensity of the spot. 
  For example, if a pixel is 5 pixels away from the spot center, its intensity 
  will be reduced by 1/5. 
  
  This is useful when you have very bright spots close to dimmer spots because 
  it reduces the influence of the bright spot on the mean intensity of the 
  dimmer spot.

  :type: boolean
  :default: ``True``  

.. confval:: Compute spots size (fit gaussian peak(s))

  If ``True``, SpotMAX will fit a 3D gaussian curve to the spots intensities. 
  This will result in more features being computed. These features are 
  described in the :ref:`spotfit-features` section. To determine which 
  pixels should be given as input to the fitting procedure for each spot, 
  SpotMAX will first perform a step called spotSIZE.
  
  Starting from a spot mask that is half the size of the minimum spot size, 
  SpotMAX will grow the masks by one voxel size in each direction. 
  At each iteration, the mean of the intensities on the surface of the newly 
  added pixels is computed. If the mean is below a limit, the spot mask 
  stops growing. 
  
  The limit is set to the median of the background (inside the cell and outside 
  of the minimum spot size mask) plus three times the background standard 
  deviation. When all the spots masks stop growing, the process ends and the 
  pixels's intensities of each spot are passed to the fitting routine. 
  
  .. note::

    If multiple spots masks are touching each other, they are 
    fitted together with as many gaussian curves as the number of touching 
    spots. 
  
  The equation of the 1D gaussian curve is the following:

  .. math::
    f(x) = \mathrm{exp}(-\frac{(x - x_0)^2}{2 \sigma_x ^ 2})
  
  where :math:`x_0` and :math:`\sigma_x` are fitting parameters and they are the center 
  of the gaussian peak and the standard devation (width), respectively. To obtain the 
  3D equation :math:`G(x, y, z)`, we simply multiply the 1D equations in each 
  direction and we add an overall amplitude :math:`A` and background :math:`B` fitting 
  parameters as follows:

  .. math::
    G(x, y, z) = A \cdot f(x) \cdot f(y) \cdot f(z) + B
  
  :type: boolean
  :default: ``False`` 

.. confval:: After spotFIT, drop spots that are too close

  If ``True``, SpotMAX will drop spots that are too close using the new spots 
  centers determined during the spotFIT step (fitting gaussian peaks). 

  If two or more peaks are within the same ellipsoid with radii equal to 
  :confval:`Spot (z, y, x) minimum dimensions (radius)` only the brightest 
  peak will be kept.

  The distances are be calculated using the ``x_fit``, ``y_fit``, and ``z_fit`` 
  coordinates. See :ref:`spotfit-coords`.

  .. note::
    You might need to allow more room for the peak center to move during the 
    fitting procedure in order to determine that two peaks are in fact only one. 
    To do this, increase :confval:`Bounds interval for the x and y peak center coord` 
    and :confval:`Bounds interval for the z peak center coord` parameters.

  :type: boolean
  :default: ``False``

.. confval:: Merge spots pairs where single peak fits better

  If ``True``, for every pair of peaks on the same spot mask (determined by the 
  :confval:`Spots segmentation method`) SpotMAX will fit two Gaussian peaks 
  and a single one. If the single one has lower root mean squared error 
  (i.e., better fit) or the two peaks merge together the dimmer peak is dropped.

  .. note:: 

    The bounds and the initial guess used for two and single peaks are the 
    same you provide in the :ref:`spotfit-params` section, except for the 
    center coordinates of the single peak. For the single peak, the initial 
    guess of the center coordinates will be the average between the two 
    peaks coordinates, while the bounds on the center will be plus/minus 
    half of the :confval:`Spot (z, y, x) minimum dimensions (radius)`. 

  :type: boolean
  :default: ``False``

.. confval:: Maximum number of spot pairs to check

  If the parameter :confval:`Merge spots pairs where single peak fits better` 
  is ``True`` you can set the maximum number of pairs to test in case there 
  are more than two peaks on the same spot mask. 

  To try all pairs set this value to ``-1``. Default is ``11`` which is just 
  a random lucky number :D. 

  :type: integer
  :default: ``11``

.. confval:: Save spots segmentation masks

  If ``True``, SpotMAX will save the segmentation masks of the spots in the same 
  folder where the spots's data is located. 
  
  The file will be named with the pattern 
  ``<basename>_run_num<run_number>_<spots_ch_name>_spots_segm_mask_<text_to_append>.npz`` 
  where ``<basename>`` is the common part of all the file names in the Position 
  folder, ``<run_number>`` is the run number defined at :confval:`Run number`, 
  the ``<spots_ch_name>`` is the text provided at the :confval:`Spots channel end name` 
  parameter, and ``<text_to_append>`` is the text provided at the 
  :confval:`Text to append at the end of the output files` 
  parameter.

  .. important::

    When :confval:`Spots detection method` is ``peak_local_max``, the masks of 
    the spots are spheroids with :confval:`Spot (z, y, x) minimum dimensions (radius)` 
    radii. When the detection method is ``label_prediction_mask``, the masks 
    of the spots are the thresholded spots' channel intensities.

  :type: boolean
  :default: ``False``

.. confval:: Save pre-processed spots image

  If ``True``, SpotMAX will save the pre-processed spots' signal image. This is 
  the image used by SpotMAX as input for spot detection.
  
  The file will be named with the pattern 
  ``<basename>_run_num<run_number>_<spots_ch_name>_preprocessed_<text_to_append>.<ext>`` 
  where ``<basename>`` is the common part of all the file names in the Position 
  folder, ``<run_number>`` is the run number defined at :confval:`Run number`, 
  the ``<spots_ch_name>`` is the text provided at the :confval:`Spots channel end name` 
  parameter, and ``<text_to_append>`` is the text provided at the 
  :confval:`Text to append at the end of the output files` 
  parameter.

  :type: boolean
  :default: ``False``

.. confval:: Skip objects where segmentation failed

  If ``True``, SpotMAX will skip those objects (e.g., the single cells) where 
  more than 25% of the spots masks determined by 
  :confval:`Spots segmentation method` are on the background. 
  
  The objects are the ones in the file provided by the 
  :confval:`Cells segmentation end name`. 

  If ``False``, SpotMAX will still detect spots in these objects, but it 
  will log a warning in the terminal, in the log file and in the final report. 

  When you segment the spots in a cell that is particularly dark or you 
  use a threshold method that is too permissive, large parts of the spots 
  masks will be found on the background. This is often a sign that the 
  segmentation of the spots failed in that particular object.

  When this happens, we recommend trying to solve this with a different 
  :confval:`Spots segmentation method` or :confval:`Spot detection threshold function`. 

  However, in some datasets, most of the spots are segmented correctly except 
  in a few cells. In this case, skipping these objects can be the right 
  solution (if you can afford it). 

  The invalid objects will not appear in the single-spot tables, while they 
  will appear with ``num_spots = 0`` and a ``1`` in the column 
  ``spots_segmentation_might_have_failed`` in the single-objects tables. 
  See the section :ref:`output-files` for more info about the saved tables.

  .. note:: 

    If you don't skip these problematic objects, analysis can take a long time 
    and many of the detections are often false positives.

  :type: boolean
  :default: ``False``

.. _spotfit-params:

SpotFIT
-------

.. confval:: Bounds interval for the x and y peak center coord

  Here you can specify the half-interval width to determine the maximum and 
  the minimum of the x and y center coordinate allowed in the fitting procedure.

  For example, if you set this to ``0.2`` and the x center coordinated detected 
  during the spot detection step is ``200``, then :math:`x_0` and :math:`y_0` in the 
  gaussian curve equation can reach a minimum of ``199.8`` and a maximum of 
  ``200.2`` during the fitting routine.

  See the :confval:`Compute spots size (fit gaussian peak(s))` parameter for more 
  details about the fitting procedure.

  :type: float
  :default: ``0.1``

.. confval:: Bounds interval for the z peak center coord

  Here you can specify the half-interval width to determine the maximum and 
  the minimum of the z center coordinate allowed in the fitting procedure.

  For example, if you set this to ``0.2`` and the z center coordinated detected 
  during the spot detection step is ``15``, then :math:`z_0` in the gaussian curve 
  equation can reach a minimum of ``14.8`` and a maximum of ``15.2`` 
  during the fitting routine.

  See the :confval:`Compute spots size (fit gaussian peak(s))` parameter for more 
  details about the fitting procedure.

  :type: float
  :default: ``0.2``

.. confval:: Bounds for sigma in x-direction

  Here you can specify the maximum and the minimum values that :math:`sigma_x` in 
  the gaussian curve equation can reach during the fitting routine.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.

  :type: string
  :default: ``0.5, spotsize_yx_radius_pxl``

.. confval:: Bounds for sigma in y-direction

  Here you can specify the maximum and the minimum values that :math:`sigma_y` in 
  the gaussian curve equation can reach during the fitting routine.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.
  
  :type: string
  :default: ``0.5, spotsize_yx_radius_pxl``

.. confval:: Bounds for sigma in z-direction

  Here you can specify the maximum and the minimum values that :math:`sigma_z` in 
  the gaussian curve equation can reach during the fitting routine.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.

  :type: string
  :default: ``0.5, spotsize_z_radius_pxl``

.. confval:: Bounds for the peak amplitude

  Here you can specify the maximum and the minimum values that :math:`A` in 
  the gaussian curve equation can reach during the fitting routine.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    are also allowed. The mathematical expression must be a valid expression 
    that can be evaluated with the Python library `pandas.eval`_.

  :type: string
  :default: ``0.0, spotsize_A_max``

.. confval:: Bounds for the peak background level

  Here you can specify the maximum and the minimum values that :math:`B` in 
  the gaussian curve equation can reach during the fitting routine.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    are also allowed. The mathematical expression must be a valid expression 
    that can be evaluated with the Python library `pandas.eval`_.

  :type: string
  :default: ``spot_B_min, inf``

.. confval:: Initial guess for sigma in x-direction

  Here you can specify the initial guess in the fitting routine for the 
  paramter :math:`sigma_x` in the gaussian curve equation.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.

  :type: string
  :default: ``spotsize_initial_radius_yx_pixel``

.. confval:: Initial guess for sigma in y-direction

  Here you can specify the initial guess in the fitting routine for the 
  paramter :math:`sigma_y` in the gaussian curve equation.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.

  :type: string
  :default: ``spotsize_initial_radius_yx_pixel``

.. confval:: Initial guess for sigma in z-direction

  Here you can specify the initial guess in the fitting routine for the 
  paramter :math:`sigma_z` in the gaussian curve equation.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.

  :type: string
  :default: ``spotsize_initial_radius_z_pixel``

.. confval:: Initial guess for the peak amplitude

  Here you can specify the initial guess in the fitting routine for the 
  paramter :math:`A` in the gaussian curve equation.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.

  :type: string
  :default: ``spotsize_A_max``

.. confval:: Initial guess for the peak background level

  Here you can specify the initial guess in the fitting routine for the 
  paramter :math:`B` in the gaussian curve equation.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.

  :type: string
  :default: ``spotsize_surface_median``

.. _custom_combined_meas:

Custom combined measurements
----------------------------

.. confval:: Custom combined measurement

  Here you can define as many single-spot custom combined measurement as you 
  want. These new measurements can then be used as features to filter valid 
  spots using the :confval:`Features and thresholds for filtering true spots` 
  parameter. Custom combined measurements can also be used to define subsequent 
  measurements.
  
  First, you need to decide on a new column name for the metric 
  (only undescores, letters, and numbers are allowed). Next, you need to define 
  a mathematical expression that combines one or more existing metrics. 
  See the section :ref:`single-spot-features` for more details on the available 
  metrics. 

  The combined measurements can be defined in the GUI or directly in the INI 
  configuration file in a new section called ``[Custom combined measurements]``. 
  
  In this example you can see two custom measurements as they are defined in 
  the INI parameters file:

  .. code-block:: ini
    
    [Custom combined measurements]
    spot_peak_to_backgr_ratio = spot_raw_max_in_spot_minimumsize_vol/background_median_z_slice_raw_image
    spot_IQR = spot_raw_q75_in_spot_minimumsize_vol - spot_raw_q25_in_spot_minimumsize_vol

  To add new measurements, in the GUI click on the |addIcon| ``Add button`` 
  beside the |infoIcon| ``Info button``.

  .. note::

    For the values you can specify any mathematical combination of the 
    available single spot features (see :ref:`single-spot-features`). Numbers 
    and ``± inf`` (i.e., no lower-upper limit) are also allowed. The 
    mathematical expression must be a valid expression that can be evaluated 
    with the Python library `pandas.eval`_.

  :type: str
  :default: ``''``


.. _config-params:

Configuration
-------------

.. confval:: Folder path of the log file

  If not specified, the default path is ``~/spotmax_appdata/logs``. 
  The log file contains useful information for debugging. Please, provide it 
  when submitting an issue on our `GitHub page`_.

  :type: string
  :default: ``~/spotmax_appdata/logs``

.. confval:: Folder path of the final report

  If not specified, the final report will be saved in the same folder of the 
  INI configuration file. 
  
  The final report contains useful information with warnings and 
  error messages that might have arose during the analysis.

  :type: string
  :default: ``''``

.. confval:: Filename of final report

  If not specified, the filename of the final report  will be a unique string 
  with a timestamp to avoid multiple analysis in parallel trying to save to the 
  same file. The final report contains useful information with warnings and 
  error messages that might have arose during the analysis.

  :type: string
  :default: ``''``

.. confval:: Disable saving of the final report

  If ``True``, the final report will not be saved.

  :type: boolean
  :default: ``False``

.. confval:: Use default values for missing parameters

  If ``True``, SpotMAX will not pause waiting for the user to choose what to do 
  with missing parameters. 
  
  It will continue the analysis with default values. Disable this only when you 
  are sure you have setup all the paramters needed. Some parameters are 
  mandatory and analysis will stop regardless.

  :type: boolean
  :default: ``False``

.. confval:: Stop analysis on critical error

  If ``False``, SpotMAX will log the error and will continue the analysis of the 
  next folder without stopping.

  :type: boolean
  :default: ``True``

.. confval:: Use CUDA-compatible GPU

  If ``True`` and CUDA libraries are installed, SpotMAX can run some of the 
  analysis steps on the GPU, significantly increasing overall analysis speed.

  :type: boolean
  :default: ``False``

.. confval:: Number of threads used by numba

  If the library `numba` is installed, here you can specify how many threads 
  should be used (we recommend to use a maximum equal to the number of CPU 
  cores available). The default value is half of the 
  CPU cores available.

  :type: integer
  :default: ``-1``

.. confval:: Reduce logging verbosity

  If ``True``, you will see almost only progress bars in the terminal during the 
  analysis.

  :type: boolean
  :default: ``False``

.. toctree:: 
  :maxdepth: 1

  ai_params