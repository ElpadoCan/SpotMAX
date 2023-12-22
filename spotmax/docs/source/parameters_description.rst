.. role:: m(math)

.. _Edit button: https://raw.githubusercontent.com/SchmollerLab/Cell_ACDC/main/cellacdc/resources/icons/edit-id.svg
.. _Create data structure: https://cell-acdc.readthedocs.io/en/latest/getting-started.html#creating-data-structures
.. _Cell-ACDC user manual: https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf
.. _Cell-ACDC: https://github.com/SchmollerLab/Cell_ACDC
.. _notebooks folder: https://github.com/ElpadoCan/spotMAX/tree/main/examples/notebooks
.. _Sato filter: https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato
.. _filters section: https://scikit-image.org/docs/stable/api/skimage.filters.html#
.. _GitHub page: https://github.com/ElpadoCan/spotMAX

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
  same structure required by `Cell-ACDC`_. Here you can find detailed instructions on how 
  to do that `Create data structure`_. You can also create the data structure 
  with Fiji/ImageJ macros or manually. See the sections ``Create data structure 
  using Fiji Macros`` and ``Manually create data structure from microscopy file(s)`` 
  of the `Cell-ACDC user manual`_.

* **Spots channel end name or path**: Last part of the file name or full path 
  of the file containing the spots channel data. The data can be a single 2D image, 
  a 3D image (z-stack or 2D over time), or a 4D image (3D over time). 
  File formats supported: ``.tif``, ``.tiff``, ``.h5``, ``.npz``, or ``.npy``.

* **Cells segmentation end name or path**: Last part of the file name or full path 
  of the file containing the masks of the segmented obejcts (e.g., single cells). The data can be a single 2D image, 
  a 3D image (z-stack or 2D over time), or a 4D image (3D over time). 
  The segmentation data must have the same YX shape of the spots channel data. 
  However, when working with time-lapse data, it can have less time-points. 
  Additionally, with z-stack data, the segmentation data can be 2D. In this case, 
  spotMAX will stack the 2D segmentation masks into 3D data with the same number of 
  z-slices of the spots channel data. Same applied when working with 3D z-stacks over time. 
  File formats supported: ``.tif``, ``.tiff``, ``.h5``, ``.npz``, or ``.npy``.

* **Reference channel end name or path**: Last part of the file name or full path 
  of the file containing the reference channel data. The reference channel is an 
  additional fluorescence channel that can aid with spot detection. For example, 
  if the spots are located on a specific sub-cellular structure, you can let spotMAX 
  segment the reference channel and keep only those spots found on the reference 
  channel. Example of reference channels are the nucleus, or the mitochondrial 
  network. The data can be a single 2D image, a 3D image (z-stack or 2D over time),
  or a 4D image (3D over time). File formats supported: ``.tif``, ``.tiff``, 
  ``.h5``, ``.npz``, or ``.npy``.

* **Spots channel segmentation end name or path**: Last part of the file name or full path 
  of the file containing the mask where to search for spots. Before detecting the 
  spots, spotMAX will segment the spots' channel with automatic thresholding or 
  a neural network. If you perform this step externally, you can provide here the 
  file containing that data and spotMAX will move directly to spot detection 
  without segmenting the spots.

* **Ref. channel segmentation end name or path**: Last part of the file name or full path 
  of the file containing the segmentation mask of the reference channel. See the 
  parameter **Reference channel end name or path** for more details about the 
  reference channel. Provide a file name here when you are performing the segmentation 
  of the reference channel externally to spotMAX. 

* **Table with lineage info end name or path**: Last part of the CSV file name or full path 
  of the CSV file containing parent-child relationship. The table must contain the 
  following columns: ``frame_i``, ``Cell_ID``, ``cell_cycle_stage``, ``relationship``, 
  and ``relative_ID``. The ``frame_i`` is the time-point index (starting from 0); 
  The ``Cell_ID`` is the ID of the segmented object (e.g., the single cells); 
  The ``cell_cycle_stage`` must be either 'G1' or 'S' depending on whether the 
  cell is one or two objects (e.g., mother+bud in budding yeast when they are 
  segmented separately). The ``relationship`` must be either 'mother' or 'bud' 
  depending on whether the cell is the mother or the daughter cell. 
  The ``relative_ID`` is the ID of the segmented object related to ``Cell_ID``. 
  When this information is provided, when a segmented object has 
  ``cell_cycle_stage = S`` it will be temporarily merged together with the 
  corresponding ``relative_ID`` for the prediction of the spots masks (i.e., the 
  areas where the spots are searched). This is very useful when two related cells 
  are segmented separately but must be considered as a unique entity. See the 
  **Spots segmentation method** for more details on how the spots masks are 
  generated. We recommend using `Cell-ACDC`_ to generate the lineage table. 

* **Run number**: An integer that will be prepended to spotMAX output files that 
  allows you to identify a specific analysis run. You can have as many runs as you 
  want. Useful when trying out different parameters and you want to compare the 
  results of the different runs. 

* **Text to append at the end of the output files**: A text to append at the end 
  of the spotMAX output files. In conjuction with **Run number**, this parameter can 
  be used to identify as specific analysis run. 

* **File extension of the output tables**: Either ``.h5`` or ``.csv``. We recommend 
  ``.h5`` when dealing with large datasets. However, ``.h5`` files can be processed 
  only with Python. You can find example notebooks on how to process these files 
  in the notebooks folder. 

METADATA
--------

* **Number of frames (SizeT)**: The number of time-points in time-lapse data. 
  Write 1 if you load static data.

* **Analyse until frame number**: Leave at 1 if you load static data. Otherwise 
  enter the frame number where the analysis should stop.

* **Number of z-slices (SizeZ)**: Leave at 1 if you don't have z-slices. 

* **Pixel width** (:m:`\mu m`): The pixel width in micrometers. This is typically given by 
  the microscope settings.

* **Pixel height** (:m:`\mu m`): The pixel height in micrometers. This is typically given by 
  the microscope settings and it's usually the same as the pixel width.

* **Voxel depth** (:m:`\mu m`): The voxel depth (in the z-direction) in micrometers. 
  This is typically given by the microscope settings. 
  Leave at 1 if you don't have z-slices.

* **Numerical aperture**: The numerical aperture of the microscope objective. 
  This is typically given by the microscope settings. This parameter will be 
  used to determine the diffraction limit (smallest spot size that can be 
  resolved with diffraction-limited microscope). For super-resolution data, you 
  can modify the size of the PSF with the **Resolution multiplier in y- and x- direction** 
  parameter.

* **Spots reporter emission wavelength (nm)**: The emission wavelength of the 
  fluorescent reporter used. As with the numerical aperture, this will be used 
  to determine the diffraction limit (smallest spot size that can be 
  resolved with diffraction-limited microscope). For super-resolution data, you 
  can modify the size of the PSF with the **Resolution multiplier in y- and x- direction** 
  parameter.

* **Spot minimum z-size** (:m:`\mu m`): Rough estimation of the smallest spot radius in 
  z-direction. As a rule of thumb you can use 2-3 times higher than the resolution 
  limit in X and Y. Another option is to visually measure this on a couple of spots. 
  The idea is that spots centers cannot be at a smaller distance than the radius of 
  the minimum size allowed. In the GUI, you can see the estimated minimum spot 
  size at the **Spot (z, y, x) minimum dimensions (radius)** line. 

* **Resolution multiplier in y- and x- direction**: This parameter allows you to modify the 
  calculated minimum spots size. The default value of 1 will result in the radius of the 
  smallest spot being the diffraction limit. Enter 2 if for example your smallest spot 
  is twice the diffraction limit. You can visually tune this on the GUI in the 
  Autotuning tab. 

* **Spot (z, y, x) minimum dimensions (radius)**: This is not a parameter. On the GUI 
  here you will see the result of minimum spot radii estimation, both in pixels and 
  micrometers. Tune the parameters above until you roughly get this right.

Pre-processing
--------------

* **Aggregate cells prior analysis**: If true, spotMAX will aggregate all the segmented objects 
  together before running the spot detection of the reference channel segmentation. 
  Activate this option if some of the objects do not have any spot. Deactivate it 
  if you have a large variation in signal's intensity across objects. Note that, 
  compared to automatic thresholding, the variation in intensity is less of a problem 
  when using the neural network. In any case, test with both options.

* **Remove hot pixels**: If true, spotMAX will run a morphological opening operation 
  on the intensity image. This will result in the removal of single bright pixels.

* **Initial gaussian filter sigma**: If greater than 0, spotMAX will apply a Gaussian 
  blur before detection. This is usually beneficial. Note that you can provide 
  a single sigma value or one for each axis (separated by a comma). 

* **Sharpen spots signal prior detection**: If true, spotMAX will apply a 
  Difference of Gaussians (DoG) filter that result in enhancing the spots. This is 
  usually beneficial. A DoG filter works by subtracting two blurred versions of the 
  image. The subtracted image is with a larger sigma (more blurring). The sigmas for 
  the two blurred images is determined with the following formula:

  .. math::
    \sigma_1 = \frac{s_{zyx}}{1 + \sqrt{2}}
  
  .. math::
    \sigma_2 = \sigma_1*\sqrt{2}
  
  where :m:`s_{zyx}` is the minimum spot size as calculated in the `METADATA`_ 
  section. The filtered image will be the result of subtracting the image blurred 
  with :m:`\sigma_2` from the image blurred with :m:`\sigma_1`.

Reference channel
-----------------

* **Segment reference channel**: If true and a reference channel name is provided 
  in the parameter **Reference channel end name or path**, spotMAX will segment the 
  reference channel. The segmentation workflow is made of the following steps: 

  1. Gaussian filter (if **Ref. channel gaussian filter sigma** > 0)
  2. Ridge filter, to enhance network-like structures (if **Sigmas used to enhance network-like structures** > 0)
  3. Automatic thresholding using the method selected by the **Ref. channel threshold function** parameter.

  Note that the **Aggregate cells prior analysis** applies here too. Do not aggregate 
  if the signal's intensities varies widely between segmented objects. 

* **Keep only spots that are inside ref. channel mask**: If true, spots whose 
  detected center lies outside the reference channel mask will be filtered out.

* **Use the ref. channel mask to determine background**: If true, the background value 
  used to compute the :ref:`Effect size (vs. backgr.)` feature is determined as the median 
  of the pixels inside the reference channel and outside of the spots. See the :ref:`Effect size (vs. backgr.)` 
  section for more details about how the spots masks are generated.

* **Ref. channel is single object (e.g., nucleus)**: If true, only the largest 
  object in the reference channel mask per single cell is kept. This is useful when 
  segmenting the nucleus for example, because artefacts that are not part of 
  the nucleus can be easily removed.

* **Ref. channel gaussian filter sigma**: If greater than 0, spotMAX will appy a 
  gaussian filter to the reference channel before segmenting it. Note that you can provide 
  a single sigma value or one for each axis (separated by a comma). 

* **Sigmas used to enhance network-like structures**: If greater than 0, spotMAX will 
  apply a ridge filter (more specifically, the `Sato filter`_) that will 
  enhance network-like structures. This parameter will require some experimentation,  
  but a good starting value is a single sigma = 1.0. If the reference channel mask 
  should be smoother you can add a second sigma = 1.0, 2.0. In the GUI, you can 
  visualize the result of the filter.

* **Ref. channel threshold function**: The automatic thresholding algorithm to use 
  when segmenting the reference channel. In the GUI, you can visualize the result 
  of all the algorithms available. You can find more details about them on the 
  scikit-image webpage at the `filters section`_.

* **Save reference channel segmentation masks**: if true, spotMAX will save the 
  segmentation masks of the reference channel in the same folder where the reference 
  channel's data is located. The file will be named with the pattern 
  ``<basename>_<ref_ch_name>_segm_mask_<text_to_append>.npz`` where ``<basename>`` 
  is the common part of all the file names in the Position folder, the ``<ref_ch_name>`` 
  is the text provided at the **Reference channel end name or path** parameter, 
  and ``<text_to_append>`` is the text provided at the **Text to append at the end of the output files** 
  parameter.


Spots channel
-------------

* **Spots detection method**: either 'Detect local peaks' or 'Label prediction mask'. 
  Choose 'Label prediction mask' when the masks of the spots after segmentation are 
  all separated. If some spots are merged, the only way to separate them is to detect 
  the local peaks. See **Spots segmentation method** for more information. 

* **Spots segmentation method**: either 'Thresholding' or 'spotMAX AI'. If you 
  choose neural network you will need to pass additional parameters for the model. 
  In the GUI you can do so by clicking on the cog button just besides the method 
  selector. If you choose thresholding, you will also need to select which thresholding 
  algorithm to use (parameter **Spot detection threshold function**). 
  During the segmentation step spotMAX will generate a binary mask from spots' 
  intensity image with potential areas where to detect spots. After this step, spotMAX 
  will separate the spots by detecting local peaks or labelling the prediction mask 
  (separate by connected component labelling) depending on the **Spots detection method** 
  parameter. In the GUI, you can visualize the output of all the thresholding 
  algoritms or of the neural network vs a specific thresholding method by clicking 
  on the compute button besides the method selector. 

* **Spot detection threshold function**: automatic thresholding algorithm to use 
  in case the Spots segmentation method is 'Thresholding'. You can find more 
  details about the available algorithms on the scikit-image webpage at 
  the `filters section`_. If the Spots segmentation method is 'spotMAX AI' 
  here you can select which thresholding algorithm to compare to the neural 
  network output.

* **Features and thresholds for filtering true spots**: list of single-spot features 
  with their threshold values (minimum and maximum allowed) that will be used to 
  filter valid spots. In the GUI you can set these by clicking on the 
  ``Set features or view the selected ones...`` button. For example, in the INI 
  configuration file you could write
  
  ::
    
    Features and thresholds for filtering true spots =
      spot_vs_ref_ch_ttest_pvalue, None, 0.025
      spot_vs_ref_ch_ttest_tstat, 0.0, None

  This example uses two features: the ``spot_vs_ref_ch_ttest_pvalue``, and the 
  ``spot_vs_ref_ch_ttest_tstat`` features (see `Statistical test (vs. ref. ch.)`_) 
  for details about these features). The thresholds, are written as ``min, max`` 
  after the feature name. Therefore, with the line ``spot_vs_ref_ch_ttest_pvalue, None, 0.025`` 
  spotMAX will keep only those spots whose p-value of the t-test against the 
  reference channel is below 0.025. Equally, wiht the ``spot_vs_ref_ch_ttest_tstat, 0.0, None`` 
  spotMAX will keep only those spots whose t-statistic of the t-test against the 
  reference channel is above 0.0. Using this syntax, you can filter using an 
  arbitrary number single-spot features described in the `Single-spot features description`_ 
  section.
 
* **Optimise detection for high spot density**: if true, spotMAX will normalise the 
  intensities within each single spot mask by the euclidean distance transform. 
  More specifically, the further away from the center a pixel is, the more its 
  intensity will be reduced before computing the mean intensity of the spot. 
  For example, if a pixel is 5 pixels away from the spot center, its intensity 
  will be reduced by 1/5. 
  This is useful when you have very bright spots close to dimmer spots because 
  it reduces the influence of the bright spot on the mean intensity of the 
  dimmer spot.

* **Compute spots size**: if true, spotMAX will fit a 3D gaussian curve to the 
  spots intensities. This will result in more features being computed. These 
  features are described in the `Spotfit features`_ section. To determine which 
  pixels should be given as input to the fitting procedure for each spot, spotMAX 
  will first perform a step called spotSIZE. Starting from a spot mask that is half 
  the size of the minimum spot size, spotMAX will grow the masks by one voxel size 
  in each direction. At each iteration, the mean of the intensities on the surface 
  of the newly added pixels is computed. If the mean is below a limit, the spot mask 
  stops growing. The limit is set to the median of the background (inside the cell 
  and outside of the minimum spot size mask) plus three times the background standard 
  deviation. When all the spots masks stop growing, the process ends and the pixels's 
  intensities of each spot are passed to the fitting routine. 
  Note that if multiple spots masks are touching each other, they are fitted together 
  with as many gaussian curves as merged spots. The equation of the 1D gaussian curve is 
  the following

  .. math::
    f(x) = e^{-\frac{(x - x_0)^2}{2 \sigma_x ^ 2}}
  
  where :m:`x_0` and :m:`\sigma_x` are fitting parameters and they are the center 
  of the gaussian peak and the standard devation (width), respectively. To obtain the 
  3D equation, we simply multiply the 1D equations in each direction and we add 
  an overall amplitude :m:`A` and background :m:`B` fitting 
  parameters as follows:

  .. math::
    g(x, y, z) = A \cdot f(x) \cdot f(y) \cdot f(z) + B


* **Save spots segmentation masks**: if true, spotMAX will save the 
  segmentation masks of the spots in the same folder where the spots's data 
  is located. Note that this is possible only when 
  ``Spots detection method = 'Label prediction mask'``.
  The file will be named with the pattern 
  ``<basename>_<spots_ch_name>_segm_mask_<text_to_append>.npz`` where ``<basename>`` 
  is the common part of all the file names in the Position folder, the ``<spots_ch_name>`` 
  is the text provided at the **Spots channel end name or path** parameter, 
  and ``<text_to_append>`` is the text provided at the **Text to append at the end of the output files** 
  parameter.


Configuration
-------------

* **Folder path of the log file**: if not specified, the default path is 
  ``~/spotmax_appdata/logs``. The log file contains useful information for debugging. 
  Please, provide it when submitting an issue on our `GitHub page`_.

* **Folder path of the final report**: if not specified, the final report will 
  be saved in the same folder of the INI configuration file. The final report contains useful information with warnings and 
  error messages that might have arose during the analysis.

* **Filename of final report**: if not specified, the filename of the final report 
  will be a unique string with a timestamp to avoid multiple analysis in 
  parallel trying to save to the same file. The final report contains useful information with warnings and 
  error messages that might have arose during the analysis.

* **Disable saving of the final report**: if true, the final report will not be 
  saved.

* **Use default values for missing parameters**: if true, spotMAX will not pause 
  waiting for the user to choose what to do with missing parameters. It will continue 
  the analysis with default values. Disable this only when you are sure you have 
  setup all the paramters needed. Some parameters are mandatory and analysis will 
  stop regardless.

* **Stop analysis on critical error**: if false, spotMAX will log the error 
  and will continue the analysis of the next folder.

* **Use CUDA-compatible GPU**: if true and CUDA libraries are installed, spotMAX 
  can run some of the analysis steps on the GPU, significantly increasing overall 
  analysis speed.

* **Number of threads used by numba**: if the library `numba` is installed, here 
  you can specify how many threads should be used (we recommend to use a maximum 
  equal to the number of CPU cores available). The default value is half of the 
  CPU cores available.

* **Reduce logging verbosity**: if true, you will see almost only progress bars 
  in the terminal during the analysis.


