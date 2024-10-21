.. _ilastik: https://www.ilastik.org/

.. _CellProfiler: https://cellprofiler.org/

.. _TrackMate: https://imagej.net/plugins/trackmate/

.. _GitHub: https://github.com/ElpadoCan/SpotMAX/issues

.. |compute| image:: ../images/compute.png
    :width: 20

.. _params-tuning:

Fine-tuning analysis parameters
===============================

Here you can find additional information on the process of choosing the 
optimal parameters. 

.. seealso:: 

    One way to choose the parameters is to use the tools available on the GUI 
    described in this section :ref:`tune-parameters-tab`. 

Before optmizing the parameters we need to identify the type of problem that 
we have. There are **two types of problems**:

1. :ref:`too-many-spots` (i.e., too many false positives)
2. :ref:`too-few-spots` (i.e, too few true positives)

Let's start with problem number 1.

.. _too-many-spots:

Too many spots detected
-----------------------

While there could be many reasons why SpotMAX is detecting too many spots, we 
can identify these main issues:

1. Oversegmentation of the spots channel
   See :confval:`Spots segmentation method` parameter.
   
2. Minimum spot size is too small
   See :confval:`Spot (z, y, x) minimum dimensions (radius)` parameter.

3. Ineffective filtering with features
   See :confval:`Features and thresholds for filtering true spots` parameter.

4. Oversegmentation of the reference channel
   Valid only when :confval:`Keep only spots that are inside ref. channel mask` 
   is active)

Oversegmentation of the spots channel before spot detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are having too many false positives the first thing to check is whether 
you have too many areas where we do not expect spots segmented using the method 
selected at the :confval:`Spots segmentation method` parameter. 

To check if true, set up all the parameters and then click on the |compute| 
compute button beside the :confval:`Spots segmentation method` parameter. Inspect the 
results and if the problem is there you can try the following steps

Better pre-processing
"""""""""""""""""""""

* **Increase smoothing**: try to increase :confval:`Initial gaussian filter sigma` 
  parameter since it might help removing noise, hence reducing the area where 
  SpotMAX will look for spots. Try values like 1.0, 2.0, or even 3.0 and beyond.

* **Activate or deactivate sharpening**: try activating/deactivating  
  :confval:`Sharpen spots signal prior detection` parameter

* **Activate aggreation**: if you have multiple objects (e.g., cells) try 
  activating :confval:`Aggregate cells prior analysis` parameter. This could 
  help because thresholding on all the cells at once can help reducing the 
  segmented areas where SpotMAX will look for spots.

.. _better-spots-segm:

Better spots segmentation method
""""""""""""""""""""""""""""""""

This might sound trivial, but make sure that you are using the best 
:confval:`Spots segmentation method`. You can tune this in the 
:ref:`tune-parameters-tab` or by visually inspecting the result of each one 
of the available methods. 

Minimum spot size is too small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another common issue for too many false positives is having a minimum spot 
size that is too small. This is the case when there are multiple detections 
within the same spot.

To fix this, increase :confval:`Resolution multiplier in y- and x- direction` 
and :confval:`Spot minimum z-size (μm)` parameters. You can visually tune 
this in the :ref:`tune-parameters-tab`. 


Ineffective filtering with features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you tried all of the above, it might be time to look into filtering valid 
spots using the features calculated by SpotMAX. You can set these at the 
:confval:`Features and thresholds for filtering true spots` parameter. 

To better understand which feature to use, read their description in the 
section :ref:`single-spot-features`. 

Some of the most used features are the :ref:`Effect size (vs. backgr.)` and 
the :ref:`stat-test-vs-ref-ch`. For example, in the tutorial :ref:`mtdna-yeast`, 
we show that it is beneficial to filter those spots whose mean intensity is 
significantly higher than the same area in the reference channel. 

On the other hand, if you want to get rid of dimmer spots (low signal-to-noise 
ratio (SNR)) any of the effect size described in the seciton 
:ref:`Effect size (vs. backgr.)` are good candidates, since the effect size 
is a measure of the SNR of the spot. 

Another combination that we found working well, is to use an ``OR`` statement 
between global and local effect sizes. For example, you could filter spots 
whose global ``OR`` local :ref:`Effect size (vs. backgr.)` are higher than a 
specific value.

.. tip:: 

    To understand what could be a good minimum effect size, run the analysis 
    without filtering valid spots, load the results into the GUI and check 
    what is the effect size of the spots you want to remove using the tools 
    available in the :ref:`inspect-results-tab`. 


Oversegmentation of the reference channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a reference channel it might be beneficial to use it. SpotMAX can 
automatically segment it and use it to filter valid spots. 

.. note:: 

    This applies only if you activate the 
    :confval:`Keep only spots that are inside ref. channel mask` parameter.

However, oversegmentation can lead to keeping spots that are instead outside of 
the reference channel. Make sure that you are segmenting the reference channel 
correctly by testing with the |compute| compute button beside the 
:confval:`Ref. channel segmentation method`. 

.. _too-few-spots:

Too few spots detected
----------------------

The reasons why SpotMAX does not detect all the true positives are essentially 
opposite to why it detects too many spots (explained above) and they are the 
followning:

1. Undersegmentation of the spots channel
   See :confval:`Spots segmentation method` parameter.
   
2. Minimum spot size is too large
   See :confval:`Spot (z, y, x) minimum dimensions (radius)` parameter.

3. Too aggressive filtering with features
   See :confval:`Features and thresholds for filtering true spots` parameter.

4. Undersegmentation of the reference channel
   Valid only when :confval:`Keep only spots that are inside ref. channel mask` 
   is active)

Undersegmentation of the spots channel before spot detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are having too many false positives the first thing to check is whether 
you have too many areas where we do not expect spots segmented using the method 
selected at the :confval:`Spots segmentation method` parameter. 

To check if true, set up all the parameters and then click on the |compute| 
compute button beside the :confval:`Spots segmentation method` parameter. Inspect the 
results and if the problem is there you can try the following steps

Better pre-processing
"""""""""""""""""""""

* **Decrease smoothing**: try to decrase :confval:`Initial gaussian filter sigma` 
  parameter since the smoothing could be too aggressive resulting in 
  dimmer spots being filtered out. Try also values below 1.0, like 0.75 or 0.5.

* **Activate or deactivate sharpening**: try activating/deactivating  
  :confval:`Sharpen spots signal prior detection` parameter

* **Deactivate aggreation**: if you have multiple objects (e.g., cells) try 
  deactivating :confval:`Aggregate cells prior analysis` parameter. This could 
  help especially if you have large variation of the signal intensities 
  between different cells.

* **Activate removal of hot pixels**: try activating/deactivating  
  :confval:`Remove hot pixels` parameter

Better spots segmentation method
""""""""""""""""""""""""""""""""

See above :ref:`better-spots-segm`.

Minimum spot size is too large
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another common issue for not enough true positives is having a minimum spot 
size that is too large. This can lead to detecting a single spot where there 
are two or more, especially when they are very close to each other.

To fix this, decrease :confval:`Resolution multiplier in y- and x- direction` 
and :confval:`Spot minimum z-size (μm)` parameters. You can visually tune 
this in the :ref:`tune-parameters-tab`. 

Too aggressive filtering with features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using features with the paramter 
:confval:`Features and thresholds for filtering true spots`, make sure 
that you are not removing too many spots. 

To better understand which feature to use, read their description in the 
section :ref:`single-spot-features`. 

Some of the most used features are the :ref:`Effect size (vs. backgr.)` and 
the :ref:`stat-test-vs-ref-ch`. For example, in the tutorial :ref:`mtdna-yeast`, 
we show that it is beneficial to filter those spots whose mean intensity is 
significantly higher than the same area in the reference channel. However, 
if we choose a **p-value** that is too low we would remove what are instead 
true spots.

On the other hand, if you are getting rid of dimmer spots using the 
:ref:`Effect size (vs. backgr.)` try reducing the minimum allowed. 

.. tip:: 

    To understand what could be a good minimum effect size, run the analysis 
    without filtering valid spots, load the results into the GUI and check 
    what is the effect size of the spots you want to remove using the tools 
    available in the :ref:`inspect-results-tab`. 


Undersegmentation of the reference channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a reference channel it might be beneficial to use it. SpotMAX can 
automatically segment it and use it to filter valid spots. 

.. note:: 

    This applies only if you activate the 
    :confval:`Keep only spots that are inside ref. channel mask` parameter.

However, undersegmentation can lead to removing spots that are inside the 
reference channel. Make sure that you are segmenting the reference channel 
correctly by testing with the |compute| compute button beside the 
:confval:`Ref. channel segmentation method`. 

Nothing works
-------------

If you tried many combinations of parameters and nothing seem to work there are 
three options:

1. **Use external software for some of the analysis steps**
2. **Train SpotMAX AI** on your data
3. **Submit your case with some sample data**


Use external software for some of the analysis steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some of the analysis steps within SpotMAX can be replaced with results you 
obtain with other software. For example, you could segment the spots or the 
reference channel with `ilastik`_, `CellProfiler`_, or `TrackMate`_ to cite a 
few, save the results to a TIFF file and provide this to SpotMAX at the 
parameters :confval:`Spots channel segmentation end name` and 
:confval:`Ref. channel segmentation end name`. If you do this, SpotMAX 
will not perform these steps and will instead use your external TIFF file. 

Train SpotMAX AI on your data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have some experience with Python (and ideally access to a GPU) you can 
easily train the SpotMAX neural network on your data. Few manually annotated 
images could actually make a big difference. 

See this repository for instructions on how to train the model on your data: 
`SpotMAX AI <https://github.com/ElpadoCan/SpotMAX-Unet>`_. 


Submit your case with some sample data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feel free to submit your case with some sample data and the parameters you 
tried so far by opening an issue on our `GitHub`_ page or by sending me an 
email at :email:`elpado6872@gmail.com`. 

Until next time! 