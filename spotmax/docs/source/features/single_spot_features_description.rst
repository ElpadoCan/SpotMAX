.. _single-spot-features:

.. role:: m(math)

Single-spot features description
================================

Description of all the features saved by spotMAX for each single spot and the 
corresponding column name.

.. contents::

.. _Background metrics from spot detection input image:

Background metrics from spot detection input image
--------------------------------------------------

These are the background metrics computed from the background pixels in the 
same image that is used to detect spots. The image used to detect spots is 
the pre-processed image (see the :ref:`Pre-processing` parameters) after the 
sharpening filter.

.. include:: _background_description.rst

* **Mean**: column name ``background_mean_spot_detection_image``.
* **Mean z-slice**: column name ``background_mean_z_slice_spot_detection_image``.
* **Sum**: column name ``background_sum_spot_detection_image``.
* **Sum z-slice**: column name ``background_sum_z_slice_spot_detection_image``.
* **Median**: column name ``background_median_spot_detection_image``.
* **Median z-slice**: column name ``background_median_z_slice_spot_detection_image``.
* **Min**: column name ``background_min_spot_detection_image``.
* **Min z-slice**: column name ``background_min_z_slice_spot_detection_image``.
* **Max**: column name ``background_max_spot_detection_image``.
* **Max z-slice**: column name ``background_max_z_slice_spot_detection_image``.
* **25 percentile**: column name ``background_25_percentile_spot_detection_image``.
* **25 percentile z-slice**: column name ``background_25_percentile_z_slice_spot_detection_image``.
* **75 percentile**: column name ``background_75_percentile_spot_detection_image``.
* **75 percentile z-slice**: column name ``background_75_percentile_z_slice_spot_detection_image``.
* **5 percentile**: column name ``background_5_percentile_spot_detection_image``.
* **5 percentile z-slice**: column name ``background_5_percentile_z_slice_spot_detection_image``.
* **95 percentile**: column name ``background_95_percentile_spot_detection_image``.
* **95 percentile z-slice**: column name ``background_95_percentile_z_slice_spot_detection_image``.
* **Standard deviation**: column name ``background_std_spot_detection_image``.
* **Standard deviation z-slice**: column name ``background_std_z_slice_spot_detection_image``.


.. _Background metrics from raw intensities:

Background metrics from raw intensities
---------------------------------------

These are the background metrics computed from the background pixels in the 
raw image. 

.. include:: _background_description.rst

* **Mean**: column name ``background_mean_raw_image``.
* **Mean z-slice**: column name ``background_mean_z_slice_raw_image``.
* **Mean local**: column name ``background_local_mean_z_slice_raw_image``.
* **Sum**: column name ``background_sum_raw_image``.
* **Sum z-slice**: column name ``background_sum_z_slice_raw_image``.
* **Sum local**: column name ``background_local_sum_z_slice_raw_image``.
* **Median**: column name ``background_median_raw_image``.
* **Median z-slice**: column name ``background_median_z_slice_raw_image``.
* **Median local**: column name ``background_local_median_z_slice_raw_image``.
* **Min**: column name ``background_min_raw_image``.
* **Min z-slice**: column name ``background_min_z_slice_raw_image``.
* **Min local**: column name ``background_local_min_z_slice_raw_image``.
* **Max**: column name ``background_max_raw_image``.
* **Max z-slice**: column name ``background_max_z_slice_raw_image``.
* **Max local**: column name ``background_local_max_z_slice_raw_image``.
* **25 percentile**: column name ``background_25_percentile_raw_image``.
* **25 percentile z-slice**: column name ``background_25_percentile_z_slice_raw_image``.
* **25 percentile local**: column name ``background_local_25_percentile_z_slice_raw_image``.
* **75 percentile**: column name ``background_75_percentile_raw_image``.
* **75 percentile z-slice**: column name ``background_75_percentile_z_slice_raw_image``.
* **75 percentile local**: column name ``background_local_75_percentile_z_slice_raw_image``.
* **5 percentile**: column name ``background_5_percentile_raw_image``.
* **5 percentile z-slice**: column name ``background_5_percentile_z_slice_raw_image``.
* **5 percentile local**: column name ``background_local_5_percentile_z_slice_raw_image``.
* **95 percentile**: column name ``background_95_percentile_raw_image``.
* **95 percentile z-slice**: column name ``background_95_percentile_z_slice_raw_image``.
* **95 percentile local**: column name ``background_local_95_percentile_z_slice_raw_image``.
* **Standard deviation**: column name ``background_std_raw_image``.
* **Standard deviation z-slice**: column name ``background_std_z_slice_raw_image``.
* **Standard deviation local**: column name ``background_local_std_z_slice_raw_image``.

.. _Background metrics from preproc. intensities:

Background metrics from preproc. intensities
--------------------------------------------

These are the background metrics computed from the background pixels in the
pre-processed image (see the :ref:`Pre-processing` parameters) before the 
sharpening filter.

.. include:: _background_description.rst

* **Mean**: column name ``background_mean_preproc_image``.
* **Mean z-slice**: column name ``background_mean_z_slice_preproc_image``.
* **Mean local**: column name ``background_local_mean_z_slice_preproc_image``.
* **Sum**: column name ``background_sum_preproc_image``.
* **Sum z-slice**: column name ``background_sum_z_slice_preproc_image``.
* **Sum local**: column name ``background_local_sum_z_slice_preproc_image``.
* **Median**: column name ``background_median_preproc_image``.
* **Median z-slice**: column name ``background_median_z_slice_preproc_image``.
* **Median local**: column name ``background_local_median_z_slice_preproc_image``.
* **Min**: column name ``background_min_preproc_image``.
* **Min z-slice**: column name ``background_min_z_slice_preproc_image``.
* **Min local**: column name ``background_local_min_z_slice_preproc_image``.
* **Max**: column name ``background_max_preproc_image``.
* **Max z-slice**: column name ``background_max_z_slice_preproc_image``.
* **Max local**: column name ``background_local_max_z_slice_preproc_image``.
* **25 percentile**: column name ``background_25_percentile_preproc_image``.
* **25 percentile z-slice**: column name ``background_25_percentile_z_slice_preproc_image``.
* **25 percentile local**: column name ``background_local_25_percentile_z_slice_preproc_image``.
* **75 percentile**: column name ``background_75_percentile_preproc_image``.
* **75 percentile z-slice**: column name ``background_75_percentile_z_slice_preproc_image``.
* **75 percentile local**: column name ``background_local_75_percentile_z_slice_preproc_image``.
* **5 percentile**: column name ``background_5_percentile_preproc_image``.
* **5 percentile z-slice**: column name ``background_5_percentile_z_slice_preproc_image``.
* **5 percentile local**: column name ``background_local_5_percentile_z_slice_preproc_image``.
* **95 percentile**: column name ``background_95_percentile_preproc_image``.
* **95 percentile z-slice**: column name ``background_95_percentile_z_slice_preproc_image``.
* **95 percentile local**: column name ``background_local_95_percentile_z_slice_preproc_image``.
* **Standard deviation**: column name ``background_std_preproc_image``.
* **Standard deviation z-slice**: column name ``background_std_z_slice_preproc_image``.
* **Standard deviation local**: column name ``background_local_std_z_slice_preproc_image``.

.. _Size of the spots metrics:

Size of the spots metrics
-------------------------

The spot mask is the spheroid with radii equal to 
:confval:`Spot (z, y, x) minimum dimensions (radius)` if 
:confval:`Spots detection method` is 'Detect local peaks'. Otherwise, when using 
'Label prediction mask' the spot mask is the actual segmentation of the 
spots.

* **Spot mask volume (voxel)**: column name ``spot_mask_volume_voxel``.
* **Spot mask volume (fL)**: column name ``spot_mask_volume_fl``.


.. _Effect size (vs. backgr.):

Effect size (vs. backgr.)
-------------------------

The effect size is a measure of Signal-to-Noise Ratio (SNR). It is a standardized 
measurement that does not depend on the absolute intensities. There are multiple ways 
to calculate the effect size (see below). 

.. include:: _effect_size_description.rst

In this case, the ``vs. backgr.`` means that the background is the negative sample, 
i.e., the Noise part in the SNR. 

.. include:: _background_description.rst

This metric is useful to determine how bright the spots are compared to the 
background. As a rule of thumb, 0.2 is a small effect, while 0.8 could mean 
a large effect. However, make sure that you explore your data before deciding 
on a threshold to filter out false positives.

Given :m:`P` the pixels intensities inside the spot, :m:`N` the background 
intensities, and :m:`\mathrm{std}` the standard deviation, spotMAX will compute 
the following effect sizes:

* **Glass**: column name ``spot_vs_backgr_effect_size_glass``. 
  Formula: :m:`(\mathrm{mean}(P) - \mathrm{mean}(N))/\mathrm{std}(N)`

* **Cohen**: column name ``spot_vs_backgr_effect_size_cohen``. 
  Formula: :m:`(\mathrm{mean}(P) - \mathrm{mean}(N))/\mathrm{std}(N \cup P)`
  where :m:`\mathrm{std}(N \cup P)` is the standard deviation of the spots and 
  background intensities pooled together. 

* **Hedge**: column name ``spot_vs_backgr_effect_size_hedge``. 
  Formula: :m:`d * c_f` where :m:`d` is the Cohen's effect size and 
  :m:`c_f = 1 - 3/(4 * \Delta n - 9)` with :m:`\Delta n` being the 
  difference between the background's and spots' number of pixels. 

* **Glass (local)**: column name ``spot_vs_local_backgr_effect_size_glass``. 
  Glass's effect size where the **background intensities are obtained from the 
  local environment around the spot** and not from the entire background mask.

* **Cohen (local)**: column name ``spot_vs_local_backgr_effect_size_glass``. 
  Cohen's effect size where the **background intensities are obtained from the 
  local environment around the spot** and not from the entire background mask.

* **Hedge (local)**: column name ``spot_vs_local_backgr_effect_size_glass``. 
  Hedge's effect size where the **background intensities are obtained from the 
  local environment around the spot** and not from the entire background mask. 

Effect size (vs. ref. ch.)
--------------------------

The effect size is a measure of Signal-to-Noise Ratio (SNR). It is a standardized 
measurement that does not depend on the absolute intensities. There are multiple ways 
to calculate the effect size (see below). 

.. include:: _effect_size_description.rst

Here, the ``vs. ref. ch.`` means that the reference channel's intensities 
inside the spots mask (see below) is the negative sample, i.e., the Noise part 
in the SNR. 

To determine if a pixel is inside or outside of the spot, spotMAX will construct 
a mask for the spots using spheroids centered on each detected spot with size 
given by the values you provide in the ``METADATA`` section of the INI parameters 
file.

.. note:: 

  If the parameter :confval:`Spots detection method` is equal to 
  ``Label prediction mask`` the spheroids are replaced with the spot mask from 
  labelling the prediction mask (i.e., segmentation of the spots).

Since we cannot compare the intensities of two different channels without any 
normalization (since they are often different stains or fluorophores and they 
are excited at different light intensities). Before computing the effect size, 
spotMAX will normalize each channel individually by dividing with the median of 
the background pixels' intensities. See the `Effect size (vs. backgr.)`_ section  
for more information about how the background mask is determined.

This metric is useful to determine how bright the spots are compared to the 
reference channel. As a rule of thumb, 0.2 is a small effect, while 0.8 could mean 
a large effect. However, make sure that you explore your data before deciding 
on a threshold to filter out false positives.

Given :m:`P` the pixels intensities inside the spot, :m:`R` the reference channel  
intensities, and :m:`std` the standard deviation, spotMAX will compute the following 
effect sizes:

* **Glass**: column name ``spot_vs_ref_ch_effect_size_glass``. 
  Formula: :m:`(\mathrm{mean}(P) - \mathrm{mean}(N))/\mathrm{std}(N)`

* **Cohen**: column name ``spot_vs_ref_ch_effect_size_cohen``. 
  Formula: :m:`(\mathrm{mean}(P) - \mathrm{mean}(N))/\mathrm{std}(N \cup P)`
  where :m:`\mathrm{std}(N \cup P)` is the standard deviation of the spots and 
  reference channel intensities pooled together.
  
* **Hedge**: column name ``spot_vs_ref_ch_effect_size_hedge``. 
  Formula: :m:`d * c_f` where :m:`d` is the Cohen's effect size and 
  :m:`c_f = 1 - 3/(4 * \Delta n - 9)` with :m:`\Delta n` being the 
  difference between the reference channel's and spots' number of pixels. 

Statistical test (vs. backgr.)
------------------------------

Welch's t-test to determine statistical significance of the difference between 
the means of two populations (spots intensities vs. background). 
The null hypothesis is that the two independent samples have identical average.

See the `Effect size (vs. backgr.)`_ section for an explanation on the meaning  
of ``vs. backgr.`` and how pixels are assigned to spots and reference 
samples.

These metrics are useful to determine if the spots are brighter than the background. 
For example, with ``spot_vs_backgr_ttest_tstat > 0`` and 
``spot_vs_backgr_ttest_pvalue < 0.025`` we would filter out spots whose mean is 
greater than the background given the statistical significance level of 0.025.

* **t-statistic**: column name ``spot_vs_backgr_ttest_tstat``. The t-statistic of 
  the test. A positive t-statistic means that the mean of the spot intensities is 
  higher than the mean of the background.
* **p-value (t-test)**: column name ``spot_vs_backgr_ttest_pvalue``. The p-value 
  associated with the alternative hypothesis.


.. _stat-test-vs-ref-ch:

Statistical test (vs. ref. ch.)
-------------------------------

Welch's t-test to determine statistical significance of the difference between 
the means of two populations (spots intensities vs. reference channel). 
The null hypothesis is that the two independent samples have identical average.

See the `Effect size (vs. ref. ch.)`_ section for an explanation on the meaning  
of ``ref. ch.``, how pixels are assigned to spots and reference 
samples, and how spots and reference channels are normalized before comparison.

These metrics are useful to determine if the spots are brighter than the reference channel. 
For example, with ``spot_vs_ref_ch_ttest_tstat > 0`` and 
``spot_vs_ref_ch_ttest_pvalue < 0.025`` we would filter out spots whose mean is 
greater than the reference channel given the statistical significance level of 0.025.

* **t-statistic**: column name ``spot_vs_ref_ch_ttest_tstat``. The t-statistic of 
  the test. A positive t-statistic means that the mean of the spot intensities is 
  higher than the mean of the reference channel.
* **p-value (t-test)**: column name ``spot_vs_ref_ch_ttest_pvalue``. The p-value 
  associated with the alternative hypothesis.


Raw intens. metrics
-------------------

Raw spots intensities distribution metrics. As the name suggested, these are 
calculated on the raw image without any filter applied to it. Note that intensities 
are converted to float data type and scaled to the range 0-1 by dividing by the maximum intensity value according 
to the data type of the image (e.g., for 8-bit the maximum is 255). This scaling, 
does not affect the relative differences between intensities. 

The pixels belonging to  a specific spot are determined by constructing a 
spehroid with radii equal to  
:confval:`Spot (z, y, x) minimum dimensions (radius)` if 
:confval:`Spots detection method` is 'Detect local peaks'. Otherwise, when using 
'Label prediction mask' the spheroids are replace by the spot mask of the 
actual segmentation of the spots.

.. note:: 

  Background correction is performed by subtracting the median of the 
  corresponding background pixels. 
  For more info, see the sections about the background metrics.

* **Intensity at spot center**: column name ``spot_center_raw_intensity``. 
  Pixel intensity of the spot center.
* **Spot to background ratio**: column name ``spot_center_raw_intens_to_backgr_median_ratio``.
  Ratio between ``spot_center_raw_intensity`` and ``background_median_raw_image``.
* **Spot to z-slice background ratio**: column name ``spot_center_raw_intens_to_backgr_z_slice_median_ratio``.
  Ratio between ``spot_center_raw_intensity`` and ``background_median_z_slice_raw_image``.
* **Mean**: column name ``spot_raw_mean_in_spot_minimumsize_vol``.
* **Background corrected mean**: column name ``spot_raw_backgr_corrected_mean_in_spot_minimumsize_vol``.
* **Z-slice background corrected mean**: column name ``spot_raw_backgr_z_slice_corrected_mean_in_spot_minimumsize_vol``.
* **Sum**: column name ``spot_raw_sum_in_spot_minimumsize_vol``.
* **Background corrected sum**: column name ``spot_raw_backgr_corrected_sum_in_spot_minimumsize_vol``.
* **Z-slice background corrected sum**: column name ``spot_raw_backgr_z_slice_corrected_sum_in_spot_minimumsize_vol``.
* **Median**: column name ``spot_raw_median_in_spot_minimumsize_vol``.
* **Min**: column name ``spot_raw_min_in_spot_minimumsize_vol``.
* **Max**: column name ``spot_raw_max_in_spot_minimumsize_vol``.
* **25 percentile**: column name ``spot_raw_q25_in_spot_minimumsize_vol``.
* **75 percentile**: column name ``spot_raw_q75_in_spot_minimumsize_vol``.
* **5 percentile**: column name ``spot_raw_q05_in_spot_minimumsize_vol``.
* **95 percentile**: column name ``spot_raw_q95_in_spot_minimumsize_vol``.
* **Standard deviation**: column name ``spot_raw_std_in_spot_minimumsize_vol``.


Preprocessed intens. metrics
----------------------------

Preprocessed spots intensities distribution metrics. These features are 
calculated on the image after it went through the gaussian filter. 
Note that the gaussian filter also scales the intensities to the range
0-1. 

The pixels belonging to  a specific spot are determined by constructing a 
spehroid with radii equal to  
:confval:`Spot (z, y, x) minimum dimensions (radius)` if 
:confval:`Spots detection method` is 'Detect local peaks'. Otherwise, when using 
'Label prediction mask' the spheroids are replace by the spot mask of the 
actual segmentation of the spots.

.. note:: 

  Background correction is performed by subtracting the median of the 
  corresponding background pixels. 
  For more info, see the sections about the background metrics.

* **Intensity at spot center**: column name ``spot_center_preproc_intensity``. 
  Pixel intensity of the spot center.
* **Spot to background ratio**: column name ``spot_center_preproc_intens_to_backgr_median_ratio``.
  Ratio between ``spot_center_preproc_intensity`` and ``background_median_preproc_image``.
* **Spot to z-slice background ratio**: column name ``spot_center_preproc_intens_to_backgr_z_slice_median_ratio``.
  Ratio between ``spot_center_preproc_intensity`` and ``background_median_z_slice_preproc_image``.
* **Mean**: column name ``spot_preproc_mean_in_spot_minimumsize_vol``.
* **Background corrected mean**: column name ``spot_preproc_backgr_corrected_mean_in_spot_minimumsize_vol``.
* **Z-slice background corrected mean**: column name ``spot_preproc_backgr_z_slice_corrected_mean_in_spot_minimumsize_vol``.
* **Sum**: column name ``spot_preproc_sum_in_spot_minimumsize_vol``.
* **Background corrected sum**: column name ``spot_preproc_backgr_corrected_sum_in_spot_minimumsize_vol``.
* **Z-slice background corrected sum**: column name ``spot_preproc_backgr_z_slice_corrected_sum_in_spot_minimumsize_vol``.
* **Median**: column name ``spot_preproc_median_in_spot_minimumsize_vol``.
* **Min**: column name ``spot_preproc_min_in_spot_minimumsize_vol``.
* **Max**: column name ``spot_preproc_max_in_spot_minimumsize_vol``.
* **25 percentile**: column name ``spot_preproc_q25_in_spot_minimumsize_vol``.
* **75 percentile**: column name ``spot_preproc_q75_in_spot_minimumsize_vol``.
* **5 percentile**: column name ``spot_preproc_q05_in_spot_minimumsize_vol``.
* **95 percentile**: column name ``spot_preproc_q95_in_spot_minimumsize_vol``.
* **Standard deviation**: column name ``spot_preproc_std_in_spot_minimumsize_vol``.

.. _spotfit-features:

SpotSIZE metrics
----------------

Features that are computed during the SpotSIZE step. This step is used to determine 
the extent of each spot by iteratively growing a spheroid centerd at each spot 
until the mean of the pixels' intensities on the surface of the spheroid is 
lower than a threshold. The threshold is determined as the median of the background 
plus 3 times the standard deviation of the background pixels' intensities. 
The pixels belonging to the final mask will be used in the spotFIT step. 

* **Background mean**: column name ``spotsize_backgr_mean``.
* **Background median**: column name ``spotsize_backgr_median``.
* **Background standard dev.**: column name ``spotsize_backgr_std``.
* **Maximum intensity inside the spot mask**: column name ``spotsize_A_max``.
* **Initial radius in xy- direction (pixel)**: column name ``spotsize_initial_radius_yx_pixel``. 
  This is the "Spot (z, y, x) minimum dimensions (radius)" parameter divided by 2.
* **Initial radius in z- direction (pixel)**: column name ``spotsize_initial_radius_z_pixel``.
  This is the "Spot (z, y, x) minimum dimensions (radius)" parameter divided by 2.
* **Mean radius xy- direction (micro-m)**: column name ``spotsize_yx_radius_um``.
* **Radius z- direction (micro-m)**: column name ``spotsize_z_radius_um``.
* **Mean radius xy- direction (pixel)**: column name ``spotsize_yx_radius_pxl``.
* **Radius z- direction (pixel)**: column name ``spotsize_z_radius_pxl``.
* **Threshold value to stop growing process**: column name ``spotsize_limit``.
* **Median of the spot's surface intensities**: column name ``spotsize_surface_median``.
* **5 percentile of the spot's surface intensities**: column name ``spotsize_surface_5perc``.
* **Mean of the spot's surface intensities**: column name ``spotsize_surface_mean``.
* **Standard dev. of the spot's surface intensities**: column name ``spotsize_surface_std``.
* **Default minium backround level allowed for spotfit**: column name ``spot_B_min``. 
  This is calculated as the mean of the intensities on the surface of all the spheorids 
  minus 3 times the standard deviation of the same intensities. If negative, 
  it is set to 0.

.. _spotfit-coords:

SpotFIT peak coordinates
------------------------

* **x-coordinate of the gaussian peak**: column name ``x_fit``.
* **y-coordinate of the gaussian peak**: column name ``y_fit``.
* **z-coordinate of the gaussian peak**: column name ``z_fit``.

SpotFIT size metrics
--------------------

Features that are computed during the gaussian fit procedure. 

* **Radius x-direction**: column name ``sigma_x_fit``.
* **Radius y-direction**: column name ``sigma_y_fit``.
* **Radius z-direction**: column name ``sigma_z_fit``.
* **Mean radius xy-direction**: column name ``sigma_yx_mean_fit``.
* **Spot volume (voxel)**: column name ``spheroid_vol_vox_fit``.

SpotFIT intens. metrics
-----------------------

* **Total integral gauss. peak**: column name ``total_integral_fit``. This is 
  the result of the analytical integration of the gaussian curve including 
  the background. 
* **Foregr. integral gauss. peak**: column name ``foreground_integral_fit``. This is 
  the result of the analytical integration of the gaussian curve excluding  
  the background.
* **Amplitude gauss. peak**: column name ``A_fit``.
* **Backgr. level gauss. peak**: column name ``B_fit``.
* **Quality factor in xy-direction**: column name ``Q_factor_yx``. 
  Ratio between ``A_fit`` and  ``sigma_yx_mean_fit``. The higher the quality 
  factor the taller and narrower the peak.
* **Quality factor in z-direction**: column name ``Q_factor_z``. 
  Ration between ``A_fit`` and  ``sigma_z_fit``. The higher the quality 
  factor the taller and narrower the peak.


SpotFIT Goodness-of-fit
-----------------------
* **RMS error gauss. fit**: column name ``RMSE_fit``. Root mean squared error 
  between fitted and predicted data. The lower this value, the better was the fit. 
* **Normalised RMS error gauss. fit**: column name ``NRMSE_fit``. RMS error 
  divided by the mean of the fitted data.
* **F-norm. RMS error gauss. fit**: column name ``F_NRMSE_fit``. Normalised RMS 
  scaled to the range 0-1 using a modified sigmoid function:
  
  .. math::
    F_{NRMSE} = \frac{2}{1 + e^{NRMSE}}

Post-analysis metrics
---------------------

* **Consecutive spots distance (pixel)**: column name ``consecutive_spots_distance_voxel``. 
  Euclidean distance between consecutive pairs of spots without a specific order. 
  Unit is pixels and the coordinates used are the detected center.
* **Consecutive spots distance ((micro-m)**: column name ``consecutive_spots_distance_um``. 
  Euclidean distance between consecutive pairs of spots without a specific order.
  Unit is pixels and the coordinates used are the detected center.
* **Consecutive spots distance from fit coords (pixel)**: column name ``consecutive_spots_distance_fit_voxel``. 
  Euclidean distance between consecutive pairs of spots without a specific order.
  Unit is pixels and the coordinates used are the fitted center from spotFIT step.
* **Consecutive spots distance from fit coords (micro-m)**: column name ``consecutive_spots_distance_fit_voxel``. 
  Euclidean distance between consecutive pairs of spots without a specific order.
  Unit is pixels and the coordinates used are the fitted center from spotFIT step.