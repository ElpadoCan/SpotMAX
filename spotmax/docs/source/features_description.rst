Features description
====================

Description of all the features saved by spotMAX and the corresponding column name.

.. contents::

.. _Effect size (vs. backgr.):

Effect size (vs. backgr.)
-------------------------

The effect size is a measure of Signal-to-Noise Ratio (SNR). It is a standardized 
measurement that does not depend on the absolute intensities. There are multiple ways 
to calculate the effect size (see below). 

In this case, the ``vs. backgr.`` means that the background is the negative sample, 
i.e., the Noise part in the SNR. 

Without a reference channel, the background is determined as the pixels outside of the spots 
and inside the segmented object (e.g., the single cell). To determine if a pixel is inside 
or outside of the spot, spotMAX will construct a mask for the spots using spheroids 
centered on each detected spot with size given by the values you provide in the 
``METADATA`` section of the INI parameters file. Note that if you are working 
with a reference channel and you set the parameter 
``Use the ref. channel mask to determine background = True`` then the backround 
will be determined as the pixels outside of the spots and inside the reference 
channel mask.

This metric is useful to determine how bright the spots are compared to the 
background. As a rule of thumb, 0.2 is a small effect, while 0.8 could mean 
a large effect. However, make sure that you explore your data before deciding 
on a threshold to filter out false positives.

Given ``P`` the pixels intensities inside the spot, ``N`` the background 
intensities, and ``std`` the standard deviation, spotMAX will compute the following 
effect sizes:

* **Glass**: column name ``spot_vs_backgr_effect_size_glass``. Formula: ``(mean(P) - mean(N))/std(N)``
* **Cohen**: column name ``spot_vs_backgr_effect_size_cohen``. Formula: ``(mean(P) - mean(N))/std(N+P)`` 
  where ``std(N+P)`` is the standard deviation of the spots and background 
  intensities pooled together. 
* **Hedge**: column name ``spot_vs_backgr_effect_size_hedge``. Formula: ``cohen_effect_size * correction_factor`` 
  where ``correction_factor = 1 - 3/(4 * Dn - 9)`` with ``Dn`` being the 
  difference number of background's and spots' pixels. 


.. _Effect size (vs. ref. ch.):

Effect size (vs. ref. ch.)
--------------------------

The effect size is a measure of Signal-to-Noise Ratio (SNR). It is a standardized 
measurement that does not depend on the absolute intensities. There are multiple ways 
to calculate the effect size (see below). 

In this case, the ``vs. ref. ch.`` means that the reference channel's intensities 
inside the spots mask (see below) is the negative sample, i.e., the Noise part 
in the SNR. 

To determine if a pixel is inside or outside of the spot, spotMAX will construct 
a mask for the spots using spheroids centered on each detected spot with size 
given by the values you provide in the ``METADATA`` section of the INI parameters 
file.

Note that we cannot compare the intensities of two different channels without any 
normalization (since they are often different stains or fluorophores and they 
are excited at different light intensities). Before computing the effect size, 
spotMAX will normalize each channel individually by dividing with the median of 
the background pixels' intensities. See the `Effect size (vs. backgr.)`_ section  
for more information about how the background mask is determined.

This metric is useful to determine how bright the spots are compared to the 
reference channel. As a rule of thumb, 0.2 is a small effect, while 0.8 could mean 
a large effect. However, make sure that you explore your data before deciding 
on a threshold to filter out false positives.

Given ``P`` the pixels intensities inside the spot, ``N`` the background 
intensities, and ``std`` the standard deviation, spotMAX will compute the following 
effect sizes:

* **Glass**: column name ``spot_vs_ref_ch_effect_size_glass``. Formula: ``(mean(P) - mean(N))/std(N)``
* **Cohen**: column name ``spot_vs_ref_ch_effect_size_cohen``. Formula: ``(mean(P) - mean(N))/std(N+P)`` 
  where ``std(N+P)`` is the standard deviation of the spots and background 
  intensities pooled together. 
* **Hedge**: column name ``spot_vs_ref_ch_effect_size_hedge``. Formula: ``cohen_effect_size * correction_factor`` 
  where ``correction_factor = 1 - 3/(4 * Dn - 9)`` with ``Dn`` being the 
  difference number of background's and spots' pixels. 


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


Raw intens. metric
------------------

Raw spots intensities distribution metrics.

* **Mean**: column name ``spot_raw_mean_in_spot_minimumsize_vol``.
* **Sum**: column name ``spot_raw_sum_in_spot_minimumsize_vol``.
* **Median**: column name ``spot_raw_median_in_spot_minimumsize_vol``.
* **Min**: column name ``spot_raw_min_in_spot_minimumsize_vol``.
* **Max**: column name ``spot_raw_max_in_spot_minimumsize_vol``.
* **25 percentile**: column name ``spot_raw_q25_in_spot_minimumsize_vol``.
* **75 percentile**: column name ``spot_raw_q75_in_spot_minimumsize_vol``.
* **5 percentile**: column name ``spot_raw_q05_in_spot_minimumsize_vol``.
* **95 percentile**: column name ``spot_raw_q95_in_spot_minimumsize_vol``.


Preprocessed intens. metric
---------------------------
* **Mean**: column name ``spot_preproc_mean_in_spot_minimumsize_vol``.
* **Sum**: column name ``spot_preproc_sum_in_spot_minimumsize_vol``.
* **Median**: column name ``spot_preproc_median_in_spot_minimumsize_vol``.
* **Min**: column name ``spot_preproc_min_in_spot_minimumsize_vol``.
* **Max**: column name ``spot_preproc_max_in_spot_minimumsize_vol``.
* **25 percentile**: column name ``spot_preproc_q25_in_spot_minimumsize_vol``.
* **75 percentile**: column name ``spot_preproc_q75_in_spot_minimumsize_vol``.
* **5 percentile**: column name ``spot_preproc_q05_in_spot_minimumsize_vol``.
* **95 percentile**: column name ``spot_preproc_q95_in_spot_minimumsize_vol``.


Spotfit size metric
-------------------
* **Radius x- direction**: column name ``sigma_x_fit``.
* **Radius y- direction**: column name ``sigma_y_fit``.
* **Radius z- direction**: column name ``sigma_z_fit``.
* **Mean radius xy- direction**: column name ``sigma_yx_mean_fit``.
* **Spot volume (voxel)**: column name ``spheroid_vol_vox_fit``.


Spotfit intens. metric
----------------------
* **Total integral gauss. peak**: column name ``total_integral_fit``.
* **Foregr. integral gauss. peak**: column name ``foreground_integral_fit``.
* **Amplitude gauss. peak**: column name ``A_fit``.
* **Backgr. level gauss. peak**: column name ``B_fit``.


Spotfit Goodness-of-fit
-----------------------
* **RMS error gauss. fit**: column name ``RMSE_fit``.
* **Normalised RMS error gauss. fit**: column name ``NRMSE_fit``.
* **F-norm. RMS error gauss. fit**: column name ``F_NRMSE_fit``.

Post-analysis metrics
---------------------

* **Consecutive spots distance**: column_name ``consecutive_spots_distance_``.