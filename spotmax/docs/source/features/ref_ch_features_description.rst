.. _ref_ch_features:

Reference channel features description
======================================

Description of all the features saved by SpotMAX for each segmented object 
(e.g., single cells, see :confval:`Cells segmentation end name` 
parameter) based on the reference channel segmentation masks (see the 
:confval:`Segment reference channel` parameter). 

There are two types of features: a) single segmented object level, called 
"whole object" (e.g., volume of the segmented reference channel in the single 
cell), and b) sub-object level, where the reference channel mask in the single 
object is separated into non-touching objects. 

These features can be used to filter the reference channel masks using the 
parameter :confval:`Features for filtering ref. channel objects` and they can 
be save to a CSV file using the paramter :confval:`Save reference channel features`. 

Background metrics - whole object
---------------------------------

* **Mean**: column name ``background_ref_ch_mean_intensity``.
* **Sum**: column name ``background_ref_ch_sum_intensity``.
* **Median**: column name ``background_ref_ch_median_intensity``.
* **Min**: column name ``background_ref_ch_min_intensity``.
* **Max**: column name ``background_ref_ch_max_intensity``.
* **25 percentile**: column name ``background_ref_ch_q25_intensity``.
* **75 percentile**: column name ``background_ref_ch_q75_intensity``.
* **5 percentile**: column name ``background_ref_ch_q05_intensity``.
* **95 percentile**: column name ``background_ref_ch_q95_intensity``.
* **Standard deviation**: column name ``background_ref_ch_std_intensity``.

Intensity metrics - whole object
--------------------------------

* **Mean**: column name ``ref_ch_mean_intensity``.
* **Background corrected mean**: column name ``ref_ch_backgr_corrected_mean_intensity``.
* **Sum**: column name ``ref_ch_sum_intensity``.
* **Background corrected sum**: column name ``ref_ch_backgr_corrected_sum_intensity``.
* **Median**: column name ``ref_ch_median_intensity``.
* **Min**: column name ``ref_ch_min_intensity``.
* **Max**: column name ``ref_ch_max_intensity``.
* **25 percentile**: column name ``ref_ch_q25_intensity``.
* **75 percentile**: column name ``ref_ch_q75_intensity``.
* **5 percentile**: column name ``ref_ch_q05_intensity``.
* **95 percentile**: column name ``ref_ch_q95_intensity``.
* **Standard deviation**: column name ``ref_ch_std_intensity``.

Morphological metrics - whole object
------------------------------------

* **Volume (voxel)**: column name ``ref_ch_vol_vox``.
* **Volume (fL)**: column name ``ref_ch_vol_um3``.
* **Number of fragments**: column name ``ref_ch_num_fragments``.

Intensity metrics - sub-object
------------------------------

* **Mean**: column name ``sub_obj_ref_ch_mean_intensity``.
* **Background corrected mean**: column name ``sub_obj_ref_ch_backgr_corrected_mean_intensity``.
* **Sum**: column name ``sub_obj_ref_ch_sum_intensity``.
* **Background corrected sum**: column name ``sub_obj_ref_ch_backgr_corrected_sum_intensity``.
* **Median**: column name ``sub_obj_ref_ch_median_intensity``.
* **Min**: column name ``sub_obj_ref_ch_min_intensity``.
* **Max**: column name ``sub_obj_ref_ch_max_intensity``.
* **25 percentile**: column name ``sub_obj_ref_ch_q25_intensity``.
* **75 percentile**: column name ``sub_obj_ref_ch_q75_intensity``.
* **5 percentile**: column name ``sub_obj_ref_ch_q05_intensity``.
* **95 percentile**: column name ``sub_obj_ref_ch_q95_intensity``.
* **Standard deviation**: column name ``sub_obj_ref_ch_std_intensity``.

Morphological metrics - sub-object
----------------------------------

* **Volume (voxel)**: column name ``sub_obj_vol_vox``.
* **Volume (fL)**: column name ``sub_obj_vol_fl``.