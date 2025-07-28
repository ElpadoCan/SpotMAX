.. _scikit-image: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops

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

Region properties - whole object
--------------------------------

These are calculated using the function ``skimage.measure.regionprops`` from 
the `scikit-image`_ library. The following properties are calculated:

* **Major axis length**: column name ``ref_ch_major_axis_length``.
* **Minor axis length**: column name ``ref_ch_minor_axis_length``.
* **Equivalent diameter**: column name ``ref_ch_equivalent_diameter``.
* **Volume (voxel)**: column name ``ref_ch_area``.
* **Solidity**: column name ``ref_ch_solidity``.
* **Extent**: column name ``ref_ch_extent``.
* **Volume of the filled region**: column name ``ref_ch_filled_area``.
* **Volume of the bounding box**: column name ``ref_ch_bbox_area``.
* **Area of the convex hull image**: column name ``ref_ch_convex_area``.
* **Euler number**: column name ``ref_ch_euler_number``.
* **Maximum Feret's diameter**: column name ``ref_ch_feret_diameter_max``.
* **Orientation (angle)**: column name ``ref_ch_orientation``.
* **Perimeter**: column name ``ref_ch_perimeter``.
* **Perimeter (Crofton)**: column name ``ref_ch_perimeter_crofton``.
* **Perimeter (Crofton)**: column name ``ref_ch_perimeter_crofton``.
* **Circularity**: column name ``ref_ch_circularity``. 
  Circularity is calculated as follows:

    .. math::
    
        \mathrm{circularity} = \frac{4\pi A}{P^2}
    
    where :math:`A` is the area of the object and :math:`P` is the perimeter 
    of the object.

* **Roundness**: column name ``ref_ch_roundness``.
  Roundness is calculated as follows:

    .. math::
    
        \mathrm{roundness} = \frac{4A}{\pi M^2}
    
    where :math:`A` is the area of the object and :math:`M` is the major  
    axis length of the object.


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

Region properties - sub-object
------------------------------

These are calculated using the function ``skimage.measure.regionprops`` from 
the `scikit-image`_ library. The following properties are calculated:

* **Major axis length**: column name ``sub_obj_ref_ch_major_axis_length``.
* **Minor axis length**: column name ``sub_obj_ref_ch_minor_axis_length``.
* **Equivalent diameter**: column name ``sub_obj_ref_ch_equivalent_diameter``.
* **Solidity**: column name ``sub_obj_ref_ch_solidity``.
* **Extent**: column name ``sub_obj_ref_ch_extent``.
* **Volume of the filled region**: column name ``sub_obj_ref_ch_filled_area``.
* **Volume of the bounding box**: column name ``sub_obj_ref_ch_bbox_area``.
* **Area of the convex hull image**: column name ``sub_obj_ref_ch_convex_area``.
* **Euler number**: column name ``sub_obj_ref_ch_euler_number``.
* **Maximum Feret's diameter**: column name ``sub_obj_ref_ch_feret_diameter_max``.
* **Orientation (angle)**: column name ``sub_obj_ref_ch_orientation``.
* **Perimeter**: column name ``sub_obj_ref_ch_perimeter``.
* **Perimeter (Crofton)**: column name ``sub_obj_ref_ch_perimeter_crofton``.
* **Circularity**: column name ``ref_ch_circularity``. 
  
  Circularity is calculated as follows:

    .. math::
    
        \mathrm{circularity} = \frac{4\pi A}{P^2}
    
    where :math:`A` is the area of the object and :math:`P` is the perimeter 
    of the object.

* **Roundness**: column name ``ref_ch_roundness``.
  
  Roundness is calculated as follows:

    .. math::
    
        \mathrm{roundness} = \frac{4A}{\pi M^2}
    
    where :math:`A` is the area of the object and :math:`M` is the major  
    axis length of the object.