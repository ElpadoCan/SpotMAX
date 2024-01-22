.. _aggr-features:

.. role:: m(math)

Aggregated features description
===============================

Description of all the features saved by spotMAX for each segmented object 
(aggregated) and the corresponding column name. These are simple aggregations 
like averaging and sum. See the related feature in the :ref:`single-spot-features` 
section for more details.

.. contents::

Spot detection
--------------
* **Number of spots**: column name ``num_spots``.
* **Number of spots inside ref. ch.**: column name ``num_spots_inside_ref_ch``.

Spotfit size metric
-------------------
* **Mean radius x- direction**: column name ``mean_sigma_x_fit``.
* **Mean radius y- direction**: column name ``mean_sigma_y_fit``.
* **Mean radius z- direction**: column name ``mean_sigma_z_fit``.
* **Std. dev. radius x- direction**: column name ``std_sigma_z_fit``.
* **Std. dev. radius y- direction**: column name ``std_sigma_y_fit``.
* **Std. dev. radius z- direction**: column name ``std_sigma_x_fit``.


Spotfit intens. metric
----------------------
* **Sum of total integral gauss. peak**: column name ``sum_tot_integral_fit``.
* **Sum of foregr. integral gauss. peak**: column name ``sum_foregr_integral_fit``.
* **Sum of amplitude gauss. peak**: column name ``sum_A_fit_fit``.
* **Mean backgr. level gauss. peak**: column name ``mean_B_fit_fit``.


Spotfit Goodness-of-fit
-----------------------
* **Mean RMS error gauss. fit**: column name ``mean_RMSE_fit``.
* **Mean normalised RMS error gauss. fit**: column name ``mean_NRMSE_fit``.
* **Mean F-norm. RMS error gauss. fit**: column name ``mean_F_NRMSE_fit``.
  
Reference channel
-----------------
* **Ref. channel volume**: column name ``ref_ch_vol_``.
* **Ref. ch. number of fragments**: column name ``ref_ch_num_fragments``.
  
Segmented objects size
----------------------
* **Area of the segmented object (pixel)**: column name ``cell_area_pxl``.
* **Area of the segmented object (micro-m^2)**: column name ``cell_area_um2``.
* **Estimated 3D volume from 2D mask (pixel)**: column name ``cell_vol_vox``. 
  To calculate cell volume from a 2D mask, the mask is first aligned along its 
  major axis. Next, it is divided into slices perpendicular to the major axis, 
  each slice with width equal to 1 pixel. 
  Assuming rotational symmetry of each 
  slice around its middle axis parallel to the mask's major axis, spotMAX computes 
  the volume of the resulting cylinder. Finally, the volumes of each cylinder 
  are summed to obtain the total volume.
* **Estimated 3D volume from 2D mask (fl)**: column name ``cell_vol_fl``. 
  Estimated 3D volume from 2D mask in pixels converted to femtoliters (equivalent 
  to :m:`\mu m^3`) using the pixel size provided in the parameters.
* **3D volume from 3D mask (voxel)**: column name ``cell_vol_vox_3D``.
* **3D volume from 3D mask (fl)**: column name ``cell_vol_fl_3D``.