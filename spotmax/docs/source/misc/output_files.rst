.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/index.html

.. _output-files:

Output files
============

Spotmax output files include **segmentation, tabular, and processed image data**. 

Segmentation and processed image data will be saved in each 
``Position_n/Images`` folder.

Tabular data will be saved in each Position folder in a sub-folder called 
``SpotMAX_output``.

Segmentation data
-----------------

Segmentation masks are saved from spots and reference channel data. These data 
are optional and can be saved by activating the parameters 
:confval:`Save spots segmentation masks` and 
:confval:`Save reference channel segmentation masks`, respectively.

The naming pattern is the following::

    <basename>_run_num<run_number>_<spots_ch_name>_spots_segm_mask_<appended_text>.npz
    <basename>_run_num<run_number>_<ref_ch_name>_ref_ch_segm_mask_<appended_text>.npz

where ``<basename>`` is the common part of all the file names in the Position 
folder, ``<run_number>`` is the run number defined at :confval:`Run number`, 
the ``<spots_ch_name>`` is the text provided at the :confval:`Spots channel end name` 
parameter, ``<ref_ch_name>`` is the text provided at the :confval:`Reference channel end name` 
parameter, and ``<appended_text>`` is the text provided at the 
:confval:`Text to append at the end of the output files` 
parameter.

Tables
------

In the ``SpotMAX_output`` folder you will find the following set of tables::

    <run_number>_0_detected_spots_<appended_text>.<ext>
    <run_number>_0_detected_spots_<appended_text>_aggregated.csv
    <run_number>_1_valid_spots_<appended_text>.<ext>
    <run_number>_1_valid_spots_<appended_text>_aggregated.csv
    <run_number>_2_spotfit_<appended_text>.<ext>
    <run_number>_2_spotfit_<appended_text>_aggregated.csv
    <run_number>_3_ref_channel_features_<appended_text>.csv
    <run_number>_4_<source_table>_<input_text>_<appended_text>.<ext>
    <run_number>_4_<source_table>_<input_text>_<appended_text>_aggregated.csv 
    <run_number>_analysis_parameters_<appended_text>.ini

where ``<run_number>`` is the number selected as the :confval:`Run number` 
parameter, ``<appended_text>`` is the text inserted at the 
:confval:`Text to append at the end of the output files` parameter, and 
``<ext>`` is either ``.csv`` or ``.h5`` as selected at the 
:confval:`File extension of the output tables` parameter. 

.. seealso:: 

    For the file ``<run_number>_3_ref_channel_features_<appended_text>.csv`` 
    see more details in the description of the :confval:`Save reference channel features` 
    parameter.

    For the files ``<run_number>_4_<source_table>_<input_text>_<appended_text>`` 
    see more details in the :ref:`inspect-results-tab` section.

The file with ``analysis_parameters`` in the name is the INI configuration file 
with all the parameters of that specific analysis run. 

The files ending with ``_aggregated`` contain features related to the single 
segmented objects (e.g., the single cells) as described in the section 
:ref:`aggr-features`, while the other files contain the features related to the 
single spots as described in the section :ref:`single-spot-features`. 

Additionally, ``0_detected_spots`` means that the file contains all the 
detected spots without any filtering, while ``1_valid_spots`` means that the 
file contains the spots after filtering based on the features selected at 
the :confval:`Features and thresholds for filtering true spots`. 

.. note:: 

    The file ``0_detected_spots`` might also contain spots that are outside 
    of the segmented objects. This is by design, because the idea is to save 
    all the detected spots. These spots will appear in the table with 
    the column ``Cell_ID`` equal to 0.

Finally, the file with ``2_spotfit`` will be created only if 
:confval:`Compute spots size (fit gaussian peak(s))` paramter is True. This 
file contains additional features determined at the spotFIT step, as described 
in the section :ref:`spotfit-features`. 

Concatenate multiple experiments results into single file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using the same data structure required by `Cell-ACDC`_ you can 
concatenate multiple Positions and multiple experiments results into a 
single table. 

To do so, run Cell-ACDC and in the small launcher window go to the menu 
on the top menu bar ``Utilies --> Concatenate --> Concatenate SpotMAX output tables...``. 

Select as many experiment and Position folders as you need and, optionally, 
select if you need to copy cell cycle annotations from the Cell-ACDC output 
file. 

The multiple Positions final table will be saved in each experiment folder 
selected in a folder called ``SpotMAX_multipos_output``. The table will have 
an additional column called ``Position_n`` that indicates from which Position 
the data on each comes from. 

If you select more than one experiment folders, Cell-ACDC will also create a 
table with the all the results from each Position and each experiment selected. 
The table will be saved in a folder of your choice (you will be asked to 
select it) and it will have two additional columns called ``experiment_folderpath`` 
and ``experiment_foldername`` to identify where the data come from.

Processed image data
--------------------

Pre-processed images are saved from spots and reference channel data. These data 
are optional and can be saved by activating the parameters 
:confval:`Save pre-processed spots image` and 
:confval:`Save pre-processed reference channel image`, respectively.

The naming pattern is the following::

    <basename>_run_num<run_number>_<spots_ch_name>_preprocessed_<appended_text>.<ext>
    <basename>_run_num<run_number>_<ref_ch_name>_preprocessed_<appended_text>.<ext>

where ``<basename>`` is the common part of all the file names in the Position 
folder, ``<run_number>`` is the run number defined at :confval:`Run number`, 
the ``<spots_ch_name>`` is the text provided at the :confval:`Spots channel end name` 
parameter, ``<ref_ch_name>`` is the text provided at the :confval:`Reference channel end name` 
parameter, ``<appended_text>`` is the text provided at the 
:confval:`Text to append at the end of the output files` 
parameter, and ``<ext>`` is the extension of the input channel file.