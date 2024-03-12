.. _output-files:

Output files
============

Output files will be saved in each Position folder in a sub-folder called 
``spotMAX_output``. In this folder you will find the following set of tables::

    <run_number>_0_detected_spots_<appended_text>.<ext>
    <run_number>_0_detected_spots_<appended_text>_aggregated.csv
    <run_number>_1_valid_spots_<appended_text>.<ext>
    <run_number>_1_valid_spots_<appended_text>_aggregated.csv
    <run_number>_2_spotfit_<appended_text>.<ext>
    <run_number>_2_spotfit_<appended_text>_aggregated.csv
    <run_number>_3_ref_channel_features_<text_to_append>.csv
    <run_number>_3_ref_channel_features_<text_to_append>.csv
    <run_number>_4_<source_table>_<input_text>_<appended_text>.<ext>
    <run_number>_4_<source_table>_<input_text>_<appended_text>_aggregated.csv 
    <run_number>_analysis_parameters_<appended_text>.ini

where ``<run_number>`` is the number selected as the :confval:`Run number` 
parameter, ``<appended_text>`` is the text inserted at the 
:confval:`Text to append at the end of the output files` parameter, and 
``<ext>`` is either ``.csv`` or ``.h5`` as selected at the 
:confval:`File extension of the output tables` parameter. 

.. seealso:: 

    For the file ``<run_number>_3_ref_channel_features_<text_to_append>.csv`` 
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

Finally, the file with ``2_spotfit`` will be created only if 
:confval:`Compute spots size (fit gaussian peak(s))` paramter is True. This 
file contains additional features determined at the spotFIT step, as described 
in the section :ref:`spotfit-features`. 