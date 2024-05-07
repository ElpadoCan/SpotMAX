.. |plus| image:: ../../../../resources/icons/math/add.svg
    :width: 20

.. _inspect-results-tab:

Inspect and edit results
========================

On this tab you can inspect and edit the results from a previous analysis. 

To load the results click on ``Load results from previous analysis...`` on the 
top-left of the tab. 

When you hover with the mouse cursor onto a spot on the image you will see its 
coordinates plus any feature you like to view.

To view more features click on the plus |plus| button beside the feature name 
and then click on the ``Click to select feature to view...`` button to select the 
feature to view. 

By default, edits are disabled, but you can enable them by activating the  
``Edit results`` toggle. To save the manually edited output, click on the 
``Save edits`` button. You will then be asked to provide additional text 
that will be added to two new files::

    <run_number>_4_<source_table>_<input_text>_<appended_text>.<ext>
    <run_number>_4_<source_table>_<input_text>_<appended_text>_aggregated.csv 

where ``<input_text>`` is the additional text you provided and ``<source_table>`` 
is the part of the table's filename that has been edited. See the 
:ref:`output-files` section for details about the other parts of the filename 
and the difference between ``_aggregated.csv`` and ``.<ext>`` files. 

.. admonition:: example

    If you load the table ``1_2_spotfit_text.h5`` and for the ``<input_text>`` 
    you decide on the text "edited", the edited table's filename will be 
    called ``1_4_2_spotfit_text_edited.h5``.

Possible edits are deleting or adding spots. You don't need to worry about 
clicking exactly on the center of the spot since the added point will snap 
to the closest maximum in the spot area. Click on a point to delete it or click 
on an area without points to add a new spot.

After adding spots, you can decide whether to quantify the features of the new 
spots by clicking on ``Compute features of new spots...``. 