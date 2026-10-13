On this tab, you can manually add or remove spots from the results of a previous
analysis. You can then save the edited results, optionally compute numerical
features for the newly added spots, and continue working with the edited files
in a later session.

Selecting the annotations to edit
---------------------------------

You can load and visualise results from multiple previous analyses, but you can
**edit only one annotation set at a time**.

Before activating edits, select exactly one tool button in the vertical toolbar
on the left. Each tool button corresponds to a loaded annotation set and uses
the same icon as the spot markers. The selected annotation set is the one that
will be edited.

If no tool button or multiple tool buttons are selected, a warning will be
displayed with instructions on how to proceed.

Activating edits
----------------

To enable manual editing, activate the ``Activate edits`` toggle in the
``Edit results`` section of the ``Inspect and edit results`` tab.

Edits are disabled by default. Make sure that the correct annotation set is
selected before activating them.

Adding and removing spots
-------------------------

Once editing is activated, you can modify the selected annotation set directly
on the image:

* **Add a spot:** left-click on the image at the desired location.
* **Remove a spot:** click on an existing spot marker.

You can optionally enable ``Snap to maximum intensity``. When this option is
enabled, newly added spots automatically snap to the nearest local maximum
intensity. This can help position the added spots more accurately.

Saving edited results
---------------------

When you are finished editing, you can save the results with or without
computing numerical features for the newly added spots.

Before saving, provide a value in ``Text to add to new filenames`` in the
``Edit results`` section of the ``Inspect and edit results`` tab. This text is
added to the filenames of the newly saved results, allowing you to distinguish
the edited files from the original analysis results.

If the required text is missing, or another prerequisite has not been met, a
warning will be displayed with instructions on how to proceed.

The edited results are saved using the following two filename formats::

    <run_number>_4_<source_table>_<input_text>_<appended_text>.<ext>
    <run_number>_4_<source_table>_<input_text>_<appended_text>_aggregated.csv

Here, ``<input_text>`` is the text provided by the user during the analysis (see :confval:``), and ``<source_table>`` identifies the source table that was edited. 
For more information about the filename components and the difference between the
``_aggregated.csv`` and ``.<ext>`` files, see :ref:`output-files`.

For example, if you edit ``3_1_valid_spots.csv`` and enter ``test`` as the
text to add to new filenames, the edited results are initially saved as::

    3_4_1_valid_spots_test.csv
    3_4_1_valid_spots_test_aggregated.csv

These files contain the manually edited spots without their numerical features.

Computing features for newly added spots
----------------------------------------

To compute numerical features for the newly added spots, click on
``Compute features of new spots...``.

SpotMAX uses the manually edited CSV file as the input coordinates and skips
spot detection, proceeding directly to feature quantification. The analysis
parameters are taken from the original analysis parameter file, with the
following parameter updated to point to the manually edited spots::

    Spots channel segmentation end name = 3_4_1_valid_spots_test

The corresponding analysis parameter file is::

    3_analysis_parameters.ini

The manually edited CSV file, ``3_4_1_valid_spots_test.csv``, is saved in both
the ``Position_n/spotMAX_output`` and ``Position_n/Images`` folders.

After quantification, SpotMAX saves the edited spots with their numerical
features in the regular SpotMAX output format. For the example above, the
resulting file is::

    3_1_valid_spots_test.csv

This file contains the manually edited spots together with their computed
numerical features. The filename follows the regular SpotMAX output naming
convention because the spots have now been quantified. See
:ref:`output-files` for additional information about the output filename
components.

Saving without computing features
---------------------------------

If you choose to save the edits without computing numerical features, the
manually edited CSV file contains the coordinates of the added spots without
their numerical features.

SpotMAX will ask you to confirm this choice before saving. This allows you to
review the decision before proceeding.

You can compute the features later by loading the manually edited file and
clicking ``Compute features of new spots...``.

Reusing manually edited files
-----------------------------

You can save the manually edited CSV file and load it in a later session to
continue working with it.

SpotMAX automatically recognises manually edited files from the ``4_``
component of their filename, such as ``3_4_1_valid_spots_test.csv``. These
files are treated as manually edited results, and SpotMAX allows them to be
overwritten when saving further edits.

This makes it possible to save intermediate edits, reload them later, and
continue refining the annotations.