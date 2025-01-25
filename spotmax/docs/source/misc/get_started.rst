.. _Cell-ACDC: https://cell-acdc.readthedocs.io/en/latest/index.html
.. _Example data SpotMAX GUI: https://hmgubox2.helmholtz-muenchen.de/index.php/s/kiY9j3zJLjQH4S2
.. _Cell-ACDC folder structure: https://cell-acdc.readthedocs.io/en/latest/data-structure.html

.. |compute| image:: ../images/compute.png
    :width: 20

Getting started
===============

.. note::
    
    If you haven't installed SpotMAX yet, follow these instructions before proceeding 
    :ref:`how-to-install`.

The simplest way to get started is to play around with the graphical user interface (GUI). 
To run the GUI follow these instructions: :ref:`how-to-run-gui`. You can download example 
data from here `Example data SpotMAX GUI`_. Alternatively, if you already cloned 
the entire repo, you will find example data in the folder 
``SpotMAX/examples/test_data_gui``.

In the GUI you can visualize the results of any previous analysis run or 
setup and run a new analysis. 

The easiest way to setup the parameters is to interactively test their effect by 
clicking on the compute button beside each "testable" parameter. 

.. tip:: 

    Before running SpotMAX you probably want to segment the objects where 
    you want to detect spots (e.g., the single cells). To do this you can use 
    our other software called `Cell-ACDC`_

Take some time to read the description of each parameter in this section 
:ref:`params-desc`. Once you are familiar with the parameters you can dive straight 
into our :ref:`tutorials`. 

When you are happy with the paramters you can either run the analysis locally or 
save the paramters to a configuration file and run the analysis in the command line 
in headless mode (without the GUI). 

.. note:: 

    The analysis always runs in the terminal, so keep an eye on that. 
    In the terminal, you will also be guided into setting up things like adding 
    or ignoring missing parameters and confirming when you are overwriting some 
    existing file (like from a previous run).

Recommended workflow
--------------------

While there are multiple ways to run SpotMAX (see the section :ref:`how-to-run`) 
and we certainly encourage you to experiment with the different modules, here 
we want to outline a **recommended workflow**. 

1. Create data structure
~~~~~~~~~~~~~~~~~~~~~~~~

In the first step, you want to organize your images in a folder structure that 
enables batch-processing and loading of the data into the GUI. 

The folder structure required is the same as for our previously published 
software called `Cell-ACDC`_, therefore we recommend starting from there. See 
here for a detailed description of the folder structure 
`Cell-ACDC folder structure`_.

2. Segment objects of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the second step you should segment the objects of interest (e.g., the single 
cells). This is done outside of SpotMAX and, again, we recommend using our
other software called `Cell-ACDC`_. 

This step is **very important** to allow SpotMAX to ignore the background when 
detecting the spots. 

.. tip:: 

    To segment the objects, you can use any software of your choice as long 
    as you save the segmentation masks inside each ``Position_n/Images`` folder. 

    The segmentation file should be named with the following pattern::

        <basename>_segm_<optional_text>.npz
    
    where ``<basename>`` is the common part at the beginning of all the files 
    inside the Position folder and ``<optional_text>`` is any text you like. 
    The file should be readable with Python using the NumPy function 
    ``np.load(segm_filepath)['arr_0']``, which is the default when saving 
    with ``np.savez_compressed(segm_filepath, segm_masks_arr)``. 

3. Select optimal parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the SpotMAX GUI and load one or more Positions. Go through each one of the 
parameters and make sure you understand their meaning. Here you find a detailed 
description :ref:`params-desc`. 

.. seealso::

    We are constantly improving this documentation and we would like to write a FAQ section. 
    If you want to help out, **feel free to submit the questions you have** on our 
    `GitHub page <https://github.com/ElpadoCan/SpotMAX/issues>`_.

Experiment with different parameters and check intermediate results by clicking 
on the |compute| compute button beside each testable parameter. Here you can 
find a guide on how to fine-tune the paramters :ref:`params-tuning`. 

4. Run the analysis on a subset of the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you think you have reasonable parameters, click on the ``Run analysis...`` 
button on the top-right of the :ref:`analysis-parameters-tab`. 

At the end of the analysis, you will be asked to visualize the results. 

.. tip:: 

    If you are working with 3D z-stack data, it can be useful to visualize 
    results in "max-projection". You can select this on the right-side of the 
    scrollbars below the image. 

If you are not happy with the results go back to step 3 and try changing the
parameters. If you are **struggling with finding good parameters**, feel free to 
send us a sample image with a description of what you tried so far. Please, 
include the log file of your best analysis run. You can send us the data 
on our `GitHub page <https://github.com/ElpadoCan/SpotMAX/issues>`_ or 
at my :email:`elpado6872@gmail.com`. 
