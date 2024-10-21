.. |compute| image:: ../../images/compute.png
    :width: 20

.. _GitHub: https://github.com/ElpadoCan/SpotMAX/issues

.. _analysis-parameters-tab:

Analysis parameters tab
=======================

On this tab you can set the parameters and interactively test the effect of 
the testable ones. 

For a detailed description of the paramters see the section :ref:`params-desc`. 
Please, take some time to read this and do not hesitate to ask us on our `GitHub`_ 
page or at my email :email:`elpado6872@gmail.com` if something is not clear. 

The testable parameters are those parameters with a compute button |compute| beside. 
Click the compute button to see the effect of that parameter in real-time. 

The parameters can be saved to a INI configuration file by clicking the button 
``Save parameters to file...`` (top-left). This file will be used to run the analysis. 

If you already have a INI parameters file, you can load its content by clicking 
on ``Load parameters from previous analysis...`` (top-left). 

On the top-right of the tab you will find the button ``Run analysis...``.

.. note:: 
    
    The analysis always runs in the terminal, so keep an eye on that. 
    In the terminal, you will also be guided into setting up things like adding 
    or ignoring missing parameters and confirming when you are overwriting some 
    existing file (like from a previous run).

On the bottom-left of the tab you will find the button ``Set measurements to save...``.
Click this button if you want to select which features to save. 

You can find a detailed description of each feature in the section :ref:`single-spot-features` 
and :ref:`aggr-features`. 

Select features to filter valid spots
-------------------------------------

A very useful parameter to remove false detections, is the parameter called 
:confval:`Features and thresholds for filtering true spots`. 

In the GUI you can set this parameter by clicking on the 
``Set features or view the selected ones...``. You will then be able to select 
one or more features and the maximum or minimum value that this feature must have 
to keep a spot. 

For example, let's say you want to filter by Signal-to-Noise Ratio (SNR) of the spots. 
The feature that quantifies the SNR is the :ref:`Effect size (vs. backgr.)`. Since 
SpotMAX computes three different types of effect sizes, in this example we will use 
the Glass' effect size. 

We want to keep only spots that have an Glass's effect size greater than 0.8. Therefore, 
you select the feature and you set the minimum to 0.8. When you save to the INI 
parameters file you will get this entry in the section ``[Spots channel]``:

.. code-block:: ini
    
    Features and thresholds for filtering true spots = 
	    spot_vs_backgr_effect_size_glass, 0.8, None

As we can see, the feature name is the column name followed by 
``, minimum value, maximum value``. In this example, we did not set a maximum value, 
hence the ``None``. 

In many cases however, we do not know what is a good minimum or maximum value. To 
get an estimate for this we can use the tools available in the :ref:`tune-parameters-tab`. 