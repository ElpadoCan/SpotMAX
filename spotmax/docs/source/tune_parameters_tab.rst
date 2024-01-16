.. _tune-parameters-tab:

Tune parameters tab
===================

This tab contains tools that will allow you to determine the best paramters for 
your dataset. 

Start by clicking on ``Start adding points`` button on the top-right of the tab. 

You can add both true spots but also spots that could be detected but are not valid 
(e.g., dimmer spots that should not be counted). 

To switch between adding true or false positives toggle ``Clicking on true spots`` 
(middle-right of the tab in the "Spots properties" section). In the same section 
you can also change the apperance of the spots or clear the selected spots. 

The first parameter that you can visually tune is the 
:ref:`Resolution multiplier in y- and x- direction <metadata>`. To tune this, 
hover on one spot and press up/down arrows on the keyboard to 
adjust the circle size to the spot. 

Once you add some spots, you can view their features in the "Features of the spot under mouse cursor". 
Select which features you want to view and hover with the mouse onto a spot. 

This step is very useful to get a sense of what could be a minimum and maximum value 
for filtering spots. For example, let's say you want to use the Glass' effect size 
(a measure of Signal-to-Noise Ratio) to filter valid spots. 
However, you don't know what could be a good minimum value. Therefore you can click 
on the dimmer valid spots and view what is their Glass' effect size.

Finally, the :ref:`Spot detection threshold function <spots-channel>` and the 
minimum and maximum values of the selected filtering features can be automatically 
tuned by spotMAX. To do so, click on the "Start autotuning" button on the top-right 
of the tab. 