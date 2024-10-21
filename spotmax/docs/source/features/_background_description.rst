Without a reference channel, the background is determined as the pixels outside 
of the spots and inside the segmented object (e.g., the single cell). 
To determine if a pixel is inside or outside of the spot, SpotMAX will 
construct a mask for the spots using spheroids centered on each detected 
spot with size given by the values you provide in the 
``METADATA`` section of the INI parameters file. 

.. note:: 

  If the parameter :confval:`Spots detection method` is equal to 
  ``Label prediction mask`` the spheroids are replaced with the spot mask from 
  labelling the prediction mask (i.e., segmentation of the spots).

Note that if you are working with a reference channel and you set the parameter 
:confval:`Use the ref. channel mask to determine background` is ``True`` then 
the backround will be determined as the pixels outside of the spots and inside 
the reference channel mask.

.. hint:: 

  The background of those metrics containing the text ``z_slice`` in the column 
  name are calculated from the center z-slice of each spot, while ``local`` 
  means that the background intensities are extracted from the surrounding of 
  each spot (in the z-slice of the spot) using a ring around the spot with 
  width specified in the parameter :confval:`Local background ring width`.