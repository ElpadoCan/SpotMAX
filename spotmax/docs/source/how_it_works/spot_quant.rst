.. _spot_quant:

Spot quantification
===================

To quantify spot features (module 4), SpotMAX uses the pixels of a spheroid 
mask with an expected spot size centred at the spot centre. See the 
parameter :confval:`Spot (z, y, x) minimum dimensions (radius)` for more 
details.

These features include several intensity distribution metrics (mean, max, 
median, quantiles, etc.) from the raw, Gaussian-filtered, and DoG-filtered 
images including different background correction strategies. 

Spots can also be further quantified with a Gaussian peak fitting procedure 
(SpotFIT). We optimised this procedure for high-density spots where multiple 
spots are fitted together with a sum of 3D Gaussian functions. 

The SpotFIT framework computes additional features, including the total integral 
of the Gaussian function, the size of the spot, the amplitude, and the 
background level. 

Calculated features are explained in this section :ref:`params-desc`. 
