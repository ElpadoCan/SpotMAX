# SpotMAX
SpotMAX is a software framework for the analysis of multi-dimensional microscopy 
data. It is written in Python 3 and it can be used from a GUI embedded in Cell-ACDC, 
from the command line, or from the Python APIs.

## Model description
The model is a U-Net 3D trained on images containing fluorescent diffraction-limited 
globular-like structures (spots). Given a single channel input image it generates 
a boolean mask of the spot areas. 

The input image can be either 2D or a 3D z-stack. For 2D images, the model will 
add a single z-slice dimension.

If you need further processing of the spots (spot center detection, validation, 
and quantification), please refer to SpotMAX [documentation](https://spotmax.readthedocs.io/en/latest/). 

## Contact

For questions or issues with this model, please reach out by one (or more) of these options:
- Opening a topic with tags `spotmax` and `bioimageio` on [image.sc](https://forum.image.sc/)
- Creating an issue on our GitHub page https://github.com/SchmollerLab/SpotMAX
- Sending an email to Francesco Padovani at [padovaf@tcd.ie](mailto:padovaf@tcd.ie)