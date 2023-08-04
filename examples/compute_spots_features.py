import numpy as np

from cellacdc.plot import imshow
from cellacdc.myutils import img_to_float

from spotmax import data
from spotmax import pipe

mitoData = data.MitoDataSnapshot()
spots_data = img_to_float(mitoData.spots_image_data())
lab = mitoData.segm_data()

spots_zyx_radii = np.array([2.5, 4.5, 4.5])
zyx_coords = np.array([
    [20, 134, 132],
    [19, 122, 168],
])

imshow(spots_data, points_coords=zyx_coords)

pipe.compute_spots_features(
    spots_data, 
    zyx_coords, 
    spots_zyx_radii,
    lab=lab, 
    gauss_sigma=0.75
)