import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

import skimage.color

cwd_path = os.path.dirname(os.path.abspath(__file__))
input_sample_filepath = os.path.join(cwd_path, 'input_sample.npy')
output_sample_filepath = os.path.join(cwd_path, 'output_sample_mask.npy')

cover_x_line = 105
cover_y_line = 80
cover_mask_color = np.array((1.0, 0.0, 0.0, 0.0))
cover_alpha = 0.5
PhysicalSizeX = 0.06725
PhysicalSizeZ = 0.35

# Load io
input_sample = np.load(input_sample_filepath)
output_sample_mask = np.load(output_sample_filepath)

# Rescale images to have better aspect ratio
output_sample_mask = skimage.transform.rescale(
    output_sample_mask, (1.0, 2.0, 1.0, 1.0), order=0
)
input_sample = skimage.transform.rescale(
    input_sample, (1.0, 2.0, 1.0, 1.0), order=0
)

# Get projections
lab_xy = output_sample_mask[0].max(axis=0)
lab_xz = output_sample_mask[0].max(axis=1)
lab_yz = output_sample_mask[0].max(axis=2).transpose()
img_xy = input_sample[0].max(axis=0)
img_xz = input_sample[0].max(axis=1)
img_yz = input_sample[0].max(axis=2).transpose()



ax_img_mapper = {
    'xz': (img_xz, lab_xz, cover_x_line, 'axvline'),
    'xy': (img_xy, lab_xy, cover_x_line, 'axvline'),
    'yz': (img_yz, lab_yz, cover_y_line, 'axhline')
}

# from cellacdc.plot import imshow
# imshow(img_xy, img_xz, img_yz, lab_xy, lab_xz, lab_yz, max_ncols=3)

# Generate cover png
# mosaic = (
# """
# A.
# BC
# """
# )
# fig = plt.figure()
# fig.subplots_adjust(
#     wspace=0, 
#     hspace=0
# )
# axd = fig.subplot_mosaic(
#     mosaic, 
# )

plt.rcParams.update({'font.size': 14})
fig = plt.figure(layout='constrained')
ax = fig.add_gridspec(top=0.7, right=0.7).subplots()
axes_xy = ax
axes_xz = axes_xy.inset_axes([0, 0.9, 1, 1], sharex=axes_xy)
axes_yx = axes_xy.inset_axes([0.8, 0, 1, 1], sharey=axes_xy)

axd = {
    'xz': axes_xz,
    'xy': axes_xy,
    'yz': axes_yx
}

for axes_id, axes in axd.items():
    img, lab, line_coord, axline_func = ax_img_mapper[axes_id]
    img = skimage.exposure.rescale_intensity(
        img, out_range=(0.0, 1.0)
    )
    img_rgba = skimage.color.gray2rgba(img)
    cover_mask = lab.astype(img_rgba.dtype)
    if axline_func == 'axvline':
        cover_mask[:, line_coord:] = 0
    else:
        cover_mask[line_coord:] = 0
    cover_mask_rgba = skimage.color.gray2rgba(cover_mask)
    cover_mask_rgba[cover_mask>0] = cover_mask_color
    cover_img = cover_alpha*img_rgba + (1-cover_alpha)*cover_mask_rgba
    cover_img[cover_mask_rgba==0] = img_rgba[cover_mask_rgba==0]

    axes.imshow(cover_img)
    axline_func = getattr(axes, axline_func)
    axline_func(line_coord, lw='3', color='limegreen')
    axes.set_xticks([])
    axes.set_xticklabels([])
    axes.set_yticks([])
    axes.set_yticklabels([])
    
axes_xz.set_ylabel('z', color='white')
axes_xy.set_ylabel('y', color='white')
axes_xy.set_xlabel('x', color='white')
axes_yx.set_xlabel('z', color='white')
    
cover_png_path = os.path.join(cwd_path, 'cover.png')
fig.savefig(
    cover_png_path, 
    bbox_inches=Bbox([[0.35, 0], [6.2, 4.9]]), 
    facecolor='0.2'
)

plt.show()

print('*'*100)
print(
    'Done. Cover saved at the following location:\n\n'
    f'  * {cover_png_path}\n'
)
print('*'*100)