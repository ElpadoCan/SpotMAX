import numpy as np

from cellacdc import colors as acdc_colors

from matplotlib.colors import LinearSegmentedColormap

def NeonGreen_plt_cmap(bkgr_color=(0.0, 0.0, 0.0)):
    neon_rgb = (0, 1, 1)
    colors = (bkgr_color, neon_rgb)
    cmap = LinearSegmentedColormap.from_list('NeonGreen', colors)
    return cmap

def mKate_plt_cmap(bkgr_color=(0.0, 0.0, 0.0)):
    neon_rgb = (1, 0, 1)
    colors = (bkgr_color, neon_rgb)
    cmap = LinearSegmentedColormap.from_list('mKate', colors)
    return cmap

def get_greedy_luts(labs: list[np.ndarray]):
    luts = []
    for lab in labs:
        if lab.ndim > 2:
            lab = lab.max(axis=0)
        cmap = acdc_colors.getFromMatplotlib('viridis')
        lut = cmap.getLookupTable(0.2, 1, lab.max()+1)
        lut[0, :3] = [25, 25, 25]
        luts.append(acdc_colors.get_greedy_lut(lab, lut))
    
    return luts

def getFromMatplotlib(*args, **kwargs):
    return acdc_colors.getFromMatplotlib(*args, **kwargs)