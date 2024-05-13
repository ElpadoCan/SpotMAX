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