import pandas as pd
import datetime
import cv2
import numpy as np
import tkinter as tk

import skimage.color

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

def isRGB(img):
    if img.ndim == 3 and img.shape[-1] == 3:
        return True
    elif 2 < img.ndim > 2:
        raise IndexError(
            f'Image is not 2D (shape = {img.shape}) '
            'and last dimension is not == 3'
        )
    else:
        return False

def mergeChannels(channel1_img, channel2_img, color, alpha):
    if not isRGB(channel1_img):
        channel1_img = skimage.color.gray2rgb(channel1_img/channel1_img.max())
    if not isRGB(channel2_img):
        if channel2_img.max() > 0:
            channel2_img = skimage.color.gray2rgb(channel2_img/channel2_img.max())
        else:
            channel2_img = skimage.color.gray2rgb(channel2_img)

    colorRGB = [v/255 for v in color][:3]
    merge = (channel1_img*(1.0-alpha)) + (channel2_img*alpha*colorRGB)
    merge = merge/merge.max()
    merge = (np.clip(merge, 0, 1)*255).astype(np.uint8)
    return merge

def objContours(obj):
    contours, _ = cv2.findContours(
        obj.image.astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    min_y, min_x, _, _ = obj.bbox
    cont = np.squeeze(contours[0], axis=1)
    cont = np.vstack((cont, cont[0]))
    cont += [min_x, min_y]
    return cont

def pdDataFrame_boolTo0s1s(df, labelsToCast, axis=0):
    df = df.copy()

    if isinstance(labelsToCast, str) and labelsToCast == 'allRows':
        labelsToCast = df.index
        axis=0

    for label in labelsToCast:
        if axis==0:
            series = df.loc[label]
        else:
            series = df[label]

        isObject = pd.api.types.is_object_dtype(series)
        isString = pd.api.types.is_string_dtype(series)
        isBool = pd.api.types.is_bool_dtype(series)

        if isBool:
            series = series.replace({True: 'yes', False: 'no'})
            df[label] = series
        elif (isObject or isString):
            series = (series.str.lower()=='true') | (series.str.lower()=='yes')
            series = series.replace({True: 'yes', False: 'no'})
            if axis==0:
                if ((df.loc[label]=='True') | (df.loc[label]=='False')).any():
                    df.loc[label] = series
            else:
                if ((df[label]=='True') | (df[label]=='False')).any():
                    df[label] = series
    return df

def seconds_to_ETA(seconds):
    seconds = round(seconds)
    ETA = datetime.timedelta(seconds=seconds)
    ETA_split = str(ETA).split(':')
    if seconds >= 86400:
        days, hhmmss = str(ETA).split(',')
        h, m, s = hhmmss.split(':')
        ETA = f'{days}, {int(h):02}h:{int(m):02}m:{int(s):02}s'
    else:
        h, m, s = str(ETA).split(':')
        ETA = f'{int(h):02}h:{int(m):02}m:{int(s):02}s'
    return ETA

class imshow_tk:
    def __init__(self, img, dots_coords=None, x_idx=1, axis=None,
                       additional_imgs=[], titles=[], fixed_vrange=False,
                       run=True):
        if img.ndim == 3:
            if img.shape[-1] > 4:
                img = img.max(axis=0)
                h, w = img.shape
            else:
                h, w, _ = img.shape
        elif img.ndim == 2:
            h, w = img.shape
        elif img.ndim != 2 and img.ndim != 3:
            raise TypeError(f'Invalid shape {img.shape} for image data. '
            'Only 2D or 3D images.')
        for i, im in enumerate(additional_imgs):
            if im.ndim == 3 and im.shape[-1] > 4:
                additional_imgs[i] = im.max(axis=0)
            elif im.ndim != 2 and im.ndim != 3:
                raise TypeError(f'Invalid shape {im.shape} for image data. '
                'Only 2D or 3D images.')
        n_imgs = len(additional_imgs)+1
        if w/h > 1:
            fig, ax = plt.subplots(n_imgs, 1, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots(1, n_imgs, sharex=True, sharey=True)
        if n_imgs == 1:
            ax = [ax]
        self.ax0img = ax[0].imshow(img)
        if dots_coords is not None:
            ax[0].plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
        if axis:
            ax[0].axis('off')
        if fixed_vrange:
            vmin, vmax = img.min(), img.max()
        else:
            vmin, vmax = None, None
        self.additional_aximgs = []
        for i, img_i in enumerate(additional_imgs):
            axi_img = ax[i+1].imshow(img_i, vmin=vmin, vmax=vmax)
            self.additional_aximgs.append(axi_img)
            if dots_coords is not None:
                ax[i+1].plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
            if axis:
                ax[i+1].axis('off')
        for title, a in zip(titles, ax):
            a.set_title(title)
        sub_win = embed_tk('Imshow embedded in tk', [800,600,400,150], fig)
        sub_win.root.protocol("WM_DELETE_WINDOW", self._close)
        self.sub_win = sub_win
        self.fig = fig
        self.ax = ax
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        if run:
            sub_win.root.mainloop()

    def _close(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

class embed_tk:
    """Example:
    -----------
    img = np.ones((600,600))
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    ax.imshow(img)

    sub_win = embed_tk('Embeddding in tk', [1024,768,300,100], fig)

    def on_key_event(event):
        print('you pressed %s' % event.key)

    sub_win.canvas.mpl_connect('key_press_event', on_key_event)

    sub_win.root.mainloop()
    """
    def __init__(self, win_title, geom, fig):
        root = tk.Tk()
        root.wm_title(win_title)
        root.geometry("{}x{}+{}+{}".format(*geom)) # WidthxHeight+Left+Top
        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas = canvas
        self.toolbar = toolbar
        self.root = root

if __name__ == '__main__':
    imshow_tk(np.random.randint(0,255,size=(256, 256)))
