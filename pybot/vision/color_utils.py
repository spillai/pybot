# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2
import numpy as np

from pybot.utils.plot_utils import plt
from matplotlib.colors import colorConverter, ListedColormap
from pybot.vision.image_utils import im_normalize

def colormap(im, min_threshold=0.01):
    mask = im<min_threshold
    if im.ndim == 1: 
        hsv = np.zeros((len(im), 3), dtype=np.uint8)
        hsv[:,0] = (im * 180).astype(np.uint8)
        hsv[:,1] = 255
        hsv[:,2] = 255
        bgr = cv2.cvtColor(hsv.reshape(-1,1,3), cv2.COLOR_HSV2BGR).reshape(-1,3)
        bgr[mask] = 0
    else: 
        hsv = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
        hsv[...,0] = (im * 180).astype(np.uint8)
        hsv[...,1] = 255
        hsv[...,2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr[mask] = 0
    return bgr

def get_color(val, colormap='jet'): 
    colormap_func = getattr(plt.cm, colormap)
    return colormap_func(val)

def get_color_by_label(labels, default='b', colormap='jet'): 
    if labels is None: 
        return default

    if colormap == 'random':
        np.random.seed(1)
        inds = np.arange(10)
        np.random.shuffle(inds)
        labels_ = inds[labels % 10]
        colormap_func = getattr(plt.cm, 'jet')
        return colormap_func(labels_ * 1.0 / (len(inds) + 1))
    else: 
        lo, hi = np.min(labels), np.max(labels)
        colormap_func = getattr(plt.cm, colormap)
        return colormap_func((labels - lo) * 1.0 / (hi - lo))

def get_random_colors(n, colormap='random'): 
    return get_color_by_label(np.arange(n), colormap=colormap)

def color_from_string(c, n): 
    return np.tile(np.array(colorConverter.to_rgb(c)), [n,1])

def color_by_lut(labels, colors):
    dlabels = np.dstack([labels, labels, labels])
    return cv2.LUT(dlabels, colors)
