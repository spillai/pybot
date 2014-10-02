import cv2
import numpy as np
import matplotlib.pylab as plt

def colormap(im, min_threshold=0.01):
    mask = im<min_threshold
    hsv = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
    hsv[...,0] = (im * 180).astype(np.uint8)
    hsv[...,1] = 255
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[mask] = 0
    return bgr

def get_color_by_label(labels, default='b'): 
    lo, hi = np.min(labels), np.max(labels)
    return plt.cm.gist_rainbow((labels - lo) * 1.0 / (hi - lo)) if labels is not None else default    
