import cv2
import numpy as np

def im_resize(im, scale=0.5, interpolation=cv2.INTER_AREA): 
    return cv2.resize(im, None, fx=scale, fy=scale, interpolation=interpolation)

def to_color(im): 
    if im.ndim == 2: 
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else: 
        return im

def to_gray(im): 
    if im.ndim == 3: 
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else: 
        return im

def gaussian_blur(im, size=3): 
    return cv2.GaussianBlur(im, (size,size), 0)

def median_blur(im, size=3): 
    return cv2.medianBlur(im, size)
