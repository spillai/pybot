import cv2
import numpy as np

def im_resize(im, scale=0.5, interpolation=cv2.INTER_AREA): 
    print 'resizing with scale', scale
    return cv2.resize(im, None, fx=scale, fy=scale, interpolation=interpolation)

def gaussian_blur(im, size=3): 
    return cv2.GaussianBlur(im, (size,size), 0)

def median_blur(im, size=3): 
    return cv2.medianBlur(im, size)
