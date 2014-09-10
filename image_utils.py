import cv2
import numpy as np

def gaussian_blur(im, size=3): 
    return cv2.GaussianBlur(im, (size,size), 0)

def median_blur(im, size=3): 
    return cv2.medianBlur(im, size)
