import cv2
import numpy as np

def sobel(im, dx=1, dy=1, blur=3): 
    if blur is None or blur == 0: 
        blur_im = im
    else: 
        blur_im = cv2.GaussianBlur(im, (blur,blur), 0)
    return cv2.Sobel(blur_im, cv2.CV_8U, dx, dy)

def sobel_threshold(im, dx=1, dy=1, blur=3, threshold=10): 
    return (sobel(im, dx=dx, dy=dy, blur=blur) > threshold).astype(np.uint8) * 255

def dilate(im, iterations=1): 
    return cv2.dilate(im, None, iterations=iterations)

def erode(im, iterations=1): 
    return cv2.erode(im, None, iterations=iterations)

def erode_dilate(im, iterations=1): 
    return dilate(erode(im, iterations=iterations), iterations)

def dilate_erode(im, iterations=1): 
    return erode(dilate(im, iterations=iterations), iterations)

def canny(im, blur=3): 
    im_blur = cv2.blur(im, (blur,blur))
    return cv2.Canny(im_blur, 50, 150, blur)
