import cv2
import numpy as np

def im_resize(im, scale=0.5, interpolation=cv2.INTER_AREA): 
    return cv2.resize(im, None, fx=scale, fy=scale, interpolation=interpolation)

def im_sample(im, sample=2): 
    return im[::2,::2]

def im_mosaic(*args): 
    items = list(args)
    H, W = items[0].shape[:2]
    sz = np.ceil(np.sqrt(len(items))).astype(int)
    for j in range(sz*sz - len(items)): 
        items.append(im.zeros_like(items[0]))

    chunks = lambda l, n: [l[x: x+n] for x in xrange(0, len(l), n)]
    mosaic = np.vstack([np.hstack(chunk) for chunk in chunks(items, sz)])
    return mosaic
        

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
