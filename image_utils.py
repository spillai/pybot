import cv2
import numpy as np

def im_resize(im, shape=None, scale=0.5, interpolation=cv2.INTER_AREA): 
    if shape is not None: 
        return cv2.resize(im, shape, interpolation=interpolation)
    else: 
        return (cv2.resize(im, shape, fx=scale, fy=scale, interpolation=interpolation) \
                if scale != 1.0 else im)

def im_sample(im, sample=2): 
    return im[::2,::2]

def im_mosaic(*args, **kwargs): 
    scale = kwargs.get('scale', 1.0)
    shape = kwargs.get('shape', None)

    items = list(args)
    N = len(items)

    # print 'Items: ', N, items
    assert N>0, 'No items to mosaic!'

    sz = np.ceil(np.sqrt(N)).astype(int)
    if shape is not None: 
        sz_w, sz_h = shape[1] / sz, shape[0] / sz
    else: 
        sz_w, sz_h = 800 / sz, 600 / sz

    for j in range(sz * sz): 
        if j < N: 
            items[j] = to_color(im_resize(items[j], shape=(sz_w, sz_h)))
        else: 
            items.append(np.zeros_like(items[-1]))

    # for j in range(sz*sz - len(items)): 
    #     items.append(np.zeros_like(items[0]))

    chunks = lambda l, n: [l[x: x+n] for x in xrange(0, len(l), n)]
    mosaic = np.vstack([np.hstack(chunk) for chunk in chunks(items, sz)])

    return im_resize(mosaic, scale=scale)
        

def to_color(im): 
    if im.ndim == 2: 
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else: 
        return im.copy()

def to_gray(im): 
    if im.ndim == 3: 
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else: 
        return im.copy()

def gaussian_blur(im, size=3): 
    return cv2.GaussianBlur(im, (size,size), 0)

def box_blur(im, size=3): 
    return cv2.boxFilter(im, -1, (size,size))

def median_blur(im, size=3): 
    return cv2.medianBlur(im, size)
