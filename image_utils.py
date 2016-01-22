import cv2
import numpy as np
from collections import deque

def flip_rb(im): 
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def im_resize(im, shape=None, scale=0.5, interpolation=cv2.INTER_AREA): 
    if shape is not None: 
        return cv2.resize(im, dsize=shape, fx=0., fy=0., interpolation=interpolation)
    else: 
        if scale <= 1.0: 
            return cv2.resize(im, None, fx=scale, fy=scale, interpolation=interpolation)
        else: 
            shape = (int(im.shape[1]*scale), int(im.shape[0]*scale))
            return im_resize(im, shape)

def im_pad(im, pad=3, value=0): 
    return cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value)

def im_sample(im, sample=2): 
    return im[::2,::2]

def im_mosaic_list(items, scale=1.0, shape=None, pad=0): 
    N = len(items)

    # print 'Items: ', N, items
    assert N>0, 'No items to mosaic!'

    sz = np.ceil(np.sqrt(N)).astype(int)
    for j in range(sz * sz): 
        if j < N: 
            if shape is not None: 
                items[j] = to_color(im_resize(items[j], shape=shape))
            else: 
                items[j] = to_color(im_resize(items[j], scale=scale))

        else: 
            items.append(np.zeros_like(items[-1]))

    chunks = lambda l, n: [l[x: x+n] for x in xrange(0, len(l), n)]
    # print len(items)
    # print [len(chunk) for chunk in chunks(items, sz)]
    # print [map(lambda ch: ch.shape, chunk) for chunk in chunks(items, sz)]
    # print [(np.hstack(chunk)).shape for chunk in chunks(items, sz)]
        
    mosaic = im_pad(np.vstack([np.hstack(chunk) for chunk in chunks(items, sz)]), pad=3)
    
    return im_resize(mosaic, scale=scale)


def im_mosaic(*args, **kwargs): 
    scale = kwargs.get('scale', 1.0)
    shape = kwargs.get('shape', None)
    pad = kwargs.get('pad', 0)

    items = list(args)
    N = len(items)

    # print 'Items: ', N, items
    assert N>0, 'No items to mosaic!'

    sz = np.ceil(np.sqrt(N)).astype(int)
    for j in range(sz * sz): 
        if j < N: 
            if shape is not None: 
                items[j] = to_color(im_resize(items[j], shape=shape))
            else: 
                items[j] = to_color(im_resize(items[j], scale=scale))

        else: 
            items.append(np.zeros_like(items[-1]))

    chunks = lambda l, n: [l[x: x+n] for x in xrange(0, len(l), n)]
    # print len(items)
    # print [len(chunk) for chunk in chunks(items, sz)]
    # print [map(lambda ch: ch.shape, chunk) for chunk in chunks(items, sz)]
    # print [(np.hstack(chunk)).shape for chunk in chunks(items, sz)]
        
    mosaic = im_pad(np.vstack([np.hstack(chunk) for chunk in chunks(items, sz)]), pad=3)
    
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

def blur_measure(im): 
    cv2.imshow('im', im)
    cv2.waitKey(0)
    return cv2.Laplacian(im, cv2.CV_64F).var()

def blur_detect(im, threshold=100): 
    return blur_measure(im) > threshold


class MosaicBuilder(object): 
    def __init__(self, filename_template, maxlen=100, shape=(1600,900), glyph_shape=(50,50)): 
        self.idx_ = 0
        self.filename_template_ = filename_template
        self.save_mosaic_ = len(self.filename_template_) > 0
        self.shape_ = shape
        
        # if '%i' not in self.filename_template_: 
        #     raise RuntimeError('Failed to parse filename template, missing %%i')
            
        self.maxlen_ = maxlen
        self.ims_ = deque(maxlen=self.maxlen_)
        self.resize_cb_ = lambda im: im_resize(im, shape=glyph_shape)
        self.mosaic_cb_ = lambda ims: im_mosaic_list(ims, shape=None)

    def add(self, im): 
        self.ims_.append(self.resize_cb_(im))
        if self.save_mosaic_ and len(self.ims_) % self.maxlen_ == 0: 
            self._save()

    def _save(self): 
        if not len(self.ims_): 
            return

        fn = self.filename_template_ % self.idx_
        cv2.imwrite(fn, self.mosaic_cb_(self.ims_))
        print('Saving mosaic: %s' % fn)

        self.idx_ += 1
        self.ims_ = deque(maxlen=self.maxlen_)

    def finalize(self): 
        if self.save_mosaic_: 
            self._save()

    @property
    def mosaic(self):
        """
        """
        return self.mosaic_cb_(list(self.ims_)) \
            if len(self.ims_) else np.zeros(shape=shape, dtype=np.uint8) 
        

