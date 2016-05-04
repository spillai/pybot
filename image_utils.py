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
        

def to_color(im, flip_rb=False): 
    if im.ndim == 2: 
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB if flip_rb else cv2.COLOR_GRAY2BGR)
    else: 
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if flip_rb else im.copy()

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

def variance_of_laplacian(im): 
    """
    Compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian
    http://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    return cv2.Laplacian(im, cv2.CV_64F).var()

def blur_measure(im): 
    """ See cv::videostab::calcBlurriness """

    H, W = im.shape[:2]
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1)
    norm_gx, norm_gy = cv2.norm(gx), cv2.norm(gy)
    return 1.0 / ((norm_gx ** 2 + norm_gy ** 2) / (H * W + 1e-6))

def blur_detect(im, threshold=7):
    """
    Negative log-likelihood on the inverse gradient norm, 
    normalized by image size
    """
    nll = -np.log(blur_measure(im))
    return nll > threshold, nll

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def im_normalize(im, lo=0, hi=255, dtype='uint8'):
    return cv2.normalize(im, alpha=lo, beta=hi, norm_type=cv2.NORM_MINMAX, dtype={'uint8': cv2.CV_8U, \
                                                                                  'float32': cv2.CV_32F, \
                                                                                  'float64': cv2.CV_64F}[dtype])

def valid_pixels(im, valid): 
    """
    Determine valid pixel (x,y) coords for the image
    """
    if valid.dtype != np.bool: 
        raise ValueError('valid_pixels requires boolean image')
    assert(im.shape == valid.shape)

    H,W = valid.shape[:2]
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    return np.dstack([xs[valid], ys[valid], im[valid]]).reshape(-1,3)


class MosaicBuilder(object): 
    def __init__(self, filename_template, maxlen=100, shape=(1600,900),
                 glyph_shape=(50,50), visualize_name='mosaics'): 
        self.idx_ = 0
        self.filename_template_ = filename_template
        self.save_mosaic_ = len(self.filename_template_) > 0
        self.shape_ = shape
        self.visualize_name_ = visualize_name
        
        self.maxlen_ = maxlen
        self.ims_ = []
        self.resize_cb_ = lambda im: im_resize(im, shape=glyph_shape)
        self.mosaic_cb_ = lambda ims: im_mosaic_list(ims, shape=None)

    def clear(self):
        self.ims_ = []
        
    def add(self, im): 
        self.ims_.append(self.resize_cb_(im))
        if len(self.ims_) % self.maxlen_ == 0: 
            if self.save_mosaic_: 
                self._save()
            else: 
                self.visualize()
            self.ims_ = [] 

    def visualize(self): 
        if not len(self.ims_): 
            return
        mosaic = self.mosaic_cb_(self.ims_)
        cv2.imshow(self.visualize_name_, mosaic)
        return

    def _save(self): 
        if not len(self.ims_): 
            return

        fn = self.filename_template_ % self.idx_
        cv2.imwrite(fn, self.mosaic_cb_(self.ims_))
        print('Saving mosaic: %s' % fn)
        self.idx_ += 1

    def finalize(self): 
        if self.save_mosaic_: 
            self._save()

    @property
    def mosaic(self):
        """
        """
        return self.mosaic_cb_(list(self.ims_)) \
            if len(self.ims_) else np.zeros(shape=self.shape_, dtype=np.uint8) 
        

