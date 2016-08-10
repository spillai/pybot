from scipy import shape, dstack, sqrt, floor, array, mean, ones, vstack, hstack, ndarray
from sys import maxint

from pyvlfeat import vl_imsmooth, vl_dsift

"""
Python rewrite of https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_phow.m
### notice no hsv support atm
### comments are largely copied from the code
Stripped from: https://github.com/shackenberg/phow_caltech101.py
"""

def vl_phow(im,
            verbose=True,
            fast=True,
            sizes=[4, 6, 8, 10],
            step=2,
            color='rgb',
            floatdescriptors=False,
            magnif=6,
            windowsize=1.5,
            contrastthreshold=0.005):

    opts = Options(verbose, fast, sizes, step, color, floatdescriptors,
                   magnif, windowsize, contrastthreshold)
    dsiftOpts = DSiftOptions(opts)

    # make sure image is float, otherwise segfault
    im = array(im, 'float32')

    # Extract the features
    imageSize = shape(im)
    if im.ndim == 3:
        if imageSize[2] != 3:
            # "IndexError: tuple index out of range" if both if's are checked at the same time
            raise ValueError("Image data in unknown format/shape")
    if opts.color == 'gray':
        numChannels = 1
        if (im.ndim == 2):
            im = vl_rgb2gray(im)
    else:
        numChannels = 3
        if (im.ndim == 2):
            im = dstack([im, im, im])
        if opts.color == 'rgb':
            pass
        elif opts.color == 'opponent':
             # from https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_phow.m
             # Note that the mean differs from the standard definition of opponent
             # space and is the regular intesity (for compatibility with
             # the contrast thresholding).
             # Note also that the mean is added pack to the other two
             # components with a small multipliers for monochromatic
             # regions.

            mu = 0.3 * im[:, :, 0] + 0.59 * im[:, :, 1] + 0.11 * im[:, :, 2]
            alpha = 0.01
            im = dstack([mu,
                         (im[:, :, 0] - im[:, :, 1]) / sqrt(2) + alpha * mu,
                         (im[:, :, 0] + im[:, :, 1] - 2 * im[:, :, 2]) / sqrt(6) + alpha * mu])
        else:
            raise ValueError('Color option ' + str(opts.color) + ' not recognized')
    if opts.verbose:
        print('{0}: color space: {1}'.format('vl_phow', opts.color))
        print('{0}: image size: {1} x {2}'.format('vl_phow', imageSize[0], imageSize[1]))
        print('{0}: sizes: [{1}]'.format('vl_phow', opts.sizes))

    frames_all = []
    descrs_all = []
    for size_of_spatial_bins in opts.sizes:
        # from https://github.com/vlfeat/vlfeat/blob/master/toolbox/sift/vl_phow.m
        # Recall from VL_DSIFT() that the first descriptor for scale SIZE has
        # center located at XC = XMIN + 3/2 SIZE (the Y coordinate is
        # similar). It is convenient to align the descriptors at different
        # scales so that they have the same geometric centers. For the
        # maximum size we pick XMIN = 1 and we get centers starting from
        # XC = 1 + 3/2 MAX(OPTS.SIZES). For any other scale we pick XMIN so
        # that XMIN + 3/2 SIZE = 1 + 3/2 MAX(OPTS.SIZES).
        # In pracrice, the offset must be integer ('bounds'), so the
        # alignment works properly only if all OPTS.SZES are even or odd.

        off = floor(3.0 / 2 * (max(opts.sizes) - size_of_spatial_bins)) + 1

        # smooth the image to the appropriate scale based on the size
        # of the SIFT bins
        sigma = size_of_spatial_bins / float(opts.magnif)
        ims = vl_imsmooth(im, sigma)

        # extract dense SIFT features from all channels
        frames = []
        descrs = []
        for k in range(numChannels):
            size_of_spatial_bins = int(size_of_spatial_bins)
            # vl_dsift does not accept numpy.int64 or similar
            f_temp, d_temp = vl_dsift(data=ims[:, :, k],
                                      step=dsiftOpts.step,
                                      size=size_of_spatial_bins,
                                      fast=dsiftOpts.fast,
                                      verbose=dsiftOpts.verbose,
                                      norm=dsiftOpts.norm,
                                      bounds=[off, off, maxint, maxint])
            frames.append(f_temp)
            descrs.append(d_temp)
        frames = array(frames)
        descrs = array(descrs)
        d_new_shape = [descrs.shape[0] * descrs.shape[1], descrs.shape[2]]
        descrs = descrs.reshape(d_new_shape)
        # remove low contrast descriptors
        # note that for color descriptors the V component is
        # thresholded
        if (opts.color == 'gray') | (opts.color == 'opponent'):
            contrast = frames[0][2, :]
        elif opts.color == 'rgb':
            contrast = mean([frames[0][2, :], frames[1][2, :], frames[2][2, :]], 0)
        else:
            raise ValueError('Color option ' + str(opts.color) + ' not recognized')
        descrs[:, contrast < opts.contrastthreshold] = 0

        # save only x,y, and the scale
        frames_temp = array(frames[0][0:3, :])
        padding = array(size_of_spatial_bins * ones(frames[0][0].shape))
        frames_all.append(vstack([frames_temp, padding]))
        descrs_all.append(array(descrs))


    frames_all = hstack(frames_all)
    descrs_all = hstack(descrs_all)
    return frames_all, descrs_all


class Options(object):
    def __init__(self, verbose, fast, sizes, step, color,
                 floatdescriptors, magnif, windowsize,
                 contrastthreshold):
        self.verbose = verbose
        self.fast = fast
        if (type(sizes) is not ndarray) & (type(sizes) is not list):
            sizes = array([sizes])
        self.sizes = sizes
        self.step = step
        self.color = color
        self.floatdescriptors = floatdescriptors
        self.magnif = magnif
        self.windowsize = windowsize
        self.contrastthreshold = contrastthreshold


class DSiftOptions(object):
    def __init__(self, opts):
        self.norm = True
        self.windowsize = opts.windowsize
        self.verbose = opts.verbose
        self.fast = opts.fast
        self.floatdescriptors = opts.floatdescriptors
        self.step = opts.step

if __name__ == "__main__":
    from scipy.misc import imread 
    im = imread('image_0001.jpg')
    frames, descrs = vl_phow(array(im, 'float32') / 255.0)    
