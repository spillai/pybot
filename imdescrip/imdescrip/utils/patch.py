# Imdescrip -- a collection of tools to extract descriptors from images.
# Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Functions for extracting, processing and displaying image patches.
    
    This file has a few useful functions for extracting and processing patches
    from images. For example, grid_patches() extracts from an overlapping dense
    grid in an image. pyramid_pooling() implements spatial pyramid pooling from
    [1]. There are also patch centring (DC component removal) and normalisation
    routines etc.

    Many of these functions are ports from the code developed for [1].

    [1] Yang, J.; Yu, K.; Gong, Y. & Huang, T. Linear spatial pyramid matching 
        using sparse coding for image classification Computer Vision and Pattern 
        Recognition, 2009. CVPR 2009. IEEE Conference on, 2009, 1794-1801

"""

import math
import numpy as np 
from scipy.misc import toimage
from image import imread_resize, rgb2gray
from progress import Progress

# Optional imports
try: 
    from matplotlib import pyplot as plt
    hasmatplotlib = True
except:
    hasmatplotlib = False


def grid_patches (image, psize, pstride):
    """ Extract a grid of (overlapping) patches from an image

    This function extracts square patches from an image in an overlapping, dense
    grid. 

    Arguments:
        image: np.array (rows, cols, channels) of an image (in memory)
        psize: int the size of the square patches to extract, in pixels.
        pstride: int the stride (in pixels) between successive patches.

    Returns:
        patches: np.array (npatches, psize**2*channels) the flattened (per row)
            image patches.
        centresx: np.array (npatches) the centres (column coords) of the patches
        centresy: np.array (npatches) the centres (row coords) of the patches

    """

    # Check and get image dimensions
    if image.ndim == 3:
        (Ih, Iw, Ic) = image.shape
    elif image.ndim == 2:
        (Ih, Iw) = image.shape
        Ic = 1
    else:
        raise ValueError('image must be a 2D or 3D np.array')

    # Make the overlapping grid
    offsetX = int(math.floor(float((Iw - psize) % pstride)/2))
    offsetY = int(math.floor(float((Ih - psize) % pstride)/2))
    spaceX = range(offsetX, Iw-psize+1, pstride)
    spaceY = range(offsetY, Ih-psize+1, pstride)
    npatches = len(spaceX)*len(spaceY)

    # Pre-allocate the returns
    rsize = (psize**2)*Ic
    patches = np.zeros((npatches, rsize))
    centresy = np.zeros(npatches)
    centresx = np.zeros(npatches)

    # Extract the patches and get the patch centres
    cnt = 0
    for sy in spaceY:           # Rows
        gridY = range(sy, sy+psize) 
        
        for sx in spaceX:       # Cols
            gridX = range(sx, sx+psize) 

            patches[cnt,:] = np.reshape(image[gridY][:,gridX], rsize)
            centresy[cnt] = sy + float(psize)/2 - 0.5;
            centresx[cnt] = sx + float(psize)/2 - 0.5;

            cnt +=1

    return patches, centresx, centresy

    
def training_patches (imnames, npatches, psize, maxdim=None, colour=False,
                        verbose=False):
    """ Extract patches from images for dictionary training

    Arguments:
        imnames: A list of image names from which to extract training patches.
        npatches: The number (int) of patches to extract from the images
        maxdim: The maximum dimension of the image in pixels. The image is
            rescaled if it is larger than this. By default there is no scaling. 
        psize: A int of the size of the square patches to extract
        verbose: bool, print progress bar

    Returns:
        An np.array (npatches, psize**2*3) for RGB or (npatches, psize**2) for
        grey of flattened image patches. NOTE, the actual npatches found may be
        less than that requested.

    """

    nimg = len(imnames)
    ppeimg = int(round(float(npatches)/nimg))
    plist = []

    # Set up progess updates
    progbar = Progress(nimg, title='Extracting patches', verbose=verbose)

    for i, ims in enumerate(imnames):
        img = imread_resize(ims, maxdim) # read in and resize the image
        spaceing = max(int(round(img.shape[1] *  ppeimg**(-0.5))), 1)
        
        # Extract patches and map to grayscale if necessary
        if (colour == False) and (img.ndim == 3):
            imgg = rgb2gray(img)
            plist.append(grid_patches(imgg, psize, spaceing)[0])
        else:
            plist.append(grid_patches(img, psize, spaceing)[0])

        progbar.update(i)

    progbar.finished()
    patches = np.concatenate(plist, axis=0)
    return np.reshape(patches, (patches.shape[0], np.prod(patches.shape[1:])))


def p_maxabs (patches):
    """ Return the maximum of the absolute values of the columns in a matrix.

    This function is used for pyramid pooling (see pyramid_pooling()), and given
    an (npatches, ndims) matrix, will return a (1, ndims) vector of the
    max(abs()) values in each column.

    Arguments:
        patches: an (npatches, ndims) array of image patches

    Returns:
        a (1, ndim) array of the max(abs()) of the patches in each column.

    """

    if patches.shape[0] > 1:
        return np.max(np.abs(patches), axis=0)
    else:
        return np.abs(patches)


def p_mean (patches):
    """ Return the mean values of the columns in a matrix.

    This function is used for pyramid pooling (see pyramid_pooling()), and given
    an (npatches, ndims) matrix, will return a (1, ndims) vector of the
    mean() values in each column.

    Arguments:
        patches: an (npatches, ndims) array of image patches

    Returns:
        a (1, ndim) array of the mean of the patches in each column.

    """

    if patches.shape[0] > 1:
        return np.mean(patches, axis=0)
    else:
        return patches


def p_max (patches):
    """ Return the max values of the columns in a matrix.

    This function is used for pyramid pooling (see pyramid_pooling()), and given
    an (npatches, ndims) matrix, will return a (1, ndims) vector of the
    max() values in each column.

    Arguments:
        patches: an (npatches, ndims) array of image patches

    Returns:
        a (1, ndim) array of the max of the patches in each column.

    """

    if patches.shape[0] > 1:
        return np.max(patches, axis=0)
    else:
        return patches


def pyramid_pooling (patches, centresx, centresy, imsize, levels=(1,2,4), 
        pfun=p_max):
    """ Spatial pyramid pooling of image patches (or codes of image patches)

    This funtion implements spatial pyramid pooling, which essentially turns a
    set of image patches, or image patch codes, into a single image descriptor.
    See [1] for more detail.

    Arguments:
        patches: an (npatches, ndims) array of image patches, or codes of image
            patches.
        centresx: an (npatches, 1) array of the x, or row, centre locations of 
            the image patches (like output from grid_patches() 
        centresy: an (npatches, 1) array of the y, or col, centre locations of 
            the image patches (like output from grid_patches() 
        imsize: a tuple (rows, cols) of the size of the original image that the
            patches were extracted from.
        levels: A tuple of ints that defines the spatial pyramid pooling levels. 
            Each level divides the image into a grid. (1, 2, 4) means one whole
            image pool, then (2x2) image pooling regions, then (4x4) image
            pooling regions. See [1] for more details.
        pfun: is the name of the pooling function to use. p_maxabs() implements
            the max abs pooling described in [1].

    Returns:
        A (1, ndims * array(levels)**2) array of all of the pooled patches/codes
        flattened.
    
    """

    # Get the number of bins in the pyramid
    Dbins = patches.shape[1]        # Dimensionality of the pyramid bins
    lbins = np.array(levels) ** 2   # Number of bins on each pyramid level
    tbins = lbins.sum()             # Total number of pyramid bins

    # pre-allocate 
    poolpatches = np.zeros((tbins, Dbins))
    cnt = 0

    # Pyramid pooling
    for (i, lev) in enumerate(levels):

        # Bin width/height
        wunit = float(imsize[1]) / lev 
        hunit = float(imsize[0]) / lev

        # Find patch-bin memberships
        binidx = np.floor(centresy / hunit) * lev + np.floor(centresx / wunit)

        # Bin the patches
        for j in range(lbins[i]):
            pidx = np.nonzero(binidx == j)[0]
            if len(pidx) > 0:
                poolpatches[cnt,:] = pfun(patches[pidx,:])
            cnt += 1

    return poolpatches.flatten() 


def norm_patches (patches, epsilon=1e-20):
    """ Normalise image patches to each be unit length.

    Arguments:
        patches: an (npatches, ndims) array of image patches, or codes of image
            patches.
        epsilon: a small non-zero value to add to the normalisation to prevent
            divide by zero. Making this term larger may also sometimes improve
            results.

    Returns:
        An (npatches, ndims) array where each row is unitised version of the
            correponding row in patches.
    """

    return patches / np.sqrt((patches ** 2).sum(axis=1) +
                        epsilon).reshape(patches.shape[0],1)


def centre_patches (patches):
    """ Centred image patches to each have a mean of zero.

    Arguments:
        patches: an (npatches, ndims) array of image patches, or codes of image
            patches.

    Returns:
        An (npatches, ndims) array where each row is centred version of the
            correponding row in patches.
    """

    return patches - np.mean(patches, axis=1).reshape(patches.shape[0],1)


if hasmatplotlib == True:
    def disp_patches (patches, colour=False):
        """ Display flattened (square) patches in a grid.
    
        This function is best for displaying the bases of learned dictionaries.
    
        Arguments:
            patches: (npatches, pixels) is a numpy np.array of all of the
                flattened image patches in each row. These will automatically be
                scalled to be displayed as images. It is assumed the original
                patches are square
            colour: boolean flag indicating whether or not these patches are
                supposed to be colour or not.
    
        """
    
        # Argument checking
        if (colour == False) and (math.sqrt(patches.shape[1])%1 != 0):
            raise ValueError('Gray image has to have square patches')
        elif (colour == True) and (math.sqrt(float(patches.shape[1])/3)%1 != 0):
            raise ValueError('Colour image has to have square patches')
    
        # Get patch size
        if colour == False:
            psize = math.sqrt(patches.shape[1])  
        else: 
            psize = math.sqrt(patches.shape[1]/3)
        
        ssize = math.ceil(math.sqrt(patches.shape[0]))
    
        # plot filters
        plt.figure()
        for i, p in enumerate(patches):
            plt.subplot(ssize, ssize, i + 1)
            if colour == False:
                plt.imshow(p.reshape(psize, psize), cmap="gray")
            else:
                plt.imshow(toimage(p.reshape(psize, psize, 3), mode='RGB'))
            plt.axis("off")
        plt.show()
