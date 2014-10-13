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

""" Some useful and generic commonly performed image operations. """

import cv
import numpy as np


def imread_resize (imname, maxdim=None):
    """ Read and resize the and image to a maximum dimension (preserving aspect)

    Arguments:
        imname: string of the full name and path to the image to be read
        maxdim: int of the maximum dimension the image should take (in pixels).
            None if no resize is to take place (same as imread).

    Returns:
        image: (height, width, channels) np.array of the image, if maxdim is not
            None, then {height, width} <= maxdim.

    Note: either gray scale or RGB images are returned depending on the original
            image's type.
    """

    # read in the image
    image = cv.LoadImageM(imname)

    # Resize image if necessary
    imgdim = max(image.rows, image.cols)
    if (imgdim > maxdim) and (maxdim is not None):
        scaler = float(maxdim)/imgdim
        imout = cv.CreateMat(int(round(scaler*image.rows)),
                             int(round(scaler*image.cols)), image.type)
        cv.Resize(image, imout)
    else:
        imout = image

    # BGR -> RGB colour conversion
    if image.type == cv.CV_8UC3:
        imcol = cv.CreateMat(imout.rows, imout.cols, cv.CV_8UC3)
        cv.CvtColor(imout, imcol, cv.CV_BGR2RGB)
        return np.asarray(imcol)
    else:
        return np.asarray(imout)


def rgb2gray (rgbim):
    """ Convert an RGB image to a gray-scale image.

    Arguments:
        rgbim: an array (height, width, 3) which is the image. If this image is
            already a gray-scale image, this function returns it directly.

    Returns:
        image: an array (height, width, 1) which is a gray-scale version of the
            image.
    """

    # Already gray 
    if rgbim.ndim == 2:
        return rgbim
    elif rgbim.ndim != 3:
        raise ValueError("Need a three channel image!")

    grayim = cv.CreateMat(rgbim.shape[0], rgbim.shape[1], cv.CV_8UC1)
    cv.CvtColor(cv.fromarray(rgbim), grayim, cv.CV_RGB2GRAY) 
    return np.asarray(grayim)

