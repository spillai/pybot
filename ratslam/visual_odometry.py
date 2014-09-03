# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
from ._globals import *

class VisualOdometry(object):
    '''Visual Odometry Module.'''

    def __init__(self):
        '''Initializes the visual odometry module.'''
        self.old_vtrans_template = np.zeros(IMAGE_ODO_X_RANGE.stop)
        self.old_vrot_template = np.zeros(IMAGE_ODO_X_RANGE.stop)
        self.odometry = [0., 0., np.pi/2]

    def _create_template(self, subimg):
        '''Compute the sum of columns in subimg and normalize it.

        :param subimg: a sub-image as a 2D numpy array.
        :return: the view template as a 1D numpy array.
        '''
        x_sums = np.sum(subimg, 0)
        avint = np.sum(x_sums, dtype=np.float32)/x_sums.size
        return x_sums/avint

    def __call__(self, img):
        '''Execute an interation of visual odometry.

        :param img: the full gray-scaled image as a 2D numpy array.
        :return: the deslocation and rotation of the image from the previous 
                 frame as a 2D tuple of floats.
        '''
        subimg = img[IMAGE_VTRANS_Y_RANGE, IMAGE_ODO_X_RANGE]
        template = self._create_template(subimg)

        # VTRANS
        offset, diff = compare_segments(
            template, 
            self.old_vtrans_template, 
            VISUAL_ODO_SHIFT_MATCH
        )
        vtrans = diff*VTRANS_SCALE

        if vtrans > 10: 
            vtrans = 0

        self.old_vtrans_template = template

        # VROT
        subimg = img[IMAGE_VROT_Y_RANGE, IMAGE_ODO_X_RANGE]
        template = self._create_template(subimg)

        offset, diff = compare_segments(
            template, 
            self.old_vrot_template,
            VISUAL_ODO_SHIFT_MATCH
        )
        vrot = offset*(50./img.shape[1])*np.pi/180;
        self.old_vrot_template = template

        # Update raw odometry
        self.odometry[2] += vrot 
        self.odometry[0] += vtrans*np.cos(self.odometry[2])
        self.odometry[1] += vtrans*np.sin(self.odometry[2])

        return vtrans, vrot
