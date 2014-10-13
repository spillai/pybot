#! /usr/bin/env python

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

""" Unit tests for the imdescrip package. """

import os
import numpy as np
import unittest 
from imdescrip.utils import patch, siftwrap, image 


class TestImdescrip (unittest.TestCase):
    """ This is a TestCase for the imgdescrip package. """


    def setUp (self):
        """ Make a few test inputs and outputs for the tests. """
       
        # Test inputs image, and patch size, stride 
        self.timg = np.vstack([np.arange(1,5), np.arange(5,9), np.arange(9,13), 
                                np.arange(13,17)])
        self.psize = 3
        self.pstride = 1
        
        # Test output patches, patch centres
        self.tpatch = np.array([[ 1,  2,  3,  5,  6,  7,  9,  10, 11],
                             [ 2,  3,  4,  6,  7,  8,  10, 11, 12],
                             [ 5,  6,  7,  9,  10, 11, 13, 14, 15],
                             [ 6,  7,  8,  10, 11, 12, 14, 15, 16]])
        self.tx = np.array([1, 2, 1, 2])
        self.ty = np.array([1, 1, 2, 2]) 
        self.tpyr = np.concatenate([self.tpatch[3,:], self.tpatch[0,:], 
                        self.tpatch[1,:], self.tpatch[2,:], self.tpatch[3,:]])

        # location of test resources 
        self.__loc__ = os.path.realpath(os.path.join(os.getcwd(), 
            os.path.dirname(__file__))) 

        self.tilist = [os.path.join(self.__loc__, 'test.jpg'),
                        os.path.join(self.__loc__, 'test2.jpg')]


    def test_grid_patches (self):
        """ Test the imgpatches.grid_patches() function. """

        # Make sure the outputs are the right shape and values
        patches, x, y = patch.grid_patches(self.timg, self.psize, self.pstride)
        self.assertTrue((patches == self.tpatch).all())
        self.assertTrue((x == self.tx).all())
        self.assertTrue((y == self.ty).all())

        # Make sure the function errors out when we expect it too
        self.assertRaises(ValueError, patch.grid_patches, 
                            np.array([1, 1, 1, 1]), self.psize, self.pstride)
    

    def test_pyramid_pooling (self):
        """ Test the imgpatches.pyramid_pooling function. """

        pyr = patch.pyramid_pooling(self.tpatch, self.tx, self.ty,
                                  self.timg.shape, (1,2), patch.p_max)
        self.assertTrue((self.tpyr == pyr).all())


    def test_norm_patches (self):
        """ Test patch contrast normalisation/unitisation. """
       
        diff = np.abs(np.ones((self.tpatch.shape[0],1)) 
                - np.sqrt((patch.norm_patches(self.tpatch)**2).sum(axis=1)))
        self.assertTrue((diff < 1e-15).all())
    
        
    def test_centre_patches (self):
        """ Test patch contrast normalisation/unitisation. """
        
        self.assertTrue((np.zeros((self.tpatch.shape[0],1)) ==
                patch.centre_patches(self.tpatch).mean(axis=1)).all())

    
    def test_imread_resize (self):
        """ Test image reader and resizer. """ 

        timg = image.imread_resize(os.path.join(self.__loc__, 'test.jpg'), 200)
        self.assertEqual(timg.shape[1], 200)
        self.assertEqual(timg.shape[0], 134)


    def test_rgb2gray (self):
        """ Test rgb 2 gray converter """

        timg = image.imread_resize(os.path.join(self.__loc__, 'test.jpg'))
        gimg = image.rgb2gray(timg)
        self.assertEqual(gimg.ndim, 2)


    def test_patches_training_patches (self):
        """ Test getting images patches for training dictionaries. """ 

        tpatches = patch.training_patches(self.tilist, 20, 16)
        self.assertEqual(tpatches.shape[1], 16**2)
        self.assertTrue(10 < tpatches.shape[0] < 30) # Not exact, but that's ok


    def test_DSIFT_patches (self):
        """ Test dense SIFT patch extraction. """

        timg = image.imread_resize(os.path.join(self.__loc__, 'test.jpg'), 200)
        patches, x, y = siftwrap.DSIFT_patches(timg, 16, 8)

        # Get an expected shapes
        self.assertTrue((patches.shape == np.array([360,128])).all()) 
        self.assertEqual(len(x), 360)
        self.assertEqual(len(y), 360)
        
        # Get reasonable range of values (bad python interface for vlfeat)
        self.assertTrue((patches != 0).any())
        self.assertTrue((patches != 255).any())
        self.assertTrue((x[0] == 7.5) and (y[0] == 7.5))
        self.assertTrue((x[-1] == 191.5) and (y[-1] == 119.5))  


    def test_DSIFT_training_patches (self):
        """ Test getting SIFT patches for training dictionaries. """ 

        tpatches = siftwrap.training_patches(self.tilist, 50, 16)
        self.assertEqual(tpatches.shape[1], 128)
        self.assertTrue(40 < tpatches.shape[0] < 60) # Not exact, but that's ok


if __name__ == '__main__':
    unittest.main()

