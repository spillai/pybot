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

""" A modified implementation of Yang et. al.'s ScSPM image descriptor [1]. """


import math
import time
import numpy as np
from hashlib import md5
from spams import omp, trainDL
from imdescrip.utils import patch as pch, siftwrap as sw
from descriptor import Descriptor


class ScSPM (Descriptor):
    """ A modified sparse coding spatial pyramid matching image descriptor.

        This class implements a modified version of Yang et. al.'s sparse code
        spatial pyramid match (ScSPM) image descriptor [1]. While the original
        descripor uses sparse code (lasso) dictionaries and image encoding, this
        uses sparse code dictionaries and orthogonal matching persuit (OMP)
        encoding. Some classification performance is lost when using OMP, but it
        is far more scalable to a large number of images. The excellent SPAMs
        library used for [2] is used for the sparse coding and OMP in this
        module. 
        
        In addition to this scalability modification, there is an option to save
        compressed ScSPM descriptors instead of the original large-dimensional
        descriptors. Compression is done using random projection, see [2] for
        more details.

        Before using the extract() method, a dictionary must be learned using
        the learn_dictionary() method. I use pickle to save this object once a
        dictionary has been learned, so then I can apply it to multiple
        datasets.

        Arguments:
            maxdim: int (default 320), the maximum dimension the images should 
                be. This will preserve the aspect ratio of the images though.
            psize: int (default 16), the (square) patch size to use for dense
                SIFT descriptor extraction
            pstride: int (default 8), the stride to use for dense SIFT
                extraction.
            active: int (default 10), the number of activations to use for OMP.
            dsize: int (default 1024), the number of dictionary elements to use.
            levels: tuple (default (1,2,4)), the type of spatial pyramid to use.
            compress_dim: int (default None), the dimension of the random
                projection matrix to use for compressing the descriptors. None
                means the descriptors are not compressed.

        Note:
            When using compression, keep the dimensionality quite large. I.e. a
            dictionary size of 1024 will lead to images descriptors of 21,504
            dimensions. To preseve classification accuracy you may want to not
            set compress_dim less than 3000 dimensions.

        References:

        [1] Yang, J.; Yu, K.; Gong, Y. & Huang, T. Linear spatial pyramid
            matching using sparse coding for image classification Computer
            Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on,
            2009, 1794-1801

        [2] Mairal, J.; Bach, F.; Ponce,J.; & Sapiro, G. Online Dictionary 
            Learning for Sparse Coding, Internationl Conference on Machine
            Learning (ICML), 2009.

        [3] Davenport, M. A.; Duarte, M. F.; Eldar, Y. C. & Kutyniok, G.
            Introduction to compressed sensing Chapter 1 Compressed Sensing:
            Theory and Applications, Cambridge University Press, 2011, 93
        
    """

    def __init__ (self, maxdim=320, psize=16, pstride=8, active=10, dsize=1024,
                    levels=(1,2,4), compress_dim=None):

        self.maxdim = maxdim
        self.psize = psize
        self.pstride = pstride 
        self.active = active
        self.levels = levels
        self.dsize = dsize
        self.compress_dim = compress_dim
        self.dic = None       # Sparse code dictionary (D)
        
        if self.compress_dim is not None:
            D = np.sum(np.array(levels)**2) * self.dsize
            self.rmat = np.random.randn(D, self.compress_dim)
            self.rmat = self.rmat / np.sqrt((self.rmat**2).sum(axis=0))
        else:
            self.rmat = None


    def extract (self, impath):
        """ Extract a ScSPM descriptor for an image.
       
        This method will return an ScSPM descriptor for an image.
        learn_dictionary() needs to be run (once) before this method can be
        used.
        
        Arguments:
            impath: str, the path to an image

        Returns:
            a ScSPM descriptor (array) for the image. This array either has
                self.dsize*sum(self.levels**2) elements, or self.compress_dim if
                not None.

        """

        if self.dic is None:
            raise ValueError('No dictionary has been learned!')

        # Get and resize image 
        img = pch.imread_resize(impath, self.maxdim) 

        # Extract SIFT patches
        patches, cx, cy = sw.DSIFT_patches(img, self.psize, self.pstride)

        # Get OMP codes 
        scpatch = np.transpose(omp(np.asfortranarray(patches.T, np.float64), 
                                self.dic, self.active, eps=np.spacing(1), 
                                numThreads=1).todense())

        # Pyramid pooling and normalisation
        fea = pch.pyramid_pooling(scpatch, cx, cy, img.shape, self.levels)
        fea = fea / math.sqrt((fea**2).sum() + 1e-10)

        if self.compress_dim is not None:
            return np.dot(fea, self.rmat)
        else:
            return fea
        

    def learn_dictionary (self, images, npatches=50000, niter=1000, njobs=-1):
        """ Learn a Sparse Code dictionary for this ScSPM.

        This method trains a sparse codes dictionary for the ScSPM descriptor
        object. This only needs to be run once before multiple calls to the
        extract() method can be made.

        Arguments:
            images: list, a list of paths to images to use for training.
            npatches: int (default 50000) number of SIFT patches to extract from
                the images to use for training the dictionary.
            niter: int (default 1000), the number of iterations of dictionary
                learning (Lasso) to perform.
            njobs: int (default -1), the number of threads to use. -1 means the
                number of threads will be equal to the number of cores.

        """

        # Get SIFT training patches 
        print('Getting training patches...')
        patches = sw.training_patches(images, npatches, self.psize, self.maxdim,
                                        verbose=True)
        patches = pch.norm_patches(patches)
        print('{0} patches requested, {1} patches found.'.format(npatches,
                patches.shape[0]))
        time.sleep(3) # Give people a chance to see this message
          
        # Learn dictionary
        print('Learning dictionary...')
        self.dic = trainDL(np.asfortranarray(patches.T, np.float64), mode=0,
                       K=self.dsize, lambda1=0.15, iter=niter, numThreads=njobs)
        print('done.')


    def get_hash (self):
        """ Get a hash (md5) of the dictionary and random matrix.

        This function returns an md5 hash for this object's dictionary, and a
        seperate hash for the for the random projection matrix if it exists. 

        Returns:
            string dicionary md5 hash.
            string projection matrix md5 hash if compress_dim is not None.

        """

        if self.dic is None:
            raise ValueError('No dictionary has been learned!')

        # Just calculating these hashes here because md5 is so fast
        diccode = md5() 
        diccode.update(self.dic)
            
        if self.rmat is None:
            return diccode.hexdigest()
        else:
            rmatcode = md5() 
            rmatcode.update(self.rmat)
            return diccode.hexdigest(), rmatcode.hexdigest() 
