# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
#
# Based on https://github.com/metalbubble/places365/

import os
import sys

_PYCAFFE_PATH = os.getenv('PYCAFFE')
assert _PYCAFFE_PATH, 'PYCAFFE environment path not set'

sys.path.append(os.path.join(_PYCAFFE_PATH, 'lib'))
sys.path.append(os.path.join(_PYCAFFE_PATH, 'python'))
import caffe; caffe.set_mode_gpu(); caffe.set_device(0)

import cv2
import pickle
import numpy as np
from pybot.utils.timer import timeit, timeitmethod

class Places365Net(object):
    """
    Layers: data, conv1, pool1, norm1, conv2, pool2, norm2, conv3, 
    conv4, conv5, pool5, fc6,fc7,fc8, prob   
    """
    def __init__(self, model_file, weights_file, labels_file): 
        if not os.path.exists(model_file) or \
           not os.path.exists(weights_file) or \
           not os.path.exists(labels_file): 
            raise ValueError('Invalid model: {}, \nweights file: {}\nlabels file: {}'
                             .format(model_file, weights_file, labels_file))

        # Init caffe with model
        self.net_ = caffe.Net(model_file, weights_file, caffe.TEST)
        self.net_.blobs['data'].reshape(1,3,227,227)

	# load input and configure preprocessing
        mean_path = os.path.join(
            _PYCAFFE_PATH,
            'python/caffe/imagenet/ilsvrc_2012_mean.npy')
        self.transformer_ = caffe.io.Transformer(
            {'data': self.net_.blobs['data'].data.shape}
        )
	self.transformer_.set_mean(
            'data', np.load(mean_path).mean(1).mean(1))
	self.transformer_.set_transpose('data', (2,0,1))
	self.transformer_.set_channel_swap('data', (2,1,0))
	self.transformer_.set_raw_scale('data', 255.0)

        # Load labels
        self.labels_ = pickle.load(open(labels_file, 'rb'))
        
    @property
    def labels(self):
        return self.labels_
        
    def top_k(self, k, return_labels=False):
        K = (k+1)
        top = self.net_.blobs['prob'].data[0].flatten().argsort()[-1:-K:-1]

        if return_labels:
            return [self.labels_[k] for k in top]
        
        return top
        
    @property
    def layers(self):
        return self.net_.blobs.keys()
        
    @timeitmethod
    def forward(self, im):
        self.net_.blobs['data'].data[...] = self.transformer_.preprocess('data', im)
        self.net_.forward()
        return 

    def extract(self, layer='fc8'): 
        return np.squeeze(self.net_.blobs[layer].data,
                          axis=0)
        
    @timeitmethod
    def describe(self, im, layer='conv5'):
        self.forward(im)
        return self.extract(layer=layer)
