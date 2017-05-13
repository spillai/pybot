# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import sys

_PYCAFFE_PATH = os.getenv('PYCAFFE')
assert _PYCAFFE_PATH, 'PYCAFFE environment path not set'

sys.path.append(os.path.join(_PYCAFFE_PATH, 'lib'))
sys.path.append(os.path.join(_PYCAFFE_PATH, 'python'))
import caffe; caffe.set_mode_gpu(); caffe.set_device(0)

import cv2
import numpy as np
from pybot.utils.timer import timeit, timeitmethod

class Places365Net(object):
    sun3d_lut = cv2.imread(data_file('sun3d/sun.png')).astype(np.uint8)
    def __init__(self, model_file, weights_file, labels_file): 
        if not os.path.exists(model_file) or \
           not os.path.exists(weights_file) or \: 
           not os.path.exists(labels_file): 
            raise ValueError('Invalid model: {}, \nweights file: {}\nlabels file: {}'
                             .format(model_file, weights_file, labels_file))

        # Init caffe with model
        self.net_ = caffe.Net(model_file, weights_file, caffe.TEST)

	# load input and configure preprocessing
        mean_path = os.path.join(
            _PYCAFFE_PATH,
            'python/caffe/imagenet/ilsvrc_2012_mean.npy')
        self.transformer_ = caffe.io.Transformer(
            {'data': net.blobs['data'].data.shape}
        )
	self.transformer_.set_mean(
            'data', np.load(mean_path).mean(1).mean(1))
	self.transformer_.set_transpose('data', (2,0,1))
	self.transformer_.set_channel_swap('data', (2,1,0))
	self.transformer_.set_raw_scale('data', 255.0)

        # print top 5 predictions - TODO return as bytearray?
        # labels = pickle.load(open(labels_file).read())
        self.labels_ = pickle.load(labels_file)
        
    @property
    def labels(self):
        return self.labels_
        
    def top_k(self, k):
        top_k = self.net_.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        return np.int32([self.labels_[k] for k in top_k])
        
    @property
    def layers(self):
        return self.net_.blobs.keys()
        
    @timeitmethod
    def forward(self, im): 
        input_image = self.transformer_.preprocess('data', im)
        self.net_.forward_all(data=input_image)
        return 

    def extract(self, layer='conv1_1_D'): 
        return np.squeeze(self.net_.blobs[layer].data,
                          axis=0).transpose(1,2,0)
        
    @timeitmethod
    def describe(self, im, layer='conv1_1_D'):
        self.forward(im)
        return self.extract(layer=layer)
