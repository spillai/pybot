# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
# 
# Based on https://github.com/alexgkendall/caffe-segnet 
# https://github.com/alexgkendall/SegNet-Tutorial

import os
import sys
import cv2
import numpy as np

_PYCAFFE_PATH = os.getenv('PYCAFFE')
assert _PYCAFFE_PATH, 'PYCAFFE environment path not set'

sys.path.append(os.path.join(_PYCAFFE_PATH, 'lib'))
sys.path.append(os.path.join(_PYCAFFE_PATH, 'python'))
import caffe; caffe.set_mode_gpu(); caffe.set_device(0)

from pybot.vision.color_utils import color_by_lut
from pybot.utils.timer import timeit, timeitmethod
from pybot.utils.dataset import data_file

@timeit
def convert_image(im, input_shape): 
    frame = cv2.resize(im, (input_shape[3],input_shape[2]), fx=0., fy=0., interpolation=cv2.INTER_AREA)
    input_image = frame.transpose((2,0,1))
    input_image = np.asarray([input_image])
    return input_image

@timeit
def segnet_extract(net, input_image, layer='conv1_1_D'):
    """
    layer options: conv1_1_D, conv1_2_D
    """
    out = net.forward_all(data=input_image)
    return np.squeeze(net.blobs[layer].data, axis=0)

class SegNet(object):
    sun3d_lut = cv2.imread(data_file('sun3d/sun.png')).astype(np.uint8)
    def __init__(self, model_file, weights_file): 
        if not os.path.exists(model_file) or \
           not os.path.exists(weights_file): 
            raise ValueError('Invalid model: {}, \nweights file: {}'
                             .format(model_file, weights_file))

        # Init caffe with model
        self.net_ = caffe.Net(model_file, weights_file, caffe.TEST)
        self.input_shape_ = self.net_.blobs['data'].data.shape    

    @property
    def layers(self):
        return self.net_.blobs.keys()
        
    @timeitmethod
    def forward(self, im): 
        input_image = convert_image(im, self.input_shape_)
        self.net_.forward_all(data=input_image)
        return 

    def extract(self, layer='conv1_1_D'): 
        return np.squeeze(self.net_.blobs[layer].data,
                          axis=0).transpose(1,2,0)
        
    @timeitmethod
    def describe(self, im, layer='conv1_1_D'):
        self.forward(im)
        return self.extract(layer=layer)

    @staticmethod
    def visualize(labels, colors):
        return color_by_lut(labels, colors)
