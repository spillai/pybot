# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import cv2
import numpy as np

import caffe
caffe.set_mode_gpu()
caffe.set_device(1)

from pybot.utils.timer import timeit, timeitmethod

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
    def __init__(self, model_file, weights_file): 
        if not os.path.exists(model_file) or \
           not os.path.exists(weights_file): 
            raise ValueError('Invalid model: {}, \nweights file: {}'
                             .format(model_file, weights_file))

        # Init caffe with model
        self.net_ = caffe.Net(model_file, weights_file, caffe.TEST)
        self.input_shape_ = self.net_.blobs['data'].data.shape    
        
    @timeitmethod
    def forward(self, im): 
        input_image = convert_image(im, self.input_shape_)
        self.net_.forward_all(data=input_image)
        return 

    def extract(self, layer='conv1_1_D'): 
        return np.squeeze(self.net_.blobs[layer].data, axis=0).transpose(1,2,0)
        
    @timeitmethod
    def describe(self, im, layer='conv1_1_D'):
        self.forward(im)
        return self.extract(response, layer=layer)
