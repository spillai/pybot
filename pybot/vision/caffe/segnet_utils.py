import os
import cv2
import numpy as np

import caffe
caffe.set_mode_gpu()

from pybot.utils.timer import timeit, timeitmethod

@timeit
def convert_image(im, input_shape): 
    # start = time.time()
    frame = cv2.resize(im, (input_shape[3],input_shape[2]), fx=0., fy=0., interpolation=cv2.INTER_AREA)
    input_image = frame.transpose((2,0,1))
    input_image = np.asarray([input_image])
    # end = time.time()
    # print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'
    return input_image

@timeit
def segnet_extract(net, input_image, layer='conv1_1_D'):
    """
    layer options: conv1_1_D, conv1_2_D
    """
    # start = time.time()
    out = net.forward_all(data=input_image)
    # end = time.time()
    # print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'
    response = np.squeeze(net.blobs[layer].data, axis=0)
    return response

class SegNetDescription(object): 
    def __init__(self, model_file, weights_file): 
        if not os.path.exists(model_file) or \
           not os.path.exists(weights_file): 
            raise ValueError('Invalid model: {}, \nweights file: {}'
                             .format(model_file, weights_file))

        # Init caffe with model
        self.net_ = caffe.Net(model_file, weights_file, caffe.TEST)
        self.input_shape_ = self.net_.blobs['data'].data.shape    

    @timeitmethod
    def describe(self, im, layer='conv1_1_D'):
        input_image = convert_image(im, self.input_shape_)
        return segnet_extract(self.net_, input_image, layer=layer)
