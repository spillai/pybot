# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import cv2
import numpy as np

import caffe
caffe.set_mode_gpu()
caffe.set_device(1)

from pybot.geometry.rigid_transform import RigidTransform, Quaternion
from pybot.utils.timer import timeit, timeitmethod

@timeit
def convert_image(im, input_shape): 
    frame = cv2.resize(im, (input_shape[3],input_shape[2]), fx=0., fy=0., interpolation=cv2.INTER_AREA)
    input_image = frame.transpose((2,0,1))
    input_image = np.asarray([input_image])
    return input_image

# @timeit
# def posenet_extract(net, input_image, meanfile_image, layer='conv1_1_D'):
#     """
#     layer options: conv1_1_D, conv1_2_D
#     """
#     out = net.forward_all(data=input_image-meanfile_image)
#     return np.squeeze(net.blobs[layer].data, axis=0)

class PoseNet(object): 
    def __init__(self, model_file, weights_file, mean_file): 
        if not os.path.exists(model_file) or \
           not os.path.exists(weights_file) or \
           not os.path.exists(mean_file): 
            raise ValueError('Invalid model: {}, \nweights file: {}, \nmean file: {}'
                             .format(model_file, weights_file, mean_file))

        # Init caffe with model
        self.net_ = caffe.Net(model_file, weights_file, caffe.TEST)
        self.mean_file_ = mean_file
        self.input_shape_ = self.net_.blobs['data'].data.shape    

        # Initialize mean file
        blob_meanfile = caffe.proto.caffe_pb2.BlobProto()
        data_meanfile = open(mean_file , 'rb' ).read()
        blob_meanfile.ParseFromString(data_meanfile)
        meanfile = np.squeeze(np.array(caffe.io.blobproto_to_array(blob_meanfile)))
        self.meanfile_ = meanfile.transpose((1,2,0))
        self.meanfile_image_ = None

    @timeitmethod
    def forward(self, im): 
        input_image = convert_image(im, self.input_shape_)
        if self.meanfile_image_ is None: 
            self.meanfile_image_ = convert_image(self.meanfile_, self.input_shape_)

        self.net_.blobs['data'].data[...] = input_image-self.meanfile_image_
        self.net_.forward()
        return 

    def extract(self, layer): 
        return np.squeeze(self.net_.blobs[layer].data)
        
    @timeitmethod
    def predict(self, im, return_rigid_transform=True):
        self.forward(im)
        predicted_q = self.extract(layer='cls3_fc_wpqr')
        predicted_x = self.extract(layer='cls3_fc_xyz')
        
        predicted_q_norm = predicted_q / np.linalg.norm(predicted_q)
        if return_rigid_transform: 
            return RigidTransform(Quaternion.from_wxyz(predicted_q_norm), tvec=predicted_x)
        return (predicted_q_norm, predicted_x)
        
