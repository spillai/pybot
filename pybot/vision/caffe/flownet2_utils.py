# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
# 
# Based on https://github.com/lmb-freiburg/flownet2

import os
import sys
import cv2
import numpy as np

_PYCAFFE_PATH = os.getenv('PYCAFFE')
_CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
assert _PYCAFFE_PATH, 'PYCAFFE environment path not set'

sys.path.append(os.path.join(_PYCAFFE_PATH, 'lib'))
sys.path.append(os.path.join(_PYCAFFE_PATH, 'python'))
import caffe;
if _CUDA_VISIBLE_DEVICES:
    caffe.set_mode_gpu(); caffe.set_device(0)
else:
    import warnings
    warnings.warn('CUDA_VISIBILE_DEVICES is not set. Caffe in CPU Mode')
    caffe.set_mode_cpu()

# from pybot.vision.color_utils import color_by_lut
from pybot.utils.timer import timeit, timeitmethod
# from pybot.utils.dataset import data_file

@timeit
def convert_image(im, input_shape):
    if len(im.shape) < 3:
        return im[np.newaxis, np.newaxis, :, :]
    else:
        return im[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]

@timeit
def convert_images(img0, img1):
    input_data = []
    input_data.append(convert_image(img0))
    input_data.append(convert_image(img1))
    return input_data

    # frame = cv2.resize(im, (input_shape[3],input_shape[2]), fx=0., fy=0., interpolation=cv2.INTER_AREA)
    # input_image = frame.transpose((2,0,1))
    # input_image = np.asarray([input_image])
    # return input_image

class FlowNet2(object):
    def __init__(self, model_file, weights_file): 
        if not os.path.exists(model_file) or \
           not os.path.exists(weights_file): 
            raise ValueError('Invalid model: {}, \nweights file: {}'
                             .format(model_file, weights_file))

        self.inited_ = False
        self.model_file_ = model_file
        self.weights_file_ = weights_file

    def _init(self, input_shape):
        height, width = input_shape[:2]

        # Input data
        vars = {}
        vars['TARGET_WIDTH'] = width
        vars['TARGET_HEIGHT'] = height

        divisor = 64.
        vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
        vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

        vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
        vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);
        
        # Setup proto file
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)
        proto = open(self.model_file_).readlines()
        for line in proto:
            for key, value in vars.items():
                tag = "$%s$" % key
                line = line.replace(tag, str(value))
            tmp.write(line)
        tmp.flush()
        
        # Init caffe with model
        self.net_ = caffe.Net(tmp.name, self.weights_file_, caffe.TEST)
        self.input_shape_ = self.net_.blobs['data'].data.shape    
        
        if not args.verbose:
            caffe.set_logging_disabled()
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()
        self.net_ = caffe.Net(tmp.name, weights_file, caffe.TEST)
            
    @property
    def layers(self):
        return self.net_.blobs.keys()
        
    @timeitmethod
    def forward(self, img0, img1):
        assert(img0.shape == img1.shape)

        # Initialize network
        if not self.inited_:
            self._init(img0.shape[:2])
            self.inited_ = True
            
        num_blobs = 2
        input_dict = {}
        input_data = convert_images(img0, img1)
        for blob_idx in range(num_blobs):
            input_dict[self.net_.inputs[blob_idx]] = input_data[blob_idx]

        self.net_.forward(**input_dict)
        
        containsNaN = False
        for name in self.net_.blobs:
            blob = self.net_.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                containsNaN = True

        if not containsNaN:
            print('Succeeded.')
            # break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    def extract(self, layer='predict_flow_final'): 
        return np.squeeze(self.net_.blobs[layer].data).transpose(1,2,0)
        
    @timeitmethod
    def describe(self, im, layer='predict_flow_final'):
        self.forward(im)
        return self.extract(layer=layer)

    # @staticmethod
    # def visualize(labels, colors):
    #     return color_by_lut(labels, colors)
