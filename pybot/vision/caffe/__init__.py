# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2
import os
os.environ["GLOG_minloglevel"] ="3"

_PYCAFFE_PATH = os.getenv('PYCAFFE')
assert _PYCAFFE_PATH, 'PYCAFFE environment path not set'
print('PYCAFFE: {}'.format(_PYCAFFE_PATH))

import numpy as np

# Initialize RCNN Caffe / GPU (set GPU via CUDA_VISIBLE_DEVICES=GPU_ID)
def resize_to(labels, im): 
    return cv2.resize(labels, (im.shape[1],im.shape[0]), fx=0., fy=0., interpolation=cv2.INTER_AREA)

# SegNet
# =================================================================================

def setup_segnet(model_file, weights_file): 
    print('=====> SegNet')
    from pybot.vision.caffe.segnet_utils import SegNet
    return SegNet(model_file, weights_file)

# Posenet
# =================================================================================

def setup_posenet(model_file, weights_file, imagemean_file): 
    print('=====> Posenet')
    from pybot.vision.caffe.posenet_utils import PoseNet
    return PoseNet(model_file, weights_file, imagemean_file)

# RCNN
# =================================================================================

def setup_rcnn(method, data_dir, net): 
    if method == 'fast_rcnn':
        print('=====> Fast RCNN')
        # fast_rcnn: ['vgg16', 'vgg_cnn_m_1024', 'caffenet']
        from pybot.vision.caffe.fast_rcnn_utils import FastRCNNDescription
        rcnn = FastRCNNDescription(data_dir, net=net)
    elif method == 'faster_rcnn':
        print('=====> Faster RCNN')
        # faster_rcnn: ['vgg16', 'zf']
        from pybot.vision.caffe.faster_rcnn_utils import FasterRCNNDescription
        rcnn = FasterRCNNDescription(
            data_dir, with_rpn=False, 
            net=net, opt_dir='fast_rcnn') # TODO opt_dir
    else: 
        raise ValueError('Unknown rcnn method {}'.format(method))

    return rcnn

class FastRCNNObjectDetector(object): 
    def __init__(self, proposer, rcnn, clf): 
        self.proposer_ = proposer
        self.rcnn_ = rcnn
        self.clf_ = clf

    def predict(self, im): 
        # 1. Propose objects, 
        # 2. Describe proposals
        # 3. Classify proposals
        bboxes = self.proposer_.process(im)
        phi = self.rcnn_.describe(im, bboxes)
        targets = self.clf_.predict(phi)
        scores = self.clf_.decision_function(phi)
        return bboxes, phi, scores, targets


class FasterRCNNObjectDetector(object): 
    def __init__(self, rcnn, clf): 
        self.rcnn_ = rcnn
        self.clf_ = clf

    def predict(self, im): 
        phi,bboxes = self.rcnn_.describe(im, boxes=None)
        bboxes = self.bboxes_.astype(np.int64)
        targets = self.clf_.predict(phi)
        scores = self.clf_.decision_function(phi)
        return bboxes, phi, scores, targets

