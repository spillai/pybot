# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
import os
import numpy as np
import cv2
os.environ["GLOG_minloglevel"] ="3"

# SegNet
# =================================================================================

def setup_segnet(model_file, weights_file): 
    print('=====> SegNet')
    from pybot.vision.caffe.segnet_utils import SegNetDescription
    return SegNetDescription(model_file, weights_file)

class SegNetFeatureExtraction(object): 
    def __init__(self, segnet_model, segnet_weights): 
        self.segnet_ = setup_segnet(segnet_model, segnet_weights)
        
    def process(self, im): 
        self.segnet_.forward(im)
        conv64 = self.segnet_.extract(layer='conv1_1_D')
        labels = self.segnet_.extract(layer='argmax')
        return labels.transpose(1,2,0).astype(np.uint8), conv64.transpose(1,2,0)

    def resize(self, labels, im): 
        return cv2.resize(labels, (im.shape[1],im.shape[0]), fx=0., fy=0., interpolation=cv2.INTER_AREA)
    
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
            net=net, opt_dir='fast_rcnn')
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

