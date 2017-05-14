#!/usr/bin/env python
"""
Do windowed detection by classifying a number of images/crops at once,
optionally using the selective search window proposal method.

This implementation follows ideas in
    Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik.
    Rich feature hierarchies for accurate object detection and semantic
    segmentation.
    http://arxiv.org/abs/1311.2524

spillai: added detect_bboxes, with fc7 output

"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
#
# Based on https://github.com/rbgirshick/fast-rcnn

import os
import sys

import numpy as np
import scipy as sp

_PYCAFFE_PATH = os.getenv('PYCAFFE')
assert _PYCAFFE_PATH, 'PYCAFFE environment path not set'

sys.path.append(os.path.join(_PYCAFFE_PATH, 'python'))
sys.path.append(os.path.join(_PYCAFFE_PATH, 'caffe-fast-rcnn', 'python'))
sys.path.append(os.path.join(_PYCAFFE_PATH, 'lib'))
import caffe; caffe.set_mode_gpu(); caffe.set_device(0)

from fast_rcnn.test import _get_blobs, _bbox_pred, _clip_boxes, nms
from fast_rcnn.config import cfg

from pybot.utils.timer import timeitmethod

def im_detect(net, im, boxes, layer='fc7'):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))

    data = net.blobs[layer].data
    return data[inv_index, :] 

    # if cfg.TEST.SVM:
    #     # use the raw scores before softmax under the assumption they
    #     # were trained as linear SVMs
    #     scores = net.blobs['cls_score'].data
    # else:
    #     # use softmax estimated probabilities
    #     scores = blobs_out['cls_prob']

    # if cfg.TEST.BBOX_REG:
    #     # Apply bounding-box regression deltas
    #     box_deltas = blobs_out['bbox_pred']
    #     pred_boxes = _bbox_pred(boxes, box_deltas)
    #     pred_boxes = _clip_boxes(pred_boxes, im.shape)
    # else:
    #     # Simply repeat the boxes, once for each class
    #     pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    # if cfg.DEDUP_BOXES > 0:
    #     # Map scores and predictions back to the original set of boxes
    #     scores = scores[inv_index, :]
    #     pred_boxes = pred_boxes[inv_index, :]

    # return scores, pred_boxes

def extract_hypercolumns(net, im, boxes): 
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)

    # # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # # (some distinct image ROIs get mapped to the same feature ROI).
    # # Here, we identify duplicate feature ROIs, so we only compute features
    # # on the unique subset.
    # if cfg.DEDUP_BOXES > 0:
    #     v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    #     hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
    #     _, index, inv_index = np.unique(hashes, return_index=True,
    #                                     return_inverse=True)
    #     blobs['rois'] = blobs['rois'][index, :]
    #     boxes = boxes[index, :]

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))

    print dir(net.blobs), net.blobs.keys(), net.blobs['conv1'].data.shape

    hypercolumns = []
    # layers = ['conv2', 'conv3', 'conv4', 'conv5']
    layers = ['norm1', 'norm2']
    layers = ['pool1', 'pool2', 'pool5']
    # layers = ['fc6', 'fc7']
    for layer in layers: 
        print layer, net.blobs[layer].data.shape
        convmap = net.blobs[layer].data
        for fmap in convmap[0]:
            # print 'fmap', fmap.shape
            upscaled = sp.misc.imresize(fmap, size=(im.shape[0], im.shape[1]),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)

    # data = net.blobs['fc7'].data
    # return data[inv_index, :] 

class FastRCNNDescription(caffe.Net): 

    NETS = {'vgg16': ('VGG16',
              'vgg16_fast_rcnn_iter_40000.caffemodel'),
    'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                       'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
    'caffenet': ('CaffeNet',
                 'caffenet_fast_rcnn_iter_40000.caffemodel')}

    def __init__(self, rcnn_dir, net='vgg_cnn_m_1024'): 
        model_file = os.path.join(
            rcnn_dir, 'models', 
            FastRCNNDescription.NETS[net][0], 'test.prototxt')
        pretrained_file = os.path.join(
            rcnn_dir, 'data', 'fast_rcnn_models', 
            FastRCNNDescription.NETS[net][1])

        if not os.path.exists(model_file) or \
           not os.path.exists(pretrained_file): 
            raise ValueError('Unknown net {}, use one of {}, \n'
                             'model: {}, \npretrained file: {}'
                             .format(net, 
                                     FastRCNNDescription.NETS.keys(), 
                                     model_file, pretrained_file))

        # Init caffe with model
        cfg.TEST.BBOX_REG = False
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)        

    @property
    def layers(self):
        return self.blobs.keys()

    @timeitmethod
    def describe(self, im, boxes, layer='fc7'):
        return im_detect(self, im, boxes, layer=layer)

    def extract(self, layer='fc7'):
        return np.squeeze(self.blobs[layer].data, axis=0).transpose(1,2,0)

    def hypercolumn(self, im, boxes):
        return extract_hypercolumns(self, im, boxes)
