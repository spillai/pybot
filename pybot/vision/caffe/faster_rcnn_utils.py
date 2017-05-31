#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
#
# Based on https://github.com/rbgirshick/py-faster-rcnn

import os
import sys
import numpy as np
import scipy as sp

_PYCAFFE_PATH = os.getenv('PYCAFFE')
assert _PYCAFFE_PATH, 'PYCAFFE environment path not set'

sys.path.append(os.path.join(_PYCAFFE_PATH, 'python'))
sys.path.append(os.path.join(_PYCAFFE_PATH, 'caffe-fast-rcnn', 'python'))
sys.path.append(os.path.join(_PYCAFFE_PATH, 'lib'))
import caffe; # caffe.set_mode_gpu(); caffe.set_device(0)

from fast_rcnn.config import cfg
cfg.TEST.HAS_RPN = True
from fast_rcnn.test import _get_blobs, nms, apply_nms
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv

from pybot.utils.timer import timeitmethod

def im_detect(net, im, boxes=None, layer='fc7', forward=False):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)
    
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    # If only requesting forward pass, return RPN bboxes or None
    if forward:
        if cfg.TEST.HAS_RPN: 
            return boxes
        else: 
            return None
    
    data = net.blobs[layer].data
    print 'Boxes: {}, data: {}'.format(data.shape, boxes.shape)
    
    if cfg.TEST.HAS_RPN: 
        return data, boxes
    else: 
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
    #     pred_boxes = bbox_transform_inv(boxes, box_deltas)
    #     pred_boxes = clip_boxes(pred_boxes, im.shape)
    # else:
    #     # Simply repeat the boxes, once for each class
    #     pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    # if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
    #     # Map scores and predictions back to the original set of boxes
    #     scores = scores[inv_index, :]
    #     pred_boxes = pred_boxes[inv_index, :]

    # return scores, pred_boxes

# def demo(net, image_name):
#     """Detect object classes in an image using pre-computed object proposals."""

#     # Load the demo image
#     im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
#     im = cv2.imread(im_file)

#     # Detect all object classes and regress object bounds
#     timer = Timer()
#     timer.tic()
#     scores, boxes = im_detect(net, im)
#     timer.toc()
#     print ('Detection took {:.3f}s for '
#            '{:d} object proposals').format(timer.total_time, boxes.shape[0])

#     # Visualize detections for each class
#     CONF_THRESH = 0.8
#     NMS_THRESH = 0.3
#     for cls_ind, cls in enumerate(CLASSES[1:]):
#         cls_ind += 1 # because we skipped background
#         cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
#         cls_scores = scores[:, cls_ind]
#         dets = np.hstack((cls_boxes,
#                           cls_scores[:, np.newaxis])).astype(np.float32)
#         keep = nms(dets, NMS_THRESH)
#         dets = dets[keep, :]
#         vis_detections(im, cls, dets, thresh=CONF_THRESH)


class FasterRCNNDescription(object): 
    NETS = {'vgg16': ('VGG16',
                      'VGG16_faster_rcnn_final.caffemodel'),
            'zf': ('ZF',
                   'ZF_faster_rcnn_final.caffemodel')}
    def __init__(self, rcnn_dir, num_proposals=300, with_rpn=False,
                 net='zf', model_dir='pascal_voc', opt_dir='fast_rcnn_end2end'): 
        """
        net: vgg16, zf
        model_dir: [pascal_voc, coco]
        opt_dir: [fast_rcnn, fast_rcnn_alt_opt, fast_rcnn_end2end]

        Layers: data, im_info, conv1, norm1, pool1, conv2, norm2,
        pool2, conv3, conv4, conv5, conv5_relu5_0_split_0,
        conv5_relu5_0_split_1, rpn_conv1,
        rpn_conv1_rpn_relu1_0_split_0, rpn_conv1_rpn_relu1_0_split_1,
        rpn_cls_score, rpn_bbox_pred, rpn_cls_score_reshape,
        rpn_cls_prob, rpn_cls_prob_reshape, rois, roi_pool_conv5, fc6,
        fc7, fc7_drop7_0_split_0, fc7_drop7_0_split_1, cls_score,
        bbox_pred, cls_prob
        """
    
        model_file = os.path.join(
            rcnn_dir, 'models', model_dir, 
            FasterRCNNDescription.NETS[net][0],
            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
            # opt_dir, 'test.prototxt')
            
        pretrained_file = os.path.join(
            rcnn_dir, 'data', 'faster_rcnn_models', 
            FasterRCNNDescription.NETS[net][1])
        
        if not os.path.exists(model_file) or \
           not os.path.exists(pretrained_file): 
            raise ValueError('Unknown net {}, use one of {}, \n'
                             'model: {}, \npretrained file: {}'
                             .format(net, 
                                     FasterRCNNDescription.NETS.keys(), 
                                     model_file, pretrained_file))

        # Init caffe with model
        cfg.TEST.HAS_RPN = with_rpn
        cfg.TEST.BBOX_REG = False
        cfg.TEST.RPN_POST_NMS_TOP_N = num_proposals

        self.net_ = caffe.Net(model_file, pretrained_file, caffe.TEST)

    @property
    def layers(self):
        return self.net_.blobs.keys()

    @timeitmethod
    def describe(self, im, boxes=None, layer='fc7'):
        return im_detect(self.net_, im, boxes=boxes, layer=layer)
        
    @timeitmethod
    def forward(self, im, boxes=None):
        return im_detect(self.net_, im, boxes=boxes, forward=True)
        
    def extract(self, layer='fc7'):
        return self.net_.blobs[layer].data

    # def hypercolumn(self, im, boxes=None):
    #     return extract_hypercolumns(self, im, boxes=boxes)

