# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2
import numpy as np

def convex_hull(pts, ccw=True): 
    """
    Returns the convex hull of points, ordering them in ccw/cw fashion

    Note: Since the orientation of the coordinate system is x-right,
    y-up, ccw is interpreted as cw in the function call.

    """ 
    assert(pts.ndim == 2 and pts.shape[1] == 2)
    return (cv2.convexHull(pts.reshape(-1,1,2), clockwise=ccw)).reshape(-1,2)


# ===========================================================================
# BBOX-functions

def bbox_inbounds(bb, shape): 
    assert(shape[1] > shape[0] and bb.shape[1] == 4)
    return np.all(np.bitwise_and(np.bitwise_and(bb[:,0] >= 0, bb[:,2] < shape[1]), \
                                 np.bitwise_and(bb[:,1] >= 0, bb[:,3] < shape[0])))

def scale_bbox(bb, scale=1.0): 
    c = np.float32([bb[0]+bb[2], bb[1]+bb[3]]) * 0.5
    s = scale
    return np.float32([(bb[0]-c[0]) * s + c[0], (bb[1]-c[1]) * s + c[1], 
                       (bb[2]-c[0]) * s + c[0], (bb[3]-c[1]) * s + c[1]])                       

def scale_bboxes(bboxes, scale=1.0): 
    return np.vstack([ scale_bbox(bb, scale=scale) for bb in bboxes])
    
def boxify_pts(pts): 
    xmin, xmax = np.min(pts[:,0]), np.max(pts[:,0])
    ymin, ymax = np.min(pts[:,1]), np.max(pts[:,1])
    return np.float32([xmin, ymin, xmax, ymax])


def bbox_pts(bbox, ccw=True):
    if ccw: 
        return np.vstack([[bbox[0], bbox[1]], 
                          [bbox[0], bbox[3]], 
                          [bbox[2], bbox[3]], 
                          [bbox[2], bbox[1]]])
    else: 
        return np.vstack([[bbox[0], bbox[1]], 
                          [bbox[2], bbox[1]], 
                          [bbox[2], bbox[3]], 
                          [bbox[0], bbox[3]]])

def bbox_area(bbox): 
    return (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

def intersection_union(bbox1, bbox2): 
    # print bbox1, bbox2
    union_ = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1]) + \
             (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
    x1, x2 = np.maximum(bbox1[0], bbox2[0]), np.minimum(bbox1[2], bbox2[2])
    if x1 >= x2: 
        return 0, union_
    y1, y2 = np.maximum(bbox1[1], bbox2[1]), np.minimum(bbox1[3], bbox2[3])
    if y1 >= y2: 
        return 0, union_
    intersection = (y2-y1) * (x2-x1) * 1.0
    I, U = intersection, union_-intersection
    return I, U

def intersection_over_union(A,B): 
    I, U = intersection_union(A, B)
    return I * 1.0 / U

def brute_force_match(bboxes_truth, bboxes_test, 
                      match_func=lambda x,y: None, dtype=np.float32):
    A = np.zeros(shape=(len(bboxes_truth), len(bboxes_test)), dtype=dtype)
    for i, bbox_truth in enumerate(bboxes_truth): 
        for j, bbox_test in enumerate(bboxes_test): 
            A[i,j] = match_func(bbox_truth, bbox_test)
    return A

def brute_force_match_coords(bboxes_truth, bboxes_test): 
    return brute_force_match(bboxes_truth, bboxes_test, 
                             match_func=lambda x,y: intersection_over_union(x['coords'], y['coords']),
                             dtype=np.float32)

def brute_force_match_target(bboxes_truth, bboxes_test): 
    return brute_force_match(bboxes_truth, bboxes_test, 
                             match_func=lambda x,y: x['target'] == y['target'], 
                             dtype=np.bool)

def match_targets(bboxes_truth, bboxes_test, intersection_th=0.5): 
    A = brute_force_match_coords(bboxes_truth, bboxes_test)
    B = brute_force_match_target(bboxes_truth, bboxes_test)
    pos = np.bitwise_and(A > intersection_th, B)
    return pos

def match_bboxes(bboxes_truth, bboxes_test, intersection_th=0.5): 
    A = brute_force_match_coords(bboxes_truth, bboxes_test)
    return A > intersection_th

def match_bboxes_and_targets(bboxes_truth, bboxes_test, targets_truth, targets_test, intersection_th=0.5):
    A = brute_force_match(bboxes_truth, bboxes_test, match_func=intersection_over_union)
    B = brute_force_match(targets_truth, targets_test, match_func=lambda x,y: x==y, dtype=np.bool)
    return np.bitwise_and(A > intersection_th, B)
    
