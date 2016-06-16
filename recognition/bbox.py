import numpy as np
from cython_bbox import bbox_overlaps as brute_force_iou

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
