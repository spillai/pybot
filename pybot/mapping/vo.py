"""
Visual odometry tools
"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
import cv2
from itertools import izip

from pybot.utils.misc import Accumulator
from pybot.utils.db_utils import AttrDict
from pybot.vision.camera_utils import Camera, CameraExtrinsic, CameraIntrinsic
from pybot.geometry.rigid_transform import Pose, RigidTransform
from pybot.vision.camera_utils import compute_essential, decompose_E
from pybot.vision.feature_detection import FeatureDetector
from pybot.vision.trackers.base_klt import OpenCVKLT

class SimpleVO(object):
    def __init__(self, calib):
        self.calib_ = calib
        self.kf_items_q_ = Accumulator(maxlen=2)

        # Setup detector params
        fast_params = FeatureDetector.fast_params
        fast_params.threshold = 20
        detector_params = dict(method='fast', grid=(8,5), max_corners=100, 
                               max_levels=1, subpixel=True,
                               params=FeatureDetector.fast_params)

        # Setup tracker params (either lk, or dense)
        lk_params = dict(winSize=(5,5), maxLevel=3)
        tracker_params = dict(method='lk', fb_check=True, params=lk_params)

        # Create KLT from detector params only
        self.klt_ = OpenCVKLT.from_params(detector_params=detector_params, 
                                          tracker_params=tracker_params, 
                                          min_tracks=100)
        

        
    def process_cv3(self, pts1, pts2):
        focal = self.calib_.fx
        pp = (self.calib_.cx, self.calib_.cy)
        E, mask = cv2.findEssentialMat(pts2, pts1,
                                       focal=focal, pp=pp, 
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, focal=focal, pp=pp)
        print E, R, t

    def process(self, im):
        # ---------------------------
        # 1. Process image: KLT tracking
        self.klt_.process(im)
        
        # Gather points, ids, flow and age
        ids, pts, age, flow = self.klt_.latest_ids, self.klt_.latest_pts, \
                              self.klt_.latest_age, self.klt_.latest_flow
        inds = age > 1
        
        # Add KF items to queue
        self.kf_items_q_.accumulate(
            AttrDict(img=im, ids=ids, pts=pts, age=age, flow=flow)
        )

        # ---------------------------
        # 2. KF-KF matching

        # Here kf1 (older), kf2 (newer)
        kf2 = self.kf_items_q_.items[-1]
        kf_ids2, kf_pts2 = kf2.ids, kf2.pts

        # Continue if only the first frame
        if len(self.kf_items_q_) < 2:
            return
        
        kf1 = self.kf_items_q_.items[-2]
        kf_ids1, kf_pts1 = kf1.ids, kf1.pts

        kf_pts1_lut = {tid: pt for (tid,pt) in izip(kf_ids1,kf_pts1)}
        kf_pts2_lut = {tid: pt for (tid,pt) in izip(kf_ids2,kf_pts2)}

        # Find matches in the newer keyframe that are consistent from 
        # the previous frame
        matched_ids = np.intersect1d(kf_ids2, kf_ids1)
        if not len(matched_ids): return 
        kf_pts1 = np.vstack([ kf_pts1_lut[tid] for tid in matched_ids ])
        kf_pts2 = np.vstack([ kf_pts2_lut[tid] for tid in matched_ids ])

        # fvis = draw_matches(frame2.img, kf_pts1, kf_pts2, colors=np.tile([0,0,255], [len(kf_pts1), 1]))
        # npts1 = len(kf_pts1)

        # ---------------------------
        # 3. FILTERING VIA Fundamental matrix RANSAC

        # Fundamental matrix estimation
        method, px_dist, conf =  cv2.cv.CV_FM_RANSAC, 3, 0.99
        (F, inliers) = cv2.findFundamentalMat(kf_pts1, kf_pts2, method, px_dist, conf)
        inliers = inliers.ravel().astype(np.bool)
        kf_pts1, kf_pts2 = kf_pts1[inliers], kf_pts2[inliers]
        matched_ids = matched_ids[inliers]
        npts2 = len(kf_pts1)

        # Test BA
        E = compute_essential(F, self.calib_.K)
        R1, R2, t = decompose_E(E)

        print RigidTransform.from_Rt(R1, t), RigidTransform.from_Rt(R2, t)

        # X = triangulate_points(cam1, kf_pts1, cam2, kf_pts2)
        # two_view_BA(cam1, kf_pts1, kf_pts2,
        #             X, frame1.pose.inverse() * frame2.pose, scale_prior=True)


        

def test_vo():
    import argparse
    from pybot.vision.imshow_utils import imshow_cv
    from pybot.utils.test_utils import test_dataset
    
    dataset = test_dataset()
    lcam = dataset.calib.left

    vo = SimpleVO(lcam)
    for im in dataset.iteritems():
        imshow_cv('im', im)

        # Process image: KLT tracking
        vo.process(im)

        # # scale
        # if scale > 0.1 and t[2] > t[0] and t[2] > 1:
        #     t = t + s * (R * t_o)
        #     R_new = R_old * R_new

if __name__ == "__main__":
    test_vo()        
