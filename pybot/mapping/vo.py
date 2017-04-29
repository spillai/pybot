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
from pybot.vision.draw_utils import draw_features, draw_matches
from pybot.vision.imshow_utils import imshow_cv, print_status
from pybot.vision.image_utils import to_color
from pybot.utils.timer import timeitmethod

from pybot.externals.lcm import draw_utils

try: 
    from pybot_vision import recoverPose
except Exception,e:
    raise RuntimeWarning('Failed to import pybot_vision.recoverPose {}'.format(e.message))

class SimpleVO(object):
    def __init__(self, calib):
        self.calib_ = calib
        self.kf_items_q_ = Accumulator(maxlen=2)
        self.poses_q_ = Accumulator(maxlen=2)
        self.poses_q_.accumulate(RigidTransform())
        
        # Setup detector params
        num_tracks = 500
        fast_params = FeatureDetector.fast_params
        fast_params.threshold = 20
        detector_params = dict(method='fast', grid=(16,9), max_corners=num_tracks, 
                               max_levels=1, subpixel=True,
                               params=FeatureDetector.fast_params)

        # Setup tracker params (either lk, or dense)
        lk_params = dict(winSize=(21,21), maxLevel=3)
        # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        tracker_params = dict(method='lk', fb_check=True, params=lk_params)

        # Create KLT from detector params only
        self.klt_ = OpenCVKLT.from_params(detector_params=detector_params, 
                                          tracker_params=tracker_params, 
                                          min_tracks=num_tracks)

        
    @timeitmethod
    def _process_pts_cv3(self, pts1, pts2):
        focal = self.calib_.fx
        pp = (self.calib_.cx, self.calib_.cy)
        E, mask = cv2.findEssentialMat(pts2, pts1,
                                       focal=focal, pp=pp, 
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, focal=focal, pp=pp)
        return RigidTransform.from_Rt(R, t.ravel())

    @timeitmethod
    def _process_pts_wrap(self, pts1, pts2):
        # Fundamental matrix estimation
        method, px_dist, conf =  cv2.cv.CV_FM_RANSAC, 1, 0.99
        (F, inliers) = cv2.findFundamentalMat(pts1, pts2, method, px_dist, conf)
        E = compute_essential(F, self.calib_.K)
        distance_threshold = 1.0
        R, t, X, mask = recoverPose(E, pts1, pts2, self.calib_.K, distance_threshold, inliers)
        return RigidTransform.from_Rt(R, t.ravel())

    @timeitmethod
    def _process_pts(self, pts1, pts2):
        
        # Fundamental matrix estimation
        method, px_dist, conf =  cv2.cv.CV_FM_RANSAC, 1, 0.99
        (F, inliers) = cv2.findFundamentalMat(pts1, pts2, method, px_dist, conf)
        inliers = inliers.ravel().astype(np.bool)
        pts1, pts2 = pts1[inliers], pts2[inliers]
        npts2 = len(pts1)

        # Compute E -> R1,R2,t
        E = compute_essential(F, self.calib_.K)
        R1, R2, t = decompose_E(E)

        # Naive strategy: Pick R with the least incremental rotation 
        rts = [RigidTransform.from_Rt(R1, t), RigidTransform.from_Rt(R2, t)]
        rts_norm = [np.linalg.norm(rt.rpyxyz[:3]) for rt in rts]
        rt = rts[1] if rts_norm[0] > rts_norm[1] \
             else rts[0]
        return rt
    
    @timeitmethod
    def process(self, im, scale=1.0):
        # ---------------------------
        # 1. Process image: KLT tracking
        self.klt_.process(im)
        
        # Gather points, ids, flow and age
        ids, pts = self.klt_.latest_ids, self.klt_.latest_pts
        
        # Add KF items to queue
        self.kf_items_q_.accumulate(
            AttrDict(img=im, ids=ids, pts=pts)
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

        # fvis = draw_matches(im, kf_pts1, kf_pts2, colors=np.tile([0,255,0], [len(kf_pts1), 1]))
        # npts1 = len(kf_pts1)
        # imshow_cv('fvis', fvis)

        # vis = to_color(im)
        # self.klt_.draw_tracks(vis, colored=True, color_type='unique')
        # imshow_cv('vis', vis)

        
        
        # ---------------------------
        # 3. FILTERING VIA Fundamental matrix RANSAC

        # self._process_pts(kf_pts1, kf_pts2)
        rt = self._process_pts(kf_pts1, kf_pts2)
        print rt
        
        R, t = rt.R, rt.tvec
        crt = self.poses_q_.latest
        cR, ct = crt.R, crt.tvec

        ct = ct + scale * cR.dot(t)
        cR = R.dot(cR)

        newp = RigidTransform.from_Rt(cR, ct.ravel())
        self.poses_q_.accumulate(newp)
        
        # newp = self.poses_q_.latest * nrt
        # print self.poses_q_.index, np.rad2deg(nrt.rpyxyz[2])
        draw_utils.publish_cameras('camera', [Pose.from_rigid_transform(self.poses_q_.index, newp)],
                                   reset=False, frame_id='camera_upright')
        draw_utils.publish_botviewer_image_t(im, jpeg=True)
        

def test_vo():
    import argparse
    from pybot.vision.imshow_utils import imshow_cv
    from pybot.utils.test_utils import test_dataset
    
    dataset = test_dataset('00')
    lcam = dataset.calib.left

    vo = SimpleVO(lcam)

    draw_utils.publish_sensor_frame('camera_upright',
                                    pose=RigidTransform.from_rpyxyz(np.pi/2,np.pi/2,0,0,0,1))

    
    ppose = None
    for f in dataset.iterframes():
        # imshow_cv('im', im)

        # Process image: KLT tracking
        scale = np.linalg.norm(ppose.tvec - f.pose.tvec) \
                if ppose is not None else 1.0
        ppose = f.pose
        vo.process(f.left, scale=scale)

        # # scale
        # if scale > 0.1 and t[2] > t[0] and t[2] > 1:
        #     t = t + s * (R * t_o)
        #     R_new = R_old * R_new

if __name__ == "__main__":
    test_vo()        
