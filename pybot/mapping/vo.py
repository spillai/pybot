"""
Visual odometry tools
"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
import cv2
from itertools import izip
from collections import deque

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

class VO(object):
    def __init__(self, calib, 
                 num_tracks=300, min_tracks=200,
                 lk_window_size=21, lk_levels=3, grid_size=30,
                 visualize=False):

        # Setup visualizations
        self.vis_ = None
        self.visualize_ = visualize

        # Setup detector params
        fast_params = FeatureDetector.fast_params
        fast_params.threshold = 20

        # Grid-sampling
        GH, GW = calib.shape[:2] / grid_size
        print('Setting grid size ({},{})'.format(GW,GH))

        # Detector
        detector_params = dict(
            method='fast', grid=(GW,GH), max_corners=num_tracks, 
            max_levels=1, subpixel=True,
            params=FeatureDetector.fast_params)
        
        # Setup tracker params (either lk, or dense)
        W = lk_window_size
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        lk_params = dict(winSize=(W,W), maxLevel=lk_levels, criteria=criteria)
        tracker_params = dict(method='lk', fb_check=True, params=lk_params)

        # Create KLT from detector params only
        self.calib_ = calib
        self.kf_q_ = Accumulator(maxlen=2)
        self.klt_ = OpenCVKLT.from_params(detector_params=detector_params, 
                                          tracker_params=tracker_params, 
                                          min_tracks=min_tracks)
        
    @property
    def initialized(self):
        return len(self.kf_q_) >= 2

    @timeitmethod
    def _process_image(self, im):
        # ---------------------------
        # 1. Process image: KLT tracking
        self.klt_.process(im)

        if self.visualize_: 
            self.vis_ = to_color(im)
            vis = to_color(im)
            vis = self.klt_.visualize(vis, colored=True)
            imshow_cv('colored-vis', vis)
        
        # Add KF items to queue
        ids, pts = self.klt_.latest_ids, self.klt_.latest_pts

        # Undistort pts before inserting into queue
        pts = self.calib_.undistort_points(pts)
        self.kf_q_.append(
            AttrDict(ids=ids, pts=pts)
        )

    @property
    def matches(self): 
        assert(self.initialized)
        
        # ---------------------------
        # 2. KF-KF matching

        # kf1 -> kf2: kf1 (older), kf2 (newer)
        kf1, kf2 = self.kf_q_[-2], self.kf_q_[-1]
        kf_ids1, kf_pts1 = kf1.ids, kf1.pts
        kf_ids2, kf_pts2 = kf2.ids, kf2.pts

        kf_pts1_lut = {tid: pt for (tid,pt) in zip(kf_ids1,kf_pts1)}
        kf_pts2_lut = {tid: pt for (tid,pt) in zip(kf_ids2,kf_pts2)}

        # Find matches in the newer keyframe that are consistent from 
        # the previous frame
        matched_ids = np.intersect1d(kf_ids2, kf_ids1)
        if not len(matched_ids):
            return [], [], []
        
        kf_pts1 = np.vstack([ kf_pts1_lut[tid] for tid in matched_ids ])
        kf_pts2 = np.vstack([ kf_pts2_lut[tid] for tid in matched_ids ])

        # Visualize
        if self.visualize_: 
            self.vis_ = draw_matches(self.vis_, kf_pts1, kf_pts2, colors=np.tile([0,0,255], [len(kf_pts1), 1]))
            npts1 = len(kf_pts1)
        
        return (matched_ids, kf_pts1, kf_pts2)

    @timeitmethod
    def process(self, im, scale=1.0):
        # 1. Process image: KLT tracking
        self._process_image(im)

        # Only continue on sufficient frames for init.
        if not self.initialized:
            return

        ids, pts1, pts2 = self.matches

        # 2. Do something
        raise NotImplementedError()
        
class NisterVO(VO):
    def __init__(self, calib, restrict_2d=False, 
                 num_tracks=300, min_tracks=200,
                 lk_window_size=21, lk_levels=3, grid_size=30,
                 visualize=False):
        super(NisterVO, self).__init__(
            calib, num_tracks=num_tracks, min_tracks=min_tracks,
            lk_window_size=lk_window_size, lk_levels=lk_levels, grid_size=grid_size,
            visualize=visualize)
        
        # Optionally restricted 2D pose estimation
        self.restrict_2d_ = restrict_2d

        # Poses for compounding
        self.poses_q_ = Accumulator(maxlen=2)
        self.poses_q_.append(RigidTransform())
        
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
        method, px_dist, conf =  cv2.cv.CV_FM_RANSAC, 1.0, 0.999
        (F, inliers) = cv2.findFundamentalMat(pts1, pts2, method, px_dist, conf)
        E = compute_essential(F, self.calib_.K)
        distance_threshold = 1.0
        R, t, X, mask = recoverPose(E, pts1, pts2, self.calib_.K, distance_threshold, inliers)
        return RigidTransform.from_Rt(R, t.ravel())

    @timeitmethod
    def _process_pts(self, pts1, pts2):
        " F matrix estimation using Nister 5-pt (X-left, Y-up, Z-fwd)"
        
        # Fundamental matrix estimation
        method, px_dist, conf =  cv2.cv.CV_FM_RANSAC, 1.0, 0.999
        (F, inliers) = cv2.findFundamentalMat(pts1, pts2, method, px_dist, conf)
        inliers = inliers.ravel().astype(np.bool)
        pts1, pts2 = pts1[inliers], pts2[inliers]
        npts2 = len(pts1)

        if self.visualize_: 
            self.vis_ = draw_matches(self.vis_, pts1, pts2, colors=np.tile([0,255,0], [len(pts1), 1]))
            imshow_cv('fvis', self.vis_)

        # Compute E -> R1,R2,t
        E = compute_essential(F, self.calib_.K)
        R1, R2, t = decompose_E(E)

        # Naive strategy: Pick R with the least incremental rotation 
        rts = [RigidTransform.from_Rt(R1, t), RigidTransform.from_Rt(R2, t)]
        rts_norm = [np.linalg.norm(rt.rpyxyz[:3]) for rt in rts]
        rt = rts[1] if rts_norm[0] > rts_norm[1] \
             else rts[0]
        return rt.scaled(1.0)
            
    @timeitmethod
    def process(self, im, scale=1.0):
        # 1. Process image: KLT tracking
        self._process_image(im)

        # Only continue on sufficient frames for init.
        if not self.initialized:
            return

        # Get matches
        kf_ids, kf_pts1, kf_pts2 = self.matches
        if not len(kf_ids): assert(0)

        # 2. FILTERING VIA Fundamental matrix RANSAC
        rt = self._process_pts(kf_pts1, kf_pts2)
        
        # Restrict 2D
        if self.restrict_2d_:
            rpyxyz = rt.to_rpyxyz(axes='sxyz')
            rpyxyz[0], rpyxyz[2], rpyxyz[4] = 0, 0, 0
            rt = RigidTransform.from_rpyxyz(*rpyxyz, axes='sxyz').scaled(rt.scale)
        
        # Sim3 scaled transformation
        crt = self.poses_q_.latest
        newp = crt.scaled(scale) * rt
        draw_utils.publish_pose_t('CAMERA_POSE', newp, frame_id='camera_upright')
        
        self.poses_q_.append(newp)
        draw_utils.publish_cameras('camera', [Pose.from_rigid_transform(self.poses_q_.index, newp)],
                                   reset=False, frame_id='camera_upright')
        # draw_utils.publish_botviewer_image_t(im, jpeg=True)

        
class SlidingWindowVO(VO):
    def __init__(self, calib, restrict_2d=False, 
                 num_tracks=300, min_tracks=200,
                 lk_window_size=21, lk_levels=3, grid_size=30,
                 visualize=False):
        super(SlidingWindowVO, self).__init__(
            calib, num_tracks=num_tracks, min_tracks=min_tracks,
            lk_window_size=lk_window_size, lk_levels=lk_levels, grid_size=grid_size,
            visualize=visualize)
        
        # Optionally restricted 2D pose estimation
        self.restrict_2d_ = restrict_2d

        # Poses for compounding
        self.poses_q_ = Accumulator(maxlen=2)
        self.poses_q_.append(RigidTransform())
            
    @timeitmethod
    def process(self, im, scale=1.0):
        # 1. Process image: KLT tracking
        self._process_image(im)

        # Only continue on sufficient frames for init.
        if not self.initialized:
            return

        # Get matches
        kf_ids, kf_pts1, kf_pts2 = self.matches
        if not len(kf_ids): assert(0)

        
    
        

def test_vo():
    from pybot.vision.imshow_utils import imshow_cv
    from pybot.utils.test_utils import test_dataset
    
    dataset = test_dataset(sequence='00', scale=0.5)
    lcam = dataset.calib.left
    poses = dataset.poses
    
    vo = NisterVO(lcam, restrict_2d=True,
                         num_tracks=500, min_tracks=300, grid_size=30,
                         lk_window_size=25, lk_levels=3, visualize=True)
    draw_utils.publish_sensor_frame('camera_upright',
                                    pose=RigidTransform.from_rpyxyz(np.pi/2,np.pi/2,0,0,0,1))

    parr = np.vstack([p.tvec for p in poses[::10]])
    draw_utils.publish_line_segments('trajectory', parr[:-1,:], parr[1:,:], c='b')
    draw_utils.publish_pose_list('poses', poses[::10], frame_id='camera')
    
    
    ppose = None
    for f in dataset.iterframes():
        scale = np.linalg.norm(ppose.tvec - f.pose.tvec) \
                if ppose is not None else 1.0

        ppose = f.pose
        vo.process(f.left, scale=scale)

def test_slidingwindowvo():
    from pybot.vision.imshow_utils import imshow_cv
    from pybot.utils.test_utils import test_dataset
    
    dataset = test_dataset(sequence='00', scale=0.5)
    lcam = dataset.calib.left
    poses = dataset.poses

    vo = SlidingWindowVO(lcam, restrict_2d=True,
                         num_tracks=500, min_tracks=300, grid_size=30,
                         lk_window_size=25, lk_levels=3, visualize=True)

    ppose = None
    for f in dataset.iterframes():
        scale = np.linalg.norm(ppose.tvec - f.pose.tvec) \
                if ppose is not None else 1.0
        
        ppose = f.pose
        vo.process(f.left, scale=scale)

    
if __name__ == "__main__":
    test_vo()        
    # test_slidingwindowvo()
