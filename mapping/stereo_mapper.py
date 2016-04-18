"""
=====================================================================
 Stereopsis-based Mapper 
=====================================================================

Map the environment using stereo data
 TODO
   1. Semi-dense stereo 

"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: TODO

import numpy as np
import cv2, os, time
import argparse
from itertools import izip
from collections import OrderedDict, deque

from bot_geometry.rigid_transform import Pose, Quaternion, \
    RigidTransform, Sim3, normalize_vec

from bot_utils.io_utils import read_config
from bot_utils.db_utils import AttrDict

from bot_vision.camera_utils import StereoCamera
from bot_vision.image_utils import to_color, to_gray, flip_rb, im_resize
from bot_vision.stereo_utils import StereoSGBM, StereoBM, CalibratedFastStereo, setup_zed_dataset
from bot_vision.color_utils import get_color_by_label
from bot_vision.image_utils import to_color, to_gray, median_blur
from bot_vision.imshow_utils import imshow_cv

import bot_externals.draw_utils as draw_utils

from pybot_vision import scaled_color_disp 
from pybot_vision import CrossRatioStereo, FastStereo
from pybot_externals import StereoELAS, StereoVISO2, MonoVISO2, ORBSLAM2

class VOMixin(object): 
    def __init__(self, camera, alg='viso2'): 
        """
        Setup visual odometry using stereo data
        """
        if not isinstance(camera, StereoCamera): 
            raise TypeError('Camera provided is not StereoCamera, but {:}'.format())

        if alg == 'viso2':        
            self.vo_ = StereoVISO2(f=camera.left.fx, 
                                   cx=camera.left.cx, cy=camera.left.cy, 
                                   baseline=camera.baseline, relative=False, project_2d=False) 
            print 'Baseline: ', camera.baseline
        # elif alg == 'orb-slam2':
        #     path = '/home/spillai/perceptual-learning/software/python/apps/config/orb_slam/'

        #     settings_fn = os.path.join(path,'Settings_zed.yaml')
        #     vocab_fn = os.path.join(path, 'ORBvoc.txt')
        #     self.vo_ = ORBSLAM2(settings=settings_fn, vocab=vocab_fn, mode=ORBSLAM2.STEREO)
        #     self.vo_.process = lambda l,r: self.vo_.process_stereo(l,r)

        #     # self.slam_ = ORBSLAM2(settings='Settings_zed.yaml')
        #     # self.slam_.initialize_baseline(Tcw)
        #     # self.slam_ = LSDMapper(self.calib_.K0)
        else: 
            raise RuntimeError('Unknown VO algorithm: %s. Use either viso2' % alg)

    def process_vo(self, left_im, right_im):
        """
        Perform stereo VO
        """

        # Perform VO and update TF
        try: 
            # Stereo-VO
            T_ct = self.vo_.process(to_gray(left_im), to_gray(right_im))

            # # Monocular vSLAM
            # T_ct = self.slam_.process(to_gray(left_im), right=to_gray(right_im))

            p_ct = RigidTransform.from_matrix(T_ct)
        except Exception as e:
            print e
            return

        # # Project to 2d
        # _, t, _, x, _, z  = p_ct.to_roll_pitch_yaw_x_y_z()
        # pose_ct = RigidTransform.from_roll_pitch_yaw_x_y_z(0, t, 0, x, 0, z)

        pose_id = self.poses_[-1].id + 1 if len(self.poses_) else 0
        pose_ct  = Pose.from_rigid_transform(pose_id, p_ct) 
        # pose_ct  = Pose.from_rigid_transform(pose_id, self.poses_[-1].oplus(p_ct)) \
        #                                      if len(self.poses_) else Pose.from_rigid_transform(pose_id, p_ct)

        return pose_ct

class StereoMapper(VOMixin): 
    def __init__(self, camera, iterations=2, vo_alg='viso2', stereo_alg='elas', draw_relative=False): 
        VOMixin.__init__(self, camera, alg=vo_alg)

        # Check VO
        if not hasattr(self, 'vo_'): 
            raise RuntimeError('Visual Odometry is not setup: VOMixin inheritance missing')

        # Dataset calib
        self.camera_ = camera
        self.idx_ = 0
        self.map_every_ = 3
        self.draw_relative_ = draw_relative

        # Rolling window length
        # Queue length, and image subsample
        self.qlen_, self.s_ = 20, 2
        self.poses_ = deque(maxlen=self.qlen_)
  
        # Setup stereo solver (VO + Stereo disparity estimation)
        # self._setup_vo(alg=vo_alg)
        self._setup_stereo(alg=stereo_alg, iterations=iterations)

        # Check stereo
        if not hasattr(self, 'stereo_'): 
            raise RuntimeError('Stereo matching is not setup: consider running _setup_stereo()')

    def process_vo(self, left_im, right_im): 
        pose_ct = VOMixin.process_vo(self, left_im, right_im)

        # Every 5 frames
        if self.idx_ % self.map_every_ != 0: 
            return False

        # Add to poses list, and fix idx for plotting
        self.poses_.append(pose_ct)
        print self.poses_[-1]

        # Move body frame of reference
        if self.draw_relative_: 
            poses = [Pose.from_rigid_transform(p.id, self.poses_[-1].inverse() * p) for p in self.poses_]
        else: 
            poses = self.poses_ 
        draw_utils.publish_pose_list('stereo_vo', poses, frame_id='camera', reset=(self.idx_ == 0))
        draw_utils.publish_pose_t('CAMERA_POSE',  pose_ct, frame_id='camera')

        # Draw matches
        # try: 
        #     matches = np.int32(self.vo_.getMatches())
        #     print('Matches {:}'.format(len(matches)))

        #     im_H, im_W = left_im.shape[:2]
        #     vis = to_color(np.hstack([left_im, right_im]))
        #     for m in matches: 
        #         cv2.line(vis, (m[0], m[1]), (m[4], m[5]), (0,255,0), thickness=1)
        #         cv2.line(vis, (im_W + m[2], m[3]), (im_W + m[6], m[7]), (0,255,0), thickness=1)

        #     imshow_cv('matches', vis)
        #     imshow_cv('matches_all', vis2)
        # except: 
        #     pass

        return True


    def _setup_stereo(self, alg='elas', iterations=2): 
        """
        Setup stereo disparity estimation
        """
        # Stereo block matcher
        if alg == 'bm':
            # stereo_params = StereoBM.default_params
            # stereo_params['minDisparity'] = 0
            # stereo_params['numDisparities'] = 128
            self.stereo_ = StereoBM()
            self.stereo_.process = lambda l,r: self.stereo_.compute(l,r)
        elif alg == 'sgbm': 
            stereo_params = StereoSGBM.params_
            stereo_params['minDisparity'] = 0
            stereo_params['numDisparities'] = 128
            self.stereo_ = StereoSGBM(params=stereo_params)
            self.stereo_.process = lambda l,r: self.stereo_.compute(l,r)
        elif alg == 'elas': 
            self.stereo_ = StereoELAS()
        elif alg == 'cross-ratio': 
            self.stereo_ = CrossRatioStereo()
        elif alg == 'fast-stereo': 
            calib = self.camera_
            self.stereo_ = FastStereo(threshold=20, 
                                      stereo_method=FastStereo.TESSELLATED_DISPARITY, 
                                      lr_consistency_check=True)
            self.stereo_.set_calibration(calib.left.K, calib.right.K, 
                                         calib.left.D, calib.right.D, calib.left.R, calib.right.R, 
                                         calib.left.P, calib.right.P, calib.Q, calib.right.t)
            self.stereo_.cost_threshold = 0.15
            self.stereo_.iterations = iterations
            # self.stereo_ = CalibratedFastStereo(stereo, self.calib_, rectify=None)
        else: 
            raise RuntimeError('Unknown stereo algorithm: %s. Use either sgbm, elas or ordered' % alg)


    def process_gt_pose(self, pose_ct): 
        """
        Perform GT pose
        """

        # Perform VO and update TF
        pose_ct.id = self.poses_[-1].id + 1 if len(self.poses_) else 0

        # Every 5 frames
        if self.idx_ % self.map_every_ != 0: 
            return False

        # Add to poses list, and fix idx for plotting
        self.poses_.append(pose_ct)

        # Move body frame of reference
        if self.draw_relative_: 
            poses = [Pose.from_rigid_transform(p.id, self.poses_[-1].inverse() * p) for p in self.poses_]
        else: 
            poses = self.poses_ 
        draw_utils.publish_pose_list('stereo_vo', poses, frame_id='camera', reset=(self.idx_ == 0))
        draw_utils.publish_pose_t('CAMERA_POSE',  pose_ct, frame_id='camera')

        return True

    def process_stereo(self, left_im, right_im): 
        """
        Perform stereo disparity estimation
        """
        # Compute stereo disparity
        disp = self.stereo_.process(to_gray(left_im), to_gray(right_im))

        # Reconstruct stereo
        im_pub, X_pub = self.camera_.reconstruct_with_texture(disp, to_color(left_im), sample=self.s_)
        m = np.bitwise_and(X_pub[:,2] < 60, X_pub[:,2] > 0.1)
        im_pub, X_pub = im_pub[m], X_pub[m]

        draw_utils.publish_cloud('stereo_cloud', X_pub, c=im_pub, 
                                 frame_id='stereo_vo', element_id=self.idx_ / self.map_every_, 
                                 reset=False)

        # Plot disparity
        disp_color = scaled_color_disp(disp)
        zmask = disp > 0

        vis = to_color(left_im)
        vis[zmask] = disp_color[zmask]

        draw_utils.publish_botviewer_image_t(im_resize(vis, scale=0.5))

        # # Ensure X_pub doesn't have points below ground plane
        # X_pub = X_pub[X_pub[:,1] < 1.65]
        # draw_utils.publish_height_map('stereo_cloud', cam_bc * X_pub_m, frame_id='body', height_axis=2)

        # imshow_cv('disparity', disp8)
        # imshow_cv('disparity_cmap', disp_color)


    def process(self, left_im, right_im): 
        if self.process_vo(to_gray(left_im), to_gray(right_im)): 
            self.process_stereo(left_im, right_im)
        self.idx_ += 1

    def process_with_gt(self, left_im, right_im, pose): 
        if self.process_gt_pose(pose): 
            self.process_stereo(left_im, right_im)
        self.idx_ += 1

    def run(self): 
        raise NotImplementedError()
