"""
Visual odometry / SLAM externals
"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
from pybot.vision.image_utils import to_gray
from pybot.vision.camera_utils import Camera, StereoCamera, RGBDCamera, CameraExtrinsic, CameraIntrinsic
from pybot.geometry.rigid_transform import Pose, RigidTransform

# from pyopengv import relative_pose_ransac as _relative_pose_ransac

# def to4x4(m):
#     T = np.eye(4, dtype=m.dtype)
#     T[:3,:] = m
#     return T

# def relative_pose_ransac(bearing_vectors1, bearing_vectors2, 
#                          sac_model="NISTER", threshold=0.01, max_iterations=1000): 
#     T_ct = _relative_pose_ransac(
#         bearing_vectors1, bearing_vectors2, sac_model, threshold, max_iterations)
#     print T_ct
#     return RigidTransform.from_matrix(to4x4(T_ct))

class VisualOdometry(object): 
    def __init__(self, camera, method='viso2'): 
        """
        Setup visual odometry using stereo data
        """
        if method == 'viso2':        
            if not isinstance(camera, StereoCamera): 
                raise TypeError('For viso2, Camera provided is not StereoCamera, but {:}'.format(type(camera)))

            from pybot_externals import StereoVISO2
            self.vo_ = StereoVISO2(f=camera.left.fx, 
                                   cx=camera.left.cx, cy=camera.left.cy, 
                                   baseline=camera.baseline, relative=False, project_2d=False) 
            print 'Baseline: ', camera.baseline
        elif method == 'fovis': 
            if not isinstance(camera, RGBDCamera): 
                raise TypeError('For fovis, Camera provided is not RGBDCamera, but {:}'.format(type(camera)))

            from pybot_externals import FOVIS                
            self.vo_ = FOVIS(rgbK=camera.rgb.K, depthK=camera.depth.K, 
                             width=int(camera.rgb.shape[1]), height=int(camera.rgb.shape[0]), 
                             shift_offset=1093.4753, projector_depth_baseline=camera.baseline)
            print 'Baseline: ', camera.baseline
        else: 
            raise RuntimeError('Unknown VO method: %s. Use either viso2' % method)
        self.method_ = method

    def process(self, left_im, right_im):
        """
        Perform stereo VO
        """

        # Perform VO and update TF
        if self.method_ == 'viso2': 
            T_ct = self.vo_.process(to_gray(left_im), to_gray(right_im))
        elif self.method_ == 'fovis' :
            if left_im.ndim != 3 or right_im.dtype != np.float32: 
                raise RuntimeError('FOVIS requires RGBD format')
            if right_im.dtype == np.float32: 
                depth_im = (right_im * 1000).astype(np.uint16)
            elif right_im.dtype == np.uint16: 
                depth_im = right_im.copy()
            else: 
                raise RuntimeError()

            T_ct = self.vo_.process(left_im, depth_im)
        else: 
            raise NotImplementedError()

        p_ct = RigidTransform.from_matrix(T_ct)
        return p_ct


class VisualSLAM(object): 
    ORB_SLAM_PATH = '/home/spillai/perceptual-learning/data/orb-slam2/'
    def __init__(self, camera, method='orb-slam2-stereo'): 
        if not isinstance(camera, CameraIntrinsic): 
            raise TypeError('For orb-slam2, Camera provided is not Camera, but {:}'.format(type(camera)))

        if 'orb-slam2' in method: 
            import os.path
            from pybot_externals import ORBSLAM2
            settings_fn = os.path.join(VisualSLAM.ORB_SLAM_PATH, 'settings_kinect.yaml')
            vocab_fn = os.path.join(VisualSLAM.ORB_SLAM_PATH, 'ORBvoc.txt')
            try: 
                mode = {'orb-slam2-mono': ORBSLAM2.MONOCULAR, 
                        'orb-slam2-stereo': ORBSLAM2.STEREO, 
                        'orb-slam2-rgbd': ORBSLAM2.RGBD}[method]
            except: 
                raise ValueError('Unknown method in orb-slam2 {}'.format(method))

            self.slam_ = ORBSLAM2(settings=settings_fn, vocab=vocab_fn, mode=mode)
            # self.slam_.set_calib(camera.K, camera.D, camera.baseline_px)

            if mode == ORBSLAM2.MONOCULAR: 
                self.process = self.process_mono
            elif mode == ORBSLAM2.STEREO: 
                self.process = self.process_stereo
            elif mode == ORBSLAM2.RGBD: 
                self.process = self.process_rgbd
            else: 
                assert(0)
        else: 
            raise RuntimeError('Unknown VO method: %s. Use either viso2' % method)
        self.method_ = method
        
    @property
    def latest_estimate(self): 
        Tcw = self.slam_.getCurrentPoseEstimate()
        if not isinstance(Tcw, np.ndarray): 
            return None
        print Tcw
        return RigidTransform.from_matrix(Tcw).inverse()

    def process_mono(self, left): 
        self.slam_.process_monocular(to_gray(left))
        # return self.latest_estimate

    def process_stereo(self, left, right): 
        self.slam_.process_stereo(to_gray(left), to_gray(right))
        # return self.latest_estimate

    def process_rgbd(self, left, depth):
        if depth.dtype == np.float32: 
            depth_im = (depth * 1000).astype(np.uint16)
        elif depth.dtype == np.uint16: 
            depth_im = depth.copy()
        self.slam_.process_rgbd(to_gray(left), depth_im)
        # return self.latest_estimate
    
