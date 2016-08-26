"""
Visual odometry tools
"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
from pybot.vision.camera_utils import Camera, StereoCamera, RGBDCamera, CameraExtrinsic, CameraIntrinsic
from pybot.geometry.rigid_transform import Pose, RigidTransform
from pybot_externals import StereoELAS, StereoVISO2, MonoVISO2, ORBSLAM2, FOVIS

class VisualOdometry(object): 
    def __init__(self, camera, alg='viso2'): 
        """
        Setup visual odometry using stereo data
        """
        if alg == 'viso2':        
            if not isinstance(camera, StereoCamera): 
                raise TypeError('For viso2, Camera provided is not StereoCamera, but {:}'.format(type(camera)))

            self.vo_ = StereoVISO2(f=camera.left.fx, 
                                   cx=camera.left.cx, cy=camera.left.cy, 
                                   baseline=camera.baseline, relative=False, project_2d=False) 
            print 'Baseline: ', camera.baseline
        elif alg == 'fovis': 
            if not isinstance(camera, RGBDCamera): 
                raise TypeError('For viso2, Camera provided is not RGBDCamera, but {:}'.format(type(camera)))
            
            self.vo_ = FOVIS(rgbK=camera.rgb.K, depthK=camera.depth.K, 
                             width=int(camera.rgb.shape[1]), height=int(camera.rgb.shape[0]), 
                             shift_offset=1093.4753, projector_depth_baseline=camera.baseline)
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
        self.alg_ = alg

    def process(self, left_im, right_im):
        """
        Perform stereo VO
        """

        # Perform VO and update TF
        # try: 
        if self.alg_ == 'viso2': 
            T_ct = self.vo_.process(to_gray(left_im), to_gray(right_im))
        elif self.alg_ == 'fovis' :
            if left_im.ndim != 3 or right_im.dtype != np.float32: 
                raise RuntimeError('FOVIS requires RGBD format')
            if right_im.dtype == np.float32: 
                depth_im = (right_im * 1000).astype(np.uint16)
            elif right_im.dtype == np.uint16: 
                depth_im = right_im.copy()
            else: 
                raise RuntimeError()

            T_ct = self.vo_.process(left_im, depth_im)

        p_ct = RigidTransform.from_matrix(T_ct)
        # except Exception as e:
        #     print e
        #     return

        # pose_id = self.poses_[-1].id + 1 if len(self.poses_) else 0
        # pose_ct  = Pose.from_rigid_transform(pose_id, p_ct) 
        # return pose_ct
        return p_ct
