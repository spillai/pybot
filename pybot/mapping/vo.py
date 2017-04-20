"""
Visual odometry tools
"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
import cv2
from pybot.vision.camera_utils import Camera, StereoCamera, RGBDCamera, CameraExtrinsic, CameraIntrinsic
from pybot.geometry.rigid_transform import Pose, RigidTransform

class SimpleVO(object):
    def __init__(self, calib):
        self.calib_ = calib

    def process(self, pts1, pts2):
        focal = self.calib_.fx
        pp = (self.calib_.cx, self.calib_.cy)
        E, mask = cv2.findEssentialMat(pts2, pts1,
                                       focal=focal, pp=pp, 
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, focal=focal, pp=pp)
        print E, R, t


def test_vo():
    import argparse
    from pybot.vision.imshow_utils import imshow_cv
    from pybot.utils.test_utils import test_dataset
    from pybot.vision.feature_detection import FeatureDetector
    from pybot.vision.trackers.base_klt import OpenCVKLT

    # Setup detector params
    fast_params = FeatureDetector.fast_params
    fast_params.threshold = 20
    detector_params = dict(method='fast', grid=(8,5), max_corners=100, 
                               max_levels=1, subpixel=True, params=FeatureDetector.fast_params)

    # Setup tracker params (either lk, or dense)
    lk_params = dict(winSize=(5,5), maxLevel=3)
    tracker_params = dict(method='lk', fb_check=True, params=lk_params)

    # Create detector from params
    det = FeatureDetector(**detector_params)

    # Create KLT from detector params only
    klt = OpenCVKLT.from_params(detector_params=detector_params, 
                                tracker_params=tracker_params, 
                                min_tracks=100)
    
    dataset = test_dataset()
    lcam = dataset.calib.left

    vo = SimpleVO(lcam)
    for im in dataset.iteritems():
        imshow_cv('im', im)

        # Process image: KLT tracking
        klt.process(im)
        
        # Gather points, ids, flow and age
        ids, pts, age, flow = klt.latest_ids, klt.latest_pts, \
                              klt.latest_age, klt.latest_flow
        inds = age > 1

if __name__ == "__main__":
    test_vo()        
