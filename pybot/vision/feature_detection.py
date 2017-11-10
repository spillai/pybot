# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2
import numpy as np
from pybot.utils.db_utils import AttrDict
from functools import reduce

def finite_and_within_bounds(xys, shape): 
    H, W = shape[:2]
    if not len(xys): 
        return np.array([])
    return np.bitwise_and(np.isfinite(xys).all(axis=1), 
                          reduce(lambda x,y: np.bitwise_and(x,y), [xys[:,0] >= 0, xys[:,0] < W, 
                                                                   xys[:,1] >= 0, xys[:,1] < H]))

def to_kpt(pt, size=1): 
    return cv2.KeyPoint(pt[0], pt[1], size)

def to_kpts(pts, size=1): 
    return [cv2.KeyPoint(pt[0], pt[1], size) for pt in pts]
    
def to_pts(kpts): 
    return np.float32([ kp.pt for kp in kpts ]).reshape(-1,2)

def kpts_to_array(kpts): 
    return np.float32([ kp.pt for kp in kpts ]).reshape(-1,2)

def get_dense_detector(step=4, levels=7, scale=np.sqrt(2)): 
    """
    Standalone dense detector instantiation
    """
    detector = cv2.FeatureDetector_create('Dense')
    detector.setInt('initXyStep', step)
    # detector.setDouble('initFeatureScale', 0.5)

    detector.setDouble('featureScaleMul', scale)
    detector.setInt('featureScaleLevels', levels)

    detector.setBool('varyImgBoundWithScale', True)
    detector.setBool('varyXyStepWithScale', False)

    # detector = cv2.PyramidAdaptedFeatureDetector(detector, maxLevel=4)
    return detector

def get_detector(detector='dense', step=4, levels=7, scale=np.sqrt(2)): 
    """ Get opencv dense-sampler or specific feature detector """
    if detector == 'dense': 
        return get_dense_detector(step=step, levels=levels, scale=scale)
    else: 
        detector = cv2.FeatureDetector_create(detector)
        return cv2.PyramidAdaptedFeatureDetector(detector, maxLevel=levels)

class AprilTagFeatureDetector(object): 
    """
    AprilTag Feature Detector (only detect 4 corner points)
    """
    default_params = AttrDict(tag_size=0.1, fx=576.09, fy=576.09, cx=319.5, cy=239.5)
    def __init__(self, tag_size=0.1, fx=576.09, fy=576.09, cx=319.5, cy=239.5): 
        try: 
            from pybot_apriltags import AprilTagsWrapper
            self.detector_ = AprilTagsWrapper(tag_size=tag_size, fx=fx, fy=fy, cx=cx, cy=cy)
        except: 
            raise ImportError('Apriltags (pybot_apriltags) is not available')

    def detect(self, im, mask=None): 
        tags = self.detector_.process(im, return_poses=False)
        kpts = []
        for tag in tags: 
            kpts.extend([cv2.KeyPoint(pt[0], pt[1], 1) for pt in tag.getFeatures()])
        return kpts

class SemiDenseFeatureDetector(object): 
    """
    Semi-Dense Feature Detector
    """
    default_params = AttrDict(step=10, threshold=20, max_corners=10000)
    def __init__(self, grid=(20,20), threshold=20, max_corners=10000): 
        try: 
            from pybot_vision import fast_proposals
            self.detector_ = cv2.GridAdaptedFeatureDetector(
                get_dense_detector(step=2, levels=1), max_corners, grid[0], grid[1])
            self.detect_edges_ = lambda im: fast_proposals(im, threshold)
        except: 
            raise ImportError('fast_proposals (pybot_vision) is not available')

    def detect(self, im, mask=None): 
        edges = self.detect_edges_(im)
        # edges1 = edges.copy()
        if mask is not None: 
            edges = np.bitwise_and(edges, mask)
        # cv2.imshow('mask', np.hstack([edges1, edges]))
        kpts = self.detector_.detect(im, mask=edges)
        return kpts


class FeatureDetector(object): 
    """
    Feature Detector class that allows for fast switching between
    several popular feature detection methods. 

    Also, you can request for variable pyramid levels of detection, 
    and perform subpixel on the detected keypoints
    """

    default_params = AttrDict(grid=(12,10), max_corners=1200, 
                              max_levels=4, subpixel=False)
    fast_params = AttrDict(threshold=10, nonmaxSuppression=True)
    gftt_params = AttrDict(maxCorners=800, qualityLevel=0.04, 
                           minDistance=5, blockSize=5)
    apriltag_params = AprilTagFeatureDetector.default_params

    detectors = { 'gftt': cv2.GFTTDetector, 
                  'fast': cv2.FastFeatureDetector, 
                  'apriltag': AprilTagFeatureDetector, 
                  'semi-dense': SemiDenseFeatureDetector }

    def __init__(self, method='fast', grid=(12,10), max_corners=1200, 
                 max_levels=4, subpixel=False, params=fast_params):

        # Determine detector type that implements detect
        try: 
            self.detector_ = FeatureDetector.detectors[method](**params)
        except Exception as e:
            raise RuntimeError('Unknown detector type: %s! Use from {:}, {:}'.format(list(FeatureDetector.detectors.keys()), e))

        # Only support grid and pyramid with gftt and fast
        if (method == 'gftt' or method == 'fast'): 
            # Establish pyramid
            if max_levels > 0: 
                self.detector_ = cv2.PyramidAdaptedFeatureDetector(
                    detector=self.detector_, maxLevel=max_levels)

            # Establish grid
            if (grid is not None and max_corners): 
                try: 
                    self.detector_ = cv2.GridAdaptedFeatureDetector(
                        self.detector_, max_corners, grid[0], grid[1])
                except: 
                    raise ValueError('FeatureDetector grid is not compatible {:}'.format(grid))

        # Check detector 
        self.check_detector()

        self.params_ = params
        self.max_levels_ = max_levels
        self.max_corners_ = max_corners
        self.grid_ = grid
        self.subpixel_ = subpixel

    @classmethod
    def from_params(cls, method='fast', grid=(12,10), max_corners=1200, 
                 max_levels=4, subpixel=False, params=fast_params):
        raise NotImplementedError()

    def check_detector(self): 
        # Check detector
        if not hasattr(self.detector_, 'detect'): 
            raise AttributeError('Detector does not implement detect method {:}'.format(dir(self.detector_)))

    @property
    def detector(self): 
        return self.detector_

    def process(self, im, mask=None, return_keypoints=False): 
        # Detect features 
        kpts = self.detector.detect(im, mask=mask)
        pts = to_pts(kpts)
        
        # Perform sub-pixel if necessary
        if self.subpixel_: self.subpixel_pts(im, pts)

        # Return keypoints, if necessary
        if return_keypoints: 
            return kpts
        
        return pts

    def subpixel_pts(self, im, pts): 
        """Perform subpixel refinement"""
        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv2.cornerSubPix(im, pts, (10, 10), (-1, -1), term)
        return
