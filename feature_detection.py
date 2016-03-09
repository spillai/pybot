import cv2
import numpy as np

# class BaseFeatureDetector(object): 
#     def __init__(self): 
#         pass

#     def detect(self, im, mask=None): 
#         raise NotImplementedError()

class AprilTagFeatureDetector(object): 
    """
    AprilTag Feature Detector (only detect 4 corner points)
    """
    default_params = dict(tag_size=0.1, fx=576.09, fy=576.09, cx=319.5, cy=239.5)
    def __init__(self, tag_size=0.1, fx=576.09, fy=576.09, cx=319.5, cy=239.5): 
        self.detector = AprilTagsWrapper()
        self.detector.set_calib(tag_size=tag_size, fx=fx, fy=fy, cx=cx, cy=cy)
    
    def detect(self, im, mask=None): 
        tags = self.detector.process(im, return_poses=False)
        kpts = []
        for tag in tags: 
            kpts.extend([cv2.KeyPoint(pt[0], pt[1], 1) for pt in tag.getFeatures()])
        return kpts

class FeatureDetector(object): 
    """
    Feature Detector class that allows for fast switching between
    several popular feature detection methods. 

    Also, you can request for variable pyramid levels of detection, 
    and perform subpixel on the detected keypoints
    """

    default_params = dict(levels=4, subpixel=False)

    fast_params = dict(default_params, type='fast', 
                                    params=dict( threshold=10, nonmaxSuppression=True ))

    gftt_params = dict(default_params, type='gftt', 
                       params=dict( maxCorners = 800, qualityLevel = 0.04, 
                                    minDistance = 5, blockSize = 5 ))

    apriltag_params = AttrDict(subpixel=False, type='apriltag', 
                               params=AprilTagFeatureDetector.default_params)

    def __init__(self, detector_type='fast', levels=4, subpixel=False)
        # FeatureDetector params
        self.params = params

        # Feature Detector, Descriptor setup
        if self.params.type == 'gftt': 
            self.detector = cv2.PyramidAdaptedFeatureDetector(
                detector=cv2.GFTTDetector(**self.params.params),  
                maxLevel=self.params.levels
            )
        elif self.params.type == 'fast': 
            
            self.detector = cv2.PyramidAdaptedFeatureDetector(
                detector=cv2.FastFeatureDetector(**self.params.params),  
                maxLevel=self.params.levels
            )
        elif self.params.type == 'apriltag': 
            self.detector = AprilTagFeatureDetector(**self.params.params)
        else: 
            raise RuntimeError('Unknown detector_type: %s! Use fast or gftt' % self.params.type)

    def process(self, im, mask=None, return_keypoints=False): 
        if im.ndim != 2: 
            raise RuntimeError('Cannot process color image')

        # Detect features 
        kpts = self.detector.detect(im, mask=mask)
        pts = to_pts(kpts)
        
        # Perform sub-pixel if necessary
        if self.params.subpixel: self.subpixel_pts(im, pts)

        # Return keypoints, if necessary
        if return_keypoints: 
            return kpts
        
        return pts

    def subpixel_pts(self, im, pts): 
        """Perform subpixel refinement"""
        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv2.cornerSubPix(im, pts, (10, 10), (-1, -1), term)
        return
