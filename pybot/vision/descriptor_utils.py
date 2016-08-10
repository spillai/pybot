import cv2
import numpy as np
from pybot.utils.db_utils import AttrDict

class FeatureDetector(object): 
    """
    Feature Detector class that allows for fast switching between
    several popular feature detection methods. 

    Also, you can request for variable pyramid levels of detection, 
    and perform subpixel on the detected keypoints
    """
    fast_detector_params = AttrDict( threshold=10, nonmaxSuppression=True )
    gftt_detector_params = AttrDict( maxCorners = 1000, qualityLevel = 0.04, 
                                     minDistance = 5, blockSize = 5 )

    default_params = AttrDict(
        detector=AttrDict(name='fast', params=fast_detector_params), 
        descriptor='SIFT', levels=4, subpixel=False, )

    def _build_detector(self, name='fast', params=fast_detector_params): 
        if name == 'gftt': 
            return cv2.GFTTDetector(**params)
        elif name == 'fast': 
            return cv2.FastFeatureDetector(**params)
        else: 
            raise RuntimeError('Unknown detector_type: %s! Use fast or gftt' % name)
        
    def __init__(self, params=default_params): 
        # FeatureDetector params
        self.params = params

        # Feature Detector, Descriptor setup
        self.detector = cv2.PyramidAdaptedFeatureDetector(
            detector=self._build_detector(**self.params.detector), 
            maxLevel=self.params.levels
        )

    def process(self, im, mask=None): 
        if im.ndim != 2: 
            raise RuntimeError('Cannot process color image')

        # Detect features 
        kpts = self.detector.detect(im, mask=mask)
        pts = np.float32([ kp.pt for kp in kpts ]).reshape(-1,2)
        
        # Perform sub-pixel if necessary
        if self.params.subpixel: self.subpixel_pts(im, pts)

        return pts

    def subpixel_pts(self, im, pts): 
        """Perform subpixel refinement"""
        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv2.cornerSubPix(im, pts, (10, 10), (-1, -1), term)
        return


def im_desc(im, pts=None, within_polygon=False): 
    return desc

