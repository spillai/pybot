'''
Lucas-Kanade tracker

@author: Sudeep Pillai (Last Edited: 07 May 2014)
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
'''

import numpy as np
import cv2, time, os.path, logging

from collections import namedtuple, deque
from bot_utils.db_utils import AttrDict

from bot_utils.plot_utils import colormap
from bot_vision.imshow_utils import imshow_cv
from bot_vision.image_utils import to_color, to_gray, gaussian_blur

from .manager import TrackManager

def draw_tracks(im, pts): 
    out = to_color(im)
    for pt in pts: 
        cv2.circle(out, tuple(map(int, pt)), 3, (0,255,0), -1, lineType=cv2.CV_AA)
    return out

def kpts_to_array(kpts): 
    return np.float32([ kp.pt for kp in kpts ]).reshape(-1,2)

class FeatureDetector(object): 
    """
    Feature Detector class that allows for fast switching between
    several popular feature detection methods. 

    Also, you can request for variable pyramid levels of detection, 
    and perform subpixel on the detected keypoints
    """

    default_detector_params = AttrDict(levels=4, subpixel=False)
    fast_detector_params = AttrDict(default_detector_params, 
                                    type='fast', params=AttrDict( threshold=10, nonmaxSuppression=True ))
    gftt_detector_params = AttrDict(default_detector_params, 
                                    type='gftt', params=AttrDict( maxCorners = 1000, qualityLevel = 0.04, 
                                                                  minDistance = 5, blockSize = 5 ))
    def __init__(self, params=default_detector_params): 
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
        else: 
            raise RuntimeError('Unknown detector_type: %s! Use fast or gftt' % self.params.type)

    def process(self, im, mask=None): 
        if im.ndim != 2: 
            raise RuntimeError('Cannot process color image')

        # Detect features 
        kpts = self.detector.detect(im, mask=mask)
        pts = np.float32([ kp.pt for kp in kpts ]).reshape(-1,2)
        
        # Perform sub-pixel if necessary
        if self.params.subpixel: self.perform_subpixel(im, pts)

        return pts

    def subpixel_pts(self, im, pts): 
        """Perform subpixel refinement"""
        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv2.cornerSubPix(im, pts, (10, 10), (-1, -1), term)
        return

def finite_and_within_bounds(xys, shape): 
    H, W = shape[:2]
    return np.bitwise_and(np.isfinite(xys).all(axis=1), 
                          reduce(lambda x,y: np.bitwise_and(x,y), [xys[:,0] >= 0, xys[:,0] < W, 
                                                                   xys[:,1] >= 0, xys[:,1] < H]))
    

class OpticalFlowTracker(object): 
    """
    General-purpose optical flow tracker class that allows for fast switching between
    sparse/dense tracking. 

    Also, you can request for variable pyramid levels of tracking, 
    and perform subpixel on the tracked keypoints
    """

    default_flow_params = AttrDict(levels=4, fb_check=True)
    klt_flow_params = AttrDict( default_flow_params, type='lk', 
                                params=AttrDict(winSize=(5,5), maxLevel=default_flow_params.levels ))
    farneback_flow_params = AttrDict( default_flow_params, type='dense', 
                                      params=AttrDict( pyr_scale=0.5, levels=default_flow_params.levels, winsize=15, 
                                                       iterations=3, poly_n=7, poly_sigma=1.5, flags=0 ))

    def __init__(self, params=klt_flow_params): 
        # FeatureDetector params
        self.params = params
        
        if self.params.type == 'lk': 
            self.track = self.sparse_track
        elif self.params.type == 'dense': 
            self.track = self.dense_track
        else: 
            raise RuntimeError('Unknown tracking type: %s! Use lk or dense' % self.params.type)

    def dense_track(self, im0, im1, p0): 
        fflow = cv2.calcOpticalFlowFarneback(im0, im1, **self.params.params)
        fflow = cv2.medianBlur(fflow, 5)

        # Initialize forward flow and propagated points
        p1 = np.ones(shape=p0.shape) * np.nan
        flow_p0 = np.ones(shape=p0.shape) * np.nan
        flow_good = np.ones(shape=p0.shape, dtype=bool)

        # Check finite value for pts, and within image bounds
        valid0 = finite_and_within_bounds(p0, im0.shape)

        # Determine finite flow at points
        xys0 = p0[valid0].astype(int)
        flow_p0[valid0] = fflow[xys0[:,1], xys0[:,0]]
        
        # Propagate
        p1 = p0 + flow_p0

        # FWD-BWD check
        if self.params.fb_check: 
            # Initialize reverse flow and propagated points
            p0r = np.ones(shape=p0.shape) * np.nan
            flow_p1 = np.ones(shape=p0.shape) * np.nan

            rflow = cv2.calcOpticalFlowFarneback(im1, im0, **self.params.params)
            rflow = cv2.medianBlur(rflow, 5)

            # Check finite value for pts, and within image bounds
            valid1 = finite_and_within_bounds(p1, im0.shape)

            # Determine finite flow at points
            xys1 = p1[valid1].astype(int)
            flow_p1[valid1] = rflow[xys1[:,1], xys1[:,0]]
            
            # Check diff
            p0r = p1 + flow_p1
            fb_good = (np.fabs(p0r-p0) < 2).all(axis=1)

            # Set only good flow 
            flow_p0[~fb_good] = np.nan
            p1 = p0 + flow_p0

        return p1

    def sparse_track(self, im0, im1, p0): 
        """
        Main tracking method using either sparse/dense optical flow
        """

        # Forward flow
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **self.params.params)
        p1[st1 == 0] = np.nan

        if self.params.fb_check: 
            # Backward flow
            p0r, st0, err0 = cv2.calcOpticalFlowPyrLK(im1, im0, p1, None, **self.params.params)
            p0r[st0 == 0] = np.nan

            # Set only good
            fb_good = (np.fabs(p0r-p0) < 2).all(axis=1)
            p1[~fb_good] = np.nan

        return p1

class BaseKLT(object): 
    """
    General-purpose KLT tracker that combines the use of FeatureDetector and 
    OpticalFlowTracker class. 
    """

    default_params = AttrDict(
        detector=FeatureDetector.fast_detector_params, 
        tracker=OpticalFlowTracker.klt_flow_params
    )
    def __init__(self, params=default_params): 

        self.log = logging.getLogger(self.__class__.__name__)

        # BaseKLT Params
        self.params = params

        # Setup Detector
        self.detector = FeatureDetector(params=self.params.detector)

        # Setup Tracker
        self.tracker = OpticalFlowTracker(params=self.params.tracker)

        # Track Manager
        self.tm = TrackManager(maxlen=10)
        
    # Pre process with graying, and gaussian blur
    def preprocess_im(self, im): 
        return cv2.GaussianBlur(to_gray(im), (3, 3), 0)        

    def draw_tracks(self, im): 
        for tid, pts in self.tm.tracks.iteritems(): 
            cv2.polylines(im,[np.vstack(pts).astype(np.int32)], False, 
                              tuple(map(int, colormap(tid % 20 / 20.0).ravel())), 
                              thickness=1, lineType=cv2.CV_AA)


# class Track(object): 
#     def __init__(self, shape, ids, pts): 
#         self.ids = np.arange(len(pts))
#         self.pts = pts
        
#     @classmethod
#     def from_track(cls, shape, track): 
#         valid = finite_and_within_bounds(track.pts, shape)
#         self.ids = track.ids[valid]
#         self.pts = track.pts[valid]

class OpenCVKLT(BaseKLT): 
    """
    KLT Tracker as implemented in OpenCV 2.4.9
    Stripped from opencv/samples/python2/lk_track.py
    """

    def __init__(self, *args, **kwargs):
        BaseKLT.__init__(self, *args, **kwargs)

        # StandardKLT Params
        self.im, self.prev_im = None, None
        self.pts, self.prev_pts = None, None
        self.add_features = True

    def reset(self): 
        self.add_features = True

    def create_mask(self, shape, pts): 
        mask = np.ones(shape=shape, dtype=np.uint8) * 255
        if pts is None: 
            return mask
            
        for pt in pts: 
            cv2.circle(mask, tuple(map(int, pt)), 9, 0, -1, lineType=cv2.CV_AA)

        return mask

    def process(self, im):
        # Preprocess
        self.im = gaussian_blur(im)

        # Track object
        pids, ppts = self.tm.ids, self.tm.pts
        if ppts is not None and len(ppts) and self.prev_im is not None: 
            pts = self.tracker.track(self.prev_im, self.im, ppts)
            self.tm.add(pts, ids=pids)

        # Check if more features required
        self.add_features = self.add_features or (ppts is not None and len(ppts) < 100)
        
        # Initialize or add more features
        if self.add_features: 
            # Extract features
            mask = self.create_mask(im.shape, self.tm.pts)            
            imshow_cv('mask', mask)

            pts = self.detector.process(self.im, mask=mask)
            self.tm.add(pts, ids=None)
            self.add_features = True


        self.prev_im = self.im.copy()
        # self.prev_pts = self.pts.copy()

    def viz(self, out): 
        valid = finite_and_within_bounds(self.tm.pts, out.shape)
        colors = colormap(np.float32(self.tm.ids % 20) / 20)
        for col, pt in zip(colors, self.tm.pts[valid]): 
            cv2.circle(out, tuple(map(int, pt)), 3, col, -1, lineType=cv2.CV_AA)

