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

from collections import namedtuple
from bot_utils.db_utils import AttrDict

class BaseKLT(object): 
    default_params = AttrDict(
        detector_type='fast'
        gftt=AttrDict( maxCorners = 1000, qualityLevel = 0.1, minDistance = 5, blockSize = 5 ), 
        lk=AttrDict( winSize=(5,5), maxLevel=4 ), 
        detector=AttrDict( maxLevel=4 ), 
        farneback=AttrDict( pyr_scale=0.5, levels=5, winsize=15, iterations=3, poly_n=7, poly_sigma=1.5, flags=0 ),
        fast=AttrDict( threshold=10, nonmaxSuppression=True ), 
    )
    def __init__(self, params=default_params): 

        self.log = logging.getLogger(self.__class__.__name__)

        # BaseKLT Params
        self.params = params

        # Feature Detector, Descriptor setup
        if params.detector_type == 'gftt': 
            self.detector = cv2.PyramidAdaptedFeatureDetector(
                detector=cv2.GFTTDetector(**self.params.gftt),  
                **self.params.detector
            )
        elif params.detector_type == 'fast': 
            self.detector = cv2.PyramidAdaptedFeatureDetector(
                detector=cv2.FastFeatureDetector(**self.params.fast),  
                **self.params.detector
            )
        else: 
            raise RuntimeError('Unknown detector_type: %s! Use fast or gftt' % params.detector_type)
        

    # Pre process with graying, and gaussian blur
    def preprocess_im(self, im): 
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)        

    # Sub-pixel refinement
    def postprocess_pts(self, im, pts): 
        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv2.cornerSubPix(im, pts, (10, 10), (-1, -1), term)
        return

    # Process: detect features, and refine
    def process_im(self, im, mask=None): 
        # Detect features 
        kpts = self.detector.detect(im, mask=None)
        pts = np.float32([ kp.pt for kp in kpts ]).reshape(-1,2)

        # Refine points
        # self.postprocess_pts(im, pts)
        return pts



class StandardKLT(BaseKLT): 
    # KLT params
    klt_params_ = AttrDict( err_th = 1.0, max_corners=1000 )

    def __init__(self, klt_params=klt_params_, **kwargs):
        BaseKLT.__init__(self, **kwargs)

        # StandardKLT Params
        self.params.klt = klt_params
        self.im, self.prev_im = None, None
        self.pts, self.prev_pts = None, None

        self.b_track = False
        self.b_init = True

    def reset(self): 
        self.b_track = False
        self.b_init = False

    def track(self, im0, im1, p0): 
        # Forward flow
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **self.params.lk)

        # Invalidate pts without good flow
        inds1 = np.where(st1 == 1)[0]
        return p1[inds1]

    def process_im(self, im):
        # Preprocess
        self.im = self.preprocess_im(im)

        # Track object
        if self.b_track and self.prev_im is not None: 
            self.pts = self.track(self.prev_im, self.im, self.prev_pts)

            # Stop tracking
            if not len(self.pts): 
                self.b_track = False

        # Store ref. frame
        if self.b_init: 
            # Extract features
            self.pts = BaseKLT.process_im(self, self.im)
            self.b_init = False
            self.b_track = True

        self.prev_im = self.im.copy()
        self.prev_pts = self.pts.copy()

        # Viz
        self.viz(im)
        
    def viz(self, im): 
        out = im.copy()
        for pt in self.pts: 
            cv2.circle(out, tuple(map(int, pt)), 3, (0,255,0), -1, lineType=cv2.LINE_AA)
        cv2.imshow('out', out)
        cv2.waitKey(100)

