'''
Lucas-Kanade tracker

Author: Sudeep Pillai 
  (First implementation: 07 May 2014)
  (Modified: 17 Oct 2014)
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

from .tracker_utils import finite_and_within_bounds, \
    TrackManager, FeatureDetector, OpticalFlowTracker

def draw_tracks(im, pts): 
    out = to_color(im)
    for pt in pts: 
        cv2.circle(out, tuple(map(int, pt)), 3, (0,255,0), -1, lineType=cv2.CV_AA)
    return out

def kpts_to_array(kpts): 
    return np.float32([ kp.pt for kp in kpts ]).reshape(-1,2)

class BaseKLT(object): 
    """
    General-purpose KLT tracker that combines the use of FeatureDetector and 
    OpticalFlowTracker class. 
    """

    default_params = AttrDict(
        detector=FeatureDetector.gftt_detector_params, 
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

    def draw_tracks(self, im): 
        for tid, pts in self.tm.tracks.iteritems(): 
            cv2.polylines(im,[np.vstack(pts).astype(np.int32)], False, 
                              tuple(map(int, colormap(tid % 20 / 20.0).ravel())), 
                              thickness=1, lineType=cv2.CV_AA)


    def viz(self, out): 
        if not len(self.tm.pts): 
            return

        valid = finite_and_within_bounds(self.tm.pts, out.shape)
        colors = colormap(np.float32(self.tm.ids % 20) / 20)
        for col, pt in zip(colors, self.tm.pts[valid]): 
            cv2.circle(out, tuple(map(int, pt)), 2, col, -1, lineType=cv2.CV_AA)




class OpenCVKLT(BaseKLT): 
    """
    KLT Tracker as implemented in OpenCV 2.4.9
\    Stripped from opencv/samples/python2/lk_track.py
    """
    def __init__(self, *args, **kwargs):
        BaseKLT.__init__(self, *args, **kwargs)

        # OpenCV KLT
        self.ims = deque(maxlen=2)
        self.add_features = True

    def reset(self): 
        self.add_features = True

    def create_mask(self, shape, pts): 
        mask = np.ones(shape=shape, dtype=np.uint8) * 255
        try: 
            for pt in pts: 
                cv2.circle(mask, tuple(map(int, pt)), 9, 0, -1, lineType=cv2.CV_AA)
        except:
            pass
        return mask

    def process(self, im):
        # Preprocess
        self.ims.append(gaussian_blur(im))

        # Track object
        pids, ppts = self.tm.ids, self.tm.pts
        if ppts is not None and len(ppts) and len(self.ims) == 2: 
            pts = self.tracker.track(self.ims[-2], self.ims[-1], ppts)
            self.tm.add(pts, ids=pids, prune=True)

        # Check if more features required
        self.add_features = self.add_features or (ppts is not None and len(ppts) < 100)

        # Initialize or add more features
        if self.add_features: 
            # Extract features
            mask = self.create_mask(im.shape, self.tm.pts)            
            # imshow_cv('mask', mask)

            pts = self.detector.process(self.ims[-1], mask=mask)
            self.tm.add(pts, ids=None, prune=False)
            self.add_features = True

