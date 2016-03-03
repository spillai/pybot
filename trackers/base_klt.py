'''
Lucas-Kanade tracker

Author: Sudeep Pillai 

    07 May 2014: First implementation
    17 Oct 2014: Minor modifications to tracker params
    28 Feb 2016: Added better support for feature addition, and re-factored
                 trackmanager with simpler/improved indexing and queuing.

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
from bot_vision.draw_utils import draw_features
from .tracker_utils import finite_and_within_bounds, \
    TrackManager, FeatureDetector, OpticalFlowTracker

def kpts_to_array(kpts): 
    return np.float32([ kp.pt for kp in kpts ]).reshape(-1,2)

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
        self.tm = TrackManager(maxlen=20)

    def draw_tracks(self, out, colored=False, color_type='unique'):
        """
        color_type: {age, unique}
        """

        N = 20
        
        if color_type == 'age': 
            cwheel = colormap(np.linspace(0, 1, N))
            cols = np.vstack([cwheel[tid % N] for idx, (tid, pts) in enumerate(self.tm.tracks.iteritems())])            
        elif color_type == 'unique': 
            cols = colormap(np.int32([pts.length for pts in self.tm.tracks.itervalues()]))            
        else: 
            raise ValueError('Color type {:} undefined, use age or unique'.format(color_type))

        for col, pts in zip(cols, self.tm.tracks.values()): 
            cv2.polylines(out, [np.vstack(pts.items).astype(np.int32)], False, 
                          tuple(map(int, col)) if colored else (0,255,0), thickness=1)

    def viz(self, out, colored=False): 
        if not len(self.tm.pts): 
            return

        N = 20
        cols = colormap(np.linspace(0, 1, N))
        valid = finite_and_within_bounds(self.tm.pts, out.shape)
        for tid, pt in zip(self.tm.ids[valid], self.tm.pts[valid]): 
            cv2.circle(out, tuple(map(int, pt)), 2, 
                       tuple(map(int, cols[tid % N])) if colored else (0,240,0),
                       -1, lineType=cv2.CV_AA)

    def matches(self): 
        p1, p2 = [], []

        for tid, idx in zip(self.tm.tracks.itervalues(), self.tm.tracks_ts.itervalues()): 
            if val < self.idx: 
                del self.tracks[tid]
                del self.tracks_ts[tid]


        for tid, pts in self.tm.tracks.iteritems(): 
            if len(pts) > 2: 
                p1.append(pts[-2,:])
                p2.append(pts[-1,:])


class OpenCVKLT(BaseKLT): 
    """
    KLT Tracker as implemented in OpenCV 2.4.9
    Stripped from opencv/samples/python2/lk_track.py
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

    def process(self, im, detected_pts=None):
        # Preprocess
        self.ims.append(gaussian_blur(im))

        # Track object
        pids, ppts = self.tm.ids, self.tm.pts
        if ppts is not None and len(ppts) and len(self.ims) == 2: 
            pts = self.tracker.track(self.ims[-2], self.ims[-1], ppts)
            self.tm.add(pts, ids=pids, prune=True)

        # Check if more features required
        self.add_features = self.add_features or ppts is None or (ppts is not None and len(ppts) < 400)
        
        # Initialize or add more features
        if self.add_features: 
            # Extract features
            mask = self.create_mask(im.shape, ppts)            
            imshow_cv('mask', mask)

            if detected_pts is None: 
                new_pts = self.detector.process(self.ims[-1], mask=mask)
            else: 
                xy = detected_pts.astype(np.int32)
                valid = mask[xy[:,1], xy[:,0]] > 0
                new_pts = detected_pts[valid]

            self.tm.add(new_pts, ids=None, prune=False)
            self.add_features = False
