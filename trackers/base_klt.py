'''
Lucas-Kanade tracker

Author: Sudeep Pillai 

    07 May 2014: First implementation
    17 Oct 2014: Minor modifications to tracker params
    28 Feb 2016: Added better support for feature addition, and re-factored
                 trackmanager with simpler/improved indexing and queuing.
    20 Mar 2016: Refactored detector/tracker initialization with params 
    21 Mar 2016: First pass at MeshKLT

====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
'''

import cv2
import numpy as np

from itertools import izip
from collections import namedtuple, deque
from bot_utils.db_utils import AttrDict

from bot_utils.plot_utils import colormap
from bot_vision.imshow_utils import imshow_cv
from bot_vision.image_utils import to_color, to_gray, gaussian_blur
from bot_vision.draw_utils import draw_features

from bot_vision.trackers import FeatureDetector, OpticalFlowTracker, LKTracker
from bot_vision.trackers import finite_and_within_bounds, to_pts, \
    TrackManager, FeatureDetector, OpticalFlowTracker, LKTracker

import time

class BaseKLT(object): 
    """
    General-purpose KLT tracker that combines the use of FeatureDetector and 
    OpticalFlowTracker class. 
    """
    default_detector_params = AttrDict(method='fast', grid=(12,10), max_corners=1200, 
                                       max_levels=4, subpixel=False, params=FeatureDetector.fast_params)
    default_tracker_params = AttrDict(method='lk', fb_check=True, 
                                      params=OpticalFlowTracker.lk_params)

    default_detector = FeatureDetector(**default_detector_params)
    default_tracker = OpticalFlowTracker.create(**default_tracker_params)

    def __init__(self, 
                 detector=default_detector, 
                 tracker=default_tracker,  
                 max_track_length=4, min_tracks=1200, mask_size=9): 

        # BaseKLT Params
        self.detector_ = detector
        self.tracker_ = tracker

        # Track Manager
        self.tm_ = TrackManager(maxlen=max_track_length)
        self.min_tracks_ = min_tracks
        self.mask_size_ = mask_size

    @classmethod
    def from_params(cls, detector_params=default_detector_params, 
                    tracker_params=default_tracker_params,
                    max_track_length=4, min_tracks=1200, mask_size=9): 

        # Setup detector and tracker
        detector = FeatureDetector(**detector_params)
        tracker = OpticalFlowTracker.create(**tracker_params)
        return cls(detector, tracker, 
                   max_track_length=max_track_length, min_tracks=min_tracks, mask_size=mask_size)

    def register_on_track_delete_callback(self, cb): 
        self.tm_.register_on_track_delete_callback(cb)

    def create_mask(self, shape, pts): 
        mask = np.ones(shape=shape, dtype=np.uint8) * 255
        try: 
            for pt in pts: 
                cv2.circle(mask, tuple(map(int, pt)), self.mask_size_, 0, -1, lineType=cv2.CV_AA)
        except:
            pass
        return mask

    def draw_tracks(self, out, colored=False, color_type='unique', max_track_length=4):
        """
        color_type: {age, unique}
        """

        N = 20
        
        if color_type == 'unique': 
            cwheel = colormap(np.linspace(0, 1, N))
            cols = np.vstack([cwheel[tid % N] for idx, (tid, pts) in enumerate(self.tm_.tracks.iteritems())])            
        elif color_type == 'age': 
            cols = colormap(np.int32([pts.length for pts in self.tm_.tracks.itervalues()]))            
        else: 
            raise ValueError('Color type {:} undefined, use age or unique'.format(color_type))

        if not colored: 
            cols = np.tile([0,240,0], [len(self.tm_.tracks), 1])

        for col, pts in izip(cols.astype(np.int64), self.tm_.tracks.itervalues()): 
            cv2.polylines(out, [np.vstack(pts.items).astype(np.int32)[-max_track_length:]], False, 
                          tuple(col), thickness=1)
            tl, br = pts.latest_item-2, pts.latest_item+2
            cv2.rectangle(out, (tl[0], tl[1]), (br[0], br[1]), (0,255,0), -1)

    def viz(self, out, colored=False): 
        if not len(self.tm_.pts): 
            return

        N = 20
        cols = colormap(np.linspace(0, 1, N))
        valid = finite_and_within_bounds(self.tm_.pts, out.shape)

        for tid, pt in izip(self.tm_.ids[valid], self.tm_.pts[valid]): 
            cv2.rectangle(out, tuple(map(int, pt-2)), tuple(map(int, pt+2)), 
                          tuple(map(int, cols[tid % N])) if colored else (0,240,0), -1)

    def matches(self, index1=-2, index2=-1): 
        tids, p1, p2 = [], [], []
        for tid, pts in self.tm_.tracks.iteritems(): 
            if len(pts) > abs(index1) and len(pts) > abs(index2): 
                tids.append(tid)
                p1.append(pts.items[index1])
                p2.append(pts.items[index2]) 
        try: 
            return tids, np.vstack(p1), np.vstack(p2)
        except: 
            return np.array([]), np.array([]), np.array([])

    def process(self, im, detected_pts=None):
        raise NotImplementedError()

    @property
    def latest_ids(self): 
        return self.tm_.ids

    @property
    def latest_pts(self): 
        return self.tm_.pts

class OpenCVKLT(BaseKLT): 
    """
    KLT Tracker as implemented in OpenCV 2.4.9
    Stripped from opencv/samples/python2/lk_track.py
    """
    def __init__(self, *args, **kwargs):
        BaseKLT.__init__(self, *args, **kwargs)

        # OpenCV KLT
        self.ims_ = deque(maxlen=2)
        self.add_features_ = True

    def reset(self): 
        self.add_features_ = True

    def process(self, im, detected_pts=None):
        # Preprocess
        self.ims_.append(gaussian_blur(to_gray(im)))

        # Track object
        pids, ppts = self.tm_.ids, self.tm_.pts
        if ppts is not None and len(ppts) and len(self.ims_) == 2: 
            pts = self.tracker_.track(self.ims_[-2], self.ims_[-1], ppts)

            # Check bounds
            valid = finite_and_within_bounds(pts, im.shape[:2])
            
            # Add pts and prune afterwards
            self.tm_.add(pts[valid], ids=pids[valid], prune=True)

        # Check if more features required
        self.add_features_ = self.add_features_ or ppts is None or (ppts is not None and len(ppts) < self.min_tracks_)
        
        # Initialize or add more features
        if self.add_features_: 
            # Extract features
            mask = self.create_mask(im.shape, ppts)            
            # imshow_cv('mask', mask)

            if detected_pts is None: 

                # Detect features
                new_kpts = self.detector_.process(self.ims_[-1], mask=mask, return_keypoints=True)
                newlen = max(0, self.min_tracks_ - len(ppts))
                new_pts = to_pts(sorted(new_kpts, key=lambda kpt: kpt.response, reverse=True)[:newlen])
            else: 
                xy = detected_pts.astype(np.int32)
                valid = mask[xy[:,1], xy[:,0]] > 0
                new_pts = detected_pts[valid]

            # Add detected features with new ids, and prevent pruning 
            self.tm_.add(new_pts, ids=None, prune=False)
            self.add_features_ = False

        return self.tm_.ids, self.tm_.pts

class MeshKLT(OpenCVKLT): 
    """
    KLT Tracker as implemented in OpenCV 2.4.9
    with basic meshing support 
    """
    def __init__(self, *args, **kwargs):
        OpenCVKLT.__init__(self, *args, **kwargs)

        from pybot_vision import DelaunayTriangulation
        from bot_utils.timer import SimpleTimer

        self.dt_ = DelaunayTriangulation()
        self.timer_ = SimpleTimer('MeshKLT')

    def process(self, im, detected_pts=None): 
        self.timer_.start()
        ids, pts = OpenCVKLT.process(self, im, detected_pts=detected_pts)
        if len(pts) > 3: 
            self.dt_.batch_triangulate(pts)
        self.timer_.stop()

        vis = to_color(im)
        dt_vis = self.dt_.visualize(vis, pts)
        # OpenCVKLT.viz(self, dt_vis, colored=True)
        OpenCVKLT.draw_tracks(self, vis, colored=True, color_type='unique', max_track_length=2)
        imshow_cv('dt_vis', np.vstack([vis, dt_vis]), wait=1)

        return ids, pts
