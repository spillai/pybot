import numpy as np
import cv2, time, os.path, logging
from copy import deepcopy

from collections import defaultdict, deque
from bot_utils.db_utils import AttrDict
from bot_utils.timer import SimpleTimer

from bot_vision.feature_detection import FeatureDetector
from bot_vision.feature_detection import to_kpt, to_kpts, to_pts, \
    finite_and_within_bounds

class IndexedDeque(object): 
    def __init__(self, maxlen=100): 
        self.items_ = deque(maxlen=maxlen)
        self.indices_ = deque(maxlen=maxlen)
        self.length_ = 0

    def append(self, index, item): 
        self.indices_.append(index)
        self.items_.append(item)
        self.length_ += 1

    def item(self, index): 
        return self.items_[index]

    def index(self, index): 
        return self.indices_[index]

    def __len__(self): 
        """ Returns the length of the deque """
        return len(self.items_)

    @property
    def latest_item(self):
        return self.items_[-1]

    @property
    def latest_index(self): 
        return self.indices_[-1]

    @property
    def items(self): 
        return self.items_

    @property
    def length(self): 
        """ 
        Returns the length of the track
        including deleted/popped items
        from the queue
        """
        return self.length_

class TrackManager(object): 
    def __init__(self, maxlen=20, on_delete_cb=lambda tracks: None): 
        # Max track length 
        self.maxlen_ = maxlen

        # Register callbacks on track delete
        self.on_delete_cb_ = on_delete_cb

        # Reset tracks before start
        self.reset()

    def reset(self): 
        self.index_ = 0

        # Tracks are a dictionary with {key: value, ... } as  
        # {track_id : , IndexedDeque [(time_index, feature), ... ]
        self.tracks_ = defaultdict(lambda: IndexedDeque(maxlen=self.maxlen_))

    def add(self, pts, ids=None, prune=True): 
        # Add only if valid and non-zero
        if not len(pts): 
            return

        # Retain valid points
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]

        # ID valid points
        max_id = np.max(self.ids) + 1 if len(self.ids) else 0
        tids = np.arange(len(pts), dtype=np.int64) + max_id if ids is None else ids[valid].astype(np.int64)
        
        # Add pts to track
        for tid, pt in zip(tids, pts): 
            self.tracks_[tid].append(self.index_, pt)

        # If features are propagated
        if prune: 
            self.prune()

        # Frame counter
        self.index_ += 1

    def prune(self): 
        # Remove tracks that are not most recent
        deleted_tracks = {}
        for tid, track in self.tracks_.items(): 
            if track.latest_index < self.index_: 
                deleted_tracks[tid] = deepcopy(self.tracks[tid])
                del self.tracks[tid]

        self.on_delete_cb_(deleted_tracks)
                
    def register_on_track_delete_callback(self, cb): 
        print('{:}: Register callback for track deletion {:}'
              .format(self.__class__.__name__, cb))
        self.on_delete_cb_ = cb

    @property
    def tracks(self): 
        return self.tracks_

    @property
    def pts(self): 
        try: 
            return np.vstack([ track.latest_item for track in self.tracks_.itervalues() ])
        except: 
            return np.array([])
        
    @property
    def ids(self): 
        return np.array(self.tracks_.keys())

    @property
    def index(self): 
        return self.index_

class OpticalFlowTracker(object): 
    """
    General-purpose optical flow tracker class that allows for fast switching between
    sparse/dense tracking. 

    Also, you can request for variable pyramid levels of tracking, 
    and perform subpixel on the tracked keypoints
    """

    lk_params = AttrDict(winSize=(5,5), maxLevel=4)
    farneback_params = AttrDict(pyr_scale=0.5, levels=3, winsize=15, 
                                iterations=3, poly_n=7, poly_sigma=1.5, flags=0)

    def __init__(self, fb_check=True): 
        self.fb_check_ = fb_check
        self.timer_ = SimpleTimer(name='optical-flow')

    @staticmethod
    def create(method='lk', fb_check=True, params=lk_params): 
        trackers = { 'lk': LKTracker, 'dense': FarnebackTracker }
        try: 
            # Determine tracker type that implements track
            tracker = trackers[method](**params)
        except: 
            raise RuntimeError('Unknown detector type: %s! Use from {:}'.format(FeatureDetector.detectors.keys()))
        return tracker

    def track(self, im0, im1, p0):
        raise NotImplementedError()

class LKTracker(OpticalFlowTracker): 
    """
    OpenCV's LK Tracker (with modifications for forward backward flow check)
    """

    default_params = OpticalFlowTracker.lk_params
    def __init__(self, fb_check=True, winSize=(5,5), maxLevel=4):
        OpticalFlowTracker.__init__(self, fb_check=fb_check)
        self.lk_params_ = AttrDict(winSize=winSize, maxLevel=maxLevel)

    def track(self, im0, im1, p0): 
        """
        Main tracking method using sparse optical flow (LK)
        """
        self.timer_.start()
        if p0 is None or not len(p0): 
            return np.array([])

        # Forward flow
        p1, st1, err1 = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **self.lk_params_)
        p1[st1 == 0] = np.nan

        if self.fb_check_: 
            # Backward flow
            p0r, st0, err0 = cv2.calcOpticalFlowPyrLK(im1, im0, p1, None, **self.lk_params_)
            p0r[st0 == 0] = np.nan
            
            # Set only good
            fb_good = (np.fabs(p0r-p0) < 2).all(axis=1)
            p1[~fb_good] = np.nan

        self.timer_.stop()
        return p1

class FarnebackTracker(OpticalFlowTracker): 
    """
    OpenCV's Dense farneback Tracker (with modifications for forward backward flow check)
    """

    default_params = OpticalFlowTracker.farneback_params
    def __init__(self, fb_check=True, pyr_scale=0.5, levels=3, winsize=15, 
                 iterations=3, poly_n=7, poly_sigma=1.5, flags=0):
        OpticalFlowTracker.__init__(self, fb_check=fb_check)
        self.farneback_params_ = AttrDict(pyr_scale=pyr_scale, levels=levels, winsize=winsize, 
                                          iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)

    def track(self, im0, im1, p0): 
        self.timer_.start()

        if p0 is None or not len(p0): 
            return np.array([])

        fflow = cv2.calcOpticalFlowFarneback(im0, im1, **self.farneback_params_)
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
        if self.fb_check_: 
            # Initialize reverse flow and propagated points
            p0r = np.ones(shape=p0.shape) * np.nan
            flow_p1 = np.ones(shape=p0.shape) * np.nan

            rflow = cv2.calcOpticalFlowFarneback(im1, im0, **self.farneback_params_)
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

        self.timer_.stop()

        return p1
