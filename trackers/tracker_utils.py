import numpy as np
import cv2, time, os.path, logging

from collections import defaultdict, deque
from bot_vision.color_utils import colormap
from bot_utils.db_utils import AttrDict

from pybot_apriltags import AprilTag, AprilTagsWrapper

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

class IndexedDeque(object): 
    def __init__(self, maxlen=100): 
        self.items_ = deque(maxlen=maxlen)
        self.indices_ = deque(maxlen=maxlen)
        self.length_ = 0

    def append(self, index, item): 
        self.indices_.append(index)
        self.items_.append(item)
        self.length_ += 1

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
        return self.length_

class TrackManager(object): 
    def __init__(self, maxlen=20): 
        self.maxlen_ = maxlen
        self.reset()

    def reset(self): 
        self.index_ = 0
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
        tids = np.arange(len(pts), dtype=np.int64) + max_id if ids is None else ids[valid]
        
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
        for tid, track in self.tracks_.items(): 
            if track.latest_index < self.index_: 
                del self.tracks[tid]

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

    def index(self): 
        return self.index_


# class BaseFeatureDetector(object): 
#     def __init__(self): 
#         pass

#     def detect(self, im, mask=None): 
#         raise NotImplementedError()

class AprilTagFeatureDetector(object): 
    """
    AprilTag Feature Detector (only detect 4 corner points)
    """
    default_detector_params = AttrDict(tag_size=0.1, fx=576.09, fy=576.09, cx=319.5, cy=239.5)
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

    default_detector_params = AttrDict(levels=4, subpixel=False)
    fast_detector_params = AttrDict(default_detector_params, type='fast', 
                                    params=AttrDict( threshold=10, nonmaxSuppression=True ))
    gftt_detector_params = AttrDict(default_detector_params, type='gftt', 
                                    params=AttrDict( maxCorners = 800, qualityLevel = 0.04, 
                                                                  minDistance = 5, blockSize = 5 ))
    apriltag_detector_params = AttrDict(subpixel=False, type='apriltag', 
                                        params=AprilTagFeatureDetector.default_detector_params)
    def __init__(self, params=fast_detector_params): 
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
        self.params = AttrDict(params)
        
        if self.params.type == 'lk': 
            self.track = self.sparse_track
        elif self.params.type == 'dense': 
            self.track = self.dense_track
        else: 
            raise RuntimeError('Unknown tracking type: %s! Use lk or dense' % self.params.type)
        from bot_utils.timer import SimpleTimer
        self.timer = SimpleTimer(name='optical-flow', iterations=100)

    def dense_track(self, im0, im1, p0): 
        self.timer.start()

        if p0 is None or not len(p0): 
            return np.array([])

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

        self.timer.stop()

        return p1

    def sparse_track(self, im0, im1, p0): 
        """
        Main tracking method using either sparse/dense optical flow
        """
        self.timer.start()
        if p0 is None or not len(p0): 
            return np.array([])

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

        self.timer.stop()
        return p1

