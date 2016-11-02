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

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import cv2
import numpy as np

from itertools import izip
from collections import namedtuple, deque

from pybot.utils.db_utils import AttrDict
from pybot.utils.timer import timeitmethod
from pybot.utils.plot_utils import colormap

from pybot.vision.imshow_utils import imshow_cv
from pybot.vision.image_utils import to_color, to_gray, gaussian_blur
from pybot.vision.draw_utils import draw_features, draw_lines


from pybot.vision.trackers import FeatureDetector, OpticalFlowTracker, LKTracker
from pybot.vision.trackers import finite_and_within_bounds, to_pts, \
    TrackManager, FeatureDetector, OpticalFlowTracker, LKTracker

class BaseKLT(object): 
    """
    General-purpose KLT tracker that combines the use of FeatureDetector and 
    OpticalFlowTracker class. 

        min_track_length:   Defines the minimum length of the track that returned. 
                            Other shorter tracks are still considered while tracking
        
        max_track_length:   Maximum deque length for track history lookup

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
                 min_track_length=2, max_track_length=4, min_tracks=1200, mask_size=9): 
        
        # BaseKLT Params
        self.detector_ = detector
        self.tracker_ = tracker

        # Track Manager
        self.tm_ = TrackManager(maxlen=max_track_length)
        self.min_track_length_ = min_track_length
        self.min_tracks_ = min_tracks
        self.mask_size_ = mask_size

    @classmethod
    def from_params(cls, detector_params=default_detector_params, 
                    tracker_params=default_tracker_params,
                    min_track_length=2, max_track_length=4, min_tracks=1200, mask_size=9): 

        # Setup detector and tracker
        detector = FeatureDetector(**detector_params)
        tracker = OpticalFlowTracker.create(**tracker_params)
        return cls(detector, tracker, 
                   min_track_length=min_track_length, max_track_length=max_track_length, 
                   min_tracks=min_tracks, mask_size=mask_size)

    def register_on_track_delete_callback(self, cb): 
        self.tm_.register_on_track_delete_callback(cb)

    def create_mask(self, shape, pts): 
        """
        Create a mask image to prevent feature extraction around regions
        that already have features detected. i.e prevent feature crowding
        """

        mask = np.ones(shape=shape, dtype=np.uint8) * 255
        all_pts = np.vstack([self.aug_pts_, pts]) if hasattr(self, 'aug_pts_') \
                  else pts
        try: 
            for pt in all_pts: 
                cv2.circle(mask, tuple(map(int, pt)), self.mask_size_, 0, -1, lineType=cv2.CV_AA)
        except:
            pass
        return mask

    def augment_mask(self, pts): 
        """
        Augment the mask of tracked features with additional features. 
        Usually, this is called when features are propagated from map points
        """
        self.aug_pts_ = pts

    def draw_tracks(self, out, colored=False, color_type='unique', min_track_length=4, max_track_length=4):
        """
        color_type: {age, unique}
        """

        N = 20
        # inds = self.confident_tracks(min_length=min_track_length)
        # if not len(inds): 
        #     return

        # ids, pts = self.latest_ids[inds], self.latest_pts[inds]
        # lengths = self.tm_.lengths[inds]

        ids, pts, lengths = self.latest_ids, self.latest_pts, self.tm_.lengths

        if color_type == 'unique': 
            cwheel = colormap(np.linspace(0, 1, N))
            cols = np.vstack([cwheel[tid % N] for idx, tid in enumerate(ids)])
        elif color_type == 'age': 
            cols = colormap(lengths)
        else: 
            raise ValueError('Color type {:} undefined, use age or unique'.format(color_type))

        if not colored: 
            cols = np.tile([0,240,0], [len(self.tm_.tracks), 1])

        for col, pts in izip(cols.astype(np.int64), self.tm_.tracks.itervalues()): 
            cv2.polylines(out, [np.vstack(pts.items).astype(np.int32)[-max_track_length:]], False, 
                          tuple(col), thickness=1)
            tl, br = np.int32(pts.latest_item)-2, np.int32(pts.latest_item)+2
            cv2.rectangle(out, (tl[0], tl[1]), (br[0], br[1]), tuple(col), -1)

    def visualize(self, out, colored=False): 
        if not len(self.latest_pts):
            return

        N = 20
        cols = colormap(np.linspace(0, 1, N))
        valid = finite_and_within_bounds(self.latest_pts, out.shape)

        for tid, pt in izip(self.latest_ids[valid], self.latest_pts[valid]): 
            cv2.rectangle(out, tuple(map(int, pt-2)), tuple(map(int, pt+2)), 
                          tuple(map(int, cols[tid % N])) if colored else (0,240,0), -1)

        return out

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
    def latest_age(self): 
        return self.tm_.lengths

    @property
    def latest_pts(self): 
        return self.tm_.pts

    @property
    def latest_flow(self): 
        return self.tm_.flow

    def confident_tracks(self, min_length=4): 
        return self.tm_.confident_tracks(min_length=min_length)

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
            mask = self.create_mask(im.shape[:2], ppts)            
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

        # Returns only tracks that have a minimum trajectory length
        # This is different from self.tm_.ids, self.tm_.pts
        return self.latest_ids, self.latest_pts

class MeshKLT(OpenCVKLT): 
    """
    KLT Tracker as implemented in OpenCV 2.4.9
    with basic meshing support 
    """
    def __init__(self, *args, **kwargs):
        OpenCVKLT.__init__(self, *args, **kwargs)

        from pybot_vision import DelaunayTriangulation
        self.dt_ = DelaunayTriangulation()

    # @timeitmethod
    def process(self, im, detected_pts=None): 
        ids, pts = OpenCVKLT.process(self, im, detected_pts=detected_pts)
        # # Mesh KLT over confident tracks 
        # inds = self.confident_tracks(min_length=4)
        # ids, pts = self.latest_ids[inds], self.latest_pts[inds]
        if len(pts) > 3: 
            self.dt_.batch_triangulate(pts)
        return ids, pts

    @property
    def triangles(self): 
        return np.hstack(self.dt_.getTriangles()).T

    @property
    def edges(self):
        return np.hstack(self.dt_.getEdges()).T

    @property
    def neighbors(self): 
        return np.hstack(self.dt_.getNeighbors()).T

    def visualize(self, im, ids, pts, colored=False): 
        vis = to_color(im)
        dt_vis = self.dt_.visualize(vis, pts)

        OpenCVKLT.draw_tracks(self, vis, colored=colored, max_track_length=10)
        out = np.vstack([vis, dt_vis])
        imshow_cv('dt_vis', out, wait=1)

        return out

def get_bbox(pts): 
    x1, x2, y1, y2 = np.min(pts[:,0]), np.max(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,1]) 
    return np.int64([x1, y1, x2, y2])

def inside_bboxes(pts, bboxes): 
    """    
    Returns the set of points that are within each bbox
    B x N boolean mask where B bboxes, N pts    
    """
    try: 
        bboxes = bboxes.astype(np.float32)
        return np.vstack([((pts[:,0] >= x1) & (pts[:,0] <= x2) & (pts[:,1] >= y1) & (pts[:,1] <= y2))
                          for (x1, y1, x2, y2) in izip(bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3])])
    except: 
        return np.array([])

class BoundingBoxKLT(OpenCVKLT): 
    """
    KLT Tracker with bounding boxes
    """
    def __init__(self, *args, **kwargs): 
        OpenCVKLT.__init__(self, *args, **kwargs)
        self.hulls_ = {}

    @property
    def initialized(self): 
        return len(self.bboxes_) > 0

    # @timeitmethod
    def process(self, im, bboxes=None): 
        """
        Propagate bounding boxes based on feature tracks
        ccw: a b c d
        """
        OpenCVKLT.process(self, im, detected_pts=None)

        # Degenerate case where no points are available to propagate
        ids, pts, flow = self.latest_ids, self.latest_pts, self.latest_flow

        # 1. Update hulls based on the newly tracked locations and 
        for hid in self.hulls_.keys(): 

            # Find the intersection of previously tracked and currently tracking
            # TODO: can potentially update the ids within the newly tracked hull, 
            # so that the propagation is more prolonged
            tids = self.hulls_[hid].ids
            common_inds, = np.where(np.in1d(ids, tids))

            # Delete the hull if no common tracked ids
            if not len(common_inds): 
                self.hulls_.pop(hid)
                continue

            # Update the hull to the latest points
            # TODO: can update the ids as well so that 
            # the tracking is more prolonged
            vpts = pts[common_inds]
            vbox = get_bbox(vpts)
            valid_pts = inside_bboxes(pts, [vbox])
            vinds, = np.where(valid_pts.ravel())
            vids = ids[np.r_[common_inds, vinds]]

            self.hulls_[hid].pts = vpts
            self.hulls_[hid].bbox = vbox
            self.hulls_[hid].ids = vids

        # 2. Add new hulls that are provided, and keep old tracked ones
        max_id = len(self.hulls_)
        
        # # Find the bounding boxes that are relatively new (IoU < 0.3)
        # if len(self.hulls_) and bboxes is not None: 
        #     hbboxes = np.vstack([hull.bbox for hull in self.hulls_.itervalues()])
        #     HB = brute_force_match(hbboxes, bboxes, 
        #                            match_func=lambda x,y: intersection_over_union(x, y),
        #                            dtype=np.float32)
        #     newinds, = np.where(HB.max(axis=0) < 0.3)
        #     bboxes = bboxes[newinds]

        valid_mask = inside_bboxes(pts, bboxes)
        for bidx, valid in enumerate(valid_mask):
            vids, vpts = ids[valid], pts[valid], 
            # vis = to_color(im)
            # vis = draw_features(vis, vpts)
            # imshow_cv('bbox_feats', vis, block=True)
            if len(vpts): 
                self.hulls_[max_id + bidx] = AttrDict(ids=vids, pts=vpts, bbox=get_bbox(vpts)) # bbox=bboxes[bidx])

        return self.ids, self.bboxes

    @property
    def bboxes(self): 
        try: 
            hboxes = np.vstack([ hull.bbox for hull in self.hulls_.itervalues() ]).reshape(-1,4)
        except: 
            hboxes = np.empty(shape=(0,4), dtype=np.int32)
        return hboxes

    @property
    def ids(self): 
        try: 
            hids = self.hulls_.keys()
        except: 
            hids = []
        return hids

    def visualize(self, vis, colored=True): 

        try: 
            tids = set(self.ids)
        except: 
            return vis

        for hid, hbox in izip(self.ids, self.bboxes): 
            cv2.rectangle(vis, (hbox[0], hbox[1]), (hbox[2], hbox[3]), (0,255,0), 1)

        vis = super(BoundingBoxKLT, self).viz(vis, colored=colored)

        # for tid, pts in self.tm_.tracks.iteritems(): 
        #     if tid not in tids: continue
        #     cv2.polylines(vis, [np.vstack(pts.items).astype(np.int32)[-4:]], False, 
        #                   (0,255,0), thickness=1)
        #     tl, br = np.int32(pts.latest_item)-2, np.int32(pts.latest_item)+2
        #     cv2.rectangle(vis, (tl[0], tl[1]), (br[0], br[1]), (0,255,0), -1)

        # OpenCVKLT.draw_tracks(self, vis, colored=colored, max_track_length=10)
        return vis
