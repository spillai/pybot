import numpy as np
from collections import defaultdict, deque
from bot_vision.color_utils import colormap

class TrackManager(object): 
    def __init__(self, maxlen=10): 
        self.maxlen = maxlen
        self.idx = 0
        self._ids, self._pts = None, None

        self.tracks_ts = dict()
        self.tracks = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, pts, ids=None): 
        # Retain valid points
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]

        # ID valid points
        max_id = np.max(self._ids) if self._ids is not None else 0
        tids = np.arange(len(pts), dtype=np.int64) + max_id + 1 if ids is None else ids[valid]
        
        if ids is not None: 
            print len(ids), len(ids[valid]), len(ids)-len(ids[valid])

        # Add pts to track
        for tid, pt in zip(tids, pts): 
            self.tracks[tid].append(pt)
            self.tracks_ts[tid] = self.idx

        # If features are propagated
        if ids is not None: 
            self.prune()
            self.idx += 1

        # Keep pts and ids up to date
        self._ids = np.array(self.tracks.keys())
        self._pts = np.vstack([ track[-1] for track in self.tracks.itervalues() ])

    def prune(self): 
        # Remove tracks that are not most recent
        # count = 0
        for tid, val in self.tracks_ts.items(): 
            if val < self.idx: 
                del self.tracks[tid]
                del self.tracks_ts[tid]
                # count += 1
        # print 'Total pruned: ', count
    @property
    def pts(self): 
        return self._pts
        
    @property
    def ids(self): 
        return self._ids

