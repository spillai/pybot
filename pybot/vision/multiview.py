# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
from collections import deque

from pybot.vision.imshow_utils import imshow_cv
from pybot.vision.camera_utils import Camera, CameraIntrinsic, CameraExtrinsic, plot_epipolar_line
from pybot.vision.image_utils import to_gray, to_color, im_resize

from pybot.vision.draw_utils import draw_features
from pybot.vision.feature_detection import FeatureDetector

class Frame(object): 
    def __init__(self, im, camera, points=None): 
        self.im_ = np.copy(im)
        self.camera_ = camera
        self.points_ = points
        
    @property
    def im(self): 
        return self.im_

    @property
    def camera(self): 
        return self.camera_

    @property
    def points(self): 
        return self.points_

class EpipolarViz(object): 
    def __init__(self, detector='fast', max_views=10, fixed_reference=False): 
        self.frames_ = deque(maxlen=max_views)
        self.fixed_reference_ = fixed_reference

        if detector == 'apriltag': 
            params = FeatureDetector.apriltag_params
        elif detector == 'fast': 
            params = FeatureDetector.fast_params
            params.params.threshold = 80
        elif detector is None:
            self.fdet_ = None
            print('Not initializing detector, provide points instead')
            return
        else: 
            raise ValueError('Unknown detector type {:}'.format(detector))

        self.fdet_ = FeatureDetector(method=detector, params=params)
 
    def add(self, im, camera, points=None): 
        if self.fixed_reference_ and not len(self.frames_): 
            self.ref_frame_ = Frame(im, camera, points=points)
            print('Reference frame {:}'.format(camera))

        self.frames_.append(Frame(im, camera, points=points))

    def visualize(self, root=0): 
        if len(self.frames_) < 2: 
            return

        if self.fixed_reference_: 
            ref = self.ref_frame_
            root = -1
        else: 
            assert(root >= 0 and root < len(self.frames_))
            try:
                ref = self.frames_[root]
            except: 
                raise RuntimeError("Unable to index to root")

        vis = {}

        # Detect features in the reference image, else
        # use provided features
        if ref.points is not None:
            pts = ref.points
        else: 
            try: 
                pts = self.fdet_.process(to_gray(ref.im))
            except: 
                return

        if not len(pts):
            return

        print(pts.shape, pts.dtype)
        
        
        # Draw epipoles across all other images/poses
        for idx, f in enumerate(self.frames_): 
            F_10 = ref.camera.F(f.camera)
            vis[idx] = plot_epipolar_line(f.im, F_10, pts, im_0=ref.im if idx == 0 else None)

            # print 'F b/w ref and idx={:}, \ncurr={:}\n\nF={:}\n'.format(idx, f.camera, F_10)

        if len(vis): 
            imshow_cv('epi_out', im_resize(np.vstack(list(vis.values())), scale=0.5))
