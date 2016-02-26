import numpy as np
from collections import deque

from .imshow_utils import imshow_cv
from .camera_utils import Camera, CameraIntrinsic, CameraExtrinsic, plot_epipolar_line
from .image_utils import to_gray, to_color, im_resize

from .trackers.base_klt import draw_features
from .trackers.tracker_utils import FeatureDetector

class Frame(object): 
    def __init__(self, im, camera): 
        self.im_ = np.copy(im)
        self.camera_ = camera
        
    @property
    def im(self): 
        return self.im_

    @property
    def camera(self): 
        return self.camera_

class EpipolarViz(object): 
    def __init__(self, max_views=10): 
        self.frames_ = deque(maxlen=max_views)

        params = FeatureDetector.fast_detector_params
        params.params.threshold = 80

        self.fdet_ = FeatureDetector(params)
 
    def add(self, im, camera): 
        self.frames_.append(Frame(im, camera))

    def visualize(self, root=-1): 
        assert(root < 0 and root >= -len(self.frames_))

        try: 
            ref_im = self.frames_[root].im
            ref_camera = self.frames_[root].camera
        except: 
            raise RuntimeError("Unable to index to root")

        vis = {}

        # Detect features in the reference image
        pts = self.fdet_.process(to_gray(ref_im))
        # vis[len(self.frames_) + root] = draw_features(to_color(ref_im), pts)

        # Draw epipoles across all other images/poses
        for idx, f in enumerate(self.frames_): 
            if idx == len(self.frames_) + root: 
                continue

            F_10 = f.camera.F(ref_camera)
            vis[idx] = plot_epipolar_line(f.im, F_10, pts, im_0=ref_im)

        if len(vis): 
            imshow_cv('epi_out', im_resize(np.vstack(vis.values()), scale=0.5))
