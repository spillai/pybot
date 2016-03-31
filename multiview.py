import numpy as np
from collections import deque

from .imshow_utils import imshow_cv
from .camera_utils import Camera, CameraIntrinsic, CameraExtrinsic, plot_epipolar_line
from .image_utils import to_gray, to_color, im_resize

from bot_vision.draw_utils import draw_features
from bot_vision.feature_detection import FeatureDetector

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
    def __init__(self, detector='apriltag', max_views=10, fixed_reference=False): 
        self.frames_ = deque(maxlen=max_views)
        self.fixed_reference_ = fixed_reference

        if detector == 'apriltag': 
            params = FeatureDetector.apriltag_params
        elif detector == 'fast': 
            params = FeatureDetector.fast_params
            params.params.threshold = 80
        else: 
            raise ValueError('Unknown detector type {:}'.format(detector))

        self.fdet_ = FeatureDetector(method=detector, params=params)
 
    def add(self, im, camera): 
        if self.fixed_reference_ and not len(self.frames_): 
            self.ref_frame_ = Frame(im, camera)
            print('Reference frame {:}'.format(camera))

        self.frames_.append(Frame(im, camera))

    def visualize(self, root=0): 
        if len(self.frames_) < 2: 
            return

        if self.fixed_reference_: 
            ref_im = self.ref_frame_.im
            ref_camera = self.ref_frame_.camera
            root = -1
        else: 
            assert(root >= 0 and root < len(self.frames_))
            try: 
                ref_im = self.frames_[root].im
                ref_camera = self.frames_[root].camera
            except: 
                raise RuntimeError("Unable to index to root")

        vis = {}

        # Detect features in the reference image
        try: 
            pts = self.fdet_.process(to_gray(ref_im))
            # pts = pts.reshape(len(pts)/4,-1,2).mean(axis=1)
        except: 
            return

        if not len(pts):
            return

        # Draw epipoles across all other images/poses
        for idx, f in enumerate(self.frames_): 
            F_10 = ref_camera.F(f.camera)
            vis[idx] = plot_epipolar_line(f.im, F_10, pts, im_0=ref_im if idx == 0 else None)

            # print 'F b/w ref and idx={:}, \ncurr={:}\n\nF={:}\n'.format(idx, f.camera, F_10)

        if len(vis): 
            imshow_cv('epi_out', im_resize(np.vstack(vis.values()), scale=0.5))
