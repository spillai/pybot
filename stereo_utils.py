import cv2, time
import numpy as np
from collections import deque

import bot_vision.color_utils as color_utils 
import bot_vision.image_utils as image_utils 

# from pybot_vision import VoxelStereoBM as _VoxelStereoBM
# from pybot_vision import EdgeStereoBM as _EdgeStereoBM
# from pybot_vision import EdgeStereo as _EdgeStereo

from pybot_externals import StereoELAS # , OrderedCostVolumeStereo
# from pybot_externals import fast_cost_volume_filtering, ordered_row_disparity

from bot_vision.camera_utils import get_calib_params
from bot_vision.image_utils import im_resize, gaussian_blur
from bot_vision.imshow_utils import imshow_cv

def colorize_stereo_disparity(disp, im=None, max_disparity=256): 
    # Display colored disparity
    disp_color = color_utils.colormap(disp.astype(np.float32) / max_disparity) 
    if im is None: 
        return disp_color 
    else: 
        return np.vstack([image_utils.to_color(im), disp_color])

class StereoSGBM: 
    # Parameters from KITTI dataset
    sad_window_size = 3

    default_params = dict( minDisparity = 0, # 16,
                    preFilterCap = 15, # 63, 
                    numDisparities = 128,
                    # SADWindowSize = sad_window_size, uniquenessRatio = 10, speckleWindowSize = 100,
                    SADWindowSize = sad_window_size, 
                    uniquenessRatio = 0, # 10, 
                    speckleWindowSize = 100, # 20,
                    speckleRange = 32, disp12MaxDiff = 1, 
                    P1 = 50, # sad_window_size*sad_window_size*4, # 8*3*3**2, # 8*3*window_size**2,
                    P2 = 800, # sad_window_size*sad_window_size*32, # 32*3*3**2, # 32*3*window_size**2, 
                    fullDP = True )

    def __init__(self, params=default_params): 
        self.params = params

        # Initilize stereo semi-global block matching
        self.sgbm = cv2.StereoSGBM(**self.params)

        # Re-map process
        self.process = self.compute

    def compute(self, left, right): 
        return self.sgbm.compute(left, right).astype(np.float32) / 16.0

class StereoBM: 
    sad_window_size = 9
    default_params = dict( preset=cv2.STEREO_BM_BASIC_PRESET, uniquenessRatio = 10, 
                           speckleWindowSize = 20, preFilterCap = 31, 
                           ndisparities=128, SADWindowSize=sad_window_size )
    def __init__(self, params=default_params): 
        self.params = params

        # Initilize stereo block matching
        self.bm = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, 128, 11) # **self.params)

        self.process = self.compute

    def compute(self, left, right): 
        return self.bm.compute(left, right).astype(np.float32) / 16.0

# class VoxelStereoBM: 
#     def __init__(self, discretize=1, do_sgm=True): 
#         self.discretize = discretize
#         if discretize > 1: 
#             # Initilize stereo block matching
#             self.stereo = _VoxelStereoBM(discretize=discretize, 
#                                          do_sgm=do_sgm,
#                                          preset=cv2.STEREO_BM_BASIC_PRESET, 
#                                          ndisparities=64, SAD_window_size=5)

#             # self.stereo = StereoBM() # **self.params)
#             # self.stereo.process = lambda l,r: self.stereo.compute(l,r)
#         else: 
#             raise RuntimeError("Discretization less than 1 not supported!")

#     def compute(self, left, right): 
#         # Compute stereo disparity
#         disp = (self.stereo.process(left, right)).astype(np.float32) / 16
#         disp = cv2.medianBlur(disp, 3)

#         # Re-scale disparity image
#         disp_out = cv2.resize(disp.astype(np.float32), (left.shape[1],left.shape[0]), 
#                               fx=self.discretize, 
#                               fy=self.discretize, 
#                               interpolation=cv2.INTER_NEAREST)

#         return disp_out


# class EdgeStereoBM: 
#     def __init__(self): 
#         # Initilize stereo block matching
#         self.stereo = _EdgeStereoBM(preset=cv2.STEREO_BM_BASIC_PRESET, 
#                                     ndisparities=64, SAD_window_size=5)

#     def compute(self, left, right): 
#         return self.stereo.process(left, right).astype(np.float32) / 16.0


# class EdgeStereo: 
#     def __init__(self): 
#         # Initilize stereo block matching
#         self.stereo = _EdgeStereo()

#     def compute(self, left, right): 
#         return self.stereo.process(left, right).astype(np.float32) / 16.0


# ================================
# Ordered stereo disparity matching
class OrderedStereoBM(object): 
    # sad_window_size = 9
    # params_ = dict( preset=cv2.STEREO_BM_BASIC_PRESET, 
    #                 ndisparities=64, 
    #                 SADWindowSize=sad_window_size )

    def __init__(self): 
        # # Init ELAS, and queue of disparities
        # self.stereo = StereoELAS()
        # self.prior_disps = deque(maxlen=5)        
        self.stereo = OrderedCostVolumeStereo()

    # def compute_with_prior(self, left_im, right_im): 
    #     # ELAS for prior disparity
    #     elas_disp = self.stereo.process(left_im, right_im)

    #     # Set maximum number of prior disparities
    #     if len(self.prior_disps) < 5: 
    #         self.prior_disps.append(np.copy(elas_disp))
    #     imshow_cv('prior_disps', reduce(lambda x, v: np.vstack([x,v]), self.prior_disps) * 1.0 / 255)

    #     # Compute median disp across past frames
    #     all_disps = reduce(lambda x, v: np.dstack([x,v]), self.prior_disps)
    #     prior_median_disp = np.median(all_disps, axis=2) if all_disps.ndim > 2 else np.copy(all_disps)

    #     # Filtered median disparity
    #     prior_median_disp[prior_median_disp < 5] = 0
    #     imshow_cv('median_disps', prior_median_disp * 1.0 / 128)

    #     # Ordered median disparity
    #     prior_median_disp = ordered_row_disparity(prior_median_disp)
    #     imshow_cv('ordered_median_disps', prior_median_disp * 1.0 / 128)

    #     # # Compute variance on past disparities
    #     # prior_var_disp = np.std(all_disps, axis=2) if all_disps.ndim > 2 else all_disps
    #     # imshow_cv('var_disps', prior_var_disp * 5.0 / 128)
    #     # print 'Minmax: ', np.min(prior_median_disp), np.max(prior_median_disp)

    #     # Fast cost volume filtering
    #     return self.compute(left_im, right_im, prior_median_disp)

    def compute(self, left, right): 

        # Fast cost volume filtering
        st = time.time()
        disp = self.stereo.process(left, right)
        assert(disp.dtype == np.float32)
        print 'Time taken to sgbm_test', time.time() - st

        # # Display 
        # imshow_cv('disparity', disp.astype(np.float32) / 128)

        # # Difference b/w median elas disp and current disparity
        # if prior_median_disp.shape == disp.shape: 
        #     imshow_cv('diff_disp', (np.abs(disp - prior_median_disp) > 2).astype(np.uint8) * 255)

        # # Compute median blur on disparity
        # dispm = cv2.medianBlur(disp, 5)
        # # imshow_cv('disp bil', dispm / 128)        

        return disp

# Class for stereo reconstruction
class StereoReconstruction(object): 
    def __init__(self, calib=None):
        self.calib = calib

    def disparity_from_plane(self, rows, height): 
        Z = height * self.calib.fy  /  (np.arange(rows) - self.calib.cy + 1e-9)
        return self.calib.fx * self.calib.baseline / Z 

    def depth_from_disparity(self, disp): 
        return self.calib.fx * self.calib.baseline / disp

    def reconstruct(self, disp): 
        """
        Reproject to 3D with calib params
        """
        X = cv2.reprojectImageTo3D(disp, self.calib.Q)
        return X

    def reconstruct_with_texture(self, disp, im, sample=1): 
        """
        Reproject to 3D with calib params and texture mapped
        """
        assert(im.ndim == 3)
        X = cv2.reprojectImageTo3D(disp, self.calib.Q)
        im_pub, X_pub = np.copy(im[::sample,::sample]).reshape(-1,3 if im.ndim == 3 else 1), \
                        np.copy(X[::sample,::sample]).reshape(-1,3)
        return im_pub, X_pub

class CalibratedFastStereo(object): 
    def __init__(self, stereo, calib_params, rectify=None): 
        self.stereo_ = stereo
        self.calib_set_ = False
        self.rectify_ = rectify

        # Only set calib if available
        if hasattr(self.stereo_, 'set_calib'): 

            # Set fake calibration parameters if None
            if calib_params is None: 
                def calibration_lambda(H,W): 
                    new_calib_params = get_calib_params(1000, 1000, W/2-0.5, H/2-0.5, 0.120)
                    return self.stereo_.set_calib(new_calib_params.P0[:3,:3], 
                                                              new_calib_params.P1[:3,:3], # K0, K1
                                                              np.zeros(5), np.zeros(5),
                                                              np.eye(3), np.eye(3),
                                                              new_calib_params.P0, new_calib_params.P1, 
                                                              new_calib_params.Q, new_calib_params.T1, 
                                                              W, H # round to closest multiple of 16
                    )
                self.set_calibration = lambda H,W: calibration_lambda(H,W)
            else: 
                self.set_calibration = lambda H,W: self.stereo_.set_calib(calib_params.P0[:3,:3], 
                                                                          calib_params.P1[:3,:3], # K0, K1
                                                                          calib_params.D1, calib_params.D0, 
                                                                          calib_params.R0, calib_params.R1, 
                                                                          calib_params.P0, calib_params.P1, 
                                                                          calib_params.Q, calib_params.T1, 
                                                                          W, H # round to closest multiple of 16
                                                                      )
        else: 
            self.set_calibration = lambda *args: None


    def strip(self, left_im, right_im): 
        sz0 = left_im.shape[:2]
        sz = np.array(list(left_im.shape)) - np.array(list(left_im.shape)) % 16
        dsz = sz0 - sz
        if not self.calib_set_: 
            self.set_calibration(sz[0], sz[1])
            self.calib_set_ = True
        return left_im[dsz[0]/2:sz[0]+dsz[0]/2, dsz[1]/2:sz[1]+dsz[1]/2], right_im[dsz[0]/2:sz[0]+dsz[0]/2, dsz[1]/2:sz[1]+dsz[1]/2]

    def process(self, left_im, right_im):
        if self.rectify_ is not None: 
            left_im, right_im = self.rectify_(left_im, right_im)
        
        lim, rim = self.strip(left_im, right_im)
        disp = np.zeros(shape=left_im.shape, dtype=np.float32)
        res = self.stereo_.process(lim, rim)
        sz = res.shape
        disp[:sz[0],:sz[1]] = res
        return disp

        
