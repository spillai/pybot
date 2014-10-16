import cv2, time
import numpy as np
from collections import deque

from pybot_vision import StereoBMCustom
from pybot_externals import StereoELAS # , OrderedCostVolumeStereo
# from pybot_externals import fast_cost_volume_filtering, ordered_row_disparity

from bot_vision.image_utils import im_resize, gaussian_blur
from bot_vision.imshow_utils import imshow_cv

class StereoSGBM: 
    # Parameters from KITTI dataset
    sad_window_size = 5

    params_ = dict( minDisparity = 0, # 16,
                    preFilterCap = 15, # 63, 
                    numDisparities = 256, # 128,
                    # SADWindowSize = sad_window_size, uniquenessRatio = 10, speckleWindowSize = 100,
                    SADWindowSize = sad_window_size, 
                    uniquenessRatio = 0, # 10, 
                    speckleWindowSize = 100, # 20,
                    speckleRange = 32, disp12MaxDiff = 1, 
                    P1 = 50, # sad_window_size*sad_window_size*4, # 8*3*3**2, # 8*3*window_size**2,
                    P2 = 800, # sad_window_size*sad_window_size*32, # 32*3*3**2, # 32*3*window_size**2, 
                    fullDP = True )


    def __init__(self, params=params_): 
        self.params = params

        # Initilize stereo semi-global block matching
        self.sgbm = cv2.StereoSGBM(**self.params)

        # Re-map process
        self.process = self.compute

    def compute(self, left, right): 
        return self.sgbm.compute(left, right).astype(np.float32) / 16.0

class StereoBM: 
    sad_window_size = 9
    params_ = dict( preset=cv2.STEREO_BM_BASIC_PRESET,
                    uniquenessRatio = 10, 
                    speckleWindowSize = 20, 
                    preFilterCap = 31, 
                    # preset=cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE, 
                    ndisparities=128, 
                    SADWindowSize=sad_window_size )

    def __init__(self, params=params_): 
        self.params = params

        # Initilize stereo block matching
        self.bm = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, 128, 11) # **self.params)

        self.process = self.compute

    def compute(self, left, right): 
        return self.bm.compute(left, right).astype(np.float32) / 16.0

class StereoSGBMDiscretized: 
    def __init__(self, discretize=1, do_sgm=True): 
        self.discretize = discretize
        if discretize > 1: 
            # Initilize stereo block matching
            self.stereo = StereoBMCustom(discretize=discretize, 
                                         do_sgm=do_sgm,
                                         preset=cv2.STEREO_BM_BASIC_PRESET, 
                                         ndisparities=64, SAD_window_size=5)

            # self.stereo = StereoBM() # **self.params)
            # self.stereo.process = lambda l,r: self.stereo.compute(l,r)
        else: 
            raise RuntimeError("Discretization less than 1 not supported!")

    def compute(self, left, right): 
        # Compute stereo disparity
        disp = (self.stereo.process(left, right)).astype(np.float32) / 16
        disp = cv2.medianBlur(disp, 3)

        # Re-scale disparity image
        disp_out = cv2.resize(disp.astype(np.float32), (left.shape[1],left.shape[0]), 
                              fx=self.discretize, 
                              fy=self.discretize, 
                              interpolation=cv2.INTER_NEAREST)

        return disp_out


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
        print 'INIT STEREO RECONSTRUCTION'

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

    def reconstruct_with_texture(self, disp, sample=1): 
        """
        Reproject to 3D with calib params and texture mapped
        """
        X = cv2.reprojectImageTo3D(disp, self.calib.Q)
        im_pub, X_pub = np.copy(left_im[::sample,::sample]).reshape(-1,3 if left_im.ndim == 3 else 1), \
                        np.copy(X[::sample,::sample]).reshape(-1,3)
        return im_pub, X_pub

        
