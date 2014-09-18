import cv2, time
import numpy as np

# import scipy.ndimage.filters as spfilters 

import bot_vision.color_utils as color_utils 
from bot_vision.imshow_utils import imshow_cv, imshow_plt, bar_plt

<<<<<<< HEAD
from fs_utils import guided_filter, CostVolumeStereo, StereoBMCustom
from bot_vision.stereo_utils import StereoSGBM, StereoSGBMDiscretized
=======
# from bot_vision.stereo_utils import StereoBM

# from fs_utils import CostVolumeStereo, StereoBMCustom
>>>>>>> ea29bc581cd7a2a55f39bb67f6d377e5c408ac41

import pyximport; pyximport.install()
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

from bot_externals.adcensus_stereo import init, ad_vol, sgm, \
    depth_discontinuity_adjustment, subpixel_enhancement

class StereoVoxels: 
    def __init__(self, calib_params, shape=[480,640], 
                 resolution=0.2, discretize=10, ndisparities = 128): 
        self.resolution = resolution
        self.ndisparities = ndisparities
        self.calib_params = calib_params
        self.discretize = discretize
        if discretize > 0: 
            self._build_voxel_map(shape, resolution)

        cv2.namedWindow("disparity_cb")
        cv2.setMouseCallback("disparity_cb", self._on_mouse)
        
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pt = (x, y)
        # print self._disp[y,x], self.disp_vox[ybin,xbin]

        xbin, ybin = self._px_to_bin(x), self._px_to_bin(y)
        bar_plt('label_costs', self.disp_vox[ybin,xbin])
        # bar_plt('intensities_left', self._left[y,x:x+self.ndisparities])
        # bar_plt('intensities_right', self._right[y,x:x+self.ndisparities])
        
    def compute(self, left, right): 
        self._left, self._right = left, right
        self._disp = self._stereo_disparity(left, right)
        return self._disp

    def _px_to_bin(self, xs): 
        return xs / self.discretize

    def _bin_to_ids(self, xbins, ybins, step): 
        return ybins * step + xbins

    def _ids_to_bin(self, ids, step): 
        return ids % step, ids / step

    def _build_voxel_map(self, shape, resolution): 
        H, W = shape
        self._xs, self._ys = np.meshgrid(np.arange(0, W), np.arange(0, H))
        self._xbin, self._ybin = self._px_to_bin(self._xs), self._px_to_bin(self._ys)
        self._max_xbin, self._max_ybin = np.max(self._xbin) + 1, np.max(self._ybin) + 1
        self._xyids = self._bin_to_ids(self._xbin, self._ybin, step=self._max_xbin)

    def _stereo_disparity(self, left, right): 

        guide_r, guide_eps, guide_gamma = 3, 0.0001, 0.11
        gamma_c, gamma_d = 0.1, 9
        thresh_grad, thresh_border = 2 / 255.0, 3 / 255.0
        
        # Load floating point images
        H,W = left.shape[:2]
        Il, Ir = (left / 255.0).astype(np.float32), (right / 255.0).astype(np.float32)

        # For each disparity
        if self.discretize > 0: 
            self.disp_vox = np.zeros(shape=(self._max_ybin, self._max_xbin, self.ndisparities), dtype=np.float64)
<<<<<<< HEAD
            # self.costs_vox = np.ones(shape=(self._max_ybin, self._max_xbin, 
            #                                 self.discretize, self.discretize, 
                                            # self.ndisparities)) * thresh_border
            # print self.costs_vox.shape

        # ================================================================
        st = time.time()
        # disp = cv2.StereoBM(cv2.STEREO_BM_PREFILTER_XSOBEL, 64, 11).compute(left, right) / 16.0
        disp = StereoSGBMDiscretized(discretize=self.discretize).compute(left, right) 
        # imshow_cv('left', np.hstack([left, right]))

        # self.disp_vox[self.disp_vox > 1000] = 1000
=======
            print self.disp_vox.shape
        # # ================================================================
        # st = time.time()
        # self.disp_vox = StereoBMCustom(discretize=self.discretize, 
        #                                preset=cv2.STEREO_BM_BASIC_PRESET, 
        #                                ndisparities=64, 
        #                                SAD_window_size=11).process(left, right) / (11 * 11 * self.discretize * self.discretize)
        # disp = np.argmin(self.disp_vox, axis=2)
>>>>>>> ea29bc581cd7a2a55f39bb67f6d377e5c408ac41

        # disp_vol = CostVolumeStereo(discretize=1, 
                                    # cost_volume_filtering='',
                                    # gamma=1.0, guide_r=5, guide_eps=0.0001,
                                    # SAD_window_size=1, 
                                    # median_post_processing=False, 
                                    # interpolate_disparities=True).compute(Il, Ir)

<<<<<<< HEAD
        # disp = np.argmin(self.disp_vox, axis=2)
        print 'Time taken for costvolume disp range %4.3f ms' % ((time.time() - st) * 1e3)
        # print cost.shape
        # imshow_cv("test_disp", cost[:,:,20] / (128 * 255))
        # cv2.waitKey(0)
=======
        # print 'Time taken for costvolume disp range %4.3f ms' % ((time.time() - st) * 1e3)
        # # print cost.shape
        # # imshow_cv("test_disp", cost[:,:,20] / (128 * 255))
        # # cv2.waitKey(0)
>>>>>>> ea29bc581cd7a2a55f39bb67f6d377e5c408ac41

        # ================================================================
        st = time.time()
        disp_vol = np.zeros(shape=(H, W, self.ndisparities)) * thresh_border

        # Pre-process mapping
        ids = np.unique(self._xyids.reshape(-1))
        xbin, ybin = self._ids_to_bin(ids, step=self._max_xbin)

        for d in range(self.ndisparities): 

            # Truncated SAD 
            tmp = np.zeros(shape=(H,W), dtype=np.float32) * thresh_border
            tmp[:,d:W] = Ir[:,:W-d]
            cost = np.abs(tmp - Il)
            # cost = cv2.GaussianBlur(np.abs(tmp - Il), (3,3), 0)
            # cost = cv2.medianBlur(np.abs(tmp - Il), 5)
            cost = cv2.bilateralFilter(np.abs(tmp - Il), 5, 5*2, 5/2)
            # cost = guided_filter(src=np.abs(tmp - Il), guidance=Il, 
            #                      radius=guide_r, eps=guide_eps)

            # Write costs
            disp_vol[:,:,d] = cost
            disp_vol[:,:self.ndisparities,d] = 0

            # Aggregation of costs
            if self.discretize >= 1: 

                # Option A. Sum the costs for each bin
                id_costs = np.bincount(self._xyids.reshape(-1), weights=cost.reshape(-1))
                self.disp_vox[ybin,xbin,d] = id_costs / self.discretize ** 2
                self.disp_vox[:,:self.ndisparities/self.discretize,d] = 0

        print 'Time taken for disp range %4.3f ms' % ((time.time() - st) * 1e3)
               
        # Zero out negative disparities
        disp_vol[:,:self.ndisparities,:] = 0
        self.disp_vox[:,:self.ndisparities/self.discretize,:] = 0
            
        # Disparity image (WTA)
        disp = np.argmin(self.disp_vox, axis=2)

        # Depth discontinuity adjustment
        # disp = depth_discontinuity_adjustment(disp, self.disp_vox)

        # Subpixel refinement
        # disp = subpixel_enhancement(disp, self.disp_vox)


        # Re-scale disparity image
        disp_out = cv2.resize(disp.astype(np.float32), (W,H), 
                              fx=self.discretize, 
                              fy=self.discretize, 
                              interpolation=cv2.INTER_NEAREST)


        # ================================================================
        # Compute voxel-mapped stereo
        if self.discretize >= 1: 

            # SGM
            st = time.time()
            h_, w_, d_ = self.disp_vox.shape[:3]
            left_thumb = cv2.resize(left, (w_,h_))    
            right_thumb = cv2.resize(right, (w_,h_))
            
            print 'Before: ', self.disp_vox.shape
            try: 
                self.disp_vox = cSGM(left_thumb, right_thumb, 
                                     self.disp_vox.astype(np.float64), self.discretize)
            except: 
                print 'Pure python-SGM '
                init(h_, w_, d_, self.discretize)
                self.disp_vox = sgm(left_thumb.astype(np.float64), 
                                    right_thumb.astype(np.float64), 
                                    self.disp_vox.astype(np.float64))
            print 'Time taken for sgm %4.3f ms' % ((time.time() - st) * 1e3)
            
            # 3-D Median filter
            # self.disp_vox = spfilters.median_filter(self.disp_vox, size=3)
            # imshow_cv("test_disp", self.disp_vox[:,:,5])

            # WTA disparity estimation
            disp2 = np.argmin(self.disp_vox, axis=2)

            # # Depth discontinuity adjustment
            # disp2 = depth_discontinuity_adjustment(disp2, self.disp_vox)

            # Subpixel refinement
            # disp2 = subpixel_enhancement(disp2, self.disp_vox)

            # Re-scale disparity image
            disp_out = cv2.resize(disp2.astype(np.float32), (W,H), 
                                  fx=self.discretize, 
                                  fy=self.discretize, 
                                  interpolation=cv2.INTER_NEAREST)

            
            imshow_cv("disparity_cb", 
                      color_utils.colormap(disp_out.astype(np.float32) / 64))
            
            return disp_out.astype(np.float32) / 64

        imshow_cv("disparity_cb", 
                  color_utils.colormap(disp_out.astype(np.float32) / 64))

        return disp.astype(np.float32) / 64
            


if __name__ == "__main__": 
    import os
    from bot_utils.kitti_helpers import kitti_stereo_calib_params

<<<<<<< HEAD
    left = cv2.imread(os.path.expanduser('~/data/dataset/sequences/08/image_0/000140.png'), 0)
    right = cv2.imread(os.path.expanduser('~/data/dataset/sequences/08/image_1/000140.png'), 0)

    # left = cv2.resize(left, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)    
    # right = cv2.resize(right, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)    

    H, W = left.shape[:2]

    DISCRETIZE = 8
    SCALE = 1. / DISCRETIZE
=======
    left = cv2.imread(
        os.path.expanduser('~/data/dataset/sequences/08/image_0/000000.png'), 0)
    right = cv2.imread(
        os.path.expanduser('~/data/dataset/sequences/08/image_1/000000.png'), 0)
>>>>>>> ea29bc581cd7a2a55f39bb67f6d377e5c408ac41

    SCALE = 1. / 2
    DISCRETIZE = 2
    
    left = cv2.resize(left, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)    
    right = cv2.resize(right, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)    

    # Stereo Voxels
    stereo = StereoVoxels(kitti_stereo_calib_params(scale=SCALE), 
                          shape=left.shape[:2], discretize=DISCRETIZE, ndisparities=64)
    disp = stereo.compute(left, right)
    cv2.waitKey(0)

