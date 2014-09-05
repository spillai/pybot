import cv2, time
import numpy as np

from bot_vision.imshow_utils import imshow_cv
from fs_utils import guided_filter

class StereoVoxels: 
    def __init__(self, calib_params, shape=[480,640], resolution=0.2, discretize=10, ndisparities = 128): 
        self.resolution = resolution
        self.ndisparities = ndisparities
        self.calib_params = calib_params
        self.discretize = discretize
        if discretize > 0: 
            self._build_voxel_map(shape, resolution)
        
    def compute(self, left, right): 
        return self._stereo_disparity(left, right)

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

        guide_r, guide_eps, guide_gamma = 9, 0.0001, 0.11
        gamma_c, gamma_d = 0.1, 9
        thresh_grad, thresh_border = 2 / 255.0, 3 / 255.0
        
        # Load floating point images
        H,W = left.shape[:2]
        Il, Ir = left / 255.0, right / 255.0

        # For each disparity
        if self.discretize > 0: 
            disp_vox = np.ones(shape=(self._max_ybin, self._max_xbin, self.ndisparities)) * thresh_border
        disp_vol = np.ones(shape=(H, W, self.ndisparities)) * thresh_border
        for d in range(self.ndisparities): 

            # Truncated SAD 
            tmp = np.ones(shape=(H,W)) * thresh_border
            tmp[:,d:W] = Ir[:,:W-d]
            cost = np.abs(tmp - Il)
            # cost = cv2.GaussianBlur(np.abs(tmp - Il), (3,3), 0)
            # cost = guided_filter(src=np.abs(tmp - Il), guidance=Il, radius=guide_r, eps=guide_eps)

            # Write costs
            disp_vol[:,:,d] = cost

            # Sum the costs for each bin
            if self.discretize > 0: 
                id_costs = np.bincount(self._xyids.reshape(-1), weights=cost.reshape(-1))
                ids = np.unique(self._xyids.reshape(-1))
                xbin, ybin = self._ids_to_bin(ids, step=self._max_xbin)

                # Write back to bins
                disp_vox[ybin,xbin,d] = id_costs

            
        # Disparity image (WTA)
        disp = np.argmin(disp_vol, axis=2)

        # Compute voxel-mapped stereo
        if self.discretize > 0: 
            disp2 = np.argmin(disp_vox, axis=2).astype(np.float32)
            disp3 = cv2.resize(disp2, (W,H), fx=self.discretize, fy=self.discretize, interpolation=cv2.INTER_NEAREST)
            imshow_cv("Disparity_vox", disp3.astype(np.float32) / 128)

        return disp
            


if __name__ == "__main__": 
    import os
    from bot_utils.kitti_helpers import kitti_stereo_calib_params

    left = cv2.imread(os.path.expanduser('~/data/dataset/sequences/08/image_0/000000.png'), 0)
    right = cv2.imread(os.path.expanduser('~/data/dataset/sequences/08/image_1/000000.png'), 0)

    H, W = left.shape[:2]

    DISCRETIZE = 4
    SCALE = 1. / DISCRETIZE

    calib_params = kitti_stereo_calib_params()
    stereo = StereoVoxels(calib_params, shape=left.shape[:2], discretize=DISCRETIZE, ndisparities=64)
    disp = stereo.compute(left, right)
    imshow_cv("Disparity 10", disp.astype(np.float32) / 128)
    cv2.waitKey(0)



    calib_params = kitti_stereo_calib_params(scale=0.1)
    left = cv2.resize(left, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)    
    right = cv2.resize(right, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)    

    stereo = StereoVoxels(calib_params, shape=left.shape[:2], discretize=0, ndisparities=64)
    disp = stereo.compute(left, right)
    disp = cv2.resize(disp, (W,H), interpolation=cv2.INTER_NEAREST)    
    imshow_cv("Disparity 0", disp.astype(np.float32) / 128)
    cv2.waitKey(0)

