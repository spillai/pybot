from bot_vision.imshow_utils import imshow_cv, trackbar_create, trackbar_value
from bot_vision.image_utils import im_resize, to_gray

from pybot_vision import FastStereo as _FastStereo
from pybot_vision import scaled_color_disp

class FastStereo(object): 
    def __init__(self, calib, threshold=10, scale=1.0, iterations=1): 
        print calib

        # Stereo Methods: CROSS_RATIO_DISPARITY, TESSELLATED_DISPARITY, PLANAR_INTERP_DISPARITY
        self.stereo = _FastStereo(threshold=threshold, 
                                  stereo_method=_FastStereo.TESSELLATED_DISPARITY, lr_consistency_check=True)
        self.stereo.set_calibration(calib.left.K, calib.right.K, 
                                    calib.left.D, calib.right.D, calib.left.R, calib.right.R, 
                                    calib.left.P, calib.right.P, calib.Q, calib.right.t)
        self.stereo.iterations = iterations

        self.calib = calib
        self.scale = scale

        # Trackbar
        trackbar_create('cost_threshold', 'disparity', 40, 100, scale=0.01)
        trackbar_create('fast_threshold', 'disparity', 20, 50, scale=1)
        trackbar_create('iterations', 'disparity', iterations, 10, scale=1)

    def process(self, left_im, right_im): 
        
        # Stereo process
        disp = self.stereo.process(to_gray(left_im), to_gray(right_im))

        # Display colored depth
        vis = scaled_color_disp(disp / self.scale)
        imshow_cv("disparity", im_resize(vis, scale=1/self.scale))
        
        # Update cost threshold for visualization
        self.stereo.fast_threshold = trackbar_value(key='fast_threshold')
        self.stereo.cost_threshold = trackbar_value(key='cost_threshold')
        self.stereo.iterations = trackbar_value(key='iterations')

        return disp

    def evaluate(self, gt_disp): 
        imshow_cv("gt", scaled_color_disp(gt_disp))               

