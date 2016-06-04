import cv2, time
import numpy as np
from collections import deque
from scipy.interpolate import LinearNDInterpolator

from bot_utils.db_utils import AttrDict

from bot_vision.camera_utils import StereoCamera
from bot_vision.image_utils import im_resize, gaussian_blur, to_color, to_gray, valid_pixels
from bot_vision.imshow_utils import imshow_cv, trackbar_create, trackbar_value
from bot_vision.color_utils import colormap
from bot_vision.calib.calibrate_stereo import StereoCalibration, get_stereo_calibration_params
        
from pybot_vision import scaled_color_disp
from pybot_externals import StereoELAS

# from pybot_externals import fast_cost_volume_filtering, ordered_row_disparity

# from pybot_vision import VoxelStereoBM as _VoxelStereoBM
# from pybot_vision import EdgeStereoBM as _EdgeStereoBM
# from pybot_vision import EdgeStereo as _EdgeStereo

def colorize_stereo_disparity(disp, im=None, max_disparity=256): 
    # Display colored disparity
    disp_color = colormap(disp.astype(np.float32) / max_disparity) 
    if im is None: 
        return disp_color 
    else: 
        return np.vstack([to_color(im), disp_color])


def disparity_interpolate(disp, fill_zeros=True): 
    # Determine valid positive disparity pixels
    xyd = valid_pixels(disp, disp > 0)

    # Nearest neighbor interpolator
    nn = LinearNDInterpolator(xyd[:,:2], xyd[:,2])

    # Interpolate all pixels
    xyd = valid_pixels(disp, np.ones(shape=disp.shape, dtype=np.bool))
    return nn(xyd[:,:2]).reshape(disp.shape[:2])


class StereoSGBM: 
    # Parameters from KITTI dataset
    sad_window_size = 5

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
    
    # From KITTI dataset
    # pre_filter_size 41 
    # pre_filter_cap 31 
    # sad_window_size 9 
    # number_of_disparities 128 
    # texture_threshold 20 
    # uniqueness_ratio 10 
    # speckleWindowSize 100 
    # speckleRange 32

    sad_window_size = 9
    default_params = dict( preset=cv2.STEREO_BM_BASIC_PRESET, uniquenessRatio = 10, 
                           speckleWindowSize = 20, preFilterCap = 31, 
                           ndisparities=128, SADWindowSize=sad_window_size )
    kitti_params = dict( preset=cv2.STEREO_BM_BASIC_PRESET, uniquenessRatio = 10, 
                           speckleWindowSize = 100, speckleRange = 32, preFilterSize = 41, preFilterCap = 31, 
                           ndisparities=128, SADWindowSize=sad_window_size )
    def __init__(self, params=default_params): 
        self.params = params

        # Initilize stereo block matching
        self.bm = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, 128, 9)

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

# class StereoReconstruction(object): 
#     raise RuntimeError('Deprecated, see camera_utils.StereoCamera')

class CalibratedStereo(object): 
    def __init__(self, left, right):
        self.cams = [left, right]
        self.undistortion_map = {}
        self.rectification_map = {}
        
        for cidx, cam in enumerate(self.cams):
            (self.undistortion_map[cidx], self.rectification_map[cidx]) = cv2.initUndistortRectifyMap(
                cam.K, cam.D, cam.R, cam.P, cam.shape[:2], cv2.CV_32FC1)

    def rectify(self, l, r): 
        """
        Rectify frames passed as (left, right) 
        Remapping is done with nearest neighbor for speed.
        """
        return [cv2.remap(l, self.undistortion_map[cidx], self.rectification_map[cidx], cv2.INTER_NEAREST)
                for cidx in range(len(self.cams))]
        

class CalibratedFastStereo(object): 
    """
    This class has been deprecated
    """
    def __init__(self, stereo, stereo_calib, rectify=None): 
        self.stereo_ = stereo
        self.rectify_ = rectify

        # Set fake calibration parameters if None
        if stereo_calib is None: 
            stereo_calib = StereoCamera.from_calib_params(1000, 1000, W/2-0.5, H/2-0.5, baseline=0.12)
        self.stereo_.set_calibration(stereo_calib.left.K, 
                                     stereo_calib.right.K, 
                                     stereo_calib.left.D, stereo_calib.right.D, 
                                     stereo_calib.left.R, stereo_calib.right.R, 
                                     stereo_calib.left.P, stereo_calib.right.P, 
                                     stereo_calib.Q, stereo_calib.right.t)

    def process(self, left_im, right_im):
        if self.rectify_ is not None: 
            left_im, right_im = self.rectify_(left_im, right_im)
        return self.stereo_.process(left_im, right_im)

def setup_zed(scale=1.0): 
    """
    Run calibration: 

    Scale (0.5): 
        python calibrate_stereo.py --scale 0.5 --rows 8 --columns 6 --square-size 9.3 zed/data/ zed/calib

    Scale (1.0): 
        python calibrate_stereo.py --scale 1.0 --rows 8 --columns 6 --square-size 9.3 zed/data/ zed/calib
    
    Print calibration: 
        python bot_vision/calib/print_calib.py zed/calib_1.0
    """
    # Saved Calibration
    # calib_path = '/home/spillai/perceptual-learning/software/python/bot_vision/calib/zed/calib'
    # calibration = StereoCalibration(input_folder=calib_path)

    # Scale
    # Determine scale from image width (720p)
    # scale = float(width) / 720.0

    # Setup one-time calibration
    # fx, fy, cx, cy = 702.429138, 702.429138, 652.789368, 360.765472
    # @ 1080: fx, fy, cx, cy = 1396.555664 * s, 1396.555664 * s, 972.651123 * s, 540.047119 * s
    
    # @ 360p
    # D0: [-0.0254902 , -0.00319033,  0.        ,  0.        ,  0.03270019]    
    # D1: [-0.03288394,  0.0149428 ,  0.        ,  0.        ,  0.01202393]
    # K0: [ 337.10210476,    0.        ,  329.15867687],
    #     [   0.        ,  337.10210476,  178.30881956],
    #     [   0.        ,    0.        ,    1.        ]

    # @ 720p
    # D0: [-0.01704945, -0.01655319,  0.        ,  0.        ,  0.04144856]
    # D1: [-0.02413693,  0.00169603,  0.        ,  0.        ,  0.023676  ]
    # K0: [ 677.57005977,    0.        ,  658.49378727],
    #     [   0.        ,  677.57005977,  358.58253284],
    #     [   0.        ,    0.        ,    1.        ]

    # @ 360p
    image_width = 360
    fx, fy, cx, cy = 337.10210476, 337.10210476, 329.15867687, 178.30881956

    # @ 720p
    # image_width = 720
    # fx, fy, cx, cy = 677.57005977463643, 677.57005977463643, 658.49378727401586, 358.58253283725276
    print 'fx, fy, cx, cy', fx, fy, cx, cy, scale

    calib_params = StereoCamera.from_calib_params(fx*scale, fy*scale, cx*scale, cy*scale, baseline=0.12)
    # calib_params = get_calib_params() # baseline_px=baseline_px * scale)
    # calib_params.D0 = np.array([0, 0, 0, 0, 0], np.float64)
    # calib_params.D0 = np.array([-0.16, 0, 0, 0, 0], np.float64)
    # calib_params.D1 = calib_params.D0
    # calib_params.image_width = image_width * scale

    return calib_params

def setup_bb(scale=1.0): 
    # Setup one-time calibration
    calib_path = '/home/spillai/perceptual-learning/software/python/bot_vision/calib/bb/calib'
    calibration = StereoCalibration(input_folder=calib_path)
    calib_params = AttrDict(get_stereo_calibration_params(input_folder=calib_path))
    return StereoCamera.from_calib_params(calib_params.fx, calib_params.fy, 
                                         calib_params.cx, calib_params.cy, baseline=calib_params.baseline)
    # return calib_params

def setup_ps3eye(scale=1.0): 
    # Setup one-time calibration
    calib_path = '/home/spillai/perceptual-learning/software/python/bot_vision/calib/ps3_stereo/calib'
    calibration = StereoCalibration(input_folder=calib_path)
    calib_params = AttrDict(get_stereo_calibration_params(input_folder=calib_path))

    print calib_params
    return StereoCamera.from_calib_params(calib_params.fx, calib_params.fy, 
                                          calib_params.cx, calib_params.cy, baseline=0.1) # calib_params.baseline)
    # return calib_params

def stereo_dataset(filename, channel='CAMERA', start_idx=0, every_k_frames=1, max_length=None, scale=1, split='vertical'): 
    from bot_externals.lcm.log_utils import LCMLogReader, ImageDecoder, StereoImageDecoder
    dataset = LCMLogReader(filename=filename, start_idx=start_idx, 
                           max_length=max_length, every_k_frames=every_k_frames, 
                           index=False, 
                           decoder=StereoImageDecoder(channel=channel,
                                                      scale=scale, split=split))
    
    def iter_frames(*args, **kwargs):
        for (t, ch, (l,r)) in dataset.iteritems(*args, **kwargs):
            yield AttrDict(left=l, right=r)

    def iter_stereo_frames(*args, **kwargs):
        for (t, ch, (l,r)) in dataset.iteritems(*args, **kwargs):
            yield l, r
            
    def iter_gt_frames(*args, **kwargs): 
        gt = StereoSGBM()
        for (t, ch, (l,r)) in dataset.iteritems(): 
            # h,w = im.shape[:2]
            # l,r = np.split(im, 2, axis=0)
            disp = gt.process(l,r)
            yield AttrDict(left=l, right=r, noc=disp, occ=disp)
            
    dataset.iter_frames = iter_frames
    dataset.iter_stereo_frames = iter_stereo_frames
    dataset.iter_gt_frames = iter_gt_frames
    return dataset

def setup_zed_dataset(filename, start_idx=0, max_length=None, every_k_frames=1, scale=1): 
    dataset = stereo_dataset(filename=filename, 
                             channel='CAMERA', start_idx=start_idx, max_length=max_length, 
                             every_k_frames=every_k_frames, scale=scale)

    # Setup one-time calibration
    calib_params = setup_zed(scale=scale)
    dataset.calib = calib_params
    dataset.scale = scale
    return dataset
 
def setup_bb_dataset(filename, start_idx=0, every_k_frames=1, max_length=None, scale=1): 
    dataset = stereo_dataset(filename=filename, start_idx=start_idx, max_length=max_length, 
                             channel='CAMERA', every_k_frames=every_k_frames, scale=scale)
    
    # Setup one-time calibration
    calib_params = setup_bb(scale=scale)
    dataset.calib = calib_params
    return dataset


def setup_ps3eye_dataset(filename, start_idx=0, max_length=None, every_k_frames=1, scale=1): 
    dataset = stereo_dataset(filename=filename, 
                             channel='CAMERA', start_idx=start_idx, max_length=max_length, 
                             every_k_frames=every_k_frames, scale=scale, split='horizontal')

    # Setup one-time calibration
    calib_params = setup_ps3eye(scale=scale)
    dataset.calib = calib_params
    dataset.scale = scale
    return dataset


# def bumblebee_stereo_calib_params_ming(scale=1.0): 
#     fx, fy = 809.53*scale, 809.53*scale
#     cx, cy = 321.819*scale, 244.555*scale
#     baseline = 0.119909
#     return get_calib_params(fx, fy, cx, cy, baseline=baseline)

# def bumblebee_stereo_calib_params(scale=1.0): 
#     fx, fy = 0.445057*640*scale, 0.59341*480*scale
#     cx, cy = 0.496427*640*scale, 0.519434*480*scale
#     baseline = 0.120018 
#     return get_calib_params(fx, fy, cx, cy, baseline=baseline)

