import os 
import numpy as np
from itertools import izip, repeat

from bot_vision.camera_utils import get_calib_params
from bot_geometry.rigid_transform import Quaternion, RigidTransform
from bot_utils.dataset_readers import FileReader, ImageDatasetReader, StereoDatasetReader
from bot_utils.db_utils import AttrDict
import bot_externals.lcm.draw_utils as draw_utils

def tsukuba_load_poses(fn): 
    """ Retrieve poses """ 
    P = np.loadtxt(os.path.expanduser(fn), dtype=np.float64, delimiter=',')
    return [ RigidTransform.from_roll_pitch_yaw_x_y_z(
        np.deg2rad(p[3]),-np.deg2rad(p[4]),np.deg2rad(p[5]),
        p[0]*.01,-p[1]*.01,-p[2]*.01, axes='sxyz') for p in P ]

def scaled_calib_params(f, cx, cy, baseline, scale=1.0): 
    f = f * scale
    cx, cy = cx * scale, cy * scale
    return get_calib_params(f, f, cx, cy, baseline=baseline)

class TsukubaStereo2012Reader(object): 
    """
    TsukubaStereo2012Reader: StereoDatasetReader + Calib

    The resolution of the images is 640x480 pixels, the baseline of the stereo 
    camera is 10cm and the focal length of the camera is 615 pixels.

    """

    def __init__(self, directory='NewTsukubaStereoDataset/', 
                 left_template='illumination/daylight/left/tsukuba_daylight_L_%05i.png', 
                 right_template='illumination/daylight/right/tsukuba_daylight_R_%05i.png', 
                 start_idx=1, max_files=50000, scale=1.0): 

        # Set args
        self.scale = scale

        # Get calib
        self.calib = scaled_calib_params(f=615, cx=319.5, cy=239.5, baseline=0.10, scale=scale)

        # Read poses
        try: 
            pose_fn = os.path.join(os.path.expanduser(directory), 'groundtruth/camera_track.txt')
            self.poses = FileReader(pose_fn, process_cb=tsukuba_load_poses)                    
        except Exception as e: 
            self.poses = repeat(None)
            raise RuntimeError('Failed to load poses properly, cannot proceed {:}'.format(e))
        draw_utils.publish_pose_list('POSES', self.poses.items, frame_id='camera')

        # Read stereo images
        self.stereo = StereoDatasetReader(directory=directory,
                                          left_template=left_template, 
                                          right_template=right_template, 
                                          start_idx=start_idx, max_files=max_files, scale=scale)
        print 'Initialized stereo dataset reader with %f scale' % scale
        self.gt = ImageDatasetReader(template=
                                     os.path.join(os.path.expanduser(directory), 
                                                  'groundtruth/disparity_maps/left/tsukuba_disparity_L_%05i.png'), 
                                     start_idx=start_idx, max_files=max_files, scale=scale)
        
    def iter_frames(self, *args, **kwargs): 
        for (left, right), pose, depth in izip(self.iter_stereo_frames(*args, **kwargs), 
                                               self.poses.iteritems(*args, **kwargs), 
                                               self.gt.iteritems(*args, **kwargs)): 
            yield AttrDict(left=left, right=right, velodyne=None, pose=pose, depth=depth)

    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo.iteritems(*args, **kwargs)

    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()

if __name__ == "__main__": 
    from bot_vision.imshow_utils import imshow_cv
    from bot_vision.image_utils import to_gray

    dataset = TsukubaStereo2012Reader(directory='~/HD1/data/NewTsukubaStereoDataset/')
    for f in dataset.iter_frames():
        lim, rim = to_gray(f.left), to_gray(f.right)
        out = np.dstack([np.zeros_like(lim), lim, rim])
        imshow_cv('left/right', out)
        imshow_cv('disp', f.depth)
