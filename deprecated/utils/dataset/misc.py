# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import numpy as np
import cv2

from pybot.utils.itertools_recipes import izip
from pybot.geometry.rigid_transform import RigidTransform
from pybot.utils.db_utils import AttrDict
from pybot.utils.dataset_readers import natural_sort, \
    read_dir, NoneReader, FileReader, DatasetReader, ImageDatasetReader, StereoDatasetReader, VelodyneDatasetReader
from pybot.vision.camera_utils import Camera, CameraIntrinsic

class VaFRICDatasetReader(object): 
    def __init__(self, directory='', scene=''): 

        # Read images
        # nfiles = number_of_files(os.path.expanduser(directory))
        # files = [os.path.join(directory, 'scene_%02i_%04i.png' % (idx/1000, idx%1000)) for idx in np.arange(nfiles)]

        path = os.path.join(directory, '%s_images_archieve' % scene)
        img_files = read_dir(os.path.expanduser(path), pattern='*.png', recursive=False, verbose=False, flatten=True)
        gt_files = map(lambda fn: fn.replace('%s_images' % scene, '%s_GT' % scene).replace('.png', '.txt'), img_files)

        def process_gt_cb(fn): 
            data = AttrDict()
            with open(fn) as f: 
                for l in f: 
                    key = l[0:l.find('=')].replace(' ', '')
                    if l.find('[') < 0: 
                        val = float(l[l.find('=')+1:l.find(';')].replace(' ', ''))
                    else: 
                        val = l[l.find('[')+1:l.find(']')]
                        val = np.array(map(lambda v: float(v), val.split(',')))
                    data[key] = val
            return data

        self.rgb = ImageDatasetReader.from_filenames(img_files)
        self.gt = DatasetReader.from_filenames(files=gt_files, process_cb=process_gt_cb)

    @staticmethod
    def read_pose(gt): 
        cam_dir, cam_up = gt.cam_dir, gt.cam_up
        z = cam_dir / np.linalg.norm(cam_dir)
        x = np.cross(cam_up, z)
        y = np.cross(z, x)

        R = np.vstack([x, y, z]).T
        t = gt.cam_pos / 1000.0
        return RigidTransform.from_Rt(R, t)

    def iteritems(self, *args, **kwargs): 
        for rgb_im, gt_info in izip(self.rgb.iteritems(*args, **kwargs), self.gt.iteritems(*args, **kwargs)): 
            yield AttrDict(img=rgb_im, gt=gt_info, pose=VaFRICDatasetReader.read_pose(gt_info))

    def iterframes(self, *args, **kwargs): 
        return self.rgb.iteritems(*args, **kwargs)

class NewCollegeDatasetReader(object): 
    """
    StereoImageDatasetReader + Calib
    """

    def __init__(self, directory='', 
                 left_template='*-left.pnm', 
                 right_template='*-right.pnm', max_files=50000, scale=1.0): 

        # Set args
        self.scale = scale

        # # Get calib
        # self.calib = kitti_stereo_calib_params(scale=scale)

        # Read stereo images
        left_files = read_dir(os.path.expanduser(directory), pattern='*-left.pnm', 
                                   recursive=False, verbose=False, flatten=True)
        print left_files[0][0]
        right_files = read_dir(os.path.expanduser(directory), pattern='*-right.pnm', 
                                    recursive=False, verbose=False, flatten=True)
        self.stereo = StereoDatasetReader.from_filenames(left_files, right_files)

        print 'Initialized stereo dataset reader with %f scale' % scale

    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo.iteritems(*args, **kwargs)

    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()


class RPGUrban(object): 
    """
    RPGUrban: ImageDatasetReader + Calib
    shape: H x W
    """
    shape = (480,640,3)
    def __init__(self, directory='', 
                 template='data/img/img%04i_0.png', 
                 velodyne_template=None, 
                 start_idx=1, max_files=100000, scale=1.0): 

        # Set args
        self.scale = scale

        # Calibration
        try: 
            calib_fn = os.path.join(os.path.expanduser(directory), 'info', 'intrinsics.txt')
            s = open(calib_fn, 'r').readlines()
            s = ''.join(s[1:])
            for item in ['\n', '[', ']', ' ']:
                s = s.replace(item,'')
            K = np.fromstring(s, sep=',').reshape(3,3)
            self.calib_ = Camera.from_intrinsics(CameraIntrinsic(K, shape=RPGUrban.shape[:2])).scaled(scale)
        except Exception,e:
            print('Failed to read calibration data: {}'.format(e))
            self.calib_ = None
        
        # Read stereo images
        try: 
            self.rgb_ = ImageDatasetReader(template=os.path.join(os.path.expanduser(directory), template), 
                                           start_idx=start_idx, max_files=max_files, scale=scale)
        except Exception,e:
            print('Failed to read rgb data: {}'.format(e))
            self.rgb_ = NoneReader()
        
        # Read poses
        def load_poses(fn):
            """
            poses: image_id tx ty tz qx qy qz qw
            """
            X = (np.fromfile(fn, dtype=np.float64, sep=' ')).reshape(-1,8)
            abs_poses = map(lambda p: RigidTransform(xyzw=p[4:], tvec=p[1:4]), X)
            rel_poses = [abs_poses[0].inverse() * p for p in abs_poses]
            return rel_poses

        try:
            pose_fn = os.path.join(os.path.expanduser(directory), 'info', 'groundtruth.txt')
            self.poses_ = FileReader(pose_fn, process_cb=load_poses)
        except Exception as e:
            self.poses_ = NoneReader()

    @property
    def calib(self):
        return self.calib_
        
    @property
    def rgb(self):
        return self.rgb_

    def iterframes(self, *args, **kwargs): 
        for im, pose in izip(self.iter_rgb_frames(*args, **kwargs), 
                             self.iter_poses(*args, **kwargs)): 
            yield AttrDict(img=im[:,:,:3], mask=im[:,:,-1], velodyne=None, pose=pose)

    def iter_poses(self, *args, **kwargs): 
        return self.poses_.iteritems(*args, **kwargs)
    
    def iter_rgb_frames(self, *args, **kwargs): 
        return self.rgb.iteritems(*args, **kwargs)

    @property
    def poses(self):
        return list(self.poses_.iteritems())

    
