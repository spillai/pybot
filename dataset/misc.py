import os
import numpy as np
import cv2

from itertools import izip
from bot_geometry.rigid_transform import RigidTransform
from bot_utils.db_utils import AttrDict
from bot_utils.dataset_readers import natural_sort, \
    read_dir, DatasetReader, ImageDatasetReader, StereoDatasetReader, VelodyneDatasetReader

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

    def iter_frames(self, *args, **kwargs): 
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
