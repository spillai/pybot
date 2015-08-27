import os
import numpy as np
import cv2

from itertools import izip
from bot_utils.db_utils import AttrDict
from bot_utils.dataset_readers import natural_sort, \
    StereoDatasetReader, VelodyneDatasetReader

from .kitti_helpers import kitti_stereo_calib_params, kitti_load_poses

class KITTIDatasetReader(object): 
    """
    KITTIDatasetReader: ImageDatasetReader + VelodyneDatasetReader + Calib
    """

    def __init__(self, directory='', 
                 sequence='', 
                 left_template='image_0/%06i.png', 
                 right_template='image_1/%06i.png', 
                 velodyne_template='velodyne/%06i.bin',
                 start_idx=0, max_files=50000, scale=1.0): 

        # Set args
        self.sequence = sequence
        self.scale = scale

        # Get calib
        self.calib = kitti_stereo_calib_params(scale=scale)

        # Read poses
        try: 
            pose_fn = os.path.join(os.path.expanduser(directory), 'poses', ''.join([sequence, '.txt']))
            self.poses = kitti_load_poses(fn=pose_fn)
        except: 
            pass

        # Read stereo images
        seq_directory = os.path.join(os.path.expanduser(directory), 'sequences', sequence)
        self.stereo = StereoDatasetReader(directory=seq_directory, 
                                          left_template=os.path.join(seq_directory,left_template), 
                                          right_template=os.path.join(seq_directory,right_template), 
                                          start_idx=start_idx, max_files=max_files, scale=scale)

        # Read velodyne
        self.velodyne = VelodyneDatasetReader(
            template=os.path.join(seq_directory,velodyne_template), 
            start_idx=start_idx, max_files=max_files
        )

        print 'Initialized stereo dataset reader with %f scale' % scale

    def iteritems(self, *args, **kwargs): 
        return self.stereo.left.iteritems(*args, **kwargs)

    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo.iteritems(*args, **kwargs)

    def iter_velodyne_frames(self, *args, **kwargs):         
        return self.velodyne.iteritems(*args, **kwargs)

    def iter_stereo_velodyne_frames(self, *args, **kwargs):         
        return izip(self.left.iteritems(*args, **kwargs), 
                    self.right.iteritems(*args, **kwargs), 
                    self.velodyne.iteritems(*args, **kwargs))

    def iter_frames(self, *args, **kwargs): 
        for (left, right), pose in izip(self.iter_stereo_frames(*args, **kwargs), self.poses): 
            yield AttrDict(left=left, right=right, velodyne=None, pose=pose)

    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()

    @property
    def velodyne_frames(self): 
        return self.iter_velodyne_frames()


class OmnicamDatasetReader(object): 
    """
    OmnicamDatasetReader: ImageDatasetReader + VelodyneDatasetReader + Calib
    """

    def __init__(self, directory='', 
                 sequence='2013_05_14_drive_0008_sync', 
                 left_template='image_02/data/%010i.png', 
                 right_template='image_03/data/%010i.png', 
                 velodyne_template='velodyne_points/data/%010i.bin',
                 start_idx=0, max_files=50000, scale=1.0): 

        # Set args
        self.sequence = sequence
        self.scale = scale

        # Get calib
        # self.calib = kitti_stereo_calib_params(scale=scale)

        # # Read poses
        # try: 
        #     pose_fn = os.path.join(os.path.expanduser(directory), 'poses', ''.join([sequence, '.txt']))
        #     self.poses = kitti_load_poses(fn=pose_fn)
        # except: 
        #     pass

        # Read stereo images
        seq_directory = os.path.join(os.path.expanduser(directory), sequence)
        print os.path.join(seq_directory, left_template)
        self.stereo = StereoDatasetReader(directory=seq_directory,
                                          left_template=os.path.join(seq_directory,left_template), 
                                          right_template=os.path.join(seq_directory,right_template), 
                                          start_idx=start_idx, max_files=max_files, scale=scale)

        # Read velodyne
        self.velodyne = VelodyneDatasetReader(
            template=os.path.join(seq_directory,velodyne_template), 
            start_idx=start_idx, max_files=max_files
        )

        print 'Initialized stereo dataset reader with %f scale' % scale

    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo.iteritems(*args, **kwargs)

    def iter_velodyne_frames(self, *args, **kwargs):         
        return self.velodyne.iteritems(*args, **kwargs)

    def iter_stereo_velodyne_frames(self, *args, **kwargs):         
        return izip(self.left.iteritems(*args, **kwargs), 
                    self.right.iteritems(*args, **kwargs), 
                    self.velodyne.iteritems(*args, **kwargs))

    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()

    @property
    def velodyne_frames(self): 
        return self.iter_velodyne_frames()

# def test_omnicam(dire):
#     return OmnicamDatasetReader(directory='/media/spillai/MRG-HD1/data/omnidirectional/')


