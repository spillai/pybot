# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os 
import numpy as np
from itertools import izip

from ..io_utils import path_exists
from ..dataset_readers import natural_sort, \
    read_dir, DatasetReader, ImageDatasetReader, StereoDatasetReader
from ..db_utils import AttrDict

from ...geometry.rigid_transform import Quaternion, RigidTransform

def load_poses(fn): 
    """ Retrieve poses """ 
    P = np.loadtxt(os.path.expanduser(fn), dtype=np.float64)
    P1 = P.reshape(-1,12)
    P = [Rt.reshape(3,4) for Rt in P1]
    return [RigidTransform.from_Rt(item[:3,:3], item[:3,3]) for item in P]


def save_poses(fn, poses): 
    """ Save poses in toon format """ 
    Rts = [pose.matrix[:3,:] for pose in poses]
    with file(fn, 'w') as outfile:
        for Rt in Rts:
            for row in Rt:
                np.savetxt(outfile, row, fmt='%-8.7f', delimiter=' ', newline=' ')
                outfile.write('\n')
            outfile.write('\n')
    return 


class StereoPOVDatasetReader(object): 
    """
    StereoPOVDatasetReader: StereoDatasetReader + Calib
    """

    def __init__(self, directory='', 
                 left_template='image_0/scene_00_%07i.png', 
                 right_template='image_1/scene_00_%07i.png', 
                 start_idx=0, max_files=50000, scale=1.0): 

        # Set args
        self.scale = scale

        # Get calib
        # self.calib = kitti_stereo_calib_params(scale=scale)

        # Read poses
        try: 
            pose_fn = os.path.join(os.path.expanduser(directory), 'poses.txt')
            self.poses = load_poses(pose_fn)
        except Exception as e: 
            raise RuntimeError('Failed to load poses properly, cannot proceed')
        

        # Read stereo images
        self.stereo = StereoDatasetReader(directory=directory,
                                          left_template=left_template, 
                                          right_template=right_template, 
                                          start_idx=start_idx, max_files=max_files, scale=scale)
        print 'Initialized stereo dataset reader with %f scale' % scale
    

    def iterframes(self, *args, **kwargs): 
        for (left, right), pose in izip(self.iter_stereo_frames(*args, **kwargs), self.poses): 
            yield AttrDict(left=left, right=right, velodyne=None, pose=pose)

    def iter_stereo_frames(self, *args, **kwargs): 
        return self.stereo.iteritems(*args, **kwargs)

    @property
    def stereo_frames(self): 
        return self.iter_stereo_frames()

if __name__ == "__main__": 
    # poses = load_poses('/home/spillai/data/stereo-poses/livingroom1_poses_0_interpolated.txt')
    # save_poses('test.txt', poses)

    from pybot.vision.imshow_utils import imshow_cv
    from pybot.vision.image_utils import to_gray

    dataset = StereoPOVDatasetReader(directory='~/data/stereo-poses/livingroom20_data')
    for f in dataset.iterframes():
        lim, rim = to_gray(f.left), to_gray(f.right)
        out = np.dstack([np.zeros_like(lim), lim, rim])
        imshow_cv('left/right', out)
