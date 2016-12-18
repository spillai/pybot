#!/usr/bin/env python
import numpy as np
import argparse

from pybot.geometry.rigid_transform import RigidTransform, Pose
from pybot.utils.dataset.misc import RPGUrban
import pybot.externals.lcm.draw_utils as draw_utils


from pybot.vision.image_utils import to_color
from pybot.vision.imshow_utils import imshow_cv

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description='Extract KLT features from video')
    parser.add_argument(
        '-d', '--directory', type=str, 
        default='', required=True, 
        help="Directory (e.g. KITTI directory with poses,sequences)")
    args = parser.parse_args()
    
    # KITTI params
    dataset = RPGUrban(directory=args.directory)
    # print 'Poses: ', len(dataset.poses)
    
    for idx, f in enumerate(dataset.iterframes()):
        print f.pose, f.im.shape
        imshow_cv('Image/Alpha', np.hstack([f.im[:,:,:3], to_color(f.im[:,:,-1])]))
        draw_utils.publish_pose_list('ground_truth_poses', [Pose.from_rigid_transform(idx, f.pose)], frame_id='camera', reset=False)

