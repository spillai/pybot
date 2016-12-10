#!/usr/bin/env python
import numpy as np
import argparse

from pybot.utils.dataset.kitti import OmnicamDatasetReader
from pybot.vision.imshow_utils import imshow_cv
from pybot.geometry.rigid_transform import Pose, RigidTransform
import pybot.externals.draw_utils as draw_utils

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(
        description='Extract KLT features from video')
    parser.add_argument(
        '-d', '--directory', type=str, 
        default='', required=True, 
        help="Directory (e.g. KITTI directory with poses,sequences)")
    args = parser.parse_args()

    # KITTI params
    dataset = OmnicamDatasetReader(directory=args.directory)
    for idx, oxts in enumerate(dataset.oxts.iteritems()):
        print(f.left.shape)
        draw_utils.publish_pose_list('ground_truth_poses', [Pose.from_rigid_transform(idx, oxts.pose)], 
                                     frame_id='origin', reset=False)

    # for idx, f in enumerate(dataset.iterframes()):
    #     # print f.left.shape
        
    #     # imshow_cv('frame', np.vstack([l,r]))
    #     # imshow_cv('frame', f.left / 2 + f.right / 2)
        
    #     # Publish ground truth poses
    #     draw_utils.publish_pose_list('ground_truth_poses', [Pose.from_rigid_transform(idx, f.pose)],
    #                                  frame_id='origin', reset=False)
        
