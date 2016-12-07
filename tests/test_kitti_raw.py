#!/usr/bin/env python
import numpy as np
import argparse

# from pybot.utils.test_utils import test_dataset
# from pybot.vision.imshow_utils import imshow_cv
# from pybot.geometry.rigid_transform import RigidTransform

from pybot.geometry.rigid_transform import RigidTransform, Pose
from pybot.utils.dataset.kitti import KITTIRawDatasetReader
import pybot.externals.lcm.draw_utils as draw_utils

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description='Extract KLT features from video')
    parser.add_argument(
        '-d', '--directory', type=str, 
        default='', required=True, 
        help="Directory (e.g. KITTI directory with poses,sequences)")
    args = parser.parse_args()
    
    # KITTI params
    dataset = KITTIRawDatasetReader(directory=args.directory,
                                    left_template=None, right_template=None, velodyne_template=None)
    
    # print dataset.oxts_fieldnames, dataset.oxts_data
    # draw_utils.publish_pose_list('ground_truth_poses',
    #                              [Pose.from_rigid_transform(0, dataset.poses[0])],
    #                              frame_id='origin', reset=False)

    draw_utils.publish_pose_list('ground_truth_poses',
                                 [Pose.from_rigid_transform(idx, p) for idx, p in enumerate(dataset.poses)],
                                 frame_id='origin', reset=False)
    
    # for idx, f in enumerate(dataset.iterframes()):
    #     draw_utils.publish_pose_list('ground_truth_poses', [Pose.from_rigid_transform(idx, f.pose)], frame_id='origin', reset=False)
    #     # draw_utils.publish_cameras('cameras', [f.pose], reset=False, frame_id='origin')
