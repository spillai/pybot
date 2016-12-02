#!/usr/bin/env python
import numpy as np

from pybot.utils.dataset.kitti import OmnicamDatasetReader
from pybot.vision.imshow_utils import imshow_cv
from pybot.geometry.rigid_transform import Pose, RigidTransform

if __name__ == "__main__": 

    # try: 
    import pybot.externals.draw_utils as draw_utils

    # KITTI params
    dataset = OmnicamDatasetReader(directory='/media/spillai/MRG-HD1/data/omnidirectional/', max_files=1000, scale=0.25)

    for idx, f in enumerate(dataset.iterframes()):
        
        # imshow_cv('frame', np.vstack([l,r]))
        imshow_cv('frame', f.left / 2 + f.right / 2)
        
        # Publish ground truth poses
        draw_utils.publish_pose_list('ground_truth_poses', [Pose.from_rigid_transform(idx, f.pose)],
                                     frame_id='origin', reset=False)

    # # Publish line segments
    # pts = np.vstack([map(lambda p: p.tvec, dataset.poses)])
    # draw_utils.publish_line_segments('ground_truth_trace', pts[:-1], pts[1:], frame_id='camera', size=1.0)

    # # Reduce dim. to (x,y,theta)
    # axes = 'szxy' # YRP
    # poses_reduced = map(lambda (roll,pitch,yaw,x,y,z): 
    #                     RigidTransform.from_rpyxyz(0,0,yaw, x, y, z, axes=axes), 
    #                     map(lambda p: p.to_rpyxyz(axes=axes), dataset.poses))
    # draw_utils.publish_pose_list('ground_truth_poses', poses_reduced, frame_id='camera',size=3.0)

    # for left_im, right_im in dataset.iter_stereo_frames(): 
    #     imshow_cv('left', left_im)


    # for pc in dataset.iter_velodyne_frames(): 
    #     X = pc[:,:3]
        
