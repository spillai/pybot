#!/usr/bin/env python
import numpy as np

from pybot.utils.test_utils import test_dataset
from pybot.vision.imshow_utils import imshow_cv
from pybot.geometry.rigid_transform import RigidTransform
from pybot.utils.dataset.kitti import KITTIRawDatasetReader

if __name__ == "__main__": 

    import pybot.externals.lcm.draw_utils as draw_utils

    # KITTI params
    dataset = KITTIRawDatasetReader(directory='/HD1/data/KITTI/data_raw/2011_09_26/2011_09_26_drive_0059_sync')
    print dataset.oxt_fieldnames

    poses = dataset.poses
    print len(poses)
    
    # for f in dataset.iterframes():
    #     print f.pose
    #     imshow_cv('frame', np.vstack([f.left, f.right]))
        

    # # Publish ground truth poses
    # draw_utils.publish_pose_list('ground_truth_poses', dataset.poses, frame_id='camera',size=3.0)

    # # Publish line segments
    # pts = np.vstack([map(lambda p: p.tvec, dataset.poses)])
    # draw_utils.publish_line_segments('ground_truth_trace', pts[:-1], pts[1:], frame_id='camera', size=1.0)

    # # Reduce dim. to (x,y,theta)
    # axes = 'szxy' # YRP
    # poses_reduced = map(lambda (roll,pitch,yaw,x,y,z): 
    #                     RigidTransform.from_rpyxyz(0,0,yaw, x, y, z, axes=axes), 
    #                     map(lambda p: p.to_rpyxyz(axes=axes), dataset.poses))
    # draw_utils.publish_pose_list('ground_truth_poses', poses_reduced, frame_id='camera',size=3.0)
        
