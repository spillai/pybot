#!/usr/bin/env python
import numpy as np

from pybot.utils.test_utils import test_dataset
from pybot.vision.imshow_utils import imshow_cv
from pybot.geometry.rigid_transform import RigidTransform

if __name__ == "__main__": 

    # try: 
    import pybot.externals.draw_utils as draw_utils
    #     draw_utils.init()
    # except: 
    #     import pybot.vision.draw_utils as draw_utils
        

    # KITTI params
    dataset = test_dataset(scale=1.0)

    for l,r in dataset.iter_stereo_frames(): 
        print l.shape
        imshow_cv('frame', np.vstack([l,r]))

    # Publish ground truth poses
    draw_utils.publish_pose_list('ground_truth_poses', dataset.poses, frame_id='camera',size=3.0)

    # Publish line segments
    pts = np.vstack([map(lambda p: p.tvec, dataset.poses)])
    draw_utils.publish_line_segments('ground_truth_trace', pts[:-1], pts[1:], frame_id='camera', size=1.0)

    # Reduce dim. to (x,y,theta)
    axes = 'szxy' # YRP
    poses_reduced = map(lambda (roll,pitch,yaw,x,y,z): 
                        RigidTransform.from_rpyxyz(0,0,yaw, x, y, z, axes=axes), 
                        map(lambda p: p.to_rpyxyz(axes=axes), dataset.poses))
    draw_utils.publish_pose_list('ground_truth_poses', poses_reduced, frame_id='camera',size=3.0)

    # for left_im, right_im in dataset.iter_stereo_frames(): 
    #     imshow_cv('left', left_im)


    # for pc in dataset.iter_velodyne_frames(): 
    #     X = pc[:,:3]
        
