#!/usr/bin/env python
import argparse
import numpy as np

from pybot.utils.test_utils import test_dataset
from pybot.utils.dataset.kitti import KITTIDatasetReader
from pybot.vision.imshow_utils import imshow_cv
from pybot.geometry.rigid_transform import RigidTransform, Pose

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(
        description='KITTI test dataset')
    parser.add_argument(
        '--velodyne', dest='velodyne', action='store_true',
        help="Process Velodyne data")
    args = parser.parse_args()

    import pybot.externals.draw_utils as draw_utils

    # KITTI params
    dataset = test_dataset(sequence='00', scale=1.0)

    try: 
        # Publish ground truth poses
        draw_utils.publish_pose_list('ground_truth_poses', dataset.poses, frame_id='camera')

        # # Publish line segments
        # pts = np.vstack([map(lambda p: p.tvec, dataset.poses)])
        # draw_utils.publish_line_segments('ground_truth_trace', pts[:-1], pts[1:], frame_id='camera')

        # # Reduce dim. to (x,y,theta)
        # axes = 'szxy' # YRP
        # poses_reduced = map(lambda (roll,pitch,yaw,x,y,z): 
        #                     RigidTransform.from_rpyxyz(0,0,yaw, x, y, z, axes=axes), 
        #                     map(lambda p: p.to_rpyxyz(axes=axes), dataset.poses))
        # draw_utils.publish_pose_list('ground_truth_poses', poses_reduced, frame_id='camera')

    except Exception, e:
        print('Failed to publish poses, {}'.format(e))
        
    # # Iterate through the dataset
    # p_bc = KITTIDatasetReader.camera2body
    # p_bv = KITTIDatasetReader.velodyne2body
    # for idx, f in enumerate(dataset.iterframes()):
    #     # imshow_cv('frame', np.vstack([f.left,f.right]))

    #     # if idx % 20 == 0: 
    #     #     draw_utils.publish_cameras(
    #     #         'poses', [Pose.from_rigid_transform(idx, f.pose)],
    #     #         frame_id='camera', zmax=5,
    #     #         reset=False, draw_faces=True)
    #     draw_utils.publish_pose_list('poses', [Pose.from_rigid_transform(idx, f.pose)], frame_id='camera', reset=False)
    #     # draw_utils.publish_pose_t('CAMERA_POSE', f.pose, frame_id='camera')
    #     print f.pose

        
    #     if args.velodyne:
    #         X = f.velodyne[::10,:3]
    #         # draw_utils.publish_cloud('cloud', X, c='b', frame_id='camera')
    #         draw_utils.publish_cloud('cloud', p_bv * X, c='b', frame_id='poses', element_id=idx)
