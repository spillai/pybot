#!/usr/bin/env python
"""
Simple exmaple of KITTI dataset reader with WebGL viewing
capabilities.

Usage:
$ python kitti.py -d ~/data/dataset/ -s 00 --velodyne
"""
import time

import argparse
import numpy as np

from pybot.geometry.rigid_transform import RigidTransform, Pose
from pybot.utils.dataset.kitti import KITTIDatasetReader
from pybot.utils.timer import SimpleTimer
from pybot.vision.imshow_utils import imshow_cv

import pybot.externals.draw_utils as draw_utils

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description='KITTI test dataset')
    parser.add_argument(
        '-s', '--sequence', type=int,
        required=True, help='KITTI sequence')
    parser.add_argument(
        '-d', '--dir', dest='directory',
        required=True, help='KITTI dataset directory')
    parser.add_argument(
        '--velodyne', dest='velodyne', action='store_true',
        help='Process Velodyne data')
    parser.add_argument(
        '--publish-ground-truth-poses', action='store_true',
        help='Publish ground truth poses ahead of time')
    parser.add_argument(
        '-v', '--publish-velodyne-every', 
        type=int, required=False, default=5, 
        help='Publish Velodyne every k frames')
    parser.add_argument(
        '-k', '--publish-camera-every', 
        type=int, required=False, default=5, 
        help='Publish camera pose every k frames')
    args = parser.parse_args()

    # Read KITTI Dataset
    if not args.velodyne:
        kwargs = dict(velodyne_template=None)
    else:
        kwargs = dict()
    dataset = KITTIDatasetReader(directory=args.directory,
                                 sequence='{:02d}'.format(args.sequence),
                                 scale=1.0, **kwargs)

    # Recover camera to velodyne extrinsics
    p_bc, p_bv = dataset.p_bc, dataset.p_bv
    p_cv = (p_bc.inverse() * p_bv).inverse()

    
    # Optionally, publish ground truth poses
    if args.publish_ground_truth_poses:
        try: 
            # Publish ground truth poses
            draw_utils.publish_pose_list('ground_truth_poses',
                                         dataset.poses, frame_id='camera')
            # Publish line segments
            pts = np.vstack([map(lambda p: p.tvec, dataset.poses)])
            draw_utils.publish_line_segments('ground_truth_trace',
                                             pts[:-1], pts[1:],
                                             c='y', frame_id='camera')
        except Exception as e:
            print('Failed to publish poses, {}'.format(e))
    
    # Iterate through frames
    timer = SimpleTimer('KITTI-example')
    for idx, f in enumerate(dataset.iterframes()):
        timer.poll()
        # imshow_cv('frame', np.vstack([f.left,f.right]))

        # Publish keyframes every 5 frames
        if idx % args.publish_velodyne_every == 0: 
            draw_utils.publish_cameras(
                'cam_poses', [Pose.from_rigid_transform(idx, f.pose)],
                frame_id='camera', zmax=2,
                reset=False, draw_faces=False, draw_edges=True)

        # Publish pose 
        draw_utils.publish_pose_list(
            'poses', [Pose.from_rigid_transform(idx, f.pose)],
            frame_id='camera', reset=False)

        # Move camera viewpoint 
        draw_utils.publish_pose_t('CAMERA_POSE', f.pose,
                                  frame_id='camera')

        # Velodyne publish
        VELODYNE_SAMPLING = 4
        if args.velodyne and idx % args.publish_velodyne_every == 0:
            # Collect velodyne point clouds, and color by height.
            # Heightmap is computed on the z-axis
            X_v = f.velodyne[::VELODYNE_SAMPLING,:3]
            carr = draw_utils.height_map(X_v[:,2], hmin=-2, hmax=4)

            # Convert velodyne cloud to camera's reference frame
            X_c = p_cv * X_v
            draw_utils.publish_cloud('cloud', X_c,
                                     c=carr, frame_id='poses',
                                     element_id=idx, reset=False)
            
