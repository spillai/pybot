import time
import numpy as np

from pybot.geometry import RigidTransform, Pose
import pybot.externals.draw_utils as draw_utils

X = np.random.rand(1000,3) * 4
draw_utils.publish_cloud('cloud', X, c='g', frame_id='camera')

poses = []
for j in range(10):
    p = RigidTransform.from_rpyxyz(np.random.rand() * 1,
                                   np.random.rand() * 1,
                                   np.random.rand() * 1,
                                   np.random.rand() * 1,
                                   np.random.rand() * 1,
                                   np.random.rand() * 1)
    if len(poses):
        p = poses[-1].oplus(p)
    poses.append(Pose.from_rigid_transform(j, p))        

draw_utils.publish_cameras('cams', poses,
                           draw_faces=False, draw_edges=True, zmax=0.7,
                           frame_id='camera')

cpose = draw_utils.get_sensor_pose(frame_id='camera')
poses.extend([cpose])
for p in poses:
    time.sleep(1)
    print('moving camera')
    draw_utils.publish_pose_t('CAMERA_POSE', p, frame_id='origin')

