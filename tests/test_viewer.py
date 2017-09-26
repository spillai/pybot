import time
import numpy as np

from pybot.geometry import RigidTransform, Pose
import pybot.externals.draw_utils as draw_utils

# =================================================
print('Publish cloud in origin reference')
X = np.random.rand(1000, 3) * 4
C = np.hstack((np.random.rand(1000, 1),np.zeros((1000,2))))

# =================================================
draw_utils.publish_cloud('cloud', X, c='b', frame_id='origin')
draw_utils.publish_line_segments('lines', X[:-1], X[1:], c='r', frame_id='origin')

# =================================================
print('Publish cloud in camera reference')
X = np.random.rand(1000,3) * 4
draw_utils.publish_cloud('cloud', X, c='g', frame_id='camera')

# =================================================
print('Create 3 poses with ids 0, 1, and 2')
p = Pose.from_rigid_transform(0, RigidTransform(tvec=[1,0,0]))
draw_utils.publish_pose_list('poses', [p], frame_id='origin', reset=False)
p = Pose.from_rigid_transform(1, RigidTransform(tvec=[2,0,0]))
draw_utils.publish_pose_list('poses', [p], frame_id='origin', reset=False)
p = Pose.from_rigid_transform(2, RigidTransform(tvec=[3,0,0]))
draw_utils.publish_pose_list('poses', [p], frame_id='origin', reset=False)

# =================================================
print('Publish cloud with respect to pose 0')
Xs = [X + 3, X + 4, X + 6]
ids = [0, 1, 2]
draw_utils.publish_cloud('cloud_with_poses', Xs, c=[C, 'r', 'g'], frame_id='poses', element_id=ids)
for j in range(3):
    time.sleep(1)
    p = Pose.from_rigid_transform(0, RigidTransform(tvec=[1,j,0]))
    draw_utils.publish_pose_list('poses', [p], frame_id='origin', reset=False)


# =================================================
print('Publish random poses')
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

# =================================================    
print('Publish cameras')
draw_utils.publish_cameras('cams', poses,
                           draw_faces=False, draw_edges=True, zmax=0.7,
                           frame_id='camera')

# =================================================    
print('Move camera')
cpose = draw_utils.get_sensor_pose(frame_id='camera')
poses.extend([cpose])
for p in poses:
    time.sleep(1)
    print('moving camera')
    draw_utils.publish_pose_t('CAMERA_POSE', p, frame_id='origin')

