import numpy as np

from pybot.geometry import RigidTransform, Pose
import pybot.externals.draw_utils as draw_utils

X = np.random.rand(1000,3) * 20
draw_utils.publish_cloud('cloud', X, c='g', frame_id='camera')

poses = []
for j in range(10):
    p = RigidTransform.from_rpyxyz(np.random.rand() * 10,
                                   np.random.rand() * 10,
                                   np.random.rand() * 10,
                                   np.random.rand() * 10,
                                   np.random.rand() * 10,
                                   np.random.rand() * 10)
    if len(poses):
        p = poses[-1].oplus(p)
    poses.append(Pose.from_rigid_transform(j, p))        

draw_utils.publish_cameras('cams', poses, draw_faces=True, draw_edges=True)


