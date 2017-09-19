import time
import numpy as np

from pybot.geometry.rigid_transform import RigidTransform, Pose
import pybot.externals.draw_utils as draw_utils

if __name__ == "__main__":
    draw_utils.publish_sensor_frame('new_frame', RigidTransform(tvec=[0,0,1]))
    # draw_utils.publish_sensor_frame('new_frame_with_ids', RigidTransform(tvec=[0,0,1]))

    # poses = [RigidTransform.from_roll_pitch_yaw_x_y_z(np.pi/180 * 10*j, 0, 0, j * 1.0, 0, 0)
    #          for j in xrange(10)]
    # draw_utils.publish_pose_list('poses', poses, frame_id='origin')
    # draw_utils.publish_pose_list('poses_new_frame', poses, frame_id='new_frame')

    # Create cloud
    X = np.random.rand(1000, 3)
    C = np.hstack((np.random.rand(1000, 1),np.zeros((1000,2))))

    draw_utils.publish_cloud('cloud', X, c='b', frame_id='origin')
    draw_utils.publish_line_segments('lines', X[:-1], X[1:], c='r', frame_id='origin')

    # Create 3 poses with ids 0, 1, and 2
    p = Pose.from_rigid_transform(0, RigidTransform(tvec=[1,0,0]))
    draw_utils.publish_pose_list('poses', [p], frame_id='origin', reset=False)
    p = Pose.from_rigid_transform(1, RigidTransform(tvec=[2,0,0]))
    draw_utils.publish_pose_list('poses', [p], frame_id='origin', reset=False)
    p = Pose.from_rigid_transform(2, RigidTransform(tvec=[3,0,0]))
    draw_utils.publish_pose_list('poses', [p], frame_id='origin', reset=False)

    # # Publish cloud with respect to pose 0
    # draw_utils.publish_cloud('cloud', X, c='b', frame_id='poses', element_id=0)

    # for j in range(3):
    #     time.sleep(1)
    #     p = Pose.from_rigid_transform(0, RigidTransform(tvec=[1,j,0]))
    #     draw_utils.publish_pose_list('poses', [p], frame_id='origin', reset=False)

    # Publish cloud with respect to pose 0
    Xs = [X + 3, X + 4, X + 6]
    ids = [0, 1, 2]
    draw_utils.publish_cloud('cloud_with_poses', Xs, c=[C, 'r', 'g'], frame_id='poses', element_id=ids)
    for j in range(3):
        time.sleep(1)
        p = Pose.from_rigid_transform(0, RigidTransform(tvec=[1,j,0]))
        draw_utils.publish_pose_list('poses', [p],
                                     frame_id='origin', reset=False)

    # draw_utils.publish_cloud('cloud', Xs, c='b', frame_id='poses', element_id=ids)



    # for j in range(10):
    #     p = Pose.from_rigid_transform(j,
    #                                   RigidTransform.from_roll_pitch_yaw_x_y_z(
    #             np.pi/180 * 10*j, 0, 0, j * 1.0, 0, 0))
    #     draw_utils.publish_pose_list('poses_new_frame_with_ids', [p],
    #                       frame_id='new_frame_with_ids', reset=False)
    #     time.sleep(1)
