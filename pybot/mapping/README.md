BaseSLAM
===
PYBOT_BACKEND environment variable
export PYBOT_BACKEND=gtsam

Basic interface to GTSAM / ISAM

    BaseSLAM.add_odom_incremental(delta)
    BaseSLAM._add_odom(xid1, xid2, delta)

    BaseSLAM.add_pose_landmarks(xid, lids, deltas)
    BaseSLAM.add_pose_landmarks_incremental(lids, pts, pts3d)

    BaseSLAM.add_point_landmarks(xid, lids, pts, pts3d)
    BaseSLAM.add_point_landmarks_incremental(lids, pts, pts3d)

    BaseSLAM.update()
    BaseSLAM.update_marginals()
    BaseSLAM.cleanup()
