import time
import numpy as np

from bot_vision.image_utils import to_color, to_gray
from bot_vision.imshow_utils import imshow_cv

from bot_utils.misc import Accumulator, PoseAccumulator, SkippedCounter, CounterWithPeriodicCallback
from bot_geometry.rigid_transform import RigidTransform, Pose, Quaternion

import bot_externals.draw_utils as draw_utils

from .gtsam import BaseSLAM, SLAM3D
from pybot_gtsam import GTSAMTags
from pybot_slam import ISAMTags, draw_tags
from pybot_apriltags import AprilTag, AprilTagsWrapper

class BaseSLAMMixin(object):
    """
    Basic Mixin SLAM
    
       All measurements are to be provided in the 
       same reference frame 
    
    """
    def __init__(self, ref_frames={}, absolute_measurements=True, visualize=True): 

        # Initialize SLAM (GTSAM / ISAM)
        # self.slam_ = GTSAMTags() # FIX: TODO
        # self.slam_ = BaseSLAM()
        self.slam_ = SLAM3D()


        # self.slam_cb_ = CounterWithPeriodicCallback(
        #     every_k=10, 
        #     process_cb=lambda:  self.slam_.save_graph("slam_fg.dot")
        # )
        # self.slam_cb_.register_callback(self.slam_, 'on_odom')

        # Poses (request for relative measurements if the odometry is absolute)
        self.pose_id_ = -1
        self.poses_ = PoseAccumulator(maxlen=1000, relative=absolute_measurements)
        self.reset_required = lambda: self.poses_.length == 0
        print 'reset: ', self.reset_required()

        # Updated ids
        self.updated_ids_ = set()

        # Visualization: Reference names
        self.vis_name_ = 'slam_vis'
        self.ref_frames_ = {}
        
        # Visualization: Publish reference frames
        for k, v in ref_frames.iteritems(): 
            draw_utils.publish_sensor_frame(k, v)

    def update(self, iterations=10): 
        # Finalize updates
        for j in range(iterations): 
            self.slam_.update()
        self.vis_slam_updates()
        self.slam_.save_graph("slam_fg.dot")

    def on_odom(self, t, p): 
        """
        Accumulate poses (input: absolute odometry) and add relative odometry to 
        factor graph with appropriate timestamp
        """
        self.poses_.accumulate(p) 
        
        if self.poses_.length < 2:
            return
            
        p_odom = (self.poses_.items[-2].inverse()).oplus(self.poses_.items[-1])
        self.pose_id_ = self.slam_.on_odom(t, p_odom.to_homogeneous_matrix())

        self.vis_odom(self.poses_)        
        self.vis_slam_updates()

        return self.pose_id_

    def on_odom_relative(self, t, p): 
        """
        Accumulate componded pose (input: relative odometry) and add relative odometry to 
        factor graph with appropriate timestamp
        """
        if self.poses_.length: 
            self.poses_.accumulate(self.poses_.latest.oplus(p))
        else: 
            self.poses_.accumulate(RigidTransform.identity())
        self.pose_id_ = self.slam_.on_odom(t, p.to_homogeneous_matrix())

        self.vis_odom(self.poses_)        
        self.vis_slam_updates()
        
        return self.pose_id_

    def on_tags(self, t, tags):
        """
        Add tags to factor graph 
        TODO: currently GTSAM is only adding Pose3-Pose3 costraint. 
        Need to incorporate Pose3-Point2 constraint from tag corners
        """
        ids = [tag.id for tag in tags]
        poses = [tag.getPose() for tag in tags]
        self.pose_id_ = self.slam_.on_pose_ids(t, ids, poses)
        print ids, poses

        # Visualize SLAM updates
        self.vis_slam_updates()

        # Visualize tags/landmarks
        p_landmarks = [ Pose.from_rigid_transform(tag.id, RigidTransform.from_homogenous_matrix(tag.getPose())) 
                        for tag in tags ]
        self.vis_landmarks(self.pose_id_, self.poses_.latest, p_landmarks)
        
        return self.pose_id_

    # def on_landmarks(self, t, poses_w_ids): 
    #     """
    #     Add pose landmarks to factor graph 
    #     Pose3-Pose3 costraint. 

    #     poses: Pose (with ID)
    #     """
    #     ids = [p.id for p in poses_w_ids]
    #     poses = [p.to_homogeneous_matrix() for p in poses_w_ids]
    #     self.pose_id_ = self.slam_.on_pose_ids(t, ids, poses)

    #     # Visualize SLAM updates
    #     self.vis_slam_updates()

    #     # Visualize tags/landmarks
    #     self.vis_landmarks(self.pose_id_, self.poses_.latest, poses_w_ids)
        
    #     return self.pose_id_

    def vis_slam_updates(self, frame_id='optcamera'): 
        """
        Update SLAM visualizations with optimized factors
        """

        # Draw poses 
        updated_poses = {pid : Pose.from_rigid_transform(
            pid, RigidTransform.from_homogenous_matrix(p)) 
                         for (pid,p) in self.slam_.poses.iteritems()}
        draw_utils.publish_pose_list('optimized_node_poses', updated_poses.values(), frame_id=frame_id, reset=self.reset_required())

        # Draw targets (constantly updated, so draw with reset)
        updated_targets = {pid : Pose.from_rigid_transform(pid, RigidTransform.from_homogenous_matrix(p)) 
                           for (pid, p) in self.slam_.targets.iteritems()}
        if len(updated_targets): 
            edges = np.vstack([draw_utils.draw_tag_edges(p) for p in updated_targets.itervalues()])
            draw_utils.publish_line_segments('optimized_node_tag', edges[:,:3], edges[:,3:6], c='r', 
                                             frame_id=frame_id, reset=True)

        # Draw edges (between landmarks and poses)
        landmark_edges = self.slam_.landmark_edges
        # print landmark_edges, len(updated_poses), len(updated_targets)
        if len(landmark_edges): 
            factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
            factor_end = np.vstack([(updated_targets[lid].tvec).reshape(-1,3) for (_, lid) in landmark_edges])

            draw_utils.publish_line_segments('optimized_factor_tag', factor_st, factor_end, c='b', 
                                             frame_id=frame_id, reset=True) 

        # Maintain updated ids
        self.updated_ids_ = set(updated_poses.keys())


    def vis_odom(self, poses, frame_id='camera'): 

        # Set ids for accumulated poses
        draw_utils.publish_pose_t('POSE', poses.latest, frame_id=frame_id)
        draw_utils.publish_pose_list('CAMERA_LATEST', [poses.latest], texts=[],
                                     frame_id=frame_id) # TODO: reset?
        draw_utils.publish_pose_list('CAMERA_POSES', [Pose.from_rigid_transform(poses.index, poses.latest)], 
                                     frame_id=frame_id, reset=(poses.length <= 1))

        if poses.length < 2: 
            return

        p_odom = (poses.items[-2].inverse()).oplus(poses.items[-1])

        factor_st = (poses.items[-2].tvec).reshape(-1,3)
        factor_end = (poses.items[-1].tvec).reshape(-1,3)

        draw_utils.publish_line_segments('measured_factor_odom', factor_st, factor_end, c='r', 
                                         frame_id=frame_id, reset=self.reset_required())
        
    
    def vis_landmarks(self, pose_id, p_latest, p_landmarks, frame_id='camera', landmark_name='tag'): 
        """
        Visualize tags with respect to latest camera frame
        """
        if not len(p_landmarks): 
            return

        # Accumulate tag detections into global reference frame
        p_Wc = p_latest
        p_Wt = [Pose.from_rigid_transform(pose_id * 20 + p.id, p_Wc.oplus(p)) for p in p_landmarks]

        # Plot RAW tag factors
        factor_st = np.tile(p_latest.tvec, [len(p_Wt),1])
        factor_end = np.vstack(map(lambda p: p.tvec, p_Wt))

        factor_ct_end = np.vstack([p.tvec for p in p_landmarks])
        factor_ct_st = np.zeros_like(factor_ct_end)

        draw_utils.publish_line_segments('measured_factor_{:}'.format(landmark_name), factor_st, factor_end, c='b', 
                                         frame_id=frame_id, reset=self.reset_required())
        edges = np.vstack([draw_utils.draw_tag_edges(p) for p in p_Wt])
        draw_utils.publish_line_segments('measured_node_{:}'.format(landmark_name), edges[:,:3], edges[:,3:6], c='b', 
                                         frame_id=frame_id, reset=self.reset_required())

        # Optionally plot as Tags
        # draw_utils.publish_pose_list('RAW_tags', p_Wt, texts=[], object_type='TAG', 
        #                              frame_id=frame_id, reset=self.reset_required())

        
        # # Plot OPTIMIZED tag factors
        # if pose_id in self.updated_ids_: 
        #     draw_utils.publish_line_segments('OPT_factors_{:}'.format(landmark_name), factor_ct_st, factor_ct_end, c='r', 
        #                                  frame_id='optimized_poses', element_id=pose_id, reset=self.reset_required())

        #     edges = np.vstack([draw_utils.draw_tag_edges(p) for p in p_landmarks])
        #     draw_utils.publish_line_segments('optimized_{:}'.format(landmark_name), edges[:,:3], edges[:,3:6], c='r', 
        #                                      frame_id='optimized_poses', element_id=pose_id, reset=self.reset_required())


class TagDetector(object): 
    def __init__(self, camera, tag_size=0.166): 

        # Setup camera (identity extrinsics)
        self.cam_ = camera 

        # Setup APRILTags detector
        self.tag_detector_ = AprilTagsWrapper()
        self.tag_detector_.set_calib(tag_size=tag_size, fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy)

    def process(self, im):
        # Detect the tags
        # im = cam.undistort(to_gray(im))
        tags = self.tag_detector_.process(to_gray(im), return_poses=True)

        # Visualize the tags
        vis = draw_tags(to_color(im), tags)
        imshow_cv('tag-detector', vis)
        # self.slam_.draw_tags(vis, tags)
        
        return tags
