import time
import numpy as np

from pybot.vision.image_utils import to_color, to_gray
from pybot.vision.imshow_utils import imshow_cv

from pybot.utils.misc import Accumulator, SkippedCounter, CounterWithPeriodicCallback
from pybot.vision.mapping.pose_utils import PoseAccumulator
from pybot.geometry.rigid_transform import RigidTransform, Pose, Quaternion

import pybot.externals.draw_utils as draw_utils

from .gtsam import BaseSLAM as GTSAM_BaseSLAM
from .gtsam import VisualSLAM as GTSAM_VisualSLAM

from pybot_gtsam import GTSAMTags
from pybot_slam import ISAMTags, draw_tags
from pybot_apriltags import AprilTag, AprilTagsWrapper


# Mahalanobis distance
# Normalized mahalanobis distance
# fbn_remus/apps/ranger_nav/Point3DLM.h
# LinAlg::Matrix<3,1,double> X = m_X-pLM.m_X;
# 	LinAlg::Matrix<3,3,double> P = (m_P+pLM.m_P).inverse();
# 	LinAlg::Matrix<1,1,double> d = (X.transpose()*P)*X;

	# LinAlg::Matrix<3,1,double> rX;
	# rX(1,1) = pA.GetX()-pB.GetX();
	# rX(2,1) = pA.GetY()-pB.GetY();
	# rX(3,1) = pA.GetZ()-pB.GetZ();
	# LinAlg::Matrix<3,3,double> iP =(pA.GetCov()+pB.GetCov()).inverse();
	# LinAlg::Matrix<3,1,double> iPrX = iP*rX;
	# double d = rX(1,1)*iPrx(1,1)+rX(2,1)*iPrX(2,1)+rX(3,1)*iPrX(3,1);

from scipy.spatial.distance import mahalanobis

def mahalanobis_distance(u, v, VI): 
    return mahalanobis(u, v, VI)

class RobotSLAMMixin(object): 
    def __init__(self, landmark_type='point', update_on_odom=False, 
                 visualize_factors=False, visualize_nodes=False, visualize_marginals=False): 
        if not isinstance(self, (GTSAM_BaseSLAM, GTSAM_VisualSLAM)): 
            raise RuntimeError('Cannot mixin without mixing with one of the SLAM classes')
        if not ((isinstance(self, GTSAM_BaseSLAM) and landmark_type == 'pose') or 
                (isinstance(self, GTSAM_VisualSLAM) and landmark_type == 'point')): 
            raise ValueError('Wrong expected landmark type for {}, provided {}'
                             .format(type(self), landmark_type))
            
        self.__poses = Accumulator(maxlen=2)
        self.landmark_type_ = landmark_type
        self.update_on_odom_ = update_on_odom

        # Visualizations
        self.visualize_factors_ = visualize_factors
        self.visualize_nodes_ = visualize_nodes
        self.visualize_marginals_ = visualize_marginals

        self.landmark_text_lut_ = {}

    def set_landmark_texts(self, landmark_lut): 
        " {landmark id -> landmark str, ... }"
        self.landmark_text_lut_ = landmark_lut

    def on_odom_absolute(self, t, p): 
        """
        Accumulate poses (input: absolute odometry) and add relative odometry to 
        factor graph with appropriate timestamp

           1. On the first odometry data, pose graph is initialized
           2. On the second and subsequent odometry measurements, 
              relative pose measurement is added 
        
        """

        # Initialize keyframes if not previously done
        # Add odometry measurements incrementally
        self.__poses.accumulate(p) 

        # 1. SLAM: Initialize
        if not self.is_initialized: 
            self.initialize(self.__poses.latest.matrix)
            return self.latest
        assert(self.__poses.length >= 2)

        # 2. SLAM: Add relative pose measurements (odometry)
        p_odom = (self.__poses.items[-2].inverse()).oplus(self.__poses.items[-1])
        self.add_odom_incremental(p_odom.matrix)

        # 3. Visualize
        # self.vis_measurements()        

        return self.latest

    def on_odom_relative(self, t, p): 
        """
        Accumulate componded pose (input: relative odometry) and add relative odometry to 
        factor graph with appropriate timestamp
        """
        # 1. SLAM: Initialize
        if not self.is_initialized: 
            self.initialize(RigidTransform.identity().matrix)
        
        # 2. SLAM: Add relative pose measurements (odometry)
        self.add_odom_incremental(p.matrix)

        # 3. Visualize
        # self.vis_measurements()

        return self.latest

    def on_point_landmarks_smart(self, t, ids, pts, keep_tracked=True): 
        self.add_point_landmarks_incremental_smart(ids, pts, keep_tracked=keep_tracked)
        if self.__poses.length >= 2: 
            self.update()
            # self.update_marginals()
            ids, pts3 = self.smart_update()

            # Publish pose
            draw_utils.publish_pose_list('gtsam-pose', [Pose.from_rigid_transform(t, self.__poses.latest)], 
                                         frame_id='camera', reset=False)
            # Publish cloud in latest pose reference frame
            if len(pts3): 
                pts3 = RigidTransform.from_matrix(self.pose(self.latest)).inverse() * pts3
            draw_utils.publish_cloud('gtsam-pc', [pts3], c='b', frame_id='gtsam-pose', element_id=[t], reset=False)

        self.vis_optimized()

        return self.latest

    def finish(self): 
        self.update()
        # self.update_marginals()
        ids, pts3 = self.smart_update()

        self.vis_optimized()

    #################
    # Visualization #
    #################

    def vis_measurements(self, frame_id='camera'): 
        poses = self.__poses
        assert(isinstance(poses, Accumulator))

        # Draw robot poses
        # draw_utils.publish_pose_t('POSE', poses.latest, frame_id=frame_id)
        # draw_utils.publish_pose_list('CAMERA_POSES', [Pose.from_rigid_transform(poses.index, poses.latest)], 
        #                              frame_id=frame_id, reset=False)

        # # Draw odometry link
        # if poses.length >= 2:

        #     p_odom = (poses.items[-2].inverse()).oplus(poses.items[-1])

        #     factor_st = (poses.items[-2].tvec).reshape(-1,3)
        #     factor_end = (poses.items[-1].tvec).reshape(-1,3)
        #     draw_utils.publish_line_segments('measured_factor_odom', factor_st, factor_end, c='r', 
        #                                      frame_id=frame_id, reset=False)


    def vis_optimized(self, frame_id='camera'): 
        """
        Update SLAM visualizations with optimized factors
        """
        if not self.visualize_nodes_ and \
           not self.visualize_factors_ and \
           not self.visualize_marginals_: 
            return 

        # if not draw_utils.has_sensor_frame('optcamera'): 
        #     draw_utils.publish_sensor_frame()

        # =========================================================
        # POSES/MARGINALS

        # Poses 
        covars = []
        updated_poses = {pid : Pose.from_rigid_transform(
            pid, RigidTransform.from_matrix(p)) 
                         for (pid,p) in self.poses.iteritems()}

        # Marginals
        if self.visualize_marginals_: 
            poses_marginals = self.poses_marginals
            triu_inds = np.triu_indices(3)
            if self.marginals_available: 
                for pid in updated_poses.keys(): 
                    covars.append(
                        poses_marginals.get(pid, np.eye(6, dtype=np.float32) * 100)[triu_inds]
                    )

        # Draw cameras (with poses and marginals)
        draw_utils.publish_cameras('optimized_node_poses', updated_poses.values(), 
                                   covars=covars, frame_id=frame_id, reset=True)

        # Draw odometry edges (between robot poses)
        if self.visualize_factors_: 
            robot_edges = self.robot_edges
            if len(robot_edges): 
                factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in robot_edges])
                factor_end = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (_, xid) in robot_edges])

                draw_utils.publish_line_segments('optimized_factor_odom', factor_st, factor_end, c='b', 
                                                 frame_id=frame_id, reset=True) 


        # =========================================================
        # TARGETS/LANDMARKS

        if self.landmark_type_ == 'pose': 
            # Draw targets (constantly updated, so draw with reset)
            updated_targets = {pid : Pose.from_rigid_transform(pid, RigidTransform.from_homogenous_matrix(p)) 
                               for (pid, p) in self.target_poses.iteritems()}
            if len(updated_targets): 
                edges = np.vstack([draw_utils.draw_tag_edges(p) for p in updated_targets.itervalues()])
                draw_utils.publish_line_segments('optimized_node_landmark', edges[:,:3], edges[:,3:6], c='r', 
                                                 frame_id=frame_id, reset=True)

            # Draw edges (between landmarks and poses)
            if self.visualize_factors_: 
                landmark_edges = self.landmark_edges
                if len(landmark_edges): 
                    factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
                    factor_end = np.vstack([(updated_targets[lid].tvec).reshape(-1,3) for (_, lid) in landmark_edges])

                    draw_utils.publish_line_segments('optimized_factor_landmark', factor_st, factor_end, c='b', 
                                                     frame_id=frame_id, reset=True) 

        elif self.landmark_type_ == 'point': 

            # Draw targets (constantly updated, so draw with reset)
            updated_targets = {pid : pt3
                               for (pid, pt3) in self.target_landmarks.iteritems()}
            if len(updated_targets): 
                points3d = np.vstack(updated_targets.values())
                draw_utils.publish_cloud('optimized_node_landmark', points3d, c='r', 
                                         frame_id=frame_id, reset=True)

            # Landmark points
            if self.visualize_nodes_: 
                covars = []
                poses = [Pose(k, tvec=v) for k, v in updated_targets.iteritems()]
                texts = [self.landmark_text_lut_.get(k, '') for k in updated_targets.keys()] \
                        if len(self.landmark_text_lut_) else []

                # Marginals
                if self.visualize_marginals_: 
                    target_landmarks_marginals = self.target_landmarks_marginals
                    triu_inds = np.triu_indices(3)
                    if self.marginals_available: 
                        for pid in updated_targets.keys(): 
                            covars.append(
                                target_landmarks_marginals.get(pid, np.ones(shape=(6,6)) * 10)[triu_inds]
                            )

                # Draw landmark points
                draw_utils.publish_pose_list('optimized_node_landmark_poses', poses, texts=texts, 
                                             covars=covars, frame_id=frame_id, reset=True)

            # Draw edges (between landmarks and poses)
            if self.visualize_factors_: 
                landmark_edges = self.landmark_edges
                if len(landmark_edges): 
                    factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
                    factor_end = np.vstack([(updated_targets[lid]).reshape(-1,3) for (_, lid) in landmark_edges])

                    draw_utils.publish_line_segments('optimized_factor_landmark', factor_st, factor_end, c='b', 
                                                     frame_id=frame_id, reset=True) 

        else: 
            raise ValueError('Unknown landmark_type {:}'.format(self.landmark_type_))

        # Maintain updated ids
        return set(updated_poses.keys())


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

        # draw_utils.publish_line_segments('measured_factor_{:}'.format(landmark_name), factor_st, factor_end, c='b', 
        #                                  frame_id=frame_id, reset=self.reset_required())
        # edges = np.vstack([draw_utils.draw_tag_edges(p) for p in p_Wt])
        # draw_utils.publish_line_segments('measured_node_{:}'.format(landmark_name), edges[:,:3], edges[:,3:6], c='b', 
        #                                  frame_id=frame_id, reset=self.reset_required())

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

# Right to left inheritance
class RobotSLAM(RobotSLAMMixin, GTSAM_BaseSLAM): 
    def __init__(self, update_on_odom=False, verbose=False, 
                 visualize_nodes=False, visualize_factors=False, visualize_marginals=False): 
        GTSAM_BaseSLAM.__init__(self, 
                                odom_noise=GTSAM_BaseSLAM.odom_noise, 
                                prior_noise=GTSAM_BaseSLAM.prior_noise, verbose=verbose)
        RobotSLAMMixin.__init__(self, landmark_type='pose', update_on_odom=update_on_odom, 
                                visualize_nodes=visualize_nodes, 
                                visualize_factors=visualize_factors, 
                                visualize_marginals=visualize_marginals)
        

class RobotVisualSLAM(RobotSLAMMixin, GTSAM_VisualSLAM): 
    def __init__(self, calib, 
                 min_landmark_obs=3, 
                 odom_noise=GTSAM_VisualSLAM.odom_noise, prior_noise=GTSAM_VisualSLAM.prior_noise, 
                 px_error_threshold=4, px_noise=[1.0, 1.0], 
                 update_on_odom=False, verbose=False, 
                 visualize_nodes=False, visualize_factors=False, visualize_marginals=False): 
        GTSAM_VisualSLAM.__init__(self, calib, 
                                  min_landmark_obs=min_landmark_obs, 
                                  px_error_threshold=px_error_threshold, 
                                  odom_noise=odom_noise, prior_noise=prior_noise, 
                                  px_noise=px_noise, verbose=verbose)
        RobotSLAMMixin.__init__(self, landmark_type='point', 
                                update_on_odom=update_on_odom, 
                                visualize_nodes=visualize_nodes, 
                                visualize_factors=visualize_factors, 
                                visualize_marginals=visualize_marginals)

    
    # def on_pose_ids(self, t, ids, poses): 
    #     print('\ton_pose_ids')
    #     for (pid, pose) in izip(ids, poses): 
    #         self.add_landmark_incremental(pid, pose)
    #     self.update()
    #     return self.latest

    # def on_landmark(self, p): 
    #     pass


# class BaseSLAM(GTSAM_BaseSLAM):
#     """
#     Basic SLAM interface
    
#        All measurements are to be provided in the 
#        same reference frame 
    
#     """
#     def __init__(self, ref_frames={}, absolute_measurements=True, visualize=True): 

#         # Initialize SLAM (GTSAM / ISAM)
#         # self.slam_ = GTSAMTags() # FIX: TODO
#         # self.slam_ = BaseSLAM()
#         self.slam_ = SLAM3D()

#         # self.slam_cb_ = CounterWithPeriodicCallback(
#         #     every_k=10, 
#         #     process_cb=lambda:  self.slam_.save_graph("slam_fg.dot")
#         # )
#         # self.slam_cb_.register_callback(self.slam_, 'on_odom')

#         # Poses (request for relative measurements if the odometry is absolute)
#         self.pose_id_ = -1
#         self.poses_ = Accumulator(maxlen=10)

#         # self.reset_required = lambda: self.poses_.length == 0
#         # print 'reset: ', self.reset_required()

#         # Updated ids
#         self.updated_ids_ = set()

#         # Visualization: Reference names
#         self.vis_name_ = 'slam_vis'
#         self.ref_frames_ = {}
        
#         # Visualization: Publish reference frames
#         for k, v in ref_frames.iteritems(): 
#             draw_utils.publish_sensor_frame(k, v)

#     @property
#     def pose_id(self): 
#         return self.slam_.latest
        
#     def update(self, iterations=1): 
#         # Finalize updates
#         for j in range(iterations): 
#             self.slam_.update()
#         self.updated_ids_ = vis_slam_updates(self.slam_)
#         self.slam_.save_graph("slam_fg.dot")

#     def on_tags(self, t, tags):
#         """
#         Add tags to factor graph 
#         TODO: currently GTSAM is only adding Pose3-Pose3 costraint. 
#         Need to incorporate Pose3-Point2 constraint from tag corners
#         """
#         ids = [tag.id for tag in tags]
#         poses = [tag.getPose() for tag in tags]
#         self.pose_id_ = self.slam_.on_pose_ids(t, ids, poses)
#         print ids, poses

#         # Visualize SLAM updates
#         self.updated_ids_ = vis_slam_updates(self.slam_)

#         # Visualize tags/landmarks
#         p_landmarks = [ Pose.from_rigid_transform(tag.id, RigidTransform.from_homogenous_matrix(tag.getPose())) 
#                         for tag in tags ]
#         self.vis_landmarks(self.pose_id_, self.poses_.latest, p_landmarks)
        
#         return self.pose_id_

#     def on_landmarks(self, t, poses_w_ids): 
#         """
#         Add pose landmarks to factor graph 
#         Pose3-Pose3 costraint. 

#         poses: Pose (with ID)
#         """
#         if not self.poses_.length: 
#             import warnings 
#             warnings.warn('Failed to add landmark since pose has not been initialized')
#             return

#         ids = [p.id for p in poses_w_ids]
#         poses = [p.matrix for p in poses_w_ids]
#         print ids, poses, self.poses_.latest, self.pose_id_

#         self.pose_id_ = self.slam_.on_pose_ids(t, ids, poses)

#         # Visualize SLAM updates
#         self.updated_ids_ = vis_slam_updates(self.slam_)

#         # Visualize tags/landmarks
#         self.vis_landmarks(self.pose_id_, self.poses_.latest, poses_w_ids)
        
#         return self.pose_id_


#     def vis_odom(self, poses, frame_id='camera'): 

#         # Set ids for accumulated poses
#         # draw_utils.publish_pose_t('POSE', poses.latest, frame_id=frame_id)
#         draw_utils.publish_pose_t('CAMERA_POSE', poses.latest, frame_id=frame_id)
#         draw_utils.publish_pose_list('CAMERA_LATEST', [poses.latest], texts=[],
#                                      frame_id=frame_id) # TODO: reset?
#         # draw_utils.publish_pose_list('CAMERA_POSES', [Pose.from_rigid_transform(poses.index, poses.latest)], 
#         #                              frame_id=frame_id, reset=(poses.length <= 1))

#         if poses.length < 2: 
#             return

#         p_odom = (poses.items[-2].inverse()).oplus(poses.items[-1])

#         factor_st = (poses.items[-2].tvec).reshape(-1,3)
#         factor_end = (poses.items[-1].tvec).reshape(-1,3)

#         draw_utils.publish_line_segments('measured_factor_odom', factor_st, factor_end, c='r', 
#                                          frame_id=frame_id, reset=False)
        
    
#     def vis_landmarks(self, pose_id, p_latest, p_landmarks, frame_id='camera', landmark_name='tag'): 
#         """
#         Visualize tags with respect to latest camera frame
#         """
#         if not len(p_landmarks): 
#             return

#         # Accumulate tag detections into global reference frame
#         p_Wc = p_latest
#         p_Wt = [Pose.from_rigid_transform(pose_id * 20 + p.id, p_Wc.oplus(p)) for p in p_landmarks]

#         # Plot RAW tag factors
#         factor_st = np.tile(p_latest.tvec, [len(p_Wt),1])
#         factor_end = np.vstack(map(lambda p: p.tvec, p_Wt))

#         factor_ct_end = np.vstack([p.tvec for p in p_landmarks])
#         factor_ct_st = np.zeros_like(factor_ct_end)

#         # draw_utils.publish_line_segments('measured_factor_{:}'.format(landmark_name), factor_st, factor_end, c='b', 
#         #                                  frame_id=frame_id, reset=self.reset_required())
#         # edges = np.vstack([draw_utils.draw_tag_edges(p) for p in p_Wt])
#         # draw_utils.publish_line_segments('measured_node_{:}'.format(landmark_name), edges[:,:3], edges[:,3:6], c='b', 
#         #                                  frame_id=frame_id, reset=self.reset_required())

#         # Optionally plot as Tags
#         # draw_utils.publish_pose_list('RAW_tags', p_Wt, texts=[], object_type='TAG', 
#         #                              frame_id=frame_id, reset=self.reset_required())

        
#         # # Plot OPTIMIZED tag factors
#         # if pose_id in self.updated_ids_: 
#         #     draw_utils.publish_line_segments('OPT_factors_{:}'.format(landmark_name), factor_ct_st, factor_ct_end, c='r', 
#         #                                  frame_id='optimized_poses', element_id=pose_id, reset=self.reset_required())

#         #     edges = np.vstack([draw_utils.draw_tag_edges(p) for p in p_landmarks])
#         #     draw_utils.publish_line_segments('optimized_{:}'.format(landmark_name), edges[:,:3], edges[:,3:6], c='r', 
#         #                                      frame_id='optimized_poses', element_id=pose_id, reset=self.reset_required())


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

# Test /home/spillai/perceptual-learning/software/python/apps/tango/tango_annotations_app.py

# import subprocess
# print ("1. Running tango slam")
# app = subprocess.Popen( [os.path.join("/home/spillai/perceptual-learning/software/python/apps/tango/tango_annotations_app.py")] )
# app.wait()

