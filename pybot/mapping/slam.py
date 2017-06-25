# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import os
import sys
import time
import numpy as np
from itertools import izip
from threading import Thread, Lock, Timer

from pybot.geometry.rigid_transform import RigidTransform, Pose, Quaternion
from pybot.vision.image_utils import to_color, to_gray
from pybot.vision.imshow_utils import imshow_cv
from pybot.utils.pose_utils import PoseAccumulator
from pybot.utils.timer import timeitmethod
from pybot.utils.misc import Accumulator, SkippedCounter, CounterWithPeriodicCallback
from pybot.externals.lcm import draw_utils

# ===============================================================================
# Import GTSAM / ISAM Backend
from pybot.mapping import cfg
if cfg.SLAM_BACKEND == 'gtsam':
    sys.stderr.write('Using GTSAM backend.\n')
    from pybot.mapping.gtsam import BaseSLAM as _BaseSLAM
    from pybot.mapping.gtsam import VisualSLAM as _VisualSLAM
    from pybot.mapping.gtsam import vec, vector, matrix
elif cfg.SLAM_BACKEND == 'isam':
    sys.stderr.write('Using iSAM backend.\n')
    raise NotImplementedError('ISAM not yet implemented')
elif cfg.SLAM_BACKEND == 'toro':
    sys.stderr.write('Using Toro backend.\n')
    from pybot.mapping.toro import BaseSLAM as _BaseSLAM
    from pybot.mapping.toro import BaseSLAM as _VisualSLAM
else:
    raise Exception('Unknown backend: {}'.format(cfg.SLAM_BACKEND))

# ===============================================================================
# BaseSLAM
_current_backend = cfg.SLAM_BACKEND

def current_backend():
    return _current_backend

POSE, POINT = range(2)

class BaseSLAM(_BaseSLAM):
    def __init__(self, update_every_k_odom=10,
                 odom_noise=cfg.ODOM_NOISE, 
                 prior_pose_noise=cfg.PRIOR_POSE_NOISE, 
                 measurement_noise=cfg.MEASUREMENT_NOISE,
                 verbose=False, export_graph=False):
        _BaseSLAM.__init__(self, odom_noise=odom_noise, prior_pose_noise=prior_pose_noise,
                           verbose=verbose, export_graph=export_graph)
        self.update_every_k_odom_ = update_every_k_odom
        self.q_poses_ = Accumulator(maxlen=2)

        self.state_lock_ = Lock()

    # def load(self, filename):
    #     raise NotImplementedError()
        
    def pose(self, k):
        return super(BaseSLAM, self).pose(k)
        
    @property
    def latest_measurement_pose(self): 
        return self.q_poses_.latest

    def measurement_pose(self, idx):
        return self.q_poses_.items[idx]

    @property
    def updated_poses(self):
        """ Upgrade to Pose from RigidTransform """
        return {pid : Pose.from_rigid_transform(pid, p)
                for (pid,p) in self.poses.iteritems()}

    @property
    def updated_targets(self):
        """ Upgrade to Pose from RigidTransform """
        return {pid : Pose.from_rigid_transform(pid, p)
                for (pid, p) in self.target_poses.iteritems()}

    def _update_check(self): 
        """
        Check whether update is required
        """
        if self.latest >= 2 and self.latest % self.update_every_k_odom_ == 0: 
            self.update()
                
        self.visualize_measurements()
    
    def initialize(self, p=None, index=0, noise=None):
        """
        Initialize factor graph with prior on node
        """
        p_init = RigidTransform.identity() \
                 if p is None else p
        self.q_poses_.append(p_init)
        super(BaseSLAM, self).initialize(p=p_init, index=index, noise=noise)

    def add_pose_prior(self, index, p, noise=None):
        """
        Add prior to node
        """
        super(BaseSLAM, self).add_pose_prior(index, p, noise=noise)

    def on_odom_absolute(self, t, p, noise=None): 
        """
        Accumulate poses (input: absolute odometry) and add relative odometry to 
        factor graph with appropriate timestamp

           1. On the first odometry data, pose graph is initialized
           2. On the second and subsequent odometry measurements, 
              relative pose measurement is added 
        
        """
        # Initialize keyframes if not previously done
        # Add odometry measurements incrementally
        self.q_poses_.append(p) 

        # 1. SLAM: Initialize
        if not self.is_initialized:
            super(BaseSLAM, self).initialize(self.q_poses_.latest)
            return self.latest
        assert(self.q_poses_.length >= 2)

        # 2. SLAM: Add relative pose measurements (odometry)
        p_odom = (self.q_poses_.items[-2].inverse()).oplus(self.q_poses_.items[-1])
        self.add_incremental_pose_constraint(p_odom, noise=noise)

        # 3. Update
        self._update_check()

        return self.latest

    def on_odom_relative(self, t, p, noise=None): 
        """
        Accumulate componded pose (input: relative odometry) and add relative odometry to 
        factor graph with appropriate timestamp
        """

        # 1. SLAM: Initialize and add (do not return if not initialized)
        if not self.is_initialized:
            p_init = RigidTransform.identity()
            self.q_poses_.append(p_init)
            super(BaseSLAM, self).initialize(p_init)
        
        # 2. SLAM: Add relative pose measurements (odometry)
        # self.q_poses_.append(p * self.q_poses_.latest)
        self.q_poses_.append(self.q_poses_.latest.oplus(p))
        self.add_incremental_pose_constraint(p, noise=noise)

        # 3. Update
        self._update_check()
        
        return self.latest

    def add_relative_pose_constraint(self, idx1, idx2, p, noise=None):
        super(BaseSLAM, self).add_relative_pose_constraint(idx1, idx2, p, noise=noise)        
        return None

    def add_pose_landmarks(self, xid, lids, poses): 
        super(BaseSLAM, self).add_pose_landmarks(xid, lids, poses) 
        return None
    
    def on_loop_closure_relative(self, t, idx1, idx2, p, noise=None): 
        """
        Accumulate componded pose (input: relative odometry) and add relative odometry to 
        factor graph with appropriate timestamp
        """
        
        # 1. SLAM: Add relative pose measurements (odometry)
        self.add_relative_pose_constraint(idx1, idx2, p, noise=noise)

        # 2. Update
        self._update_check()

        return self.latest
        
    def on_pose_landmarks(self, t, ids, poses): 
        self.add_pose_landmarks_incremental(ids, poses)

        # 2. Update
        self._update_check()

        return self.latest

    def update(self, iterations=1): 
        self._update(iterations=iterations)
        self._update_estimates()
        # self._update_marginals()

    def batch_solve(self):
        self._batch_solve()
        self._update_estimates()
        
    def finish(self, iterations=10): 
        for j in range(iterations): 
            self.update()

    def visualize_measurements(self):
        # Publish latest pose
        latest_idx = self.latest
        latest_pose = self.q_poses_.latest
        draw_utils.publish_pose_list(
            'current_pose',
            [Pose.from_rigid_transform(latest_idx, latest_pose)], 
            frame_id=self.frame_id_, reset=False)
        
        # Visualize all poses
        self.visualize_poses()
        
    def visualize_poses(self, pose_type=POSE):
        if not self.visualize_measurements_:
            return

        with self.state_lock_:
            # Draw odom pose
            draw_utils.publish_pose_list(
                self.name_ + 'measured_odom', 
                [Pose.from_rigid_transform(self.latest, self.latest_measurement_pose)], 
                frame_id=self.frame_id_, reset=False)
            
            # Draw odom factor
            if self.visualize_factors_ and self.latest >= 1: 
                factor_st = (self.measurement_pose(-2).tvec).reshape(-1,3)
                factor_end = (self.measurement_pose(-1).tvec).reshape(-1,3)
                draw_utils.publish_line_segments(
                    self.name_ + 'measured_factor_odom', 
                    factor_st, factor_end, c='r', 
                    frame_id=self.frame_id_, reset=False)

    def visualize_optimized_poses(
            self, 
            visualize_nodes=False, 
            visualize_measurements=False,
            visualize_factors=False,
            visualize_marginals=False,
            pose_type=POSE, name='SLAM_', frame_id='camera'):
        
        with self.state_lock_:
            # Poses 
            updated_poses = self.updated_poses
            texts, covars = map(str, updated_poses.keys()), []
            
            # Marginals
            if visualize_marginals: 
                poses_marginals = self.poses_marginals
                triu_inds = np.triu_indices(3)
                if self.marginals_available: 
                    for pid in updated_poses.keys(): 
                        covars.append(
                            poses_marginals.get(pid, np.eye(6, dtype=np.float32) * 100)[triu_inds]
                        )

            # Draw cameras (with poses and marginals)
            if pose_type == POSE: 
                draw_utils.publish_cameras(
                    name + 'optimized_node_poses', updated_poses.values(), 
                    covars=covars, frame_id=frame_id, draw_edges=False, reset=True)
            elif pose_type == POINT: 
                draw_utils.publish_cloud(
                    name + 'optimized_node_poses_pc', np.vstack(map(lambda p: p.tvec, updated_poses.values())), 
                    c='b', frame_id=frame_id, reset=True)
            else:
                raise ValueError()
            
            # Draw odometry edges (between robot poses)
            if visualize_factors: 
                robot_edges = self.robot_edges
                if len(robot_edges): 
                    factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in robot_edges])
                    factor_end = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (_, xid) in robot_edges])

                    draw_utils.publish_line_segments(
                        name + 'optimized_factor_odom', factor_st, factor_end, c='b', 
                        frame_id=frame_id, reset=True) 
            
    def visualize_optimized_landmarks(
            self,
            visualize_nodes=False, 
            visualize_measurements=False,
            visualize_factors=False,
            visualize_marginals=False,
            landmark_type=POSE, name='SLAM_', frame_id='camera'):
        
        with self.state_lock_:

            # Draw targets (constantly updated, so draw with reset)
            updated_targets = self.updated_targets
            if visualize_nodes and len(updated_targets):
                poses = updated_targets.values()
                texts, covars = map(str, updated_targets.keys()), []

                if landmark_type == POSE: 
                    draw_utils.publish_pose_list(
                        name + 'optimized_node_landmark_poses', poses, texts=texts, 
                        covars=covars, frame_id=frame_id, reset=True)
                elif landmark_type == POINT:
                    draw_utils.publish_cloud(
                        name + 'optimized_node_landmark_poses_pc',
                        np.vstack(map(lambda p: p.tvec, poses.values())), 
                        c='b', frame_id=frame_id, reset=True)
                else:
                    raise ValueError()
            
            # Draw edges (between landmarks and poses)
            if visualize_factors: 
                updated_poses = self.updated_poses
                landmark_edges = self.landmark_edges
                if len(landmark_edges) and len(updated_poses) and len(updated_targets): 
                    factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
                    factor_end = np.vstack([(updated_targets[lid].tvec).reshape(-1,3) for (_, lid) in landmark_edges])
                    draw_utils.publish_line_segments(
                        name + 'optimized_factor_landmark',
                        factor_st, factor_end, c='b', 
                        frame_id=frame_id, reset=True) 

            # # Landmark points
            # if visualize_nodes: 
            #     covars = []
            #     poses = [Pose(k, tvec=v) for k, v in updated_targets.iteritems()]
            #     # texts = [self.landmark_text_lut_.get(k, '') for k in updated_targets.keys()] \
            #     #         if len(self.landmark_text_lut_) else []

            #     # Marginals
            #     if self.visualize_marginals_: 
            #         target_landmarks_marginals = self.target_landmarks_marginals
            #         triu_inds = np.triu_indices(3)
            #         if self.marginals_available: 
            #             for pid in updated_targets.keys(): 
            #                 covars.append(
            #                     target_landmarks_marginals.get(pid, np.ones(shape=(6,6)) * 10)[triu_inds]
            #                 )

            #     # Draw landmark points
            #     draw_utils.publish_pose_list(name + 'optimized_node_landmark_poses', poses, texts=texts, 
            #                                  covars=covars, frame_id=frame_id, reset=True)

            # # Draw edges (between landmarks and poses)
            # if visualize_factors: 
            #     landmark_edges = self.landmark_edges
            #     if len(landmark_edges): 
            #         factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
            #         factor_end = np.vstack([(updated_targets[lid]).reshape(-1,3) for (_, lid) in landmark_edges])

            #         draw_utils.publish_line_segments(name + 'optimized_factor_landmark', factor_st, factor_end, c='b', 
            #                                          frame_id=frame_id, reset=True) 
            
class VisualSLAM(BaseSLAM, _VisualSLAM):
    def __init__(self, calib, min_landmark_obs=cfg.VSLAM_MIN_LANDMARK_OBS,
                 update_every_k_odom=10, 
                 odom_noise=cfg.ODOM_NOISE, prior_pose_noise=cfg.PRIOR_POSE_NOISE,
                 prior_point3d_noise=cfg.PRIOR_POINT3D_NOISE,
                 px_error_threshold=4, px_noise=cfg.PX_MEASUREMENT_NOISE, verbose=False):
        _VisualSLAM.__init__(self, calib, min_landmark_obs=min_landmark_obs, prior_point3d_noise=prior_point3d_noise,
                             px_error_threshold=px_error_threshold, px_noise=px_noise, verbose=verbose)
        BaseSLAM.__init__(self, update_every_k_odom=update_every_k_odom,
                          odom_noise=odom_noise, prior_pose_noise=prior_pose_noise, verbose=verbose)
        assert(hasattr(self, 'smart_update'))
        self.smart_ = True
        
    @property
    def updated_targets(self):
        return {pid : pt3
                for (pid, pt3) in self.target_landmarks.iteritems()}

    def _update_check(self): 
        """
        Check whether update is required
        """
        if self.latest == 3 or \
           (self.latest >= 3 and self.latest % self.update_every_k_odom_ == 0):
            
            self.update()
            lids, pts3_w = self.smart_update()
            self.update()
            
            if not len(pts3_w):
                return
                
        # Convert points in the latest reference frame
        self.visualize_measurements()

            
    def add_landmark_prior(self, index, p, noise=None):
        """
        Add prior to node
        """
        super(BaseSLAM, self).add_landmark_prior(index, p, noise=noise)

    def on_point_landmarks_smart(self, xid, ids, pts, keep_tracked=True): 
        assert(self.smart_)
        self.add_point_landmarks_smart(xid, ids, pts, keep_tracked=keep_tracked)

        # Update
        self._update_check()
        
        return self.latest

    def finish(self):
        super(BaseSLAM, self).finish()
        lids, pts3 = self.smart_update()
        if self.verbose_:
            print('{} :: Finished/Solved in {:4.2f} s'
                  .format(self.pretty_name, time.time() - self.slam_mixin_timing_st_))

    def visualize_measurements(self):
        # Publish latest pose
        latest_idx = self.latest
        latest_pose = self.q_poses_.latest
        draw_utils.publish_pose_list(
            'current_pose', [Pose.from_rigid_transform(latest_idx, latest_pose)], 
            frame_id='camera', reset=False)
        
        # # Publish cloud in latest pose reference frame
        # pts3_c = self.pose(self.latest).inverse() * pts3_w
        # draw_utils.publish_cloud('current_cloud', [pts3_c], c='b', frame_id='current_pose', element_id=[latest_idx], reset=False)
        
    def visualize_optimized_landmarks(
            self,
            visualize_nodes=False, 
            visualize_measurements=False,
            visualize_factors=False,
            visualize_marginals=False,
            landmark_type=POINT, name='SLAM_',
            frame_id='camera'):
        
        with self.state_lock_:

            # Draw targets (constantly updated, so draw with reset)
            updated_poses = self.updated_poses
            updated_targets = self.updated_targets
            if len(updated_targets): 
                points3d = np.vstack(updated_targets.values())
                draw_utils.publish_cloud(name + 'optimized_node_landmark', points3d, c='r', 
                                         frame_id=frame_id, reset=True)

            # Landmark points
            if visualize_nodes: 
                covars = []
                poses = [Pose(k, tvec=v) for k, v in updated_targets.iteritems()]
                # texts = [self.landmark_text_lut_.get(k, '') for k in updated_targets.keys()] \
                #         if len(self.landmark_text_lut_) else []

                # Marginals
                if visualize_marginals: 
                    target_landmarks_marginals = self.target_landmarks_marginals
                    triu_inds = np.triu_indices(3)
                    if self.marginals_available: 
                        for pid in updated_targets.keys(): 
                            covars.append(
                                target_landmarks_marginals.get(pid, np.ones(shape=(6,6)) * 10)[triu_inds]
                            )

            # Draw edges (between landmarks and poses)
            if visualize_factors: 
                landmark_edges = self.landmark_edges
                if len(landmark_edges): 
                    factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
                    factor_end = np.vstack([(updated_targets[lid]).reshape(-1,3) for (_, lid) in landmark_edges])

                    draw_utils.publish_line_segments(
                        name + 'optimized_factor_landmark', factor_st, factor_end, c='b', 
                        frame_id=frame_id, reset=True) 
                    
    
def with_visualization(
        cls,
        name='SLAM_', frame_id='camera',
        offset_optimized=np.zeros(3),
        visualize_nodes=True, 
        visualize_measurements=False, 
        visualize_factors=True, visualize_marginals=False,
        pose_type='pose', landmark_type='point',
        visualize_every=1, write_every=0, ): 

    class SLAM_vis(cls):
        def __init__(self, *args, **kwargs): 
            cls.__init__(self, *args, **kwargs)
            
            self.name_ = name
            self.frame_id_ = frame_id
            self.opt_frame_id_ = frame_id + '_optimized'

            # Create optimized frame (offset by z m)
            pose = draw_utils.get_sensor_pose(frame_id)
            pose.tvec += offset_optimized
            draw_utils.publish_sensor_frame(self.opt_frame_id_, pose)

            self.visualize_nodes_ = visualize_nodes
            self.visualize_factors_ = visualize_factors
            self.visualize_marginals_ = visualize_marginals
            self.visualize_measurements_ = visualize_measurements
            try:
                self.pose_type_ = {'pose': POSE, 'point': POINT}[pose_type]
            except Exception, e:
                raise ValueError('Unknown pose type {}'.format(pose_type))
            try: 
                self.landmark_type_ = {'pose': POSE, 'point': POINT}[landmark_type]
            except Exception, e:
                raise ValueError('Unknown landmark type {}'.format(landmark_type))

            self.visualize_every_ = visualize_every
            if visualize_every:
                self.publish_cb_ = CounterWithPeriodicCallback(
                    every_k=visualize_every, process_cb=self.visualize_optimized)

            self.write_every_ = write_every
            if write_every:
                self.write_cb_ = CounterWithPeriodicCallback(
                    every_k=write_every, process_cb=self.save_dot_graph)
                
        def save_dot_graph(self):
            self.save_graph('test.dot')

        def update(self, iterations=1): 
            super(SLAM_vis, self).update(iterations=iterations)
            if self.visualize_every_: 
                self.publish_cb_.poll()
            if self.write_every_:
                self.write_cb_.poll()
            
        def finish(self):
            super(SLAM_vis, self).finish()
            self.visualize_optimized()

        def visualize_measurements(self):
            if self.visualize_measurements_:
                super(SLAM_vis, self).visualize_measurements()

        @timeitmethod
        def visualize_optimized(self):
            """
            Update SLAM visualizations with optimized factors
            """
            self.visualize_optimized_poses()
            self.visualize_optimized_landmarks()

        def visualize_optimized_poses(self):
            super(SLAM_vis, self).visualize_optimized_poses(
                visualize_nodes=self.visualize_nodes_,
                visualize_measurements=self.visualize_measurements_, 
                visualize_factors=self.visualize_factors_, 
                visualize_marginals=self.visualize_marginals_,
                pose_type=self.pose_type_, 
                name=self.name_, frame_id=self.opt_frame_id_)

        def visualize_optimized_landmarks(self):
            super(SLAM_vis, self).visualize_optimized_landmarks(
                visualize_nodes=self.visualize_nodes_,
                visualize_measurements=self.visualize_measurements_, 
                visualize_factors=self.visualize_factors_, 
                visualize_marginals=self.visualize_marginals_,
                landmark_type=self.landmark_type_,
                name=self.name_, frame_id=self.opt_frame_id_)

    return SLAM_vis

def RobotSLAM(*args, **kwargs): 
    """
    name='SLAM_', frame_id='camera',
    visualize_every=0.5,
    visualize_nodes=True, 
    visualize_measurements=False, 
    visualize_factors=True, visualize_marginals=False
    """
    kwargs.setdefault('landmark_type', 'pose')
    
    return with_visualization(BaseSLAM, *args, **kwargs)


def RobotVisualSLAM(*args, **kwargs): 
    """
    name='SLAM_', frame_id='camera',
    visualize_every=0.5,
    visualize_nodes=True, 
    visualize_measurements=False, 
    visualize_factors=True, visualize_marginals=False
    """
    kwargs.setdefault('landmark_type', 'point')
    kwargs.setdefault('frame_id', 'camera')
    
    return with_visualization(VisualSLAM, *args, **kwargs)

# RobotVisualSLAM = with_visualization(
#     VisualSLAM,
#     frame_id='camera',
#     visualize_measurements=True)



            # =========================================================
            # TARGETS/LANDMARKS

            # if self.landmark_type_ == 'pose': 

    #         elif self.landmark_type_ == 'point': 

    #         else: 
    #             raise ValueError('Unknown landmark_type {:}'.format(self.landmark_type_))

    #         # Maintain updated ids
    #         return set(updated_poses.keys())


        # def visualize_landmarks(self, ids, poses):
        #     if not self.visualize_measurements_:
        #         return

        #     # Draw landmark pose
        #     draw_utils.publish_pose_list(self.name_ + 'measured_landmarks', 
        #                                  [Pose.from_rigid_transform(pid + self.latest * 100, self.latest_measurement_pose * p) 
        #                                   for (pid, p) in izip(ids, poses)], 
        #                                  # texts=[str(pid) for pid in ids],
        #                                  frame_id=self.frame_id_, reset=False)

        #     if self.visualize_factors_: 
        #         # Draw odom factor
        #         factor_st = np.vstack([np.zeros((1,3)) for _ in poses])
        #         factor_end = np.vstack([p.tvec.reshape(-1,3) for p in poses])
        #         draw_utils.publish_line_segments(self.name_ + 'measured_factor_landmark', factor_st, factor_end, c='b', 
        #                                          frame_id=self.name_ + 'measured_odom', element_id=self.latest, reset=False) 

        # def on_odom_absolute(self, t, p, noise=None): 
        #     BaseSLAM.on_odom_absolute(self, t, p, noise=noise)
        #     self.visualize_poses(p, relative=False)

        # def on_odom_relative(self, t, p, noise=None): 
        #     BaseSLAM.on_odom_relative(self, t, p, noise=noise)
        #     self.visualize_poses(p, relative=True)

        # def on_pose_landmarks(self, t, ids, poses): 
        #     BaseSLAM.on_pose_landmarks(self, t, ids, poses)
        #     self.visualize_landmarks(ids, poses)

        
    
# class BaseSLAMWithViz(cls): 
#     def __init__(self, name='SLAM_', frame_id='origin',
#                  update_every_k_odom=10, 
#                  visualize_every=2.0,
#                  visualize_measurements=False, 
#                  visualize_factors=True, visualize_marginals=False): 

#         BaseSLAM.__init__(self, update_every_k_odom=update_every_k_odom)
#         self.name_ = name
#         self.frame_id_ = frame_id

#         self.visualize_every_ = visualize_every
#         self.last_viz_t_ = time.time()
#         self.last_viz_lcount_ = 0
        
#         self.visualize_factors_ = visualize_factors
#         self.visualize_marginals_ = visualize_marginals
#         self.visualize_measurements_ = visualize_measurements

#     def update(self): 
#         super(BaseSLAMWithViz, self).update()

#         now = time.time()
#         if now - self.last_viz_t_ > self.visualize_every_ and \
#            self.target_poses_count - self.last_viz_lcount_ > 0: 
#             self.visualize_optimized()
#             self.last_viz_t_ = now
#             self.last_viz_lcount_ = self.target_poses_count
            
#     def finish(self): 
#         super(BaseSLAMWithViz, self).finish()
#         self.visualize_optimized()

#     @timeitmethod
#     def visualize_optimized(self):
#         """
#         Update SLAM visualizations with optimized factors
#         """
#         self.visualize_optimized_poses()
#         self.visualize_optimized_landmarks()
        
#     def visualize_optimized_poses(self):
#         with self.state_lock_:
#             # Poses 
#             covars = []
#             updated_poses = self.updated_poses

#             # Marginals
#             if self.visualize_marginals_: 
#                 poses_marginals = self.poses_marginals
#                 triu_inds = np.triu_indices(3)
#                 if self.marginals_available: 
#                     for pid in updated_poses.keys(): 
#                         covars.append(
#                             poses_marginals.get(pid, np.eye(6, dtype=np.float32) * 100)[triu_inds]
#                         )

#             # Draw cameras (with poses and marginals)
#             draw_utils.publish_cameras(self.name_ + 'optimized_node_poses', updated_poses.values(), 
#                                        covars=covars, frame_id=self.frame_id_, draw_edges=False, reset=True)

#             # Draw odometry edges (between robot poses)
#             if self.visualize_factors_: 
#                 robot_edges = self.robot_edges
#                 if len(robot_edges): 
#                     factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in robot_edges])
#                     factor_end = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (_, xid) in robot_edges])

#                     draw_utils.publish_line_segments(self.name_ + 'optimized_factor_odom', factor_st, factor_end, c='b', 
#                                                      frame_id=self.frame_id_, reset=True) 

#     def visualize_optimized_landmarks(self):
#         with self.state_lock_:
#             # Draw targets (constantly updated, so draw with reset)
#             updated_poses = self.updated_poses
#             updated_targets = self.updated_targets
            
#             if len(updated_targets): 
#                 # texts = [self.landmark_text_lut_.get(k, str(k)) for k in updated_targets.keys()]
#                 draw_utils.publish_pose_list(self.name_ + 'optimized_node_landmark', updated_targets.values(), 
#                                              texts=[str(k) for k in updated_targets.keys()], 
#                                              frame_id=self.frame_id_, reset=True)

#                 # edges = np.vstack([draw_utils.draw_tag_edges(p) for p in updated_targets.itervalues()])
#                 # draw_utils.publish_line_segments('optimized_node_landmark', edges[:,:3], edges[:,3:6], c='r', 
#                 #                                  frame_id=frame_id, reset=True)
#                 # draw_utils.publish_tags('optimized_node_landmark', updated_targets.values(), 
#                 #                         texts=map(str, updated_targets.keys()), draw_nodes=True, draw_edges=True, 
#                 #                         frame_id=frame_id, reset=True)

#             # Draw edges (between landmarks and poses)
#             if self.visualize_factors_: 
#                 landmark_edges = self.landmark_edges
#                 if len(landmark_edges): 
#                     factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
#                     factor_end = np.vstack([(updated_targets[lid].tvec).reshape(-1,3) for (_, lid) in landmark_edges])

#                     draw_utils.publish_line_segments(self.name_ + 'optimized_factor_landmark',
#                                                      factor_st, factor_end, c='b', 
#                                                      frame_id=self.frame_id_, reset=True) 

            
        

#         # =========================================================
#         # TARGETS/LANDMARKS

#         # if self.landmark_type_ == 'pose': 

# #         elif self.landmark_type_ == 'point': 

# #             # Draw targets (constantly updated, so draw with reset)
# #             updated_targets = {pid : pt3
# #                                for (pid, pt3) in self.target_landmarks.iteritems()}
# #             if len(updated_targets): 
# #                 points3d = np.vstack(updated_targets.values())
# #                 draw_utils.publish_cloud('optimized_node_landmark', points3d, c='r', 
# #                                          frame_id=frame_id, reset=True)

# #             # Landmark points
# #             if self.visualize_nodes_: 
# #                 covars = []
# #                 poses = [Pose(k, tvec=v) for k, v in updated_targets.iteritems()]
# #                 texts = [self.landmark_text_lut_.get(k, '') for k in updated_targets.keys()] \
# #                         if len(self.landmark_text_lut_) else []

# #                 # Marginals
# #                 if self.visualize_marginals_: 
# #                     target_landmarks_marginals = self.target_landmarks_marginals
# #                     triu_inds = np.triu_indices(3)
# #                     if self.marginals_available: 
# #                         for pid in updated_targets.keys(): 
# #                             covars.append(
# #                                 target_landmarks_marginals.get(pid, np.ones(shape=(6,6)) * 10)[triu_inds]
# #                             )

# #                 # Draw landmark points
# #                 draw_utils.publish_pose_list('optimized_node_landmark_poses', poses, texts=texts, 
# #                                              covars=covars, frame_id=frame_id, reset=True)

# #             # Draw edges (between landmarks and poses)
# #             if self.visualize_factors_: 
# #                 landmark_edges = self.landmark_edges
# #                 if len(landmark_edges): 
# #                     factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
# #                     factor_end = np.vstack([(updated_targets[lid]).reshape(-1,3) for (_, lid) in landmark_edges])

# #                     draw_utils.publish_line_segments('optimized_factor_landmark', factor_st, factor_end, c='b', 
# #                                                      frame_id=frame_id, reset=True) 

# #         else: 
# #             raise ValueError('Unknown landmark_type {:}'.format(self.landmark_type_))

# #         # Maintain updated ids
# #         return set(updated_poses.keys())

            
#     def visualize_poses(self, p, relative=True):
#         if not self.visualize_measurements_:
#             return

#         # Draw odom pose
#         draw_utils.publish_pose_list(self.name_ + 'measured_odom', 
#                                      [Pose.from_rigid_transform(self.latest, self.latest_measurement_pose)], 
#                                      frame_id=self.frame_id_, reset=False)

#         # Draw odom factor
#         if self.visualize_factors_ and self.latest >= 1: 
#             factor_st = (self.measurement_pose(-2).tvec).reshape(-1,3)
#             factor_end = (self.measurement_pose(-1).tvec).reshape(-1,3)
#             draw_utils.publish_line_segments(self.name_ + 'measured_factor_odom', 
#                                              factor_st, factor_end, c='r', 
#                                              frame_id=self.frame_id_, reset=False)

#     def visualize_landmarks(self, ids, poses):
#         if not self.visualize_measurements_:
#             return
        
#         # Draw landmark pose
#         draw_utils.publish_pose_list(self.name_ + 'measured_landmarks', 
#                                      [Pose.from_rigid_transform(pid + self.latest * 100, self.latest_measurement_pose * p) 
#                                       for (pid, p) in izip(ids, poses)], 
#                                      # texts=[str(pid) for pid in ids],
#                                      frame_id=self.frame_id_, reset=False)

#         if self.visualize_factors_: 
#             # Draw odom factor
#             factor_st = np.vstack([np.zeros((1,3)) for _ in poses])
#             factor_end = np.vstack([p.tvec.reshape(-1,3) for p in poses])
#             draw_utils.publish_line_segments(self.name_ + 'measured_factor_landmark', factor_st, factor_end, c='b', 
#                                              frame_id=self.name_ + 'measured_odom', element_id=self.latest, reset=False) 

#     def on_odom_absolute(self, t, p, noise=None): 
#         BaseSLAM.on_odom_absolute(self, t, p, noise=noise)
#         self.visualize_poses(p, relative=False)

#     def on_odom_relative(self, t, p, noise=None): 
#         BaseSLAM.on_odom_relative(self, t, p, noise=noise)
#         self.visualize_poses(p, relative=True)

#     def on_pose_landmarks(self, t, ids, poses): 
#         BaseSLAM.on_pose_landmarks(self, t, ids, poses)
#         self.visualize_landmarks(ids, poses)
        
# def SLAM(visualize=False, **kwargs):
#     # kwargs: name='SLAM_', frame_id='origin'
#     if visualize: 
#         return BaseSLAMWithViz(**kwargs)
#     else: 
#         return BaseSLAM(**kwargs)

# class RobotSLAMMixin(object): 
#     def __init__(self, landmark_type='point', smart=False, update_on_odom=False, 
#                  visualize_factors=False, visualize_nodes=False, visualize_marginals=False): 
#         if not isinstance(self, (_BaseSLAM, _VisualSLAM)): 
#             raise RuntimeError('Cannot mixin without mixing with one of the SLAM classes')
#         if not ((isinstance(self, _BaseSLAM) and landmark_type == 'pose') or 
#                 (isinstance(self, _VisualSLAM) and landmark_type == 'point')): 
#             raise ValueError('Wrong expected landmark type for {}, provided {}'
#                              .format(type(self), landmark_type))

#         self.landmark_type_ = landmark_type
#         self.update_on_odom_ = update_on_odom

#         # Visualizations
#         self.visualize_factors_ = visualize_factors
#         self.visualize_nodes_ = visualize_nodes
#         self.visualize_marginals_ = visualize_marginals

#         self.landmark_text_lut_ = {}
#         self.smart_ = smart
#         self.slam_mixin_timing_st_ = time.time()

#     @property
#     def is_smart(self): 
#         return self.smart_

#     def set_landmark_texts(self, landmark_lut): 
#         " {landmark id -> landmark str, ... }"
#         self.landmark_text_lut_ = landmark_lut


#     def finish(self): 
#         for j in range(10): 
#             self.update()
#         self.update_marginals()
#         if self.smart_: 
#             ids, pts3 = self.smart_update()
#         self.vis_optimized()
#         print('{} :: Finished/Solved in {:4.2f} s'.format(self.__class__.__name__, time.time() - self.slam_mixin_timing_st_))

#     #################
#     # Visualization #
#     #################

#     def vis_measurements(self, frame_id='camera'): 
#         poses = self.q_poses_
#         assert(isinstance(poses, Accumulator))

#         # Draw robot poses
#         # draw_utils.publish_pose_t('POSE', poses.latest, frame_id=frame_id)
#         # draw_utils.publish_pose_list('CAMERA_POSES', [Pose.from_rigid_transform(poses.index, poses.latest)], 
#         #                              frame_id=frame_id, reset=False)

#         # Draw odometry link
#         if poses.length >= 2:
#             p_odom = (poses.items[-2].inverse()).oplus(poses.items[-1])
#             factor_st = (poses.items[-2].tvec).reshape(-1,3)
#             factor_end = (poses.items[-1].tvec).reshape(-1,3)
#             draw_utils.publish_line_segments('measured_factor_odom', factor_st, factor_end, c='r', 
#                                              frame_id=frame_id, reset=False)

#     def vis_optimized(self, frame_id='camera'): 
#         """
#         Update SLAM visualizations with optimized factors
#         """
#         if not self.visualize_nodes_ and \
#            not self.visualize_factors_ and \
#            not self.visualize_marginals_: 
#             return 

#         # if not draw_utils.has_sensor_frame('optcamera'): 
#         #     draw_utils.publish_sensor_frame()

#         # =========================================================
#         # POSES/MARGINALS

#         # Poses 
#         covars = []
#         updated_poses = {pid : Pose.from_rigid_transform(
#             pid, RigidTransform.from_matrix(p)) 
#                          for (pid,p) in self.poses.iteritems()}

#         # Marginals
#         if self.visualize_marginals_: 
#             poses_marginals = self.poses_marginals
#             triu_inds = np.triu_indices(3)
#             if self.marginals_available: 
#                 for pid in updated_poses.keys(): 
#                     covars.append(
#                         poses_marginals.get(pid, np.eye(6, dtype=np.float32) * 100)[triu_inds]
#                     )

#         # Draw cameras (with poses and marginals)
#         draw_utils.publish_cameras('optimized_node_poses', updated_poses.values(), 
#                                    covars=covars, frame_id=frame_id, draw_edges=False, reset=True)

#         # Draw odometry edges (between robot poses)
#         if self.visualize_factors_: 
#             robot_edges = self.robot_edges
#             if len(robot_edges): 
#                 factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in robot_edges])
#                 factor_end = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (_, xid) in robot_edges])

#                 draw_utils.publish_line_segments('optimized_factor_odom', factor_st, factor_end, c='b', 
#                                                  frame_id=frame_id, reset=True) 


#         # =========================================================
#         # TARGETS/LANDMARKS

#         if self.landmark_type_ == 'pose': 
#             # Draw targets (constantly updated, so draw with reset)
#             updated_targets = {pid : Pose.from_rigid_transform(pid, RigidTransform.from_matrix(p)) 
#                                for (pid, p) in self.target_poses.iteritems()}
#             if len(updated_targets): 
#                 texts = [self.landmark_text_lut_.get(k, str(k)) for k in updated_targets.keys()]
#                 draw_utils.publish_pose_list('optimized_node_landmark', updated_targets.values(), 
#                                              texts=texts, reset=True)

#                 # edges = np.vstack([draw_utils.draw_tag_edges(p) for p in updated_targets.itervalues()])
#                 # draw_utils.publish_line_segments('optimized_node_landmark', edges[:,:3], edges[:,3:6], c='r', 
#                 #                                  frame_id=frame_id, reset=True)
#                 # draw_utils.publish_tags('optimized_node_landmark', updated_targets.values(), 
#                 #                         texts=map(str, updated_targets.keys()), draw_nodes=True, draw_edges=True, 
#                 #                         frame_id=frame_id, reset=True)

#                 # Draw landmark points
#                 draw_utils.publish_pose_list('optimized_node_landmark_poses', poses, texts=texts, 
#                                              covars=covars, frame_id=frame_id, reset=True)

#             # Draw edges (between landmarks and poses)
#             if self.visualize_factors_: 
#                 landmark_edges = self.landmark_edges
#                 if len(landmark_edges): 
#                     factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
#                     factor_end = np.vstack([(updated_targets[lid].tvec).reshape(-1,3) for (_, lid) in landmark_edges])

#                     draw_utils.publish_line_segments('optimized_factor_landmark', factor_st, factor_end, c='b', 
#                                                      frame_id=frame_id, reset=True) 

#         elif self.landmark_type_ == 'point': 

#             # Draw targets (constantly updated, so draw with reset)
#             updated_targets = {pid : pt3
#                                for (pid, pt3) in self.target_landmarks.iteritems()}
#             if len(updated_targets): 
#                 points3d = np.vstack(updated_targets.values())
#                 draw_utils.publish_cloud('optimized_node_landmark', points3d, c='r', 
#                                          frame_id=frame_id, reset=True)

#             # Landmark points
#             if self.visualize_nodes_: 
#                 covars = []
#                 poses = [Pose(k, tvec=v) for k, v in updated_targets.iteritems()]
#                 texts = [self.landmark_text_lut_.get(k, '') for k in updated_targets.keys()] \
#                         if len(self.landmark_text_lut_) else []

#                 # Marginals
#                 if self.visualize_marginals_: 
#                     target_landmarks_marginals = self.target_landmarks_marginals
#                     triu_inds = np.triu_indices(3)
#                     if self.marginals_available: 
#                         for pid in updated_targets.keys(): 
#                             covars.append(
#                                 target_landmarks_marginals.get(pid, np.ones(shape=(6,6)) * 10)[triu_inds]
#                             )

#             # Draw edges (between landmarks and poses)
#             if self.visualize_factors_: 
#                 landmark_edges = self.landmark_edges
#                 if len(landmark_edges): 
#                     factor_st = np.vstack([(updated_poses[xid].tvec).reshape(-1,3) for (xid, _) in landmark_edges])
#                     factor_end = np.vstack([(updated_targets[lid]).reshape(-1,3) for (_, lid) in landmark_edges])

#                     draw_utils.publish_line_segments('optimized_factor_landmark', factor_st, factor_end, c='b', 
#                                                      frame_id=frame_id, reset=True) 

#         else: 
#             raise ValueError('Unknown landmark_type {:}'.format(self.landmark_type_))

#         # Maintain updated ids
#         return set(updated_poses.keys())


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

#         draw_utils.publish_line_segments('measured_factor_{:}'.format(landmark_name), factor_st, factor_end, c='b', 
#                                          frame_id=frame_id, reset=self.reset_required())
#         edges = np.vstack([draw_utils.draw_tag_edges(p) for p in p_Wt])
#         draw_utils.publish_line_segments('measured_node_{:}'.format(landmark_name), edges[:,:3], edges[:,3:6], c='b', 
#                                          frame_id=frame_id, reset=self.reset_required())

#         # # Optionally plot as Tags
#         # draw_utils.publish_pose_list('RAW_tags', p_Wt, texts=[], object_type='TAG', 
#         #                              frame_id=frame_id, reset=self.reset_required())

        
#         # # Plot OPTIMIZED tag factors
#         # if pose_id in self.updated_ids_: 
#         #     draw_utils.publish_line_segments('OPT_factors_{:}'.format(landmark_name), factor_ct_st, factor_ct_end, c='r', 
#         #                                  frame_id='optimized_poses', element_id=pose_id, reset=self.reset_required())

#         #     edges = np.vstack([draw_utils.draw_tag_edges(p) for p in p_landmarks])
#         #     draw_utils.publish_line_segments('optimized_{:}'.format(landmark_name), edges[:,:3], edges[:,3:6], c='r', 
#         #                                      frame_id='optimized_poses', element_id=pose_id, reset=self.reset_required())
        
# Right to left inheritance
# class _RobotSLAM(RobotSLAMMixin, BaseSLAM): 
#     def __init__(self, odom_noise=ODOM_NOISE, prior_noise=PRIOR_NOISE, verbose=False,
#                  smart=False, landmark_type='point', 
#                  update_on_odom=False, visualize_nodes=False, visualize_factors=False, visualize_marginals=False): 
#         BaseSLAM.__init__(self, odom_noise=odom_noise, 
#                                 prior_noise=prior_noise, verbose=verbose)
        
#         RobotSLAMMixin.__init__(self, smart=smart, landmark_type=landmark_type, 
#                                 update_on_odom=update_on_odom, 
#                                 visualize_nodes=visualize_nodes, 
#                                 visualize_factors=visualize_factors, 
#                                 visualize_marginals=visualize_marginals)

# class _RobotSLAM

# class _RobotVisualSLAM(VisualSLAM): 
#     def __init__(self, calib, min_landmark_obs=3, 
#                  odom_noise=ODOM_NOISE, prior_noise=PRIOR_NOISE, 
#                  px_error_threshold=4, px_noise=[1.0, 1.0], verbose=False)
#     # , 
#     #              landmark_type='point', smart=False, update_on_odom=False, 
#     #              visualize_factors=False, visualize_nodes=False, visualize_marginals=False): 
#         VisualSLAM.__init__(self, calib, 
#                             min_landmark_obs=min_landmark_obs, 
#                             px_error_threshold=px_error_threshold, 
#                             odom_noise=odom_noise, prior_noise=prior_noise, 
#                             px_noise=px_noise, verbose=verbose)
#         # RobotSLAM.__init__(self, smart=smart, landmark_type=landmark_type, 
#         #                         update_on_odom=update_on_odom, 
#         #                         visualize_nodes=visualize_nodes, 
#         #                         visualize_factors=visualize_factors, 
#         #                         visualize_marginals=visualize_marginals)
        


# class TagDetector(object): 
#     def __init__(self, camera, tag_size=0.166): 

#         # Setup camera (identity extrinsics)
#         self.cam_ = camera 

#         # Setup APRILTags detector
#         from pybot_apriltags import AprilTag, AprilTagsWrapper, draw_tags
        
#         self.tag_detector_ = AprilTagsWrapper()
#         self.tag_detector_.set_calib(tag_size=tag_size, fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy)

#     def process(self, im):
#         # Detect the tags
#         # im = cam.undistort(to_gray(im))
#         tags = self.tag_detector_.process(to_gray(im), return_poses=True)

#         # Visualize the tags
#         vis = draw_tags(to_color(im), tags)
#         imshow_cv('tag-detector', vis)
#         # self.slam_.draw_tags(vis, tags)
        
#         return tags

