"""Pose Graph interface with Toro"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import sys
import numpy as np
np.set_printoptions(precision=2, suppress=True)

# from collections import deque, defaultdict, Counter, namedtuple
# from itertools import izip
from threading import Lock, RLock

from pybot.geometry.rigid_transform import RigidTransform, Quaternion
from pybot.utils.timer import SimpleTimer, timeitmethod
from pybot.utils.db_utils import AttrDict
from pybot.utils.misc import print_red, print_yellow, print_green
from pybot.mapping import cfg

from pytoro import TreeOptimizer3, Transformation3, tf3_from_vec, tf3_to_vec

FLOAT = np.float64

def rt_vec(rt=RigidTransform.identity()):
    return tf3_from_vec(np.hstack([rt.tvec, rt.wxyz]).astype(FLOAT))

def rt_from_vec(v):
    return RigidTransform(tvec=v[:3], xyzw=Quaternion.from_xyzw(v[3:]))

class BaseSLAM(object):
    """
    BASIC Pose Graph interface with Toro
    
    Factor graph is constructed and grown dynamically, with
    Pose3-Pose3 constraints, and iteratively optimized.

    Params: 
        xs: Robot poses
        ls: Landmark measurements
        xls: Edge list between x and l 

    Todo: 
        Updated slam every landmark addition

    """
    def __init__(self, 
                 odom_noise=cfg.ODOM_NOISE, 
                 prior_pose_noise=cfg.PRIOR_POSE_NOISE, 
                 measurement_noise=cfg.MEASUREMENT_NOISE,
                 verbose=False, export_graph=False):

        # Toro interface

        self.pg_ = TreeOptimizer3()
        self.idx_ = -1
        self.verbose_ = verbose

        # Optimized robot state
        self.state_lock_ = Lock()
        self.xs_ = {}
        self.xxs_ = []

        self.xcovs_ = {}
        self.current_ = None

    def initialize(self, p=None, index=0, noise=None): 
        if self.verbose_:
            print_red('{}::initialize index: {}={}'
                      .format(self.__class__.__name__, index, p))
            print_red('{:}::add_pose_prior {}={}'
                      .format(self.__class__.__name__, index, p))

        pose0 = p if p is not None else RigidTransform.identity()
        with self.state_lock_:
            self.pg_.addVertex(index, rt_vec(pose0))
            self.xs_[index] = pose0 
        self.idx_ = index

    def add_incremental_pose_constraint(self, delta, noise=None):
        # Add prior on first pose
        if not self.is_initialized:
            self.initialize(p=None, index=0)

        # Predict pose and add as initial estimate
        with self.state_lock_: 
            assert(self.latest + 1 not in self.xs_)
            assert(self.latest in self.xs_)
            pred_pose = self.xs_[self.latest].oplus(delta)
            self.pg_.addVertex(self.latest + 1, rt_vec(pred_pose))
            self.xs_[self.latest + 1] = pred_pose
            
        # Add odometry factor
        self.add_relative_pose_constraint(self.latest, self.latest+1, delta, noise=noise)
        self.idx_ += 1
        
    def add_relative_pose_constraint(self, xid1, xid2, delta, noise=None): 
        if self.verbose_:
            print_red('{}::add_odom {}->{} = {}'
                      .format(self.__class__.__name__, xid1, xid2, delta))

        # Add odometry factor
        sxyz, srpy = 0.01, 0.001
        inf_m = 1. / noise.astype(FLOAT) if noise else \
                1. / FLOAT([sxyz, sxyz, sxyz, srpy, srpy, srpy])
        # assert(isinstance(delta, RigidTransform))
        pdelta = rt_vec(delta)

        self.pg_.addEdge(xid1, xid2, pdelta, inf_m)
        # import warnings; warnings.warn('Ignoring odom noise')
        
        # x_id1, x_id2 = symbol('x', xid1), symbol('x', xid2)
        # self.graph_.add(BetweenFactorPose3(x_id1, x_id2, 
        #                                    pdelta, Diagonal.Sigmas(noise)
        #                                    if noise is not None
        #                                    else self.odo_noise_))

        # # Predict pose and add as initial estimate
        # with self.state_lock_: 
        #     if xid2 not in self.xs_: 
        #         pred_pose = self.xs_[xid1].oplus(pdelta)
        #         self.initial_.insert(x_id2, pred_pose)
        #         self.xs_[xid2] = pred_pose

        #     # Add to edges
        #     self.xxs_.append((xid1, xid2))

    @timeitmethod
    def _update(self): 
        # print('.')
        # print('_update {}'.format(self.idx_))
        
        # Get current estimate
        self.current_ = self.pg_.vertices()

    @timeitmethod
    def _update_estimates(self): 
        if not self.estimate_available:
            raise RuntimeError('Estimate unavailable, call update first')

        with self.state_lock_: 
            # Extract and update landmarks and poses
            for k,v in self.current_.iteritems():
                self.xs_[k] = rt_from_vec(v)

    @property
    def latest(self): 
        return self.idx_

    @property
    def index(self): 
        return self.idx_

    @property
    def is_initialized(self): 
        return self.latest >= 0

    @property
    def poses_count(self): 
        " Robot poses: Expects poses to be Pose3 "
        return len(self.xs_)

    @property
    def poses(self): 
        " Robot poses: Expects poses to be Pose3 "
        return self.xs_

    def pose(self, k): 
        return self.xs_[k]

    # @property
    # def poses_marginals(self): 
    #     " Marginals for Robot poses: Expects poses to be Pose3 "
    #     return self.xcovs_ 

    # def pose_marginal(self, node_id): 
    #     return self.xcovs_[node_id]


    
    # @property
    # def target_poses(self): 
    #     " Landmark Poses: Expects landmarks to be Pose3 "
    #     return {k: v.matrix() for k,v in self.ls_.iteritems()}

    # @property
    # def target_poses_count(self): 
    #     " Landmark Poses: Expects landmarks to be Pose3 "
    #     return len(self.ls_)

    # def target_pose(self, k): 
    #     return self.ls_[k].matrix()
        
    # @property
    # def target_landmarks(self): 
    #     " Landmark Points: Expects landmarks to be Point3 " 
    #     return {k: v.vector().ravel() for k,v in self.ls_.iteritems()}

    # @property
    # def target_landmarks_count(self): 
    #     " Landmark Points: Expects landmarks to be Point3 " 
    #     return len(self.ls_)

    # def target_landmark(self, k): 
    #     return self.ls_[k].vector().ravel()
        
    # @property
    # def target_poses_marginals(self): 
    #     " Marginals for Landmark Poses: Expects landmarks to be Pose3 "
    #     return self.lcovs_
        
    # @property
    # def target_landmarks_marginals(self): 
    #     " Marginals for Landmark Points: Expects landmarks to be Point3 " 
    #     return self.lcovs_

    # def landmark_marginal(self, node_id): 
    #     return self.lcovs_[node_id]
        
    # @property
    # def landmark_edges(self): 
    #     return self.xls_

    @property
    def robot_edges(self): 
        return self.xxs_

    @property
    def estimate_available(self): 
        return self.current_ is not None and len(self.current_) > 0

    # @property
    # def marginals_available(self): 
    #     return len(self.xcovs_) > 0 or len(self.lcovs_) > 0

    # def save_graph(self, filename):
    #     # with self.slam_lock_: 
    #     self.slam_.saveGraph(filename)
            
