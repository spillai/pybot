"""SLAM interface with GTSAM"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
np.set_printoptions(precision=2, suppress=True)

from collections import deque, defaultdict, Counter
from itertools import izip
from threading import Lock, RLock

from pybot.utils.timer import SimpleTimer, timeitmethod
from pybot.utils.db_utils import AttrDict
from pybot.utils.misc import print_red, print_yellow, print_green

from pygtsam import Symbol, extractPose2, extractPose3, extractPoint3, extractKeys
from pygtsam import symbol as _symbol
from pygtsam import Point2, Rot2, Pose2, \
    PriorFactorPose2, BetweenFactorPose2, \
    BearingRangeFactorPose2Point2
from pygtsam import Point3, Rot3, Pose3, \
    PriorFactorPose3, BetweenFactorPose3
from pygtsam import SmartFactor
from pygtsam import Cal3_S2, SimpleCamera, simpleCamera
from pygtsam import StereoPoint2, Cal3_S2Stereo, \
    GenericStereoFactor3D, GenericProjectionFactorPose3Point3Cal3_S2
from pygtsam import NonlinearEqualityPose3
from pygtsam import Isotropic
from pygtsam import Diagonal, Values, Marginals
from pygtsam import ISAM2, NonlinearOptimizer, \
    NonlinearFactorGraph, LevenbergMarquardtOptimizer, DoglegOptimizer

def symbol(ch, i): 
    return _symbol(ord(ch), i)

def vector(v): 
    return np.float64(v)

def matrix(m): 
    return np.float64(m)

def vec(*args):
    return vector(list(args)) 

_odom_noise = np.ones(6) * 0.01
_prior_noise = np.ones(6) * 0.001
_measurement_noise = np.ones(6) * 0.4

class BaseSLAM(object):
    odom_noise = _odom_noise
    prior_noise = _prior_noise
    measurement_noise = _measurement_noise
    """
    Basic SLAM interface with GTSAM::ISAM2

    This is a basic interface that allows hot-swapping factors without
    having to write much boilerplate and templated code.
    
    Factor graph is constructed and grown dynamically, with
    Pose3-Pose3 constraints, and finally optimized.

    Params: 
        xs: Robot poses
        ls: Landmark measurements
        xls: Edge list between x and l 

    Todo: 
        Updated slam every landmark addition

    """
    def __init__(self, 
                 odom_noise=_odom_noise, 
                 prior_noise=_prior_noise, 
                 measurement_noise=_measurement_noise,
                 verbose=False, export_graph=False):
 
        # ISAM2 interface
        self.slam_ = ISAM2()
        self.slam_lock_ = Lock()

        self.idx_ = -1
        self.verbose_ = verbose

        # Factor graph storage
        self.graph_ = NonlinearFactorGraph()
        self.initial_ = Values()
        
        # Pose3D measurement
        self.measurement_noise_ = Diagonal.Sigmas(measurement_noise)
        self.prior_noise_ = Diagonal.Sigmas(prior_noise)
        self.odo_noise_ = Diagonal.Sigmas(odom_noise)

        # Optimized robot state
        self.state_lock_ = Lock()
        self.xs_ = {}
        self.ls_ = {}
        self.xls_ = []
        self.xxs_ = []

        self.xcovs_ = {}
        self.lcovs_ = {}
        self.current_ = None

        # Timestamped look up for landmarks, and poses
        self.timer_ls_ = defaultdict(list)
        self.export_graph_ = export_graph

        # Graph visualization
        # if self.export_graph_: 
        #     self.gviz_ = nx.Graph()

    def initialize(self, p_init=None, index=0, noise=None): 
        # print_red('\t\t{:}::add_p0 index: {:}'.format(self.__class__.__name__, index))

        x_id = symbol('x', index)
        pose0 = Pose3(p_init) if p_init is not None else Pose3()
            
        self.graph_.add(
            PriorFactorPose3(x_id, pose0,
                             Diagonal.Sigmas(noise)
                             if noise is not None
                             else self.prior_noise_)
        )
        self.initial_.insert(x_id, pose0)
        with self.state_lock_: 
            self.xs_[index] = pose0
        self.idx_ = index

        # Add node to graphviz
        # if self.export_graph_: 
        #     self.gviz_.add_node(x_id) # , label='x0')
        #     self.gviz_.node[x_id]['label'] = 'X %i' % index

        #     # Add prior factor to graphviz
        #     p_id = symbol('p', index)
        #     self.gviz_.add_edge(p_id, symbol('x', index))
        #     self.gviz_.node[p_id]['label'] = 'P %i' % index
        #     self.gviz_.node[p_id]['color'] = 'blue'
        #     self.gviz_.node[p_id]['style'] = 'filled'
        #     self.gviz_.node[p_id]['shape'] = 'box'

    def add_prior(self, index, p, noise=None): 
        x_id = symbol('x', index)
        pose = Pose3(p)
        self.graph_.add(
            PriorFactorPose3(x_id, pose, Diagonal.Sigmas(noise)
                             if noise is not None
                             else self.prior_noise_)
        )
        
    def add_odom_incremental(self, delta, noise=None): 
        """
        Add odometry measurement from the latest robot pose to a new
        robot pose
        """
        # Add prior on first pose
        if not self.is_initialized:
            self.initialize()

        # Add odometry factor
        self.add_relative_pose_constraint(self.latest, self.latest+1, delta, noise=noise)
        self.idx_ += 1

    def add_relative_pose_constraint(self, xid1, xid2, delta, noise=None): 
        # print_red('\t\t{:}::add_odom {:}->{:}'.format(self.__class__.__name__, xid1, xid2))

        # Add odometry factor
        pdelta = Pose3(delta)
        x_id1, x_id2 = symbol('x', xid1), symbol('x', xid2)
        self.graph_.add(BetweenFactorPose3(x_id1, x_id2, 
                                           pdelta, Diagonal.Sigmas(noise)
                                           if noise is not None
                                           else self.odo_noise_))
        
        # Predict pose and add as initial estimate
        with self.state_lock_: 
            if xid2 not in self.xs_: 
                pred_pose = self.xs_[xid1].compose(pdelta)
                self.initial_.insert(x_id2, pred_pose)
                self.xs_[xid2] = pred_pose

            # Add to edges
            self.xxs_.append((xid1, xid2))

        # Add edge to graphviz
        # if self.export_graph_: 
        #     self.gviz_.add_edge(x_id1, x_id2)
        #     self.gviz_.node[x_id2]['label'] = 'X ' + str(xid2)

    def add_pose_landmarks(self, xid, lids, deltas, noise=None): 
        if self.verbose_: 
            print_red('\t\t{:}::add_landmark x{:} -> lcount: {:}'
                      .format(self.__class__.__name__, xid, len(lids)))

        # Add Pose-Pose landmark factor
        x_id = symbol('x', xid)
        l_ids = [symbol('l', lid) for lid in lids]
        
        # Add landmark poses
        noise = Diagonal.Sigmas(noise) if noise is not None \
                else self.measurement_noise_
        assert(len(l_ids) == len(deltas))
        for l_id, delta in izip(l_ids, deltas): 
            pdelta = Pose3(delta)
            self.graph_.add(BetweenFactorPose3(x_id, l_id, pdelta, noise))

        with self.state_lock_: 

            # Add to landmark measurements
            self.xls_.extend([(xid, lid) for lid in lids])

            # Add landmark edge to graphviz
            # if self.export_graph_: 
            #     self.gviz_.add_edge(x_id, l_id)

            # Initialize new landmark pose node from the latest robot
            # pose. This should be done just once
            for (l_id, lid, delta) in izip(l_ids, lids, deltas): 
                if lid not in self.ls_:
                    try: 
                        pred_pose = self.xs_[xid].compose(Pose3(delta))
                        self.initial_.insert(l_id, pred_pose)
                        self.ls_[lid] = pred_pose
                        self.timer_ls_[xid].append(lid)
                    except: 
                        raise KeyError('Pose {:} not available'
                                       .format(xid))

                    # Label landmark node
                    # if self.export_graph_: 
                    #     self.gviz_.node[l_id]['label'] = 'L ' + str(lid)            
                    #     self.gviz_.node[l_id]['color'] = 'red'
                    #     self.gviz_.node[l_id]['style'] = 'filled'
            
        return 

    def add_pose_landmarks_incremental(self, lid, delta, noise=None): 
        """
        Add landmark measurement (pose3d) 
        from the latest robot pose to the
        specified landmark id
        """
        self.add_pose_landmarks(self.latest, lid, delta, noise=noise)

    def add_point_landmarks(self, xid, lids, pts, pts3d, noise=None): 
        if self.verbose_: 
            print_red('\t\tadd_landmark_points xid:{:}-> lid count:{:}'
                      .format(xid, len(lids)))
        
        # Add landmark-ids to ids queue in order to check
        # consistency in matches between keyframes. This 
        # allows an easier interface to check overlapping 
        # ids across successive function calls.
        self.lid_count_ += Counter(lids)

        # if len(ids_q_) < 2: 
        #     return

        # Add Pose-Pose landmark factor
        x_id = symbol('x', xid)
        l_ids = [symbol('l', lid) for lid in lids]

        noise = Diagonal.Sigmas(noise) if noise is not None \
                else self.image_measurement_noise_
        assert(len(l_ids) == len(pts) == len(pts3d))
        for l_id, pt in izip(l_ids, pts):
            self.graph_.add(
                GenericProjectionFactorPose3Point3Cal3_S2(
                    Point2(vec(*pt)), noise, x_id, l_id, self.K_))

        with self.state_lock_: 

            # # Add to landmark measurements
            # self.xls_.extend([(xid, lid) for lid in lids])

            # Add landmark edge to graphviz
            # if self.export_graph_: 
            #     for l_id in l_ids: 
            #         self.gviz_.add_edge(x_id, l_id)

            # Initialize new landmark pose node from the latest robot
            # pose. This should be done just once
            for (l_id, lid, pt3) in izip(l_ids, lids, pts3d): 
                if lid not in self.ls_: 
                    try: 
                        pred_pt3 = self.xs_[xid].transform_from(Point3(vec(*pt3)))
                        self.initial_.insert(l_id, pred_pt3)
                        self.ls_[lid] = pred_pt3
                        self.timer_ls_[xid].append(lid)
                    except Exception, e: 
                        raise RuntimeError('Initialization failed ({:}). xid:{:}, lid:{:}, l_id: {:}'
                                           .format(e, xid, lid, l_id))

                    # Label landmark node
                    # if self.export_graph_: 
                    #     self.gviz_.node[l_id]['label'] = 'L ' + str(lid)            
                    #     self.gviz_.node[l_id]['color'] = 'red'
                    #     self.gviz_.node[l_id]['style'] = 'filled'
        
        return 

    def add_point_landmarks_incremental(self, lids, pts, pts3d, noise=None): 
        """
        Add landmark measurement (image features)
        from the latest robot pose to the
        set of specified landmark ids
        """
        self.add_point_landmarks(self.latest, lids, pts, pts3d, noise=noise)

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
        return {k: v.matrix() for k,v in self.xs_.iteritems()}

    def pose(self, k): 
        return self.xs_[k].matrix()
        
    @property
    def target_poses(self): 
        " Landmark Poses: Expects landmarks to be Pose3 "
        return {k: v.matrix() for k,v in self.ls_.iteritems()}

    @property
    def target_poses_count(self): 
        " Landmark Poses: Expects landmarks to be Pose3 "
        return len(self.ls_)

    def target_pose(self, k): 
        return self.ls_[k].matrix()
        
    @property
    def target_landmarks(self): 
        " Landmark Points: Expects landmarks to be Point3 " 
        return {k: v.vector().ravel() for k,v in self.ls_.iteritems()}

    @property
    def target_landmarks_count(self): 
        " Landmark Points: Expects landmarks to be Point3 " 
        return len(self.ls_)

    def target_landmark(self, k): 
        return self.ls_[k].vector().ravel()

    @property
    def poses_marginals(self): 
        " Marginals for Robot poses: Expects poses to be Pose3 "
        return self.xcovs_ 
        
    @property
    def target_poses_marginals(self): 
        " Marginals for Landmark Poses: Expects landmarks to be Pose3 "
        return self.lcovs_
        
    @property
    def target_landmarks_marginals(self): 
        " Marginals for Landmark Points: Expects landmarks to be Point3 " 
        return self.lcovs_

    def pose_marginal(self, node_id): 
        return self.xcovs_[node_id]

    def landmark_marginal(self, node_id): 
        return self.lcovs_[node_id]
        
    @property
    def landmark_edges(self): 
        return self.xls_

    @property
    def robot_edges(self): 
        return self.xxs_

    @property
    def estimate_available(self): 
        return self.current_ is not None

    @property
    def marginals_available(self): 
        return len(self.xcovs_) > 0 or len(self.lcovs_) > 0

    def save_graph(self, filename):
        with self.slam_lock_: 
            self.slam_.saveGraph(filename)

    # def save_dot_graph(self, filename): 
    #     nx.write_dot(self.gviz_, filename)
    #     # nx.draw_graphviz(self.gviz_, prog='neato')
    #     # nx_force_draw(self.gviz_)

    @timeitmethod
    def _update(self): 
        # print('.')

        # Update ISAM with new nodes/factors and initial estimates
        with self.slam_lock_: 
            self.slam_.update(self.graph_, self.initial_)
            self.slam_.update()

        # Get current estimate
        with self.slam_lock_: 
            self.current_ = self.slam_.calculateEstimate()
            
    def _batch_solve(self):
        # Optimize using Levenberg-Marquardt optimization
        with self.slam_lock_:
            opt = LevenbergMarquardtOptimizer(self.graph_, self.initial_)
            self.current_ = opt.optimize();
            
    def _update_estimates(self): 
        if not self.estimate_available:
            raise RuntimeError('Estimate unavailable, call update first')

        poses = extractPose3(self.current_)
        landmarks = extractPoint3(self.current_)

        with self.state_lock_: 
            # Extract and update landmarks and poses
            for k,v in poses.iteritems():
                if k.chr() == ord('l'): 
                    self.ls_[k.index()] = v
                elif k.chr() == ord('x'): 
                    self.xs_[k.index()] = v
                else: 
                    raise RuntimeError('Unknown key chr {:}'.format(k.chr))

            # Extract and update landmarks
            for k,v in landmarks.iteritems():
                if k.chr() == ord('l'): 
                    self.ls_[k.index()] = v
                else: 
                    raise RuntimeError('Unknown key chr {:}'.format(k.chr))

        self.graph_.resize(0)
        self.initial_.clear()
        # self.cleanup()
        
        # if self.index % 10 == 0 and self.index > 0: 
        #     self.save_graph("slam_fg.dot")
        #     self.save_dot_graph("slam_graph.dot")

    def _update_marginals(self): 
        if not self.estimate_available:
            raise RuntimeError('Estimate unavailable, call update first')

        # Retrieve marginals for each of the poses
        with self.slam_lock_: 
            for xid in self.xs_: 
                self.xcovs_[xid] = self.slam_.marginalCovariance(symbol('x', xid))

            for lid in self.ls_: 
                self.lcovs_[lid] = self.slam_.marginalCovariance(symbol('l', lid))

    def cleanup(self): 

        clean_l = []
        idx = self.latest
        for index in self.timer_ls_.keys(): 
            if abs(idx-index) > 20: 
                lids = self.timer_ls_[index]
                for lid in lids: 
                    self.ls_.pop(lid)
                    clean_l.append(lid)
                self.timer_ls_.pop(index)
                
        # clean_x = []
        # for index in self.xs_.keys(): 
        #     if abs(idx-index) > 20: 
        #         self.xs_.pop(index)
        #         clean_x.append(index)

        print clean_l
        

class VisualSLAM(BaseSLAM): 
    def __init__(self, calib, min_landmark_obs=3, 
                 odom_noise=_odom_noise, prior_noise=_prior_noise,
                 px_error_threshold=4, px_noise=[1.0, 1.0], verbose=False):
        BaseSLAM.__init__(self, odom_noise=odom_noise, prior_noise=prior_noise, verbose=verbose)

        self.px_error_threshold_ = px_error_threshold
        self.min_landmark_obs_ = min_landmark_obs
        assert(self.min_landmark_obs_ >= 2)

        # Define the camera calibration parameters
        # format: fx fy skew cx cy
            
        # Calibration for specific instance
        # that is maintained across the entire
        # pose-graph optimization (assumed static)
        self.K_ = Cal3_S2(calib.fx, calib.fy, 0.0, calib.cx, calib.cy)

        # Counter for landmark observations
        self.lid_count_ = Counter()
            
        # Dictionary pointing to smartfactor set
        # for each landmark id
        # self.lid_factors_ = defaultdict(SmartFactor)
        # self.lid_factors_ = defaultdict(lambda: dict(
        #     in_graph=False, factor=SmartFactor(rankTol=1, linThreshold=-1, manageDegeneracy=False)))
        self.lid_factors_ = defaultdict(lambda: dict(in_graph=False, factor=SmartFactor()))
        self.lid_update_needed_ = np.int64([])
        
        # Measurement noise (2 px in u and v)
        self.image_measurement_noise_ = Diagonal.Sigmas(vec(*px_noise))

    def add_point_landmarks_incremental_smart(self, lids, pts, keep_tracked=True): 
        """
        Add landmark measurement (image features)
        from the latest robot pose to the
        set of specified landmark ids
        """
        self.add_point_landmarks_smart(self.latest, lids, pts, keep_tracked=keep_tracked)

    @timeitmethod
    def add_point_landmarks_smart(self, xid, lids, pts, keep_tracked=True): 
        """
        keep_tracked: Maintain only tracked measurements in the smart factor list; 
        The alternative is that all measurements are added to the smart factor list
        """
        # print_red('\t\t{:}::add_landmark_points_smart {:}->{:}'.format(self.__class__.__name__, xid, len(lids)))
        if self.verbose_: 
            print_red('\t\t{:}::add_landmark_points_smart {:}->{:}'
                      .format(self.__class__.__name__, xid, lids))

        # Mahalanobis check before adding points to the 
        # factor graph
        # self.check_point_landmarks(xid, lids, pts)
        
        # Add landmark-ids to ids queue in order to check 
        # consistency in matches between keyframes. This 
        # allows an easier interface to check overlapping 
        # ids across successive function calls.

        # Only maintain lid counts for previously tracked 
        for lid in self.lid_count_.keys():
            if lid not in self.lid_factors_: 
                self.lid_count_.pop(lid)

        # Add new tracks to the counter
        self.lid_count_ += Counter(lids)

        # Add Pose-Pose landmark factor
        x_id = symbol('x', xid)
        l_ids = [symbol('l', lid) for lid in lids]
        
        assert(len(l_ids) == len(pts))
        for (lid, l_id, pt) in izip(lids, l_ids, pts):

            # If the landmark is already initialized, 
            # then add to graph
            if lid in self.ls_:
                # print_yellow('Adding graph measurement: {:}'.format(lid))

                # # Add projection factors 
                # self.graph_.add(GenericProjectionFactorPose3Point3Cal3_S2(
                #     Point2(vec(*pt)), self.image_measurement_noise_, x_id, l_id, self.K_))
                
                # Add to landmark measurements
                self.xls_.append((xid, lid))

            # In case the landmarks have not been initialized, add 
            # as a smart factor and delay until multiple views have
            # been registered
            else: 
                # print_yellow('Adding smartfactor measurement: {:}'.format(lid))
                # Insert smart factor based on landmark id
                self.lid_factors_[lid]['factor'].add_single(
                    Point2(vec(*pt)), x_id, self.image_measurement_noise_, self.K_
                )
                
        # Keep only successively tracked features
        if not keep_tracked: 
            # Add smartfactors to the graph only if that 
            # landmark ID is no longer visible. setdiff1d
            # returns the set of IDs that are unique to 
            # `smart_lids` (previously tracked) but not 
            # in `lids` (current)
            smart_lids = np.int64(self.lid_factors_.keys())

            # Determine old lids that are no longer tracked and add
            # only the ones that have at least min_landmark_obs
            # observations. Delete old factors that have insufficient
            # number of observations

            dropped_lids = np.setdiff1d(smart_lids, lids)
            for lid in dropped_lids:
                self.lid_factors_.pop(lid)

        if self.verbose_: 
            self.print_stats()
        return 

    def print_stats(self): 
        print_red('\tLID factors: {}\n'
                  '\tLID count: {}\n'.format(len(self.lid_factors_), len(self.lid_count_)))

    @timeitmethod
    def smart_update(self, delete_factors=True): 
        """
        Update the smart factors and add 
        to the graph. Once the landmarks are 
        extracted, remove them from the factor list
        """
        current = self.slam_.calculateEstimate()

        ids, pts3 = [], []
        for lid in self.lid_factors_.keys(): 

            # No need to initialize smart factor if already 
            # added to the graph OR 
            # Cannot incorporate factor without sufficient observations
            lid_factor = self.lid_factors_[lid]
            if lid_factor['in_graph'] or self.lid_count_[lid] < self.min_landmark_obs_:
                continue

            l_id = symbol('l', lid)
            smart = lid_factor['factor']

            # Cannot do much when degenerate or behind camera
            if smart.isDegenerate() or smart.isPointBehindCamera():
                continue

            # Check smartfactor reprojection error 
            err = smart.error(current)
            if err > self.px_error_threshold_ or err <= 0.0:
                continue

            # Add triangulated smart factors back into the graph for
            # complete point-pose optimization Each of the projection
            # factors, including the points, and their initial values
            # are added back to the graph. Optionally, we can choose
            # to subsample and add only a few measurements from the
            # set of original measurements

            x_ids = smart.keys()
            pts = smart.measured()
            assert len(pts) == len(x_ids)

            # Add each of the smart factor measurements to the 
            # factor graph
            for x_id,pt in zip(x_ids, pts): 
                # self.graph_.add(GenericProjectionFactorPose3Point3Cal3_S2(
                #     pt, self.image_measurement_noise_, x_id, l_id, self.K_))
                
                # # Add to landmark measurements
                # self.xls_.append((Symbol(x_id).index(), lid))
                pass

            # Initialize the point value, set in_graph, and
            # remove the smart factor once point is computed
            pt3 = smart.point_compute(current)

            del self.lid_factors_[lid]['factor']
            self.lid_factors_.pop(lid)

            # self.lid_factors_[lid]['in_graph'] = True
            # self.lid_factors_[lid]['factor'] = None

            # Provide initial estimate to factor graph
            assert(lid not in self.ls_)
            if lid not in self.ls_: 
                self.initial_.insert(l_id, pt3)
            self.ls_[lid] = pt3
            self.timer_ls_[self.latest].append(lid)

            # Add the points for visualization 
            ids.append(lid)
            pts3.append(pt3.vector().ravel())

            # if delete_factors: 
            #     # Once all observations are incorporated, 
            #     # remove feature altogether. Can be deleted
            #     # as long as the landmarks are initialized
            #     self.lid_count_.pop(lid)

        try: 
            ids, pts3 = np.int64(ids).ravel(), np.vstack(pts3)            
            assert(len(ids) == len(pts3))
            return ids, pts3
        except Exception, e:
            # print('Could not return pts3, {:}'.format(e))
            return np.int64([]), np.array([])        

    # def check_point_landmarks(self, xid, lids, pts): 
    #     print_red('\t\t{:}::check_point_landmarks {:}->{:}, ls:{:}'.format(
    #         self.__class__.__name__, xid, len(lids), len(self.ls_)))


    #     print 'new landmark', xid, lids

    #     if not len(self.ls_): 
    #         return

    #     # Recover pose
    #     camera = SimpleCamera(self.xs_[xid], self.K_)
    #     # Project landmarks onto current frame, 
    #     # and check mahalanobis distance
    #     for (pt, lid) in izip(pts, lids): 
    #         if lid not in self.ls_: 
    #             continue

    #         print 'new landmark', xid, lid
        

    #         # Project feature onto camera and check
    #         # distance metric
    #         pred_pt = camera.project(self.ls_[lid])
    #         print('LID {:} Distance {:}'.format(lid, pred_pt.vector().ravel()-pt))
            
    #         # pred_cov = camera.project(self.lcovs_[lid])
    #         # mahalanobis_distance(pt, pred_pt)

