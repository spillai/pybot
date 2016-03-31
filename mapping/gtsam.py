import numpy as np
import networkx as nx

from collections import deque, defaultdict, Counter
from itertools import izip
from bot_utils.misc import print_red, print_green

from pygtsam import extractPose2, extractPose3, extractKeys
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

np.set_printoptions(precision=2, suppress=True)

def symbol(ch, i): 
    return _symbol(ord(ch), i)

def vector(v): 
    return np.float64(v)

def matrix(m): 
    return np.float64(m)

def vec(*args):
    return vector(list(args)) 

def plot2DTrajectory(values, linespec, marginals=[]): 
    pass

    # poses = extractPose2(values)
    # X = poses[:,1]
    # Y = poses[:,2]
    # theta = poses[:,3]

    # # if len(marginals): 
    # #     C = np.cos(theta)
    # #     S = np.sin(theta)

    # print values, marginals

class BaseSLAM(object): 
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

    def __init__(self, calib=None): 
        # ISAM2 interface
        self.slam_ = ISAM2()
        self.idx_ = -1

        # Define the camera calibration parameters
        # format: fx fy skew cx cy
        if calib is not None: 
            
            # Calibration for specific instance
            # that is maintained across the entire
            # pose-graph optimization (assumed static)
            self.K_ = Cal3_S2(calib.fx, calib.fy, 0.0, calib.cx, calib.cy)

            # Counter for landmark observations
            self.lid_count_ = Counter()
            
            # Dictionary pointing to smartfactor set
            # for each landmark id
            self.lid_factors_ = defaultdict(SmartFactor)
            # self.lid_factors_ = defaultdict(lambda: SmartFactor(rankTol=1, linThreshold=-1, manageDegeneracy=True))
            self.lid_to_xids_ = defaultdict(list)

            self.lid_update_needed_ = np.int64([])

            # Measurement noise (1 px in u and v)
            self.image_measurement_noise_ = Diagonal.Sigmas(vec(4.0, 4.0))

            # # Mainly meant for synchronization of 
            # # ids across time frames for SFM/VSLAM 
            # # related tasks
            # self.lids_q_ = deque(maxlen=2)
            # self.xid_q_ = deque(maxlen=2)
            # self.pts_q_ = defaultdict(list)
            # self.pts3d_q_ = defaultdict(list)

        # Factor graph storage
        self.graph_ = NonlinearFactorGraph()
        self.initial_ = Values()
        
        # Pose3D measurement
        self.measurement_noise_ = Isotropic.Sigma(6, 0.4)
        self.prior_noise_ = Isotropic.Sigma(6, 0.01)
        self.odo_noise_ = Isotropic.Sigma(6, 0.01)

        # Optimized robot state
        self.xs_ = {}
        self.ls_ = {}
        self.xls_ = []

        # Graph visualization
        self.gviz_ = nx.Graph()

    @property
    def poses(self): 
        return {k: v.matrix for k,v in self.xs_.iteritems()}
        
    @property
    def targets(self): 
        return {k: v.matrix for k,v in self.ls_.iteritems()}
        
    @property
    def target_landmarks(self): 
        return {k: v for k,v in self.ls_.iteritems()}
        
    @property
    def edges(self): 
        return self.xls_

    def initialize(self, p_init=None, index=0): 
        print_red('\t\t{:}::add_p0 index: {:}'.format(self.__class__.__name__, index))
        x_id = symbol('x', index)
        pose0 = Pose3(p_init) if p_init is not None else Pose3()
            
        self.graph_.add(
            PriorFactorPose3(x_id, pose0, self.prior_noise_)
        )
        self.initial_.insert(x_id, pose0)
        self.xs_[index] = pose0
        self.idx_ = index

        # Add node to graphviz
        self.gviz_.add_node(x_id) # , label='x0')
        self.gviz_.node[x_id]['label'] = 'X %i' % index

        # Add prior factor to graphviz
        p_id = symbol('p', index)
        self.gviz_.add_edge(p_id, symbol('x', index))
        self.gviz_.node[p_id]['label'] = 'P %i' % index
        self.gviz_.node[p_id]['color'] = 'blue'
        self.gviz_.node[p_id]['style'] = 'filled'
        self.gviz_.node[p_id]['shape'] = 'box'

    def add_odom_incremental(self, delta): 
        """
        Add odometry measurement from the latest robot pose to a new
        robot pose
        """
        # Add prior on first pose
        if not self.is_initialized:
            self.initialize()

        # Add odometry factor
        self.add_odom(self.latest, self.latest+1, delta)
        self.idx_ += 1

    def add_odom(self, xid1, xid2, delta): 
        print_red('\t\t{:}::add_odom {:}->{:}'.format(self.__class__.__name__, xid1, xid2))

        # Add odometry factor
        pdelta = Pose3(delta)
        x_id1, x_id2 = symbol('x', xid1), symbol('x', xid2)
        self.graph_.add(BetweenFactorPose3(x_id1, x_id2, 
                                           pdelta, self.odo_noise_))
        
        # Predict pose and add as initial estimate
        pred_pose = self.xs_[xid1].compose(pdelta)
        self.initial_.insert(x_id2, pred_pose)
        self.xs_[xid2] = pred_pose

        # Add edge to graphviz
        self.gviz_.add_edge(x_id1, x_id2)
        self.gviz_.node[x_id2]['label'] = 'X ' + str(xid2)

    def add_landmark(self, xid, lid, delta): 
        print_red('\t\t{:}::add_landmark {:}->{:}'.format(self.__class__.__name__, xid, lid))

        # Add Pose-Pose landmark factor
        x_id = symbol('x', xid)
        l_id = symbol('l', lid)
        
        # Add landmark pose
        pdelta = Pose3(delta)
        self.graph_.add(BetweenFactorPose3(x_id, l_id, pdelta, 
                                           self.measurement_noise_))

        # Add to landmark measurements
        self.xls_.append((xid, lid))

        # Add landmark edge to graphviz
        self.gviz_.add_edge(x_id, l_id)

        # Initialize new landmark pose node from the latest robot
        # pose. This should be done just once
        if lid not in self.ls_:
            try: 
                pred_pose = self.xs_[xid].compose(pdelta)
                self.initial_.insert(l_id, pred_pose)
                self.ls_[lid] = pred_pose
            except: 
                raise KeyError('Pose {:} not available'
                               .format(xid))

            # Label landmark node
            self.gviz_.node[l_id]['label'] = 'L ' + str(lid)            
            self.gviz_.node[l_id]['color'] = 'red'
            self.gviz_.node[l_id]['style'] = 'filled'
            
        return 

    def add_landmark_points_smart(self, xid, lids, pts): 
        print_red('\t\t{:}::add_landmark_points_smart {:}->{:}'.format(self.__class__.__name__, xid, len(lids)))
        
        # Add landmark-ids to ids queue in order to check
        # consistency in matches between keyframes. This 
        # allows an easier interface to check overlapping 
        # ids across successive function calls.
        self.lid_count_ += Counter(lids)

        # Add Pose-Pose landmark factor
        x_id = symbol('x', xid)
        l_ids = [symbol('l', lid) for lid in lids]
        
        assert(len(l_ids) == len(pts))
        for (lid, l_id, pt) in izip(lids, l_ids, pts):
            # Insert smart factor based on landmark id
            self.lid_factors_[lid].add_single(Point2(vec(*pt)), x_id, self.image_measurement_noise_, self.K_)
            self.lid_to_xids_[lid].append(xid)

        # Add to landmark measurements
        self.xls_.extend([(xid, lid) for lid in lids])

        # Add smartfactors to the graph only if that 
        # landmark ID is no longer visible. setdiff1d
        # returns the set of IDs that are unique to 
        # `smart_lids` (previously tracked) but not 
        # in `lids` (current)
        smart_lids = np.int64(self.lid_factors_.keys())

        # Determine old lids that are no longer tracked
        # and add only the ones that have at least 3
        # observations. Delete old factors that have 
        # insufficient number of observations
        add_lids = []
        old_lids = np.setdiff1d(smart_lids, lids)
        for lid in old_lids:
            degenerate = self.lid_factors_[lid].isDegenerate()
            if self.lid_count_[lid] >= 3: 
                self.graph_.add(self.lid_factors_[lid])
                add_lids.append(lid)
                assert(not degenerate)
                # print_green('[{:}] Sufficient factors ({:}) : Adding lid {:} '.format(degenerate, self.lid_count_[lid], lid))
            else: 
                # print_red('[{:}] Insufficient factors ({:}) : Removing lid {:} '.format(degenerate, self.lid_count_[lid], lid))
                del self.lid_factors_[lid]
        self.lid_update_needed_ = np.union1d(self.lid_update_needed_, add_lids)
        print 'Difference, to be added to graph', len(add_lids)
        
        # # Add landmark edge to graphviz
        # for l_id in l_ids: 
        #     self.gviz_.add_edge(x_id, l_id)
        
        # # Initialize new landmark pose node from the latest robot
        # # pose. This should be done just once
        # for (l_id, lid, pt3) in izip(l_ids, lids, pts3d): 
        #     if lid not in self.ls_: 
        #         try: 
        #             pred_pt3 = self.xs_[xid].transform_from(Point3(vec(*pt3)))
        #             self.initial_.insert(l_id, pred_pt3)
        #             self.ls_[lid] = pred_pt3
        #         except Exception, e: 
        #             raise RuntimeError('Initialization failed ({:}). xid:{:}, lid:{:}, l_id: {:}'
        #                                .format(e, xid, lid, l_id))

        #         # # Label landmark node
        #         # self.gviz_.node[l_id]['label'] = 'L ' + str(lid)            
        #         # self.gviz_.node[l_id]['color'] = 'red'
        #         # self.gviz_.node[l_id]['style'] = 'filled'
        
        return 


    def add_landmark_points(self, xid, lids, pts, pts3d): 
        print_red('\t\tadd_landmark_points xid:{:}-> lid count:{:}'.format(xid, len(lids)))
        
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

        assert(len(l_ids) == len(pts) == len(pts3d))
        for l_id, pt in izip(l_ids, pts):
            self.graph_.add(
                GenericProjectionFactorPose3Point3Cal3_S2(
                    Point2(vec(*pt)), self.image_measurement_noise_, x_id, l_id, self.K_))

        # Add to landmark measurements
        self.xls_.extend([(xid, lid) for lid in lids])

        # # Add landmark edge to graphviz
        # for l_id in l_ids: 
        #     self.gviz_.add_edge(x_id, l_id)
        
        # Initialize new landmark pose node from the latest robot
        # pose. This should be done just once
        for (l_id, lid, pt3) in izip(l_ids, lids, pts3d): 
            if lid not in self.ls_: 
                try: 
                    pred_pt3 = self.xs_[xid].transform_from(Point3(vec(*pt3)))
                    self.initial_.insert(l_id, pred_pt3)
                    self.ls_[lid] = pred_pt3
                except Exception, e: 
                    raise RuntimeError('Initialization failed ({:}). xid:{:}, lid:{:}, l_id: {:}'
                                       .format(e, xid, lid, l_id))

                # # Label landmark node
                # self.gviz_.node[l_id]['label'] = 'L ' + str(lid)            
                # self.gviz_.node[l_id]['color'] = 'red'
                # self.gviz_.node[l_id]['style'] = 'filled'
        
        return 

    def add_landmark_points_incremental_smart(self, lids, pts): 
        """
        Add landmark measurement (image features)
        from the latest robot pose to the
        set of specified landmark ids
        """
        self.add_landmark_points_smart(self.latest, lids, pts)

    def add_landmark_points_incremental(self, lids, pts, pts3d): 
        """
        Add landmark measurement (image features)
        from the latest robot pose to the
        set of specified landmark ids
        """
        self.add_landmark_points(self.latest, lids, pts, pts3d)

    def add_landmark_incremental(self, lid, delta): 
        """
        Add landmark measurement (pose3d) 
        from the latest robot pose to the
        specified landmark id
        """
        self.add_landmark(self.latest, lid, delta)

    @property
    def latest(self): 
        return self.idx_

    @property
    def index(self): 
        return self.idx_

    @property
    def is_initialized(self): 
        return self.latest >= 0

    def save_graph(self, filename): 
        self.slam_.saveGraph(filename)

    def save_dot_graph(self, filename): 
        nx.write_dot(self.gviz_, filename)
        # nx.draw_graphviz(self.gviz_, prog='neato')
        # nx_force_draw(self.gviz_)

    def update(self): 
        # Update ISAM with new nodes/factors and initial estimates
        self.slam_.update(self.graph_, self.initial_)
        self.slam_.update()

        # Get current estimate
        current = self.slam_.calculateEstimate()
        poses = extractPose3(current)

        # Extract and update landmarks and poses
        for k,v in poses.iteritems():
            if k.chr() == ord('l'): 
                self.ls_[k.index()] = v
            elif k.chr() == ord('x'): 
                self.xs_[k.index()] = v
            else: 
                raise RuntimeError('Unknown key chr {:}'.format(k.chr))

        self.graph_.resize(0)
        self.initial_.clear()

        if self.index % 10 == 0 and self.index > 0: 
            self.save_graph("slam_fg.dot")
            self.save_dot_graph("slam_graph.dot")

    def smart_update(self): 
        """
        Update the smart factors and add 
        to the graph. Once the landmarks are 
        extracted, remove them from the factor list
        """
        current = self.slam_.calculateEstimate()

        ids, pts3 = [], []
        for lid in self.lid_update_needed_: 
            smart = self.lid_factors_[lid]

            l_id = symbol('l', lid)

            # Add triangulated smart factors back into the graph
            # for complete point-pose optimization
            # Each of the projection factors, including the
            # points, and their initial values are added back to the 
            # graph. Optionally, we can choose to subsample and add
            # only a few measurements from the set of original 
            # measurements
            if not smart.isDegenerate():
                pts = smart.measured()
                assert len(pts) == len(self.lid_to_xids_[lid])

                # Add each of the smart factor measurements to the 
                # factor graph
                for xid,pt in zip(self.lid_to_xids_[lid], pts): 
                    self.graph_.add(GenericProjectionFactorPose3Point3Cal3_S2(
                        pt, self.image_measurement_noise_, symbol('x', xid), l_id, self.K_))
                
                # Initialize the point value
                pt3 = smart.point_compute(current)
                self.initial_.insert(l_id, pt3)
                self.ls_[lid] = pt3

                # Add the points for visualization 
                ids.append(lid)
                pts3.append(pt3.vector().ravel())

            del self.lid_factors_[lid]
            del self.lid_count_[lid]

        # Reset lid updates for newer ids 
        self.lid_update_needed_ = np.int64([])

        try: 
            ids, pts3 = np.int64(ids).ravel(), np.vstack(pts3)
            assert(len(ids) == len(pts3))
            return ids, pts3
        except Exception, e:
            print('Could not return pts3, {:}'.format(e))
            return np.int64([]), np.array([])        

class SLAM3D(BaseSLAM): 
    def __init__(self, update_on_odom=False): 
        BaseSLAM.__init__(self)
        self.update_on_odom_ = update_on_odom

    def on_odom(self, t, odom): 
        print('\ton_odom')
        self.add_odom_incremental(odom)
        if self.update_on_odom_: self.update()
        return self.latest

    def on_pose_ids(self, t, ids, poses): 
        print('\ton_pose_ids')
        for (pid, pose) in izip(ids, poses): 
            self.add_landmark_incremental(pid, pose)
        self.update()
        return self.latest

    def on_landmark(self, p): 
        pass

# class Tag3D(BaseSLAM): 
#     def __init__(self, K): 
#         BaseSLAM.__init__(self)

#         # TODO: Cal3_S2 datatype
#         self.K_ = None

#     @staticmethod
#     def construct_tag_corners(tag_size): 
#         s = tag_size / 2.0
#         return [Point3(s,s,0), Point3(s,-s,0), 
#                 Point3(-s,-s,0), Point3(-s,s,0)]  

#     def set_calib(self, K): 
#         pass

#     def on_odom(self, t, odom): 
#         self.add_odom_incremental(odom)
#         self.update()
#         return self.latest

#     def on_tags(self, t, tags, use_corners=False): 
#         """
#         Add tag measurements as separate  
#         Pose3-Pose3 constraints        
#         """
#         for tag in tags: 
#             delta = Pose3(tag.getPose())
#             self.add_landmark_incremental(tag.getId(), delta)

# #     def on_tag_corners(self, t, tags): 
# #         """
# #         Add tag measurements as 4 separate 
# #         Pose3-Point3 constraints        
# #         """
# #         camera = SimpleCamera(self.prev_pose_)
# #         pass

def createPoints(): 
    # Create the set of ground-truth landmarks
    points = [Point3(10.0,10.0,10.0), 
              Point3(-10.0,10.0,10.0), 
              Point3(-10.0,-10.0,10.0), 
              Point3(10.0,-10.0,10.0), 
              Point3(10.0,10.0,-10.0), 
              Point3(-10.0,10.0,-10.0), 
              Point3(-10.0,-10.0,-10.0), 
              Point3(10.0,-10.0,-10.0)]
    return points

def createPoses(): 
    radius = 30.0
    theta = 0.0
    up, target = Point3(0,0,1), Point3(0,0,0)

    poses = []
    for i in range(8): 
        theta += 2 * np.pi / 8
        position = Point3(radius * np.cos(theta), radius * np.sin(theta), 0.0)
        camera = SimpleCamera.Lookat(eye=position, target=target, upVector=up)
        pose = camera.pose()
        poses.append(camera.pose())

    return poses

def test_odometryExample(): 
    print("test_odmetryExample\n")
    print("=================================")

    # Add a prior on the first pose, setting it to the origin A prior factor
    # consists of a mean and a noise model (covariance matrix)
    prior_mean = Pose2(0, 0, 0)
    prior_noise  = Diagonal.Sigmas(vec(0.3, 0.3, 0.1), True)

    graph = NonlinearFactorGraph()
    graph.add(PriorFactorPose2(1, prior_mean, prior_noise))

    # Add odometry factors
    odometry = Pose2(2., 0., 0.)

    # For simplicity, we will use the same noise model for each odometry factor
    odometry_noise = Diagonal.Sigmas(vec(0.2, 0.2, 0.1), True)

    # Create odometry (Between) factors between consecutive poses
    graph.add(BetweenFactorPose2(1, 2, odometry, odometry_noise))
    graph.add(BetweenFactorPose2(2, 3, odometry, odometry_noise))

    block_str = '================'
    graph.printf(s="%s\nFactor Graph:\n" % block_str)

    # Create the data structure to hold the initialEstimate estimate to the
    # solution For illustrative purposes, these have been deliberately set to
    # incorrect values
    initial = Values()
    initial.insert(1, Pose2(0.5, 0.0, 0.2))
    initial.insert(2, Pose2(2.3, 0.1, -0.2))
    initial.insert(3, Pose2(4.1, 0.1, 0.1))

    # Optimize using Levenberg-Marquardt optimization
    opt = LevenbergMarquardtOptimizer(graph, initial)
    result = opt.optimize();
    result.printf("%s\nFinal Result:\n" % block_str);

    # Calculate and print marginal covariances for all variables
    marginals = Marginals(graph, result)
    print('x1 covariance:\n %s' % marginals.marginalCovariance(1))
    print('x2 covariance:\n %s' % marginals.marginalCovariance(2))
    print('x3 covariance:\n %s' % marginals.marginalCovariance(3))

    plot2DTrajectory(result, [], marginals)


def test_PlanarSLAMExample(): 
    print("test_PlanarSLAMExample\n")
    print("=================================")

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Create the keys we need for this simple example
    x1, x2, x3 = symbol('x', 1), symbol('x', 2), symbol('x', 3)
    l1, l2 = symbol('l', 1), symbol('l', 2)
    
    # Add a prior on pose x1 at the origin. A prior factor consists of a mean
    # and a noise model (covariance matrix)
    prior = Pose2(0.0, 0.0, 0.0) #  prior mean is at origin
    prior_noise = Diagonal.Sigmas(vec(0.3, 0.3, 0.1), True) #  30cm std on x,y, 0.1 rad on theta
    graph.add(PriorFactorPose2(x1, prior, prior_noise)) #  add directly to graph
    
    # Add two odometry factors
    odometry = Pose2(2.0, 0.0, 0.0) #  create a measurement for both factors (the same in this case)
    odometry_noise = Diagonal.Sigmas(vec(0.2, 0.2, 0.1), True) #  20cm std on x,y, 0.1 rad on theta
    graph.add(BetweenFactorPose2(x1, x2, odometry, odometry_noise))
    graph.add(BetweenFactorPose2(x2, x3, odometry, odometry_noise))

    # Add Range-Bearing measurements to two different landmarks
    # create a noise model for the landmark measurements
    measurement_noise = Diagonal.Sigmas(vec(0.1, 0.2), True) #  0.1 rad std on bearing, 20cm on range
    
    # create the measurement values - indices are (pose id, landmark id)
    bearing11 = Rot2.fromDegrees(45)
    bearing21 = Rot2.fromDegrees(90)
    bearing32 = Rot2.fromDegrees(90)
    range11 = np.sqrt(4.0+4.0)
    range21 = 2.0
    range32 = 2.0
    
    # Add Bearing-Range factors
    graph.add(
        BearingRangeFactorPose2Point2(x1, l1, 
                                      bearing11, range11, measurement_noise))
    graph.add(
        BearingRangeFactorPose2Point2(x2, l1, 
                                      bearing21, range21, measurement_noise))
    graph.add(
        BearingRangeFactorPose2Point2(x3, l2, 
                                      bearing32, range32, measurement_noise))  
    
    #  Print
    graph.printf("Factor Graph:\n");
    
    #  Create (deliberately inaccurate) initial estimate
    initialEstimate = Values()
    initialEstimate.insert(x1, Pose2(0.5, 0.0, 0.2))
    initialEstimate.insert(x2, Pose2(2.3, 0.1,-0.2))
    initialEstimate.insert(x3, Pose2(4.1, 0.1, 0.1))
    initialEstimate.insert(l1, Point2(1.8, 2.1))
    initialEstimate.insert(l2, Point2(4.1, 1.8))
  
    #  Print
    initialEstimate.printf("Initial Estimate:\n")
    
    #  Optimize using Levenberg-Marquardt optimization. The optimizer
    #  accepts an optional set of configuration parameters, controlling
    #  things like convergence criteria, the type of linear system solver
    #  to use, and the amount of information displayed during optimization.
    #  Here we will use the default set of parameters.  See the
    #  documentation for the full set of parameters.
    optimizer = LevenbergMarquardtOptimizer(graph, initialEstimate)
    result = optimizer.optimize()
    result.printf("Final Result:\n")
    
    #  Calculate and print marginal covariances for all variables
    marginals = Marginals(graph, result)
    print(marginals.marginalCovariance(x1), "x1 covariance")
    print(marginals.marginalCovariance(x2), "x2 covariance")
    print(marginals.marginalCovariance(x3), "x3 covariance")
    print(marginals.marginalCovariance(l1), "l1 covariance")
    print(marginals.marginalCovariance(l2), "l2 covariance") 

def test_StereoVOExample():
    print("test_StereoVOExample\n")
    print("=================================")

    # Assumptions
    #  - For simplicity this example is in the camera's coordinate frame
    #  - X: right, Y: down, Z: forward
    #  - Pose x1 is at the origin, Pose 2 is 1 meter forward (along Z-axis)
    #  - x1 is fixed with a constraint, x2 is initialized with noisy values
    #  - No noise on measurements
    x1, x2 = symbol('x', 1), symbol('x', 2)
    l1, l2, l3 = symbol('l', 1), symbol('l', 2), symbol('l', 3)

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # add a constraint on the starting pose
    first_pose = Pose3()
    graph.add(NonlinearEqualityPose3(x1, first_pose))

    # Create realistic calibration and measurement noise model
    # format: fx fy skew cx cy baseline
    K = Cal3_S2Stereo(1000, 1000, 0, 320, 240, 0.2)
    stereo_model = Diagonal.Sigmas(vec(1.0, 1.0, 1.0))

    ## Add measurements
    # pose 1
    graph.add(GenericStereoFactor3D(
        StereoPoint2(520, 480, 440), stereo_model, x1, l1, K))
    graph.add(GenericStereoFactor3D(
        StereoPoint2(120,  80, 440), stereo_model, x1, l2, K))
    graph.add(GenericStereoFactor3D(
        StereoPoint2(320, 280, 140), stereo_model, x1, l3, K))

    # pose 2
    graph.add(GenericStereoFactor3D(
        StereoPoint2(570, 520, 490), stereo_model, x2, l1, K))
    graph.add(GenericStereoFactor3D(
        StereoPoint2( 70,  20, 490), stereo_model, x2, l2, K))
    graph.add(GenericStereoFactor3D(
        StereoPoint2(320, 270, 115), stereo_model, x2, l3, K))


    ## Create initial estimate for camera poses and landmarks
    initialEstimate = Values()
    initialEstimate.insert(x1, first_pose)

    # noisy estimate for pose 2
    initialEstimate.insert(x2, Pose3(Rot3(), Point3(0.1,-.1,1.1)))
    expected_l1 = Point3( 1,  1, 5)
    initialEstimate.insert(l1, expected_l1)
    initialEstimate.insert(l2, Point3(-1,  1, 5))
    initialEstimate.insert(l3, Point3( 0,-.5, 5))

    ## optimize
    optimizer = LevenbergMarquardtOptimizer(graph, initialEstimate)
    result = optimizer.optimize()

    block_str = '================'
    graph.printf(s="%s\nFactor Graph:\n" % block_str)
    result.printf("%s\nFinal Result:\n" % block_str);
    # print extractPose3(result)

    ## check equality for the first pose and point
    # pose_x1 = result.at(x1)
    # CHECK('pose_x1.equals(first_pose,1e-4)',
    #       pose_x1.equals(first_pose,1e-4))
    
    # point_l1 = result.at(l1)
    # CHECK('point_1.equals(expected_l1,1e-4)',
    #       point_l1.equals(expected_l1,1e-4))

def test_SFMExample_SmartFactor(): 
    print("test_SFMExample_SmartFactor\n")
    print("=================================")

    # Define the camera calibration parameters
    # format: fx fy skew cx cy
    K = Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0)

    # Define the camera observation noise model
    measurement_noise = Isotropic.Sigma(2, 1.0)

    # Create the set of ground-truth landmarks and poses
    points = createPoints()
    poses = createPoses()
    # print poses, points

    # Create a factor graph
    graph = NonlinearFactorGraph()

    # Simulated measurements from each camera pose, adding them to the factor graph
    for j, pointj in enumerate(points): 

        # every landmark represent a single landmark, we use shared pointer to init the factor, and then insert measurements.
        smartfactor = SmartFactor()

        for i, posei in enumerate(poses): 

            # generate the 2D measurement
            camera = SimpleCamera(posei, K)
            measurement = camera.project(pointj)

            # call add() function to add measurement into a single factor, here we need to add:
            #    1. the 2D measurement
            #    2. the corresponding camera's key
            #    3. camera noise model
            #    4. camera calibration
            smartfactor.add_single(measurement, i, measurement_noise, K)

        # insert the smart factor in the graph
        graph.add(smartfactor)
    
    # Add a prior on pose x0. This indirectly specifies where the origin is.
    # 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
    pose_noise = Diagonal.Sigmas(vec(0.3, 0.3, 0.3, 0.1, 0.1, 0.1))
    graph.add(PriorFactorPose3(0, poses[0], pose_noise))

    # Because the structure-from-motion problem has a scale ambiguity, the problem is
    # still under-constrained. Here we add a prior on the second pose x1, so this will
    # fix the scale by indicating the distance between x0 and x1.
    # Because these two are fixed, the rest of the poses will be also be fixed.
    graph.add(PriorFactorPose3(1, poses[1], pose_noise)) # add directly to graph

    graph.printf("Factor Graph:\n")

    # Create the initial estimate to the solution
    # Intentionally initialize the variables off from the ground truth
    initialEstimate = Values()
    delta = Pose3(Rot3.rodriguez(-0.1, 0.2, 0.25), Point3(0.05, -0.10, 0.20))

    for i, posei in enumerate(poses): 
        initialEstimate.insert(i, posei.compose(delta))
    initialEstimate.printf("Initial Estimates:\n")

    # Optimize the graph and print results
    result = DoglegOptimizer(graph, initialEstimate).optimize()
    result.printf("Final results:\n")

    # A smart factor represent the 3D point inside the factor, not as a variable.
    # The 3D position of the landmark is not explicitly calculated by the optimizer.
    # To obtain the landmark's 3D position, we use the "point" method of the smart factor.
    landmark_result = Values()
    for j, pointj in enumerate(points): 

        # The output of point() is in boost::optional<gtsam::Point3>, as sometimes
        # the triangulation operation inside smart factor will encounter degeneracy.

        # The graph stores Factor shared_ptrs, so we cast back to a SmartFactor first
        # c++ -> py: smart.point -> smart.point_compute
        smart = graph[j]
        if smart is not None: 
            point = smart.point_compute(result)
        
         # ignore if boost::optional return NULL
        if point is not None:
            landmark_result.insert(j, point)

        # print point

    keys = extractKeys(landmark_result)
    # for key in keys: 
    #     landmark_result.atPose3()
    # landmark_result.printf("Landmark results:\n")
            
if __name__ == "__main__": 
    # test_odometryExample()
    # test_PlanarSLAMExample()
    # test_StereoVOExample()
    test_SFMExample_SmartFactor()
    
