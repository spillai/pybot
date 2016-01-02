import numpy as np
from itertools import izip

from pygtsam import symbol as _symbol

from pygtsam import Point3, Rot3, Pose3, \
    PriorFactorPose3, BetweenFactorPose3
    
from pygtsam import Point2, Rot2, Pose2, \
    PriorFactorPose2, BetweenFactorPose2, \
    BearingRangeFactorPose2Point2

from pygtsam import Isotropic
from pygtsam import Diagonal, Values, Marginals
from pygtsam import ISAM2, NonlinearOptimizer, \
    NonlinearFactorGraph, LevenbergMarquardtOptimizer

np.set_printoptions(precision=2, suppress=True)

def symbol(ch, i): 
    return _symbol(ord(ch), i)

def vector(v): 
    return np.float64(v)

def matrix(m): 
    return np.float64(m)

def vec(*args):
    return vector(list(args)) 

def plot2DTrajectory(values, linespec, marginals): 
    print values, marginals

class BaseSLAM(object): 
    def __init__(self): 
        self.slam_ = ISAM2()
        self.idx_ = -1

        # Factor graph storage
        self.graph_ = NonlinearFactorGraph()
        self.initial_ = Values()
        self.landmark_edges_ = []
        self.landmarks_ = set()

        # Pose3D measurement
        self.measurement_noise_ = Isotropic.Sigma(6, 0.4)
        self.prior_noise_ = Isotropic.Sigma(6, 0.01)
        self.odo_noise_ = Isotropic.Sigma(6, 0.01)

        # TODO: Update slam every landmark addition
        self.xs_ = {}
        self.ls_ = {}

        self.poses_ = {}
        self.targets_ = {}

    def get_poses(self): 
        return self.poses_

    def get_targets(self): 
        return self.targets_

    def get_landmark_edges(self): 
        return self.landmark_edges_

    def initialize(self, p_init=None): 
        x_id = symbol('x', 0)
        pose0 = Pose3(p_init) if p_init else Pose3()
            
        self.graph_.add(PriorFactorPose3(x_id, pose0, self.prior_noise_))
        self.initial_.insert(x_id, pose0)
        self.prev_pose_ = pose0
        self.idx_ = 0

        self.xs_[0] = pose0

    def add_odom_incremental(self, delta): 
        print('\t\tadd_odom_incr {:}->{:}'.format(self.latest(), self.latest()+1))
        # Add prior on first pose
        if self.latest() < 0: 
            self.initialize()

        # Add odometry factor
        self.add_odom(self.latest(), self.latest()+1, delta)
        self.idx_ += 1

    def add_odom(self, xid1, xid2, delta): 
        # Add odometry factor
        pdelta = Pose3(delta)
        x_id1, x_id2 = symbol('x', xid1), symbol('x', xid2)
        self.graph_.add(BetweenFactorPose3(x_id1, x_id2, 
                                           pdelta, self.odo_noise_))

        
        # Predict pose and add as initial estimate
        # TODO: get latest estiamte for x_id1, and 
        # compose with delta to find initial estimate
        pred_pose = self.prev_pose_.compose(pdelta)
        self.initial_.insert(x_id2, pred_pose)
        self.prev_pose_ = pred_pose

        self.xs_[xid2] = pred_pose

    def add_landmark(self, xid, lid, delta): 
        print('\t\tadd_landmark {:}->{:}'.format(xid, lid))

        # Add Pose-Pose landmark factor
        x_id = symbol('x', xid)
        l_id = symbol('l', lid)
        
        # Add landmark pose
        pdelta = Pose3(delta)
        self.graph_.add(BetweenFactorPose3(x_id, l_id, pdelta, 
                                           self.measurement_noise_))

        # Add to landmark measurements
        self.landmark_edges_.append((self.latest(), lid))

        if lid not in self.landmarks_: 
            pred_pose = self.prev_pose_.compose(pdelta)
            self.initial_.insert(l_id, pred_pose)
            self.landmarks_.add(lid)
            self.ls_[lid] = pred_pose

        return 

    def add_landmark_incremental(self, lid, delta): 
        self.add_landmark(self.latest(), lid, delta)
        pass

    def latest(self): 
        return self.idx_
        
    def index(self): 
        return self.idx_

    def save_graph(self, filename): 
        self.slam_.saveGraph(filename)

    def update(self): 
        self.slam_.update(self.graph_, self.initial_)
        self.slam_.update()

        # Get current estimate
        current = self.slam_.calculateEstimate()

        # self.targets_ = {}
        # self.poses_ = {}

        # # Update xs, ls
        # for xid in self.xs_.keys(): 
        #     self.xs_[xid] = current.at(symbol('x', xid))
        # for lid in self.ls_.keys(): 
        #     self.ls_[lid] = current.at(symbol('l', lid))

        # print self.xs_
        # for item in current: 
        #     print item

        self.graph_.resize(0)
        self.initial_.clear()

        self.save_graph("slam_fg.dot")

# class SLAM2D(BaseSLAM): 
#     def __init__(self): 
#         BaseSLAM.__init__(self)

#     def on_odom(self, p): 
#         pass

#     def on_landmark(self, p): 
#         pass

# class PlanarSLAM(object): 
#     def __init__(self): 
#         pass

#     def on_odom(self): 
#         pass

#     def on_landmark(self): 
#         pass
        
class SLAM3D(BaseSLAM): 
    def __init__(self): 
        BaseSLAM.__init__(self)

    def on_odom(self, t, odom): 
        print('\ton_odom')
        self.add_odom_incremental(odom)
        self.update()
        return self.latest()

    def on_pose_ids(self, t, ids, poses): 
        print('\ton_pose_ids')
        for (pid, pose) in izip(ids, poses): 
            self.add_landmark_incremental(pid, pose)
        self.update()
        return self.latest()

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
#         return self.latest()

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

        
def test_odometryExample(): 

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


if __name__ == "__main__": 
    test_odometryExample()
    test_PlanarSLAMExample()
    
    
