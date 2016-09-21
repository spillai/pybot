"""
Test SLAM interface with GTSAM
"""
print(__doc__)

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np

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

from gtsam import symbol, vector, vec, matrix

# Helper functions for tests
# ======================================================================

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


# Tests
# ======================================================================

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
    test_odometryExample()
    print('OK')
    test_PlanarSLAMExample()
    print('OK')
    test_StereoVOExample()
    print('OK')
    test_SFMExample_SmartFactor()
    print('OK')
    
