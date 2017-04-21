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

    from collections import defaultdict
    sfactors = defaultdict(SmartFactor)

    for i, posei in enumerate(poses):
        # generate the 2D measurement
        camera = SimpleCamera(posei, K)
        for j, pointj in enumerate(points):
            measurement = camera.project(pointj)
            print measurement

            sfactors[j].add_single(measurement, i, measurement_noise, K)

    # insert the smart factor in the graph
    for item in sfactors.itervalues(): 
        graph.add(item)

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
    test_SFMExample_SmartFactor()
    print('OK')
    
