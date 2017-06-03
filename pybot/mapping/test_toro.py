"""
Test SLAM interface with Toro
"""
print(__doc__)

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np
from pybot.geometry.rigid_transform import RigidTransform


# Tests
# ======================================================================

def test_solveFromFile():
    import os
    import sys
    import time
    from pytoro import TreeOptimizer3

    
    overrideCovariances = False
    twoDimensions = False

    pg = TreeOptimizer3()

    pg.verboseLevel = 0
    pg.restartOnDivergence = False

    filename = '/home/spillai/perceptual-learning/software/externals/toro-pod/toro/data/3D/sphere_smallnoise.graph'
    print('Loading graph file {}... '.format(filename))
    if not pg.load(filename, overrideCovariances, twoDimensions):
        print('FATAL ERROR: Could not read file. Abrting.')
        sys.exit(1)
    print 'V / E', pg.nvertices, pg.nedges
    print('Done')

    # Reduce nodes
    print('Loading equivalence constraints and collapsing nodes... ')
    pg.loadEquivalences(filename)
    print(' #nodes: {} #edges: {}'.format(pg.nvertices, pg.nedges))

    # Compress indices (-ic) def: false
    print('Compressing indices... ')
    # pg.compressIndices()
    print('Done')

    # Tree type  (-st, -mst) def: st
    print('Incremental tree construction... ')
    pg.buildSimpleTree() # st

    # print('MST construction... ')
    # ppg.buildMST(first_vertex_id) # mst
    print('Done')

    # Initialize on tree (-nib) def: true
    print('Computing initial guess from observations... ')
    pg.initializeOnTree()
    print('Done')

    print('Initializing the optimizer...')
    pg.initializeTreeParameters()
    pg.initializeOptimization(compare_mode='level');
    l = pg.totalPathLength()
    nEdges = pg.nedges
    apl = l / nEdges
    print('Done')
    print('Average path length={}'.format(apl))
    print('Complexity of an iteration={}'.format(l))


    # stripped_filename,_ = os.path.splitext(filename)
    # print('Saving starting graph... ')
    # output = stripped_filename + '-treeopt-initial.graph'
    # pg.save(output)
    # print('Done')

    # output = stripped_filename + '-treeopt-initial.dat'
    # pg.saveGnuplot(output)
    # print('Done')

    # error_output = stripped_filename + '-treeopt-error.dat'
    # # ofstream errorStream;


    # ignore preconditioner (-ip)
    ignorePreconditioner = False

    print('**** Starting optimization ****')
    st = time.time()
    corrupted = False
    for j in range(100): 
        pg.iterate([], ignorePreconditioner);
        ei = pg.error()
        error = ei['error']

        print('Iteration {} RotGain={}'.format(j, pg.rotGain))
        print('   global error = {}   error/constraint = {}'.format(error, error / nEdges))
        print('mte={} mre={} are={} ate={}'.format(ei['mte'], ei['mre'], ei['are'], ei['ate']))

        if (ei['mre'] > (np.pi / 2) * (np.pi / 2)):
            corrupted = True
        else: 
            corrupted = False

    print('TOTAL TIME= {} s.'.format(time.time() - st))    


    # print('Saving files...(graph file)')
    # output = stripped_filename+ '-treeopt-final.graph'
    # pg.save(output)
    # print('...(gnuplot file)...')
    # output = stripped_filename + '-treeopt-final.dat'
    # pg.saveGnuplot(output)
    # # errorStream.close();
    # print('Done')


    
fixed_yaw = lambda: RigidTransform.from_rpyxyz(0, 0, 0.2, 0, 0.2, 0)
rand_yaw = lambda: RigidTransform.from_rpyxyz(0, 0,
                                              0.2, # (np.random.random() - 0.5) * np.pi/6,
                                              0, # (np.random.random() - 0.5) * 2.0,
                                              0.2, # np.random.random() * 1,
                                              0)

def test_odometryExample(): 
    print("test_odmetryExample\n")
    print("=================================")

    from toro import BaseSLAM
    from pybot.geometry.rigid_transform import Pose
    from pybot.externals.lcm.draw_utils import publish_pose_list

    # Init
    slam = BaseSLAM(verbose=True)

    slam.pg_.verboseLevel = 0
    slam.pg_.restartOnDivergence = False
    
    for j in range(10): 
        slam.add_incremental_pose_constraint(rand_yaw()) 
    slam._update(iterations=1)
    slam._update_estimates()
    poses = [Pose.from_rigid_transform(k,v) for k,v in slam.poses.iteritems()]
    publish_pose_list('original_poses', poses, frame_id='origin')
     
    slam.add_relative_pose_constraint(0,9,RigidTransform.from_rpyxyz(0,0,np.pi/2,-1.0,2.0,0))
    slam._update(iterations=100)
    slam._update_estimates()

    poses = [Pose.from_rigid_transform(k,v) for k,v in slam.poses.iteritems()]
    publish_pose_list('optimized_poses', poses, frame_id='origin')


def test_RobotSLAM(): 
    print("test_RobotSLAM\n")
    print("=================================")

    from pybot.mapping import cfg
    cfg.SLAM_BACKEND = 'toro'
    from pybot.mapping.slam import RobotSLAM

    # Initalize SLAM object with visualization
    slam_cls = RobotSLAM(frame_id='origin',
                         visualize_nodes=True, visualize_measurements=True,
                         visualize_factors=True, visualize_marginals=False,
                         pose_type='pose')
    slam = slam_cls(verbose=True)

    # Initialization: slam.initialize(index=0, p=RigidTransform.identity())
    # on_odom_relative automatically initializes
    # if pose graph is not already intiailized
    for j in range(10):
        rt = fixed_yaw()
        slam.on_odom_relative(j, rt)
    slam.update()

    # Add loop closure constraint
    slam.on_loop_closure_relative(None, 0,9,RigidTransform.from_rpyxyz(0,0,np.pi/2,-1.0,2.0,0))
    slam.update(iterations=10)
    slam.finish()

def test_toro_pose_graph_file(): 
    print("test_toro_pose_graph_file\n")
    print("=================================")

    from pybot.mapping import cfg
    cfg.SLAM_BACKEND = 'toro'
    from pybot.mapping.slam import RobotSLAM

    # Initalize SLAM object with visualization
    slam_cls = RobotSLAM(frame_id='origin',
                         visualize_every=1, visualize_nodes=True, visualize_measurements=True,
                         visualize_factors=True, visualize_marginals=False,
                         pose_type='point')
    slam = slam_cls(verbose=True)

    # Load pose graph file
    slam.load('data/sphere_mednoise.graph')
    for j in range(100): 
        slam.update(iterations=1)
    slam.finish()
    
if __name__ == "__main__": 
    test_odometryExample()
    print('OK')

    test_RobotSLAM()
    print('OK')

    test_toro_pose_graph_file()
    print('OK')
