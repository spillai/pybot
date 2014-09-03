#trajectory_tools.py
'''
A set of utility functions for working with joint and robot trajectories.
'''

__docformat__ = "restructuredtext en"

import roslib; roslib.load_manifest('pr2_python')
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from arm_navigation_msgs.msg import MultiDOFJointState, MultiDOFJointTrajectoryPoint, RobotState

import numpy as np


def normalize_joint_state(joint_state):
    '''
    Normalize the joint angles to be between -\pi and \pi

    **Args:**

        **joint_state (sensor_msgs.msg.JointState):** joint state   

    **Returns:**
        A sensor_msgs.msg.JointState in which all angles are between -\pi and \pi.
    '''
    #assume these are all revolute hahaha
    joint_state.position = list(joint_state.position)
    for p in range(len(joint_state.position)):
        joint_state.position[p] = (joint_state.position[p] % (2.0*np.pi)) - np.pi
    return joint_state

def joint_trajectory_point_to_joint_state(joint_tp, joint_names):
    '''
    Convert a joint trajectory point to a state

    **Args:**

        **joint_tp (trajectory_msgs.msg.JointTrajectoryPoint):** joint trajectory point

        **joint_names ([string]):** List of joint names (not defined in joint trajectory points)
    
    **Returns:**
        The corresponding sensor_msgs.msg.JointState
    '''
    js = JointState()
    js.name = joint_names
    js.position = joint_tp.positions
    return js

def joint_state_to_joint_trajectory_point(joint_state):
    '''
    Convert a joint state to a trajectory point

    **Args:**

        **joint_state (sensor_msgs.msg.JointState):** joint state

    **Returns:**
        The corresponding trajectory_msgs.msg.JointTrajectoryPoint
    '''
    tp = JointTrajectoryPoint()
    tp.positions = joint_state.position
    return tp

def multi_dof_state_to_multi_dof_trajectory_point(mds_state):
    '''
    Convert a multi DOF state to a trajectory point

    **Args:**

        **mds_state (arm_navigation_msgs.mgs.MultiDOFJointState):** Multi-DOF joint state

    **Returns:**
        Corresponding arm_navigation_msgs.msg.MultiDOFJointTrajectoryPoint
    '''
    mds_pt = MultiDOFJointTrajectoryPoint()
    mds_pt.poses = mds_state.poses
    return mds_pt

def multi_dof_trajectory_point_to_multi_dof_state(mds_point, joint_names, frame_ids, child_frame_ids, stamp):
    '''
    Convert a multi DOF trajectory point to a state

    **Args:**

        **mds_point (arm_navigation_msgs.msg.MultiDOFJointTrajectoryPoint):** joint trajectory point

        **joint_names ([string]):** Joint names

        **frame_ids ([string]):** Parent frame IDs

        **child_frame_ids ([string]):** Child frame IDs
        
        **stamp (time): Timestamp**
        
    **Returns:**
        Corresponding arm_navigation_msgs.msg.MultiDOFJointState
    '''
    mds_state = MultiDOFJointState()
    mds_state.stamp = stamp
    mds_state.joint_names = joint_names
    mds_state.frame_ids = frame_ids
    mds_state.child_frame_ids = child_frame_ids
    mds_state.poses = mds_point.poses
    return mds_state

def last_state_on_joint_trajectory(trajectory):
    '''
    Returns the last state on a joint trajectory

    **Args:**
        
        **trajectory (trajectory_msgs.msg.JointTrajectory):** joint trajectory

    **Returns:**
        The last state on the trajectory as a sensor_msgs.msg.JointState
    '''
    if not trajectory.points:
        return None
    return joint_trajectory_point_to_joint_state(trajectory.points[-1], trajectory.joint_names)

def last_state_on_robot_trajectory(trajectory):
    '''
    Returns the last state on a robot trajectory

    **Args:**

        **trajectory (arm_navigation_msgs.msg.RobotTrajectory):** robot trajectory

    **Returns:**
        The last state on the trajectory as an arm_navigation_msgs.msg.RobotState
    '''
    robot_state = RobotState()
    robot_state.joint_state = last_state_on_joint_trajectory(trajectory.joint_trajectory)
    if not robot_state.joint_state: return None
    mdf_traj = trajectory.multi_dof_joint_trajectory
    robot_state.multi_dof_joint_state = multi_dof_trajectory_point_to_multi_dof_state\
        (mdf_traj.points[-1], mdf_traj.joint_names, mdf_traj.frame_ids, mdf_traj.child_frame_ids, mdf_traj.stamp)
    return robot_state

def add_state_to_front_of_joint_trajectory(joint_state, trajectory, time=1.0):
    '''
    Puts a state on the front of a joint trajectory

    **Args:**

        **joint_state (sensor_msgs.msg.JointState):** state to add to front of trajectory

        **trajectory (trajectory_msgs.msg.JointTrajectory):** trajectory

        *time (double):* Time in seconds allowed to get to this state

    **Returns:**
        A joint trajectory with joint_state appended to the front.  If the trajectory was safe to execute, it is 
        still safe to execute.  Also appends the joint state to the passed in trajectory.
    '''
    trajectory.points.insert(0, joint_state_to_joint_trajectory_point(joint_state))
    for p in trajectory.points:
        p.time_from_start += rospy.Duration(time)
    #remove crazy spinnies!
    return unnormalize_trajectory(trajectory)

def add_state_to_front_of_robot_trajectory(robot_state, trajectory, time=1.0):
    '''
    Puts a state on the front of a robot trajectory

    **Args:**
        **robot_state (arm_navigation_msgs.msg.RobotState):** state to add to front of trajectory
        
        **trajectory (arm_navigation_msgs.msg.RobotTrajectory):** trajectory

        *time (double):* Time in seconds allowed to get to this state

    **Returns:**
        A robot trajectory with robot_state appended to the front.  If the trajectory was safe to execute, it is 
        still safe to execute.  Also appends the robot state to the passed in trajectory.
    '''
    trajectory.joint_trajectory = add_state_to_front_of_joint_trajectory(robot_state.joint_state, 
                                                                         trajectory.joint_trajectory, time=time)
    trajectory.multi_dof_joint_trajectory.points.insert\
        (0, multi_dof_state_to_multi_dof_trajectory_point(robot_state.multi_dof_joint_state))
    for p in trajectory.multi_dof_joint_trajectory.points:
        p.time_from_start += rospy.Duration(time)
    return unnormalize_trajectory(trajectory.joint_trajectory)

def convert_path_to_trajectory(trajectory, time_per_pt=0.05):
    '''
    Takes a joint trajectory that might be normalized or not have times filled in and returns one that is safe to
    execute.  

    This is not a trajectory filter; it simply makes sure the points are reasonably spaced in time and the
    trajectory is unnormalized.

    **Args:**
    
        **trajectory (trajectory_msgs.msg.JointTrajectory):** trajectory to convert

        *time_per_pt (seconds):* The minimum time between each point on the trajectory.  If some times have already 
        been assigned, this will preserve them unless they are less than this time.

    **Returns:**
        A sensor_msgs.msg.JointStateTrajectory that is safe to execute (assuming time_per_pt was reasonable)

    **TODO:**
        * Make this take velocity restrictions instead of time_per_pt
    '''
    last_time = 0
    for p in trajectory.points:
        if p.time_from_start.to_sec() <= last_time:
            p.time_from_start = rospy.Duration(last_time+time_per_pt)    
        last_time = p.time_from_start.to_sec()
    return unnormalize_trajectory(trajectory)

def unnormalize_trajectory(trajectory):
    '''
    Minimizes distance between angles on the trajectory for continuous joints.

    For robots that have continuous joints, joint angles along a trajectory should be continuous rather than always
    between -\pi and \pi.  This function takes a joint trajectory and "unnormalizes" it so that the absolute distance
    between the angles is minimized.  If you see robot joints spinning crazily chances are you need to unnormalize
    your trajectories.

    **Args:**

        **trajectory (trajectory_msgs.msg.JointTrajectory):** trajectory to unnormalize
        
    **Returns:**
        An unnormalized trajectory_msgs.msg.JointTrajectory.  Also unnormalizes the passed in trajectory.

    **TODO:**
        * Check for continuous joints instead of hard-coding forearm_roll and wrist_roll.
    '''
    i = 1
    for j in range(1, len(trajectory.points)):
        newpositions = []
        for i in range(len(trajectory.joint_names)):
            curr = trajectory.points[j].positions[i]
            if trajectory.joint_names[i][1:] ==\
                    '_forearm_roll_joint' or\
                    trajectory.joint_names[i][1:] ==\
                    '_wrist_roll_joint':
                #this really should check if the joints
                #are revolute... not this silly thing
                prev = trajectory.points[j-1].positions[i]
                while prev - curr > np.pi:
                    curr += 2.0*np.pi
                while curr - prev > np.pi:
                    curr -= 2.0*np.pi
            newpositions.append(curr)
        trajectory.points[j].positions = tuple(newpositions)
    return trajectory

def set_joint_state_in_robot_state(joint_state, robot_state):
    '''
    Set the joint states of a robot state

    **Args:**

        **joint_state (sensor_msgs.msg.JointState):** joint state

        **robot_state (arm_navigation_msgs.msg.RobotState):** robot state

    **Returns:**
        An arm_navigation_msgs.msg.RobotState in which the joints corresponding to those in joint_state have been set
        to have the values in joint_state.  Also sets these joints in the passed in robot_state.
    '''
    robot_state.joint_state.position = list(robot_state.joint_state.position)
    robot_state.joint_state.name = list(robot_state.joint_state.name)
    for i in range(len(joint_state.name)):
        found = False
        for j in range(len(robot_state.joint_state.name)):
            if joint_state.name[i] == robot_state.joint_state.name[j]:
                robot_state.joint_state.position[j] = joint_state.position[i]
                found = True
                break
        if not found:
            robot_state.joint_state.name.append(joint_state.name[i])
            robot_state.joint_state.name.position.append(joint_state.position[i])
    return robot_state
