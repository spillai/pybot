#arm_planner.py
# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Jenny Barry, Jon Binney, Sachin Chitta
#         Willow Garage
'''
Defines the ArmPlanner class for planning with the PR2 arms.
'''

__docformat__ = "restructuredtext en"

import roslib
roslib.load_manifest('pr2_python')

import rospy
from pr2_python.transform_state import TransformState
from arm_navigation_msgs.msg import RobotState, ArmNavigationErrorCodes, MotionPlanRequest
from arm_navigation_msgs.srv import GetMotionPlan, FilterJointTrajectoryWithConstraints,\
    FilterJointTrajectoryWithConstraintsRequest, GetStateValidity
from interpolated_ik_motion_planner.srv import SetInterpolatedIKMotionPlanParams,\
    SetInterpolatedIKMotionPlanParamsRequest
from kinematics_msgs.msg import PositionIKRequest
from kinematics_msgs.srv import GetPositionIK, GetKinematicSolverInfo, GetConstraintAwarePositionIK,\
    GetPositionIKRequest, GetConstraintAwarePositionIKRequest, GetPositionFK, GetPositionFKRequest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from object_manipulation_msgs.srv import GraspPlanning, GraspPlanningRequest
from household_objects_database_msgs.msg import DatabaseModelPose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from visualization_msgs.msg import MarkerArray,Marker

from pr2_python.hand_description import HandDescription
from pr2_python.planning_scene_interface import get_planning_scene_interface
from pr2_python.exceptions import ArmNavError

import pr2_python.conversions as conv
import pr2_python.trajectory_tools as tt
import pr2_python.geometry_tools as gt
import pr2_python.visualization_tools as vt

import numpy as np
import copy

GRASPS_VISUALIZATION_PUB='/grasps_visualization'
'''
Name of visualization topic.
'''

PLANNER_NAME='/ompl_planning/plan_kinematic_path'
'''
Name of planning service.
'''

SHORTEST_FILTERABLE_TRAJECTORY = 3
'''
Any trajectory shorter than this length cannot be filtered (weird things happen when trajectories of length 2 are
filtered)
'''

JOINT_LIMIT_PAD = 0.01
'''
The padding used when finding a state within joint angles.
'''

DEFAULT_DATABASE_GRASP_PLANNER = '/objects_database_node/database_grasp_planning'
'''
The default grasp planner used (with the database)
'''


class ArmPlanner:
    '''
    This class can be used to plan for the PR2 arms.  

    This allows you to plan several motions in a row before executing them.  This class can be used for collision 
    free planning, trajectory filtering, and interpolated IK plannig.  The function you should mainly use to plan 
    for the arms is plan_collision_free.

    The class also has a number of arm-related functions, such as forward and inverse kinematics,
    that are related to planning, although not planning themselves.  It also has some utility functions involving 
    working with joint states and robot states where it is necessary to know joint names (functions for which it
    is not necessary to use joint names are in trajectory_tools.py).

    This class relies heavily on the planning scene interface.  It makes no calls to TF.

    **Attributes:**

        **arm_name (string):** The name of the arm for which to plan ('right_arm' or 'left_arm')
        
        **hand (hand_description.HandDescription):** Hand description for the arm.

        **joint_names ([string]):** Names of the arm joints

        **joint_limits ([arm_navigation_msgs.msg.JointLimits]):** Limits for the arm joints

        **kinematics_info_service (rospy.ServiceProxy):** Service for retreiving kinematic information about the arm

        **get_state_validity_service (rospy.ServiceProxy):** Service for ascertaining state validity.
        Be aware that this service has several bugs at the moment.
    '''
    def __init__(self, arm_name):
        '''
        Constructor for ArmPlanner.

        **Args:**
            
            **arm_name (string):** The arm ('left_arm' or 'right_arm') for which to plan
        '''
        rospy.loginfo("Creating arm planner...")

        #public variables
        self.arm_name = arm_name
        self.hand = HandDescription(arm_name)
        
        self.kinematics_info_service = rospy.ServiceProxy\
            ('/pr2_'+self.arm_name+ '_kinematics/get_ik_solver_info', GetKinematicSolverInfo)
        rospy.loginfo('Waiting for kinematics solver info service')
        self.kinematics_info_service.wait_for_service()

        self.get_state_validity_service = rospy.ServiceProxy('/planning_scene_validity_server/get_state_validity',
                                                             GetStateValidity)
        rospy.loginfo('Waiting for state validity service.')
        self.get_state_validity_service.wait_for_service()


        #find the joint names
        info = self.kinematics_info_service()
        self.joint_names = info.kinematic_solver_info.joint_names
        self.joint_limits = info.kinematic_solver_info.limits
        for l in range(len(self.joint_limits)):
            #for some reason this isn't filled in on return
            self.joint_limits[l].joint_name = self.joint_names[l]


        #private variables
        self._psi = get_planning_scene_interface()
        self._transformer = TransformState()

        #services (private)
        self._move_arm_planner = rospy.ServiceProxy(PLANNER_NAME, GetMotionPlan)
        rospy.loginfo('Waiting for move arm planner')
        self._move_arm_planner.wait_for_service()
        self._interpolated_ik_planning_service = rospy.ServiceProxy\
            ('/'+self.arm_name[0]+'_interpolated_ik_motion_plan', GetMotionPlan)
        rospy.loginfo('Waiting for interpolated IK planning service')
        self._interpolated_ik_planning_service.wait_for_service()
        self._interpolated_ik_parameter_service =rospy.ServiceProxy\
            ('/'+self.arm_name[0]+'_interpolated_ik_motion_plan_set_params', SetInterpolatedIKMotionPlanParams)
        rospy.loginfo('Waiting for interpolated IK parameter setting service')
        self._interpolated_ik_parameter_service.wait_for_service()
        self._filter_trajectory_with_constraints_service = rospy.ServiceProxy\
            ('/trajectory_filter_server/filter_trajectory_with_constraints', FilterJointTrajectoryWithConstraints)
        rospy.loginfo('Waiting for trajectory filter with constraints service')
        self._filter_trajectory_with_constraints_service.wait_for_service()
        self._collision_aware_ik_service = rospy.ServiceProxy\
            ('/pr2_'+self.arm_name+'_kinematics/get_constraint_aware_ik', GetConstraintAwarePositionIK)
        rospy.loginfo('Waiting for collision aware IK service')
        self._collision_aware_ik_service.wait_for_service()
        self._ik_service = rospy.ServiceProxy('/pr2_'+self.arm_name+'_kinematics/get_ik', GetPositionIK)
        rospy.loginfo('Waiting for IK service')
        self._ik_service.wait_for_service()
        self._fk_service = rospy.ServiceProxy('/pr2_'+self.arm_name+'_kinematics/get_fk', GetPositionFK)
	rospy.loginfo('Waiting for FK service')
	self._fk_service.wait_for_service()


        database_grasp_planner_name = rospy.get_param('/object_manipulator_1/default_database_planner',
                                                      DEFAULT_DATABASE_GRASP_PLANNER)
        print 'default database grasp planner'
        print database_grasp_planner_name
        self.database_grasp_planner = rospy.ServiceProxy(database_grasp_planner_name, GraspPlanning)
        rospy.loginfo('Waiting for database grasp planner')
        self.database_grasp_planner.wait_for_service()

        self._grasps_pub = rospy.Publisher(GRASPS_VISUALIZATION_PUB, MarkerArray, latch=True)

        rospy.loginfo("Arm planner created")

    def arm_joint_state(self, robot_state=None, fail_if_joint_not_found=True):
        '''
        Returns the joint state of the arm in the current planning scene state or the passed in state.

        **Args:**

            *robot_state (arm_navigation_msgs.msg.RobotState):* robot state from which to find the joints.  
            If None, will use the current robot state in the planning scene interface
            
            *fail_if_joint_not_found (boolean):* Raise a value error if an arm joint is not found in the robot state

        **Returns:**
            A sensor_msgs.msg.JointState containing just the arm joints as they are in robot_state

        **Raises:**

            **ValueError:** if fail_if_joint_not_found is True and an arm joint is not found in the robot state
        '''
        if not robot_state:
            robot_state = self._psi.get_robot_state()
        joint_state = JointState()
        joint_state.header = robot_state.joint_state.header
        for n in self.joint_names:
            found = False
            for i in range(len(robot_state.joint_state.name)):
                if robot_state.joint_state.name[i] == n:
                    joint_state.name.append(n)
                    joint_state.position.append(robot_state.joint_state.position[i])
                    found = True
                    break
            if not found and fail_if_joint_not_found:
                raise ValueError('Joint '+n+' is missing from robot state')
        return joint_state

    def arm_robot_state(self, robot_state=None, fail_if_joint_not_found=True):
        '''
        Returns the joint state of the arm (as a robot state) in the current planning scene state or the passed in
        state.

        **Args:**

            *robot_state (arm_navigation_msgs.msg.RobotState):* robot state from which to find the joints.  
            If None, will use the current robot state in the planning scene interface
            
            *fail_if_joint_not_found (boolean):* Raise a value error if an arm joint is not found in the robot state

        **Returns:**
            An arm_navigation_msgs.msg.RobotState containing just the arm joints as they are in robot_state
        
        **Raises:**

            **ValueError:** if fail_if_joint_not_found is True and an arm joint is not found in the robot state
        '''
        joint_state = self.arm_joint_state(robot_state=robot_state, fail_if_joint_not_found=fail_if_joint_not_found)
        robot_state = RobotState()
        robot_state.joint_state = joint_state
        return robot_state

    def joint_positions_to_joint_state(self, joint_pos):
        '''
        Converts a list of joint positions to a joint state.

        **Args:**

            **joint_pos ([double]):** An array of joint positions

        **Returns:**
            A sensor_msgs.msg.JointState corresponding to these joint positions.  Assumes they are in the same order
            as joint_names.
        '''
        joint_state = JointState()
        joint_state.name = copy.copy(self.joint_names)
        joint_state.position = joint_pos
        return joint_state

    def set_joint_positions_in_robot_state(self, joint_pos, robot_state):
        '''
        Sets joint positions in a robot state.

        **Args:**

            **joint_pos ([double]):** An array of joint positions
            
            **robot_state (arm_navigation_msgs.msg.RobotState):** A robot state in which to set the joint angles 
            equal to joint_pos

        **Returns:**
            An arm_navigation_msgs.msg.RobotState in which the joint angles corresponding to this arm are set to
            joint_pos (assumes these are in the same order as joint_names) and all other joints match the passed
            in robot_state.  Also sets these joints in the passed in robot state.
        '''
        return tt.set_joint_state_in_robot_state(self.joint_positions_to_joint_state(joint_pos), robot_state)

    def joint_trajectory(self, joint_trajectory):
        '''
        Returns just the part of the trajectory corresponding to the arm.

        **Args:**
        
            **joint_trajectory (trajectory_msgs.msg.JointTrajectory):** Trajectory to convert

        **Returns:**
            A trajectory_msgs.msg.JointTrajectory corresponding to only this arm in joint_trajectory

        **Raises:**
            
            **ValueError:** If some arm joint is not found in joint_trajectory
        '''
        arm_trajectory = JointTrajectory()
        arm_trajectory.header = copy.deepcopy(joint_trajectory.header)
        arm_trajectory.joint_names = copy.copy(self.joint_names)
        indexes = [-1]*len(self.joint_names)
        for (i, an) in enumerate(self.joint_names):
            for (j, n) in enumerate(joint_trajectory.joint_names):
                if an == n:
                    indexes[i] = j
                    break
            if indexes[i] < 0:
                raise ValueError('Joint '+n+' is missing from joint trajectory')
        
        for joint_point in joint_trajectory.points:
            arm_point = JointTrajectoryPoint()
            arm_point.time_from_start = joint_point.time_from_start
            for i in indexes:
                arm_point.positions.append(joint_point.positions[i])
            arm_trajectory.points.append(arm_point)
        return arm_trajectory


    def get_ik(self, pose_stamped, collision_aware=True, starting_state=None,
               seed_state=None, timeout=5.0, ordered_collisions=None):
        '''
        Solves inverse kinematics problems.

        **Args:**

            **pose_stamped (geometry_msgs.msg.PoseStamped):** The pose for which to get joint positions

            *collision_aware (boolean):* If True, returns a solution that does not collide with anything in the world

            *starting_state (arm_navigation_msgs.msg.RobotState):* The returned state will have all non-arm joints
            matching this state.  If no state is passed in, it will use the current state in the planning scene
            interface.

            *seed_state (arm_navigation_msgs.msg.RobotState):* A seed state for IK.  If no state is passed it, it 
            will use the current state in planning scene interface.

            *timeout (double):* Time in seconds that IK is allowed to run

        **Returns:**
            A kinematics_msgs.srv.GetConstraintAwarePositionIKResponse if collision_aware was True and a
            kinematics_msgs.srv.GetPosiitonIKResponse if collision_aware was False.  The robot state returned has
            the arm joints set to the IK solution if found and all other joints set to that of starting_state.
            
        **Raises:**

            **rospy.ServiceException:** if there is a problem with the service call

        **Warning:**
            Calls an IK service which may use TF for transformations!  Probably best to only use with pose stampeds
            defined in the robot frame (convert them yourself using the planning scene interface).
        '''
        rospy.logdebug('Solving IK for\n'+str(pose_stamped))
        pos_req = PositionIKRequest()
        pos_req.ik_link_name = self.hand.hand_frame
        pos_req.pose_stamped = pose_stamped
        if not starting_state: 
            starting_state = self._psi.get_robot_state()
        if not seed_state:
            seed_state = self.arm_robot_state(starting_state)
        pos_req.ik_seed_state = seed_state
        pos_req.robot_state = starting_state
        if collision_aware:
            self._psi.add_ordered_collisions(ordered_collisions)
            coll_req = GetConstraintAwarePositionIKRequest()
            coll_req.ik_request = pos_req
            coll_req.timeout = rospy.Duration(timeout)
            coll_res = self._collision_aware_ik_service(coll_req)
            coll_res.solution = tt.set_joint_state_in_robot_state(coll_res.solution.joint_state,
                                                                  copy.deepcopy(starting_state))
            self._psi.remove_ordered_collisions(ordered_collisions)
            return coll_res
        coll_req = GetPositionIKRequest()
        coll_req.ik_request = pos_req
        coll_req.timeout = rospy.Duration(timeout)
        coll_res = self._ik_service(coll_req)
        coll_res.solution = tt.set_joint_state_in_robot_state(coll_res.solution.joint_state,
                                                              copy.deepcopy(starting_state))
        return coll_res

    def get_fk(self, fk_links, robot_state=None, frame_id=None):
        '''
        Solves forward kinematics.  

        In general, it is better to use the planning scene or the state transformer get_transform function between 
        robot links rather than this function.
        
        **Args:**

            **fk_links ([string]):** Links for which you want FK solutions

            *robot_state (arm_navigation_msgs.mgs.RobotState):* The state of the robot during forward kinematics.
            If no state is passed in the state in the planning scene will be used.

            *frame_id (string):* The frame ID in which you want the returned poses.  Note that the FK service may use
            TF so we recommend only using the robot frame.  If no frame is specified, the robot frame is used.
           
        **Returns:**
            A list of geometry_msgs.msg.PoseStamped corresponding to forward kinematic solutions for fk_links.

        **Raises:**

            **rospy.ServiceException:** if there is a problem with the service call

        **Warning:**
            Calls a service which may use TF!  Recommended that you only ask for poses in the robot frame.
        '''
        req = GetPositionFKRequest()
        req.header.frame_id = frame_id
        if not frame_id:
            req.header.frame_id = self._psi.robot_frame
        req.header.stamp = rospy.Time(0)
        req.robot_state = robot_state
        if not robot_state:
            req.robot_state = self._psi.get_robot_state()
        req.fk_link_names = fk_links
        res = self._fk_service(req)
        if res.error_code.val != res.error_code.SUCCESS or not res.pose_stamped:
            raise ArmNavError('Forward kinematics failed', error_code=res.error_code)
        return res.pose_stamped

    def get_hand_frame_pose(self, robot_state=None, frame_id=None):
        '''
        Returns the pose of the hand in the current or passed in robot state.

        Note that this function uses the TransformState get_transform function rather than FK.

        **Args:**

            *robot_state (arm_navigation_msgs.msg.RobotState):* The robot state in which you want to find 
            the pose of the hand frame.  If nothing is passed in, returns the pose of the hand frame in the robot 
            state in the planning scene interface.

            *frame_id (string):* The frame in which you want the pose of the hand frame.  If nothing is passed in, 
            returns in the robot frame.

        **Returns:**
            A geometry_msgs.msg.PoseStamped that is the position of the hand frame.
        '''
        if frame_id == None:
            frame_id = self._psi.robot_frame
        if robot_state == None:
            robot_state = self._psi.get_robot_state()
        trans = self._transformer.get_transform(self.hand.hand_frame, frame_id, robot_state)
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose = conv.transform_to_pose(trans.transform)
        return ps

    def hand_marker(self, robot_state=None, ns='', r = 0.0, g = 0.0, b = 1.0, a = 0.8, scale = 1.0):
        '''
        Returns a MarkerArray of the hand in the current or passed in state.

        **Args:**

            *robot_state (arm_navigation_msgs.msg.RobotState):* The state in which to draw the hand.  If no state
            is specified, the robot state in the planning scene interface is used.

            *ns (string):* Marker namespace
            
            *r (double):* Red value (between 0 and 1)
            
            *g (double):* Green value (between 0 and 1)
            
            *b (double):* Blue value (between 0 and 1)

            *a (double):* Alpha value (between 0 and 1)

            *scale (double):* Scale

        **Returns:**
            A visualization_msgs.msg.MarkerArray that when published shows the hand in the position specified by
            the robot state.

        **Raises:**
            
            **rospy.ServiceException:** if there is a problem calling the service for getting the robot markers
        '''
        
        if not robot_state:
            robot_state = self._psi.get_robot_state()
        return vt.robot_marker(robot_state, link_names=self.hand.hand_links, ns=ns, r=r, g=g, b=b, a=a, scale=scale)


    def get_closest_joint_state_in_limits(self, robot_state=None):
        '''
        Finds the closest joint state to the passed in or current robot state that is within joint limits.

        **Args:**
        
            *robot_state (arm_navigation_msgs.msg.RobotState):* The robot state.  If no state is passed in 
            the current state in the planning scene interface is used.

        **Returns:**
            A sensor_msgs.msg.JointState that is the closest state in which the arm joints are all within the joint
            limits.  The return has only the arm joints, but it also sets the joints in the passed in robot state.
        '''
        return self.arm_joint_state(self.get_closest_state_in_limits(robot_state=robot_state))

    def get_closest_state_in_limits(self, robot_state=None):
        '''
        Finds the closest robot state to the passed in or current robot state that is within joint limits.

        **Args:**

            *robot_state (arm_navigation_msgs.msg.RobotState):* The robot state.  If no state is passed in 
            the current state in the planning scene interface is used.

        **Returns:**
            An arm_navigation_msgs.msg.RobotState that is the closest state in which the arm joints are all within 
            the joint limits.  Also sets the joints in the passed in robot state.
        '''
        if not robot_state:
            robot_state = self._psi.get_robot_state()
        robot_state.joint_state.position = list(robot_state.joint_state.position)
        #is this outside the joint limits?
        #if so, modify it slightly so that it is not
        for j in range(len(robot_state.joint_state.name)):
            for limit in self.joint_limits:
                if limit.joint_name == robot_state.joint_state.name[j]:
                    jp = robot_state.joint_state.position[j]
                    if limit.has_position_limits:
                        if jp < limit.min_position:
                            robot_state.joint_state.position[j] = limit.min_position+JOINT_LIMIT_PAD
                        elif jp > limit.max_position:
                            robot_state.joint_state.position[j] = limit.max_position-JOINT_LIMIT_PAD
                    break
        return robot_state

    def filter_trajectory(self, trajectory, motion_plan_request=None):
        '''
        Filters a joint trajectory and assigns times using the joint trajectory filter service.

        **Args:**

            **trajectory (trajectory_msgs.msg.JointTrajectory):** The trajectory to be filtered.  All times must be 0.

            *motion_plan_request (arm_navigation_msgs.msg.MotionPlanRequest):* If passed in the trajectory 
            filter will respect the starting state, goal constriants, and path constraints.  It will also append the 
            starting state from the motion plan request to the front of the trajectory.

        **Returns:**
            A trajectory_msgs.msg.JointTrajectory that has been filtered and had times assigned.

        **Raises:**

            **exceptions.ArmNavError:** if the trajectory cannot be filtered.  This is almost always because the 
            trajectory did not actually reach the goal state or had a collision.  Note that OMPL returns bad plans 
            with reasonable frequency so if trajectory filtering fails, you want to re-plan.
        '''
        if len(trajectory.points) < SHORTEST_FILTERABLE_TRAJECTORY:
            rospy.logwarn('Not filtering trajectory of length %d.  Too small.',
                          len(trajectory.points))
            trajectory = tt.convert_path_to_trajectory(trajectory)
            if motion_plan_request:
                trajectory = tt.add_state_to_front_of_joint_trajectory\
                    (self.arm_joint_state(robot_state=motion_plan_request.start_state), trajectory)
            return trajectory
        req = FilterJointTrajectoryWithConstraintsRequest()
        req.trajectory = trajectory
        #these have to be in the world frame... go figure
        if motion_plan_request:
            req.path_constraints =\
                self._psi.transform_constraints(self._psi.world_frame, motion_plan_request.path_constraints)
            req.goal_constraints =\
                self._psi.transform_constraints(self._psi.world_frame, motion_plan_request.goal_constraints)
            req.start_state = motion_plan_request.start_state
            req.group_name = motion_plan_request.group_name
        else:
            req.start_state = self._psi.get_robot_state()
            req.group_name = self.arm_name
        req.allowed_time = rospy.Duration(1.5)
        ec = ArmNavigationErrorCodes()
        try:
            traj_resp = self._filter_trajectory_with_constraints_service(req)
            ec = traj_resp.error_code
        except rospy.ServiceException:
            #this almost certainly means the trajectory doesn't reach the goal
            ec.val = ec.GOAL_CONSTRAINTS_VIOLATED
        if ec.val != ec.SUCCESS:
            raise ArmNavError('Trajectory filter failed probably because OMPL returned a bad plan.', 
                              error_code=ec, trajectory=trajectory)
        traj = traj_resp.trajectory
        if motion_plan_request:
            traj = tt.add_state_to_front_of_joint_trajectory\
                (self.arm_joint_state(robot_state=motion_plan_request.start_state), traj)
        return traj

    
    def plan_pose_collision_free(self, pose_stamped, starting_state=None, ordered_collisions=None, timeout=15.0, 
                                 bounds=None, planner_id='', ntries=3):
        '''
        **Deprecated.**  

        Use plan_collision_free.
        '''
        starting_state = self.get_closest_state_in_limits(robot_state=starting_state)
        goal = conv.pose_stamped_to_motion_plan_request(pose_stamped, self.hand.hand_frame, self.arm_name,
                                                        starting_state, timeout=timeout,
                                                        bounds=bounds, planner_id=planner_id)

        return self.plan_collision_free(goal, ordered_collisions=ordered_collisions, ntries=ntries)

    def plan_joints_collision_free(self, joint_state, starting_state=None, ordered_collisions=None, timeout=15.0,
                                   bounds=None, planner_id='', ntries=3):
        '''
        **Deprecated.**  

        Use plan_collision_free.
        '''
        starting_state = self.get_closest_state_in_limits(robot_state=starting_state)
        goal = conv.joint_state_to_motion_plan_request(joint_state, self.hand.hand_frame, self.arm_name,
                                                       starting_state, timeout=timeout,
                                                       bounds=bounds, planner_id=planner_id)
        goal_state = tt.set_joint_state_in_robot_state(joint_state, copy.deepcopy(starting_state))
        rospy.loginfo('Position of wrist is\n'+str(self._transformer.get_transform
                                                   ('l_wrist_roll_link', self._psi.robot_frame, goal_state)))
        rospy.loginfo('Position of fingertip is\n'+str(self._transformer.get_transform('l_gripper_r_finger_tip_link',
                                                                                       self._psi.robot_frame,
                                                                                       goal_state)))
        rospy.loginfo('Collision objects are\n')
        cos = self._psi.collision_objects()
        for co in cos: 
            if 'table' not in co.id:
                rospy.loginfo(str(co))
        rospy.loginfo('Attached collision objects are\n')
        aos = self._psi.attached_collision_objects()
        for ao in aos:
            rospy.loginfo(ao)

        return self.plan_collision_free(goal, ordered_collisions=ordered_collisions, ntries=ntries)

        
    def plan_collision_free(self, goal_in, starting_state=None, ordered_collisions=None, timeout=15.0, bounds=None,
                            planner_id='', ntries=3):
        '''
        Plans collision free paths for the arms.  

        The trajectory returned from this function is safe to execute.  This function may alter the planning scene 
        during planning, but returns it to its initial state when finished.

        **Args:**

            **goal_in (geometry_msgs.msg.PoseStamped, sensor_msgs.msg.JointState or
            arm_navigation_msgs.msg.MotionPlanRequest):** The goal for the arm.  The pose goal should be for
            the hand frame defined in hand (see hand_description.py).

            *starting_state (arm_navigation_msgs.msg.RobotState):* The state at which planning starts.  If 
            no state is passed in, this will use the current state in the planning scene interface as the starting 
            state.
            
            *ordered_collisions (arm_navigation_msgs.msg.OrderedCollisionOperations):* Any ordered 
            collisions not already in the planning scene diff that you want applied during planning.

            *timeout (double):* The time in seconds allowed for planning.
            
            *bounds ([double] or [[double]]):* The bounds on the goal defining how close the trajectory must get to 
            the goal.  For a pose goal, this should be a list of 6 numbers corresponding to 
            (x_allowed_error, y_allowed_error, z_allowed_error, roll_allowed_error, pitch_allowed_error,
            yaw_allowed_error).  For a joint goal, this should be a list of two lists corresponding to 
            tolerance above or tolerance below.  For a motion plan request, this field is ignored.  There are 
            defaults defined in conversions.py if this is not passed in.
            
            *planner_id (string):* The ID of a specific preferred planner.  If the empty string is passed in the 
            default planner (ROS parameter) is used.

            *ntries (int):* OMPL returns bad plans sometimes.  This is the number of calls to OMPL allowed before
            giving up.  The function will only retry if OMPL returns success, but trajectory filtering returns
            failure.  This can lead to this function taking ntries*timeout.  If you pass in less than 1, the
            function will try once.

        **Returns:**
            A trajectory_msgs.msg.JointTrajectory from starting_state to the goal_in.  This trajectory has already 
            been filtered and is safe to execute.

        **Raises:**

            **exceptions.ArmNavError:** if a plan cannot be found.

            **rospy.ServiceException:** if there is a problem with the call to the planning service

        **TODO:**
            Check that this works even if starting state has a different base pose.
        '''
        if type(goal_in) == PoseStamped:
            goal = conv.pose_stamped_to_motion_plan_request\
                (goal_in, self.hand.hand_frame, self.arm_name,\
                     self.get_closest_state_in_limits(robot_state=starting_state), timeout=timeout, bounds=bounds,\
                     planner_id=planner_id)
        elif type(goal_in) == JointState:
            goal = conv.joint_state_to_motion_plan_request\
                (goal_in, self.hand.hand_frame, self.arm_name,\
                     self.get_closest_state_in_limits(robot_state=starting_state), timeout=timeout, bounds=bounds,\
                     planner_id=planner_id)
        elif type(goal_in) == MotionPlanRequest:
            goal = goal_in
        else:
            raise ArmNavError('Unsupported goal type for planning: '+str(type(goal_in)))

        #save the old planning scene robot state (so we can come back to it later)
        #current_ps_state = self._psi.get_robot_state()
        #plan in the starting state we've just been passed in
        #we plan from this anyway so let's not send a planning scene for it
        #self._psi.set_robot_state(goal.start_state)
        #add in the ordered collisions
        self._psi.add_ordered_collisions(ordered_collisions)
        rospy.loginfo('Calling plan collision free')
        rospy.logdebug('With goal\n'+str(goal))
        for i in range(max(1,ntries)):
            filter_error = None
            plan = self._move_arm_planner(goal)
            if plan.error_code.val == plan.error_code.SUCCESS:
                trajectory = plan.trajectory.joint_trajectory
                try:
                    trajectory = self.filter_trajectory(trajectory, motion_plan_request=goal)    
                except ArmNavError, filter_error:
                    continue
                #self._psi.set_robot_state(current_ps_state)
                self._psi.remove_ordered_collisions(ordered_collisions)
                return trajectory
        #self._psi.set_robot_state(current_ps_state)
        self._psi.remove_ordered_collisions(ordered_collisions)
        if filter_error:
            raise filter_error

        raise ArmNavError('Unable to plan collision-free trajectory', error_code=plan.error_code, 
                          trajectory=plan.trajectory.joint_trajectory)    


    def plan_interpolated_ik(self, pose_stamped, starting_pose=None, ordered_collisions=None, bounds = None, 
                             starting_state=None, min_acceptable_distance=None, collision_aware=True,
                             reverse=False, resolution=0.005, nsteps=0, consistent_angle=np.pi/6.0, 
                             steps_before_abort=0, collision_check_resolution=2, max_joint_vels=None, 
                             max_joint_accs=None):
        '''
        Plans a path that follows a straight line in Cartesian space.  
        
        This function is useful for tasks like grasping where it is necessary to ensure that the gripper moves in a
        straight line relative to the world.  The trajectory returned from this function is safe to execute.  
        This function may alter the planning scene during planning, but returns it to its initial state when 
        finished.

        There are a lot of possible arguments to this function but in general the defaults work well.

        **Args:**

            **pose_stamped (geomety_msgs.msg.PoseStamped):** The ending pose for the hand frame defined in hand (see
            hand_description.py)

            *starting_pose (geometry_msgs.msg.PoseStamped):* The starting pose of the hand frame.  If None, this will
            use the current pose of the hand frame in the starting state

            *ordered_collisions (arm_navigation_msgs.msg.OrderedCollisionOperations):* Any additional collision
            operations besides those in the planning scene interface you want to use during planning.

            *bounds ([double]):* Acceptable errors for the goal position as (x, y, z, roll, pitch, yaw).  If nothing
            is passed in, uses the defaults defined in conversions.py

            *starting_state (arm_navigation_msgs.msg.RobotState):* The state of the robot at the start of the plan if
            reverse is False and at the end of the plan if reverse is True.  If you pass in a starting_pose that 
            does not match the starting state, the planner will use the starting_pose not the stating_state.  If you
            pass in a starting state and no starting pose, the planner will use the hand frame pose in the starting 
            state.  If you pass in no starting state, but you do pass in a starting_pose, the planner will solve for 
            a collision free IK solution for the starting state.  If you pass in no starting state or starting pose,
            the planner will use the current robot state in the planning scene interface.  If reverse is False, the 
            starting state will be appended to the front of the trajectory.

            *min_acceptable_distance (double):* If the planner finds a path of at least this distance (in meters), 
            it is a success.  This must be greater than zero; to disable, set to None.  

            *collision_aware (boolean):* Set to False if you want no collision checking to be done

            *reverse (boolean):* Set to True if you want the planner to start planning at pose_stamped and try to
            plan towards starting_pose.  The trajectory returned will still end at pose_stamped.

            *resolution (double):* The resolution in centimeters between points on the trajectory in Carteisan space.
            Will only be used if nsteps=0.

            *nsteps (int):* The number of steps you want on the trajectory.  If nsteps is set, resolution will be
            ignored.
            
            *consistent_angle (double):* If any joint angle between two points on this trajectory changes by more 
            than this amount, the planning will fail.
            
            *steps_before_abort (int):* The number of invalid steps (no IK solution etc) allowed before the planning
            fails.
            
            *collision_check_resolution (int):* Collisions will be checked every collision_check_resolution steps
            
            *max_joint_vels ([double]):* Maximum allowed joint velocities.  If not passed in, will be set to 0.1 for
            all joints.
            
            *max_joint_accs ([double]):* Maximum allowed joint accelerations.  If not passed in, only velocity
            constraints will be used.

        **Returns:**
            A trajectory_msgs.msg.JointTrajectory that is safe to execute in which the hand frame moves in a straight
            line in Cartesian space.
           
        **Raises:**
        
            **exceptions.ArmNavError:** if no plan can be found.

            **rospy.ServiceException:** if there is a problem with the call to the planning or parameter service
        '''
        #prior_state = self._psi.get_robot_state()
        self._psi.add_ordered_collisions(ordered_collisions)
        #if starting_state:
            #starting_state = self.get_closest_state_in_limits(robot_state=starting_state)
            #self._psi.set_robot_state(starting_state)
        if not starting_pose:
            starting_pose = self.get_hand_frame_pose(robot_state=starting_state, frame_id=pose_stamped.header.frame_id)
        else:
            starting_pose = self._psi.transform_pose_stamped(pose_stamped.header.frame_id, starting_pose)
        if starting_state:
            #check that it matches
            if not reverse:
                chk_pose = starting_pose
            else:
                chk_pose = pose_stamped
            starting_fk = self.get_hand_frame_pose(robot_state=starting_state, 
                                                   frame_id=chk_pose.header.frame_id)
            if not gt.near(starting_fk.pose, chk_pose.pose):
                rospy.logwarn('Input starting state does not match starting pose.  '+
                              'Solving for an IK solution instead')
                rospy.logdebug('Starting FK is\n'+str(starting_fk)+'\nCheck pose is\n'+str(chk_pose))
                rospy.logdebug('Euclidean distance is: '+
                               str(gt.euclidean_distance(starting_fk.pose.position, chk_pose.pose.position))+
                               ', angular distance is: '+
                               str(gt.quaternion_distance(starting_fk.pose.orientation, chk_pose.pose.orientation)))
                starting_state = None
        if not starting_state:
            if reverse:
                ik_sol = self.get_ik(pose_stamped, collision_aware=collision_aware)
            else:
                ik_sol = self.get_ik(starting_pose, collision_aware=collision_aware)
            if ik_sol.error_code.val != ik_sol.error_code.SUCCESS:
                rospy.logerr('Starting pose for interpolated IK had IK error '+str(ik_sol.error_code.val))
                raise ArmNavError('Starting pose for interpolated IK had no IK solution', 
                                  error_code = ik_sol.error_code)
            starting_state = ik_sol.solution
            #self._psi.set_robot_state(starting_state)
        rospy.logdebug('Planning interpolated IK from\n'+str(starting_pose)+'\nto\n'+str(pose_stamped))
        init_state = RobotState()
        init_state.joint_state = starting_state.joint_state
        init_state.multi_dof_joint_state.frame_ids.append(starting_pose.header.frame_id)
        init_state.multi_dof_joint_state.child_frame_ids.append(self.hand.hand_frame)
        init_state.multi_dof_joint_state.poses.append(starting_pose.pose)
        goal = conv.pose_stamped_to_motion_plan_request(pose_stamped, self.hand.hand_frame, self.arm_name,
                                                        init_state, bounds=bounds)
        dist = gt.euclidean_distance(pose_stamped.pose.position, starting_pose.pose.position)
        if nsteps == 0:
            if resolution == 0:
                rospy.logwarn('Resolution and steps were both zero in interpolated IK.  '+
                              'Using default resolution of 0.005')
                resolution = 0.005
            nsteps = int(dist/resolution)
        res = dist/nsteps
        req = SetInterpolatedIKMotionPlanParamsRequest()
        req.num_steps = nsteps
        req.consistent_angle = consistent_angle
        req.collision_check_resolution = collision_check_resolution
        req.steps_before_abort = steps_before_abort
        req.collision_aware = collision_aware
        req.start_from_end = reverse
        if max_joint_vels:
            req.max_joint_vels = max_joint_vels
        if max_joint_accs:
            req.max_joint_accs = max_joint_accs
        self._interpolated_ik_parameter_service(req)
        rospy.loginfo('Calling interpolated ik motion planning service.  Expecting '+str(nsteps)+' steps')
        rospy.logdebug('Sending goal\n'+str(goal))
        ik_resp = self._interpolated_ik_planning_service(goal)
        self._psi.remove_ordered_collisions(ordered_collisions)
        #self._psi.set_robot_state(prior_state)
        traj = ik_resp.trajectory.joint_trajectory
        first_index = 0
        rospy.logdebug('Trajectory error codes are '+str([e.val for e in ik_resp.trajectory_error_codes]))
        if reverse:
            for first_index in range(len(ik_resp.trajectory_error_codes)):
                e = ik_resp.trajectory_error_codes[first_index]
                if e.val == e.SUCCESS:
                    break
	last_index = 0
	e = ArmNavigationErrorCodes()
	e.val = e.SUCCESS
        for last_index in range(first_index,len(ik_resp.trajectory_error_codes)+1):
            if last_index == len(ik_resp.trajectory_error_codes):
                #the whole trajectory works
                break
            e = ik_resp.trajectory_error_codes[last_index]
            if e.val != e.SUCCESS:
                rospy.logerr('Interpolated IK failed with error '+str(e.val)+' on step ' +str(last_index)+
                             ' after distance '+ str((last_index+1-first_index)*res))
                last_index -= 1
                break
        rospy.logdebug('First index = '+str(first_index)+', last index = '+str(last_index))
        distance = (last_index-first_index)*res
        traj.points = traj.points[first_index:max(0,last_index)]
        rospy.loginfo('Interpolated IK returned trajectory with '+str(len(traj.points))+' points')
        if e.val != e.SUCCESS and (not min_acceptable_distance or distance < min_acceptable_distance):
            raise ArmNavError('Interpolated IK failed after '+str(last_index-first_index)+' steps.', error_code=e,
                              trajectory_error_codes = ik_resp.trajectory_error_codes, trajectory=traj)
        if not reverse or not traj.points:
            return tt.add_state_to_front_of_joint_trajectory(self.arm_joint_state(robot_state=starting_state), traj)
        return traj

    def get_grasps(self, model_id, model_pose):
        '''
        Get the set of grasps for a model, given a particular model pose
        **Args:**

            **model_id:** The model id corresponding to the database being used

            *model_pose:* The pose of the model       
        '''
        request = GraspPlanningRequest()
        request.arm_name = 'right_arm'
        db_pose = DatabaseModelPose()
        db_pose.pose = model_pose
        db_pose.model_id = model_id
        db_pose.confidence = 0.5
        request.target.potential_models.append(db_pose)
        request.target.reference_frame_id = model_pose.header.frame_id

        response = self.database_grasp_planner(request)
        rospy.loginfo('Found = '+str(len(response.grasps))+' grasps')
        self._visualize_grasps(response.grasps,model_pose.header.frame_id)
        return response.grasps

    def _visualize_grasps(self, grasps, frame_id):
        '''
        Visualize all the grasps
        **Args:**
            **grasps:** The set of grasps to be visualized
            **frame_id:**  The frame id in which the grasps are defined
        '''
        mm = Marker()
        action = mm.ADD
        marray2 = MarkerArray()
        for i in range(len(grasps)):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.type = marker.ARROW
            marker.action = action
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            if i==0:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            marker.id = i
            marker.pose = grasps[i].grasp_pose
            marray2.markers.append(marker)
        self._grasps_pub.publish(marray2)        
        return True
