# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
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
# Author: Jon Binney
#arm_planner.py
'''
Defines the ArmMover class for moving the PR2 arms.
'''

__docformat__ = "restructuredtext en"


import time, copy, threading
import numpy as np
import roslib

import rospy
import actionlib
import actionlib_msgs.msg
import arm_navigation_msgs.msg
from arm_navigation_msgs.msg import ArmNavigationErrorCodes as ArmNavErrorCodes
from arm_navigation_msgs.srv import GetStateValidityRequest
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pr2_controllers_msgs.msg import JointTrajectoryGoal, JointTrajectoryAction
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from pr2_python.world_interface import WorldInterface
from pr2_python.hand_description import HandDescription
from pr2_python.controller_manager_client import ControllerManagerClient
from pr2_python.cartesian_controller_interface import CartesianControllerInterface
from pr2_python.arm_planner import ArmPlanner
from pr2_python import conversions
from pr2_python import trajectory_tools
from pr2_python.exceptions import ArmNavError, ActionFailedError


DEFAULT_PLANNER_SERVICE_NAME = 'ompl_planning/plan_kinematic_path'
'''
The planning service used by default.
'''

class MovementHandle:
    _idnum = 0
    def __init__(self, task_func=None, task_args=None):
        self._idnum = MovementHandle._idnum
        MovementHandle._idnum += 1

        self.task_func = task_func
        self.task_args = task_args

        self._state_lock = threading.Lock()
        self._in_progress = True
        self._cancel_requested = False
        self._reached_goal = False
        self._exceptions = []

    def __hash__(self):
        return hash(self._idnum)

    def __eq__(self, other):
        return self._idnum == other._idnum

    def wait(self):
        '''Wait for movement to complete.
        '''
        r = rospy.Rate(100)
        while self.in_progress():
            r.sleep()
        return self.reached_goal()
        
    def cancel(self):
        '''
        Request to cancel the current move.
        
        After calling this function, call in_progress() to find out when the move has actually
        been stopped.
        '''
        with self._state_lock:
            self._cancel_requested = True
    
    def in_progress(self):
        '''
        Check whether the move is currently in progress.
        
        **Returns:**
            (boolean): True if there is a move in progress, False otherwise.
        '''
        with self._state_lock:
            return self._in_progress
    
    def reached_goal(self):
        '''
        Get whether the movement reached the goal.

        **Returns:**
           reached_goal (boolean): True if the last arm move reached its goal.
        
        **Raises:**
        
            **exceptions.ArmNavError:** if there is a move currently in progress.
        '''
        with self._state_lock:
            if self._in_progress:
                raise ArmNavError('Result requested before move has finished')
            return self._reached_goal
        
    def get_errors(self):
        '''
        Get a list of the errors (as exception objects) raised during the move.
        
        **Raises:**
        
            **exceptions.ArmNavError:** if there is a move currently in progress
        '''
        with self._state_lock:
            if self._in_progress:
                raise ArmNavError('Result requested before move has finished')            
            return copy.copy(self._exceptions)

    def _set_in_progress(self, in_progress=True):
        with self._state_lock:
            self._in_progress = in_progress

    def _set_reached_goal(self, reached_goal=True):
        with self._state_lock:
            self._reached_goal = reached_goal

    def _add_error(self, ex):
        with self._state_lock:
            self._exceptions.append(ex)

    def _get_cancel_requested(self):
        with self._state_lock:
            return self._cancel_requested or rospy.is_shutdown()

class ArmMoverWorker(threading.Thread):
    def __init__(self, arm_name):
        '''Worker class that can move or query the arm. By making several of these,
        the arm mover class can do multiple movements/queries at once.
        '''
        self._arm_name = arm_name
        threading.Thread.__init__(self)
        self.daemon = True

        self._current_handle = None
        self._task_cv = threading.Condition()


    def assign_task(self, handle):
        ''' Assign a task to this worker.

        **Args:**
            handle (MovementHandle): Handle to the movement being assigned.

        **Raises:**
            ArmNavError if already doing another task.
        '''
        self._task_cv.acquire()
        try:
            if self._current_handle is not None:
                raise ArmNavError('Movement requested while other movement still runnning!')
            self._current_handle = handle
            self._task_cv.notify()
        finally:
            self._task_cv.release()
    
    def run(self):
        '''Loop and wait to be assigned a task.
        '''
        self._initialize()

        while not rospy.is_shutdown():
            self._task_cv.acquire()
            while (self._current_handle is None) and (not rospy.is_shutdown()):
                self._task_cv.wait(0.01)
            self._task_cv.release()

            if rospy.is_shutdown():
                break

            try:
                rospy.logdebug('ArmMoverWorker starting task: %s(%s)' % (
                        str(self._current_handle.task_func), str(self._current_handle.task_args)))
                self._current_handle.task_func(self, *self._current_handle.task_args)
            except Exception as e:
                self._current_handle._add_error(e)
            
            self._task_cv.acquire()
            self._current_handle._set_in_progress(False)
            self._current_handle = None
            self._task_cv.release()

    def _initialize(self):
        '''Connect up all services and action clients.
        '''
        self._world_interface = WorldInterface()
        self._controller_manager = ControllerManagerClient()
        if self._arm_name in ['right_arm', 'left_arm']:
            self._group_name = self._arm_name
            self._planner = ArmPlanner(self._arm_name)
            self._hand_description = HandDescription(self._arm_name)

            arm_abbr = self._arm_name[0]
            self._joint_controller = '%s_arm_controller' % arm_abbr                
            self._cartesian_controller = '%s_cart' % arm_abbr
            
            self._move_arm_client = actionlib.SimpleActionClient(
                'move_%s' % self._arm_name, arm_navigation_msgs.msg.MoveArmAction)
            self._wait_for_action_server(self._move_arm_client)
            
            jt_action_name = '/%s_arm_controller/joint_trajectory_action' % arm_abbr
            self._joint_trajectory_client = actionlib.SimpleActionClient(jt_action_name, JointTrajectoryAction)
            self._wait_for_action_server(self._joint_trajectory_client)
            
            self._cart_interface = CartesianControllerInterface(self._arm_name)
        elif self._arm_name == 'both':
            self._joint_controller = 'two_arm_controller'
            jt_two_arm_action_name = '/two_arm_controller/joint_trajectory_action'
            self._joint_trajectory_client = actionlib.SimpleActionClient(jt_two_arm_action_name, JointTrajectoryAction)
            self._wait_for_action_server(self._joint_trajectory_client)
        else:
            raise ValueError('Invalid arm name for worker: %s' % self._arm_name)

    def _wait_for_action_server(self, action_client, max_wait=10., wait_increment=0.1):
        '''
        Wait for the action server corresponding to this action client to be ready.
        
        **Args:**
            **action_client (actionlib.SimpleActionClient):** client for an action

            *max_wait (float):* Total number of seconds to wait before failing

            *wait_increment (float):* Number or seconds to wait between checks to rospy.is_shutdown()
        
        **Raises:**

            **exceptions.ActionFailedError:** if max_wait seconds elapses without server being ready.
        '''
        for ii in range(int(round(max_wait / wait_increment))):
            if rospy.is_shutdown():
                raise ActionFailedError('Could not connect to action server (rospy shutdown requested)')
            
            if action_client.wait_for_server(rospy.Duration(wait_increment)):
                return
        raise ActionFailedError('Could not connect to action server (timeout exceeeded)')

    def _move_to_goal(self, goal,
                try_hard=False, collision_aware_goal=True, planner_timeout=5., ordered_collisions=None,
                bounds=None, planner_id='', cartesian_timeout=5.0):
        '''
        Move the specified arm to the given goal.
        
        This function should only get called indirectly by calling move_arm_to_goal.
        '''
        try:
            reached_goal = False
            
            # check which kind of goal we were given
            if type(goal) == PoseStamped:
                goal_is_pose = True
            elif type(goal) == JointState:
                goal_is_pose = False
            else:
                raise ArmNavError('Invalid goal type %s' % str(type(goal)))
    
            rospy.loginfo('Attempting to use move arm to get to goal')
            try:
                self._move_to_goal_using_move_arm(
                    goal, planner_timeout, ordered_collisions, bounds, planner_id)
                reached_goal = True
            except ArmNavError as e:
                self._current_handle._add_error(e)
                rospy.loginfo('Move arm failed: %s' % str(e))
    
            if (not reached_goal) and try_hard:
                rospy.loginfo('Attempting to move directly to goal')
                try:
                    self._move_to_goal_directly(goal, planner_timeout, bounds, collision_aware=True)
                    reached_goal = True
                except ArmNavError as e:
                    self._current_handle._add_error(e)
                    rospy.loginfo('Collision aware IK failed: %s' % str(e))
    
            if (not reached_goal) and try_hard and (not collision_aware_goal) and goal_is_pose:
                rospy.loginfo('Attempting to move directly to goal, ignoring collisions')
                try:
                    self._move_to_goal_directly(goal, planner_timeout, bounds, collision_aware=False)
                    reached_goal = True
                except ArmNavError as e:
                    self._current_handle._add_error(e)
                    rospy.loginfo('Non-collision aware IK failed: %s' % str(e))
    
            if (not reached_goal) and try_hard and (not collision_aware_goal) and goal_is_pose:
                rospy.loginfo('Attempting to use cartesian controller to move towards goal')
                try:
                    self._move_to_goal_using_cartesian_control(goal, cartesian_timeout, bounds)
                    reached_goal = True
                except ArmNavError as e:
                    self._current_handle._add_error(e)
        finally:
            self._current_handle._set_reached_goal(reached_goal)


    def _move_into_joint_limits(self):
        '''
        Moves the arm into joint limits if it is outside of them.

        This cannot be a collision free move but it is almost always very very short.

        **Raises:**

            **exceptions.ArmNavError:** if IK fails

            **rospy.ServiceException:** if there is a problem calling the IK service

            **ValueError:** if the goal type is wrong
        '''
        joint_state = self._planner.get_closest_joint_state_in_limits()
        self._move_to_goal_directly(joint_state, trajectory_time=0.5)
        self._current_handle._set_reached_goal(True)

    def _move_out_of_collision(self, move_mag=0.3, num_tries=100):
        '''
        Tries to find a small movement that will take the arm out of collision.

        **Args:**

            *move_mag (float):* Max magnitude in radians of movement for each joint.
            
            *num_tries (int):* Number of random joint angles to try before giving up.
            
        **Returns:**
            succeeded (boolean): True if arm was sucessfully moved out of collision.
        '''
        req = GetStateValidityRequest()
        req.robot_state = self.world_interface.get_robot_state()
        req.check_collisions = True
        req.check_path_constraints = False
        req.check_joint_limits = False
        req.group_name = self._arm_name
        res = self._planner.get_state_validity_service(req)
        if res.error_code.val == ArmNavErrorCodes.SUCCESS:
            rospy.logdebug('Current state not in collision')
            return False

        joint_state = self._planner.arm_joint_state()
        current_joint_position = np.array(joint_state.position)
        for ii in range(num_tries):
            joint_position = current_joint_position + np.random.uniform(
                -move_mag, move_mag, (len(joint_state.position),))
            joint_state.position = list(joint_position)
            trajectory_tools.set_joint_state_in_robot_state(joint_state, req.robot_state)
            res = self._planner.get_state_validity_service(req)
            in_collision = (res.error_code.val != ArmNavErrorCodes.SUCCESS)
            rospy.logdebug('%s in collision: %s' % (str(joint_position), str(in_collision)))
            if not in_collision:
                self._move_to_goal_directly(joint_state, None, None, collision_aware=False)
                self._current_handle._set_reached_goal(True)
        self._current_handle._set_reached_goal(False)
    
    def _call_action(self, action_client, goal):
        '''
        Call an action and wait for it to complete.
        
        **Returns:** 
            Result of action.

        **Raises:** 
            
            **exceptions.ArmNavError:** if action fails.
        '''
        action_client.send_goal(goal)
        gs = actionlib_msgs.msg.GoalStatus()

        r = rospy.Rate(100)
        while True:
            if self._current_handle._get_cancel_requested():
                raise ActionFailedError('Preempted (cancel requested)')
            state = action_client.get_state()
            if state in [gs.PENDING, gs.ACTIVE, gs.PREEMPTING, gs.RECALLING]:
                # action is still going
                pass
            elif state in [gs.PREEMPTED, gs.REJECTED, gs.RECALLED, gs.LOST]:
                raise ArmNavError('Action call failed (%d)!' % (state,))
            elif state in [gs.SUCCEEDED, gs.ABORTED]:
                return action_client.get_result() 
            r.sleep()

    def _move_to_goal_directly(self, goal, planner_timeout=5.0, bounds=None,
            collision_aware=True, trajectory_time=5.0):
        '''
        Move directly to the goal.
        
        No planning, just interpolated joint positions.
        
        If goal is a PoseStamped, does IK to find joint positions to put the end effector in that pose.
        Then executes a trajectory where the only point is the goal joint positions.
        
        Note: planner_timeout, collision_aware and bounds only apply to the IK, and so are not used when
        the goal is already a JointState

        **Raises:**

            **exceptions.ArmNavError:** if IK fails

            **rospy.ServiceException:** if there is a problem calling the IK service

            **ValueError:** if the goal type is wrong
        '''
        if type(goal) == JointState:
            joint_state = goal
        elif type(goal) == PoseStamped:
            ik_res = self._planner.get_ik(goal, collision_aware=collision_aware, starting_state=None,
                seed_state=None, timeout=planner_timeout)
            if not ik_res.error_code.val == ArmNavErrorCodes.SUCCESS:
                raise ArmNavError('Unable to get IK for pose', ik_res.error_code)
            joint_state = ik_res.solution.joint_state
        else:
            raise ValueError('Invalid goal type: %s' % str(type(goal)))
        
        trajectory = JointTrajectory()
        trajectory.joint_names = self._planner.joint_names
        jtp = JointTrajectoryPoint()
        jtp.positions = joint_state.position
        jtp.time_from_start = rospy.Duration(trajectory_time)
        trajectory.points.append(jtp)
        self._execute_joint_trajectory(trajectory)

        # should actually check this...
        self._current_handle._set_reached_goal(True)
        
    def _move_to_goal_using_move_arm(self, goal, planner_timeout, ordered_collisions, bounds, planner_id=''):
        '''
        Try using the MoveArm action to get to the goal.
        '''
        self._controller_manager.switch_controllers(start_controllers=[self._joint_controller])
        current_state = self._world_interface.get_robot_state()
        link_name = self._hand_description.hand_frame
        
        if type(goal) == JointState:
            mp_request = conversions.joint_state_to_motion_plan_request(
                goal, link_name, self._group_name, current_state,
                timeout=planner_timeout, bounds=bounds, planner_id=planner_id)
        elif type(goal) == PoseStamped:
            mp_request = conversions.pose_stamped_to_motion_plan_request(
                goal, link_name, self._group_name, starting_state=current_state, 
                timeout=planner_timeout, bounds=bounds, planner_id=planner_id)
        else:
            raise ValueError('Invalid goal type %s' % str(type(goal)))
        
        ma_goal = arm_navigation_msgs.msg.MoveArmGoal()
        ma_goal.motion_plan_request = mp_request
        if ordered_collisions:
            ma_goal.operations = ordered_collisions
        ma_goal.planner_service_name = DEFAULT_PLANNER_SERVICE_NAME
        
        # send goal to move arm
        res = self._call_action(self._move_arm_client, ma_goal)
        if res == None:
            raise ArmNavError('MoveArm failed without setting result')
        elif not res.error_code.val == ArmNavErrorCodes.SUCCESS:
            raise ArmNavError('MoveArm failed', res.error_code)
        else:
            self._current_handle._set_reached_goal(True)

    def _move_to_goal_using_cartesian_control(self, goal, timeout, bounds):
        if type(goal) == PoseStamped:
            pose_stamped = goal
        else:
            raise ValueError('Invalid goal type for cartesian control: %s' % str(type(goal)))
        self._controller_manager.switch_controllers(start_controllers=[self._cartesian_controller])
        self._cart_interface.set_desired_pose(pose_stamped)
        start_time = time.time()
        r = rospy.Rate(100)
        try:
            print 'Current handle'
            print self._current_handle._get_cancel_requested()
            while not self._current_handle._get_cancel_requested():
                print 'Inside while loop'
                # ignores bounds right now and uses defaults... fixme
                if self._cart_interface.reached_desired_pose():
                    self._current_handle._set_reached_goal(True)
                    return
                if (time.time() - start_time) > timeout:
                    raise ArmNavError('Cartesian control move time out', 
                                      ArmNavErrorCodes(ArmNavErrorCodes.TIMED_OUT))
                r.sleep()
        finally:
            self._cart_interface.cancel_desired_pose()
                                   
    def _execute_joint_trajectory(self, trajectory):
        '''
        Executes the given trajectory, switching controllers first if needed.

        **Args:**

            **trajectory (trajectory_msgs.msg.JointTrajectory):** Trajectory to execute.
        '''
        self._controller_manager.switch_controllers(start_controllers=[self._joint_controller])
        goal = JointTrajectoryGoal()
        goal.trajectory = trajectory
        jt_res = self._call_action(self._joint_trajectory_client, goal)

        # should actually check this
        self._current_handle._set_reached_goal(True)

        return jt_res
    
    def _execute_two_arm_trajectory(self, trajectory):
        '''
        Executes the given trajectory, switching controllers first if needed.

        **Args:**

        **trajectory (trajectory_msgs.msg.JointTrajectory):** Trajectory to execute.
        '''
        self._controller_manager.switch_controllers(start_controllers=[self._joint_controller])
        goal = JointTrajectoryGoal()
        goal.trajectory = trajectory
        jt_res = self._call_action(self._joint_trajectory_client, goal)

        # should actually check this
        self._current_handle._set_reached_goal(True)
        
        return jt_res
                

class ArmMover:
    def __init__(self):
        self._state_lock = threading.Lock()

        # to avoid synchronization problems when running left and right arm movements
        # at the same time, we start separate clients for each arm and for both arms, and for
        # control and queries
        self._workers = {}        
        for arm_name in ['right_arm', 'left_arm', 'both']:
            rospy.loginfo('Initializing ArmMoverWorker for arm: %s' % arm_name)
            worker = ArmMoverWorker(arm_name)
            self._workers[arm_name] = worker
            worker.start()

        # these just used for queries in the main thread
        self._planners = {
            'right_arm': ArmPlanner('right_arm'),
            'left_arm': ArmPlanner('left_arm')
            }

    def _get_worker(self, arm_name):
        # should add check here to see if its already in use -jdb
        return self._workers[arm_name]

    def _check_arm_in_use(self, arm_name):
        conflicts_with = {
            'right_arm':['right_arm', 'both'],
            'left_arm':['left_arm', 'both'],
            'both': ['right_arm', 'left_arm', 'both']
            }
        with self._state_lock:
            for other_arm_name in conflicts_with[arm_name]:
                handle = self._arms[other_arm_name]['current_handle']
                if handle is not None and handle.in_progress():
                    raise ArmNavError('Move requested for arm: %s, but move already in progress for arm: %s' % 
                                      arm_name, other_arm_name)

    def move_to_goal(self, arm_name, goal,
                try_hard=False, collision_aware_goal=False, planner_timeout=5., ordered_collisions=None,
                bounds=None, planner_id='', cartesian_timeout=5.0, blocking=True):
        '''
        Moves the arm to the desired goal.

        **Args:**
            **arm_name (str): Name of arm to use; either 'right_arm' or 'left_arm'.

            **goal (geometry_msgs.msg.PoseStamped or sensor_msgs.msg.JointState):** If the goal is a PoseStamped,
            will try to move the end effector to that pose. If the goal is a JointState, will try to move
            the arm to those joint positions.
            
            *try_hard (boolean):* If True, and there is no collision free trajectory, allow trajectories with 
            collisions.
            
            *collision_aware_goal (boolean):* If True, disallow trajectories with collisions at the goal state,
            even if try_hard is True. (can still try to move through collisions to get there)
            
            *planner_timeout (float):* Timeout (in seconds) for the motion planner.

            *ordered_collisions (arm_navigation_msgs.msg.OrderedCollisionOperations):* Extra collision operations to
            use when planning.
            
            *bounds (sequence):* Sequence with 6 elements,
            [x_bound, y_bound, z_bound, roll_bound, pitch_bound, yaw_bound].
            
            *planner_id (string):* Planner to use.
            
            *cartesian_timeout (float):* If the cartesian controller ends up getting used, how long to use it for.
            
            *blocking (boolean):* If True, will not return until move is completed or failed. If False, will execute
            action in another thread, and the user can use in_progress() to check whether move is still in
            progress, and call cancel() to stop the move.

        **Returns:**
            reached_goal (boolean) if called with blocking set to True

        '''
        if not arm_name in ['right_arm', 'left_arm']:
            raise ValueError('Invalid arm name for move_to_goal(): %s' % arm_name)

        return self._ex_task(
            ArmMoverWorker._move_to_goal, blocking, arm_name, goal, try_hard, collision_aware_goal, planner_timeout,
            ordered_collisions, bounds, planner_id, cartesian_timeout)

    def move_to_goal_using_move_arm(self, arm_name, goal, planner_timeout=None, ordered_collisions=None,
                                    bounds=None, planner_id='', blocking=True):
        return self._ex_task(
            ArmMoverWorker._move_to_goal_using_move_arm, blocking, arm_name, goal, planner_timeout,
            ordered_collisions, bounds, planner_id)

    def move_to_goal_using_cartesian_control(self, arm_name, goal, timeout=None, bounds=None, blocking=True):
        if not arm_name in ['right_arm', 'left_arm']:
            raise ValueError('Invalid arm name for move_to_goal(): %s' % arm_name)
        return self._ex_task(
            ArmMoverWorker._move_to_goal_using_cartesian_control, blocking, arm_name, goal, timeout, bounds)

    def execute_joint_trajectory(self, arm_name, trajectory, blocking=True):
        '''
        Executes the given trajectory, switching controllers first if needed.

        **Args:**

            **trajectory (trajectory_msgs.msg.JointTrajectory):** Trajectory to execute.
        '''
        return self._ex_task(ArmMoverWorker._execute_joint_trajectory, blocking, arm_name, trajectory)

    def move_into_joint_limits(self, arm_name, blocking=True):
        return self._ex_task(ArmMoverWorker._move_into_joint_limits, blocking, arm_name)

    def move_out_of_collision(self, arm_name, move_mag=0.3, num_tries=100, blocking=True):
        return self._ex_task(ArmMoverWorker._move_out_of_collision, blocking, arm_name, move_mag, num_tries)

    def execute_two_arm_trajectory(self, trajectory, blocking=True):
        '''
        Executes the given trajectory, switching controllers first if needed.

        **Args:**

            **trajectory (trajectory_msgs.msg.JointTrajectory):** Trajectory to execute.
        '''
        return self.ex_task(ArmMoverWorker._execute_two_arm_trajectory, blocking, 'both', trajectory)

    
    def get_joint_state(self, arm_name):
        '''
        Get the current joint state for an arm.

        **Returns:**
            (sensor_msg.msg.JointState): Current joint state.
        '''
        if not arm_name in ['right_arm', 'left_arm']:
            raise ArmNavError('Invalid arm_name passed to get_joint_state(): %s' % arm_name)

        return self._planners[arm_name].arm_joint_state()

    def is_current_state_in_collision(self, arm_name, check_joint_limits = False):
        '''
        Tells you whether the current arm pose is in collision.

        **Returns:**
            succeeded (boolean): True if arm is in collision.
        '''
        req = GetStateValidityRequest()
        req.robot_state = self._world_interface.get_robot_state()
        req.check_collisions = True
        req.check_path_constraints = False
        req.check_joint_limits = check_joint_limits
        req.group_name = arm_name
        res = self._planners[arm_name].get_state_validity_service(req)
        if res.error_code.val == ArmNavErrorCodes.SUCCESS:
            rospy.logdebug('Current state not in collision')
            return False
        else:
            rospy.logdebug('Current state in collision')
            return True
            
    def is_in_collision(self, arm_name, joint_state, check_joint_limits=False):
        '''
        Tells you whether the current arm pose is in collision.

        **Returns:**
            succeeded (boolean): True if arm is in collision.
        '''
        req = GetStateValidityRequest()
        req.robot_state = self._world_interface.get_robot_state()
        trajectory_tools.set_joint_state_in_robot_state(joint_state, req.robot_state)
        req.check_collisions = True
        req.check_path_constraints = False
        req.check_joint_limits = check_joint_limits
        req.group_name = self._arm_name
        res = self._planners[arm_name].get_state_validity_service(req)
        if res.error_code.val == ArmNavErrorCodes.SUCCESS:
            rospy.logdebug('Current state not in collision')
            return False
        else:
            rospy.logdebug('Current state in collision')
            return True
        
    def _ex_task(self, func, blocking, arm_name, *args):
        '''Execute the given function in a worker thread. If blocking is True, wait for the action
        to finish. Otherwise, return a handle to the action
        '''
        worker = self._get_worker(arm_name)
        handle = MovementHandle(func, args)
        worker.assign_task(handle)
        if blocking:
            handle.wait()
        return handle
