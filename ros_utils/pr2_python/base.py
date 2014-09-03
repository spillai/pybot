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
# Author: Bhaskara Marthi

'''
Defines the Base class for moving the PR2 base
'''

__docformat__ = "restructuredtext en"

import roslib
roslib.load_manifest('pr2_python')
import rospy
import threading
import move_base_msgs.msg as mbm
import actionlib as al
import actionlib_msgs.msg as am
import geometry_msgs.msg as gm
import nav_msgs.msg as nav_msgs
import tf.transformations as trans
import pr2_python.exceptions as ex
import sensor_msgs.msg as sm
import arm_navigation_msgs.msg as anm
import copy
import sbpl_3dnav_planner.srv as sbpl
import transform_listener as tl
import pose_follower_3d.srv as pose_follower_3d
from math import sqrt
import numpy as np
import tf

class Base():
    """
    Represents the PR2 Base
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._js_lock = threading.Lock()
        self._js_counter = 0
        self._last_pose = None
        self._last_state = None
        self._sub = rospy.Subscriber('pose_gossip',
                                     gm.PoseWithCovarianceStamped,
                                     self._save_pose)
        self._js_sub = rospy.Subscriber('joint_states', sm.JointState,
                                        self._save_joint_state)
        self._check_pose_srv = rospy.ServiceProxy('sbpl_full_body_planning/'
                                                   'collision_check',
                                                 sbpl.FullBodyCollisionCheck)
        self._base_pose_srv = rospy.ServiceProxy('sbpl_full_body_planning/'
                                                 'find_base_poses',
                                                 sbpl.GetBasePoses)
        self._follow_trajectory_srv = rospy.ServiceProxy('pose_follower_3d/'
                                                         'follow_trajectory',
                                                         pose_follower_3d.FollowTrajectory)
        self._move_out_of_collision_srv = rospy.ServiceProxy('pose_follower_3d/'
                                                             'push_out_of_collision',
                                                             pose_follower_3d.PushOutOfCollision)
        self._ac = al.SimpleActionClient('move_base', mbm.MoveBaseAction)
        self._min_distance = 0.1
        self._min_angle = 0.1
        rospy.loginfo("Waiting for move base action server...")
        #self._ac.wait_for_server()
        rospy.loginfo("Move base action server ready")

    def move_to(self, x, y, theta):
        """
        Moves base to a 2d pose specified in the map frame.  Possible outcomes:

        * The function returns successfully.  In this case, the robot is guaranteed to be at the goal pose.
        * An ActionFailed exception is thrown if navigation fails.
        * The robot loops forever.  This should only happen in situations involving malicious humans.
        """

        goal = mbm.MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = '/map'
        goal.target_pose.pose = _to_pose(x, y, theta)
        rospy.loginfo("Sending base goal ({0}, {1}, {2}) and waiting for result".\
                      format(x, y, theta))
        self._ac.send_goal(goal)
        self._ac.wait_for_result()

        if self._ac.get_state() != am.GoalStatus.SUCCEEDED:
            raise ex.ActionFailedError()


    def _get_current_pose_helper(self):
        """
        :returns: Robot pose, as ((x,y,z), (x,y,z,w))
        """
        listener = tl.get_transform_listener()
        fixed_frame = "/map"
        base_frame = "/base_footprint"
        wait = rospy.Duration(1.0)
        latest = rospy.Time(0)
        while not rospy.is_shutdown():
            try:
                listener.waitForTransform(fixed_frame, base_frame, latest, wait)
                return listener.lookupTransform(fixed_frame, base_frame, latest)
                return (tr.x, tr.y, trans.euler_from_quaternion(rot)[2])
            except tf.Exception:
                print "Waiting for transform from {0} to {1}".\
                      format(base_frame, fixed_frame)

    def get_current_pose(self):
        """
        :returns: Robot pose, as (x,y, theta).  Blocks until transform available.
        """
        tr, rot = self._get_current_pose_helper()
        return (tr[0], tr[1], trans.euler_from_quaternion(rot)[2])

    def get_current_pose_stamped(self):
       """
       :returns: Robot pose as a geometry_msgs.msg.PoseWithCovarianceStamped object
       """
       m = gm.PoseStamped()#WithCovarianceStamped()
       m.header.frame_id = "/map"
       m.header.stamp = rospy.Time.now()
       tr, rot = self._get_current_pose_helper()
       m.pose.position = gm.Point(*tr)
       m.pose.orientation = gm.Quaternion(*rot)
       return m



    def move_manipulable_pose(self, x, y, z, group='torso', try_hard=False):
        """
        Moves base to 2d pose from which an object can be manipulated.

        :param x: x coordinate of object
        :param y: y coordinate of object
        :param z: z coordinate of object
        :keyword group: One of 'torso', 'head', 'left_arm', or 'right_arm'.

        :raise exceptions.ActionFailedError: If the base movement failed
        :raise exceptions.AllYourBasePosesAreBelongToUs: If no valid base poses could be found
        """
        assert group in ['left_arm', 'right_arm', 'torso', 'head']
        poses = self.get_base_poses(x, y, z, group)
        if poses is None or not len(poses):
            raise ex.AllYourBasePosesAreBelongToUs((x,y,z))
        if not try_hard:
            pose = poses[0].pose
            try:
                self.move_to(pose.position.x, pose.position.y, _yaw(pose.orientation))
            except ex.ActionFailedError(), e:
                raise e
        else:
            poses_sorted =self._sort_poses(poses)
            for i in range(len(poses_sorted)):
                pose = poses_sorted[i].pose
                try:
                    self.move_to(pose.position.x, pose.position.y, _yaw(pose.orientation))                    
                except ex.ActionFailedError(), e:
                    raise e            
                return True
        
    def move_to_look(self, x, y, z, try_hard=False):
        self.move_manipulable_pose(x, y, z, group='head', try_hard=try_hard)        

    def _sort_poses(self, poses):
        poses_and_dists = [(p, self._dist_between(self.get_current_pose_stamped(), p)) for p in poses]
        poses_and_dists.sort(key=lambda pair: pair[1])
#        if len(poses_and_dists) == 0:
#            return None
        poses_in_dist_order = [pair[0] for pair in poses_and_dists]
        return poses_in_dist_order   

    def _dist_between(self,pos1,pos2):
        """Call as dist_between(position1, position2) or
        dist_between(x1, y1, x2, y2).
        
        """
        x1 = pos1.pose.position.x
        x2 = pos2.pose.position.x
        y1 = pos1.pose.position.y
        y2 = pos2.pose.position.y
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _save_pose(self, m):
        with self._lock:
            self._last_pose = m

    def _save_joint_state(self, m):
        # No need to save this 100 times a second
        self._js_counter += 1
        if self._js_counter % 20 == 0:
            with self._lock:
                if self._last_pose is None:
                    return
                with self._js_lock:
                    self._last_state = anm.RobotState()
                    self._last_state.joint_state.position = m.position
                    self._last_state.joint_state.name = m.name
                    md = self._last_state.multi_dof_joint_state
                    md.frame_ids.append('map')
                    md.child_frame_ids.append('base_footprint')
                    md.poses.append(self._last_pose.pose.pose)
                       
    def get_base_poses(self, x, y, z, group='head', sort=False):
        pose = gm.PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = gm.Point(x, y, z)
        pose.pose.orientation = _to_quaternion(0)
        while not rospy.is_shutdown():
            with self._js_lock:
                if self._last_state is not None:
                    robot_state = copy.deepcopy(self._last_state)
                    break
            rospy.loginfo("Waiting for joint state")
            rospy.sleep(1.0)
        poses = self._base_pose_srv(object_pose=pose, robot_state=robot_state, group_name=group).base_poses
        if sort:
            return self._sort_poses(poses)
        return poses

    def check_base_pose(self, pose_stamped):
        while not rospy.is_shutdown():
            with self._js_lock:
                if self._last_state is not None:
                    robot_state = copy.deepcopy(self._last_state)
                    break
            rospy.loginfo("Waiting for joint state")
            rospy.sleep(1.0)
        robot_states = []
        robot_state.multi_dof_joint_state.poses[0] = copy.deepcopy(pose_stamped.pose)
        robot_states.append(robot_state)
        error_codes = self._check_pose_srv(robot_states=robot_states).error_codes
        if not len(error_codes) > 0:
            raise ex.AllYourBasePosesAreBelongToUs((x,y,z))
        if error_codes[0].val == error_codes[0].SUCCESS:
            return True
        else:
            return False        

    def _interpolate_and_check_base_poses(self, my_poses, min_distance=0.1,min_rotation=0.25):
        while not rospy.is_shutdown():
            with self._js_lock:
                if self._last_state is not None:
                    robot_state = copy.deepcopy(self._last_state)
                    break
            rospy.loginfo("Waiting for joint state")
            rospy.sleep(1.0)
        return_poses = []
        for i in range(len(my_poses)-1):
            poses = []
            poses.append(my_poses[i])
            poses.append(my_poses[i+1])
            interpolated_poses = self._interpolate_poses(poses=poses,min_distance=min_distance,min_rotation=min_rotation)
            robot_states = []
            for pose in interpolated_poses:
                robot_state.multi_dof_joint_state.poses[0] = pose.pose
                robot_states.append(robot_state)
            error_codes = self._check_pose_srv(robot_states=robot_states).error_codes
            motion_ok = True
            for j in range(len(error_codes)):#check if all error codes are ok
                if error_codes[j].val != error_codes[j].SUCCESS:
                    motion_ok = False
                    break
            if motion_ok:
                if i == 0:
                    return_poses.extend(interpolated_poses)
                else:
                    return_poses.extend(interpolated_poses[0:len(interpolated_poses)-1])
            else:
                break
        print 'Return Poses'
        for pose in return_poses:
          print pose
        return return_poses

    def _shortest_angular_distance(self,angle_1,angle_2):
        result = self._normalize_angle_positive(self._normalize_angle_positive(angle_2) - self._normalize_angle_positive(angle_1))
        if result > np.pi:
          result = -(2*np.pi - result)
        return self._normalize_angle(result)

    def _normalize_angle(self,angle):
        a = self._normalize_angle_positive(angle)
        if a > np.pi:
            a = a - 2.0 * np.pi;
        return a
  

    def _normalize_angle_positive(self,angle):
      angle = ((angle % (2*np.pi)) + 2*np.pi) % (2*np.pi)
      return angle

    def _interpolate_poses(self, poses, min_distance, min_rotation):
      return_poses = []
      for i in range(len(poses)-1):          
          yaw_start = _yaw(poses[i].pose.orientation)
          yaw_end = _yaw(poses[i+1].pose.orientation)
          xd = poses[i+1].pose.position.x - poses[i].pose.position.x
          yd = poses[i+1].pose.position.y - poses[i].pose.position.y
          distance = sqrt(xd*xd+yd*yd)
          delta_angle = self._shortest_angular_distance(yaw_start,yaw_end)
          num_segments = -1
          if np.abs(distance) > min_distance:
              num_segments = np.ceil(distance/min_distance)
          if np.abs(delta_angle) > min_rotation:
              num_segments_angle = np.ceil(np.abs(delta_angle)/min_rotation)
              if num_segments_angle > num_segments:
                  num_segments = num_segments_angle
          if num_segments > 0:#should always be at least 2              
              for j in range(int(num_segments)):#last one will be put in by the next pass
                  x = poses[i].pose.position.x + j * (poses[i+1].pose.position.x - poses[i].pose.position.x)/num_segments
                  y = poses[i].pose.position.y + j * (poses[i+1].pose.position.y - poses[i].pose.position.y)/num_segments
                  theta = self._normalize_angle(yaw_start + j * delta_angle/num_segments) 
                  new_pose = gm.PoseStamped()
                  new_pose.header.frame_id = 'map'
                  new_pose.header.stamp = rospy.Time.now()
                  new_pose.pose.position.x = x
                  new_pose.pose.position.y = y
                  new_pose.pose.position.z = 0
                  new_pose.pose.orientation = _to_quaternion(theta)
                  return_poses.append(new_pose)
          else:
              return_poses.append(poses[i])
      # make sure the last pose is appended
      return_poses.append(poses[-1])
      return return_poses


    def follow_trajectory(self, path_x, path_y, path_theta):
        length = len(path_x)
        if (len(path_x) != len(path_y) or len(path_x) != len(path_theta)):
            rospy.logerror("All fields in path must have same size")
            return false
        path = nav_msgs.Path()
        for i in range(len(path_x)):
            pose = gm.PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = rospy.Time.now()
            pose.pose.position = gm.Point(path_x[i],path_y[i],0.0)
            pose.pose.orientation = _to_quaternion(path_theta[i])
            path.poses.append(pose)
        
        path.poses = self._interpolate_and_check_base_poses(path.poses)
        if not path.poses:
            return False
        success = self._follow_trajectory_srv(path=path)
        return False

    def move_out_of_collision(self):
        """
        Tries to move base out of collision
        * The function returns True if it could move the robot out of collision.
        """
        resp = self._move_out_of_collision_srv()
        return resp.recovered
            
        
def _yaw(q):
    e = trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
    return e[2]

def _to_quaternion(yaw):
    return gm.Quaternion(*trans.quaternion_from_euler(0, 0, yaw))

def _to_pose(x, y, theta):
    return gm.Pose(gm.Point(x, y, 0), _to_quaternion(theta))
                   
