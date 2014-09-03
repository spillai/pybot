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
'''
Defines the CartesianControllerInterface for working with the Cartesian controller.
'''

__docformat__ = "restructuredtext en"

import threading, copy, time
import numpy as np
from scipy import linalg

import roslib
roslib.load_manifest('pr2_python')

import rospy
import tf
from tf import transformations
from geometry_msgs.msg import PoseStamped

from pr2_python.transform_listener import get_transform_listener
from pr2_python import geom, conversions
from pr2_python.hand_description import HandDescription

# frame that the cartesian controller expects
CONTROLLER_FRAME = '/torso_lift_link'

class CartesianControllerInterface:
    '''
    Class that helps send pose goals to the jacobian transpose cartesian controller.
    
    Make sure that you load and start the cartesian controller (r/l_cart) before trying
    to send desired poses.
    '''

    def __init__(self, arm_name, goal_frame_id=None):
        '''
        Constructor for CartesianControllerInterface.

        **Args**
        
            **arm_name (string):** 'right_arm' or 'left_arm'.
        '''
        if not arm_name in ['left_arm', 'right_arm']:
            raise ValueError('Invalid arm name: %s' % arm_name)
        self._arm_name = arm_name
        self._goal_link = '%s_wrist_roll_link' % arm_name[0]
        self._transform_listener = get_transform_listener()
        arm_abbr = arm_name[0]
        self._desired_pose_pub = rospy.Publisher('%s_cart/command_pose' % arm_abbr, PoseStamped)

        self._hand_description = HandDescription(arm_name)

        self.goal_frame_id = goal_frame_id
        if self.goal_frame_id is None:
            self.br = None
        else:
            self.br = tf.TransformBroadcaster()
        
        self._state_lock = threading.Lock()
        self._desired_pose_stamped = None
        self._delta_dist = None
        self._pose_sending_thread = threading.Thread(target=self._pose_sending_func)
        self._pose_sending_thread.daemon = True
        self._pose_sending_thread.start()

    def set_desired_pose(self, ps, delta_dist=0.012):
        '''
        Set the desired pose (starts arm motion).

        **Args:**

            **ps (geometry_msgs.msg.PoseStamped):** Goal pose.

            *delta_dist (float):* Distance (in meters) from current pose to put desired pose.
            This is updated as the arm moves towards the ultimate goal, so that the desired
            pose is always the same distance away, since the torques applied by the jacobian
            transpose controller depend on distance from the goal.
        '''
        with self._state_lock:
            self._desired_pose_stamped = ps
            self._delta_dist = delta_dist

    def reached_desired_pose(self, max_trans_err=0.02, max_rpy_err=0.04):
        '''
        Checks whether controller has reached the desired pose.

        **Args:**

            *max_trans_err (float):* Max translational error (in meters) allow in each dimension.
            
            *max_rpy_err (float):* Max rotational angle (in radians) allow for roll, pitch, or yaw.

        **Returns:**
            reached_desired (boolean): True if arm has reached the desired pose, False otherwise.
        '''
        with self._state_lock:
            ps_des = copy.deepcopy(self._desired_pose_stamped)

        self._transform_listener.waitForTransform(
            CONTROLLER_FRAME, ps_des.header.frame_id, rospy.Time(0), rospy.Duration(4.0))
        ps_des.header.stamp = rospy.Time(0)
        ps_des = self._transform_listener.transformPose(CONTROLLER_FRAME, ps_des)
        torso_H_wrist_des = conversions.pose_to_mat(ps_des.pose)

        # really shouldn't need to do waitForTransform here, since rospy.Time(0) means "use the most
        # recent available transform". but without this, tf raises an ExtrapolationException. -jdb
        self._transform_listener.waitForTransform(
            CONTROLLER_FRAME, self._goal_link, rospy.Time(0), rospy.Duration(4.0))
        trans, rot = self._transform_listener.lookupTransform(CONTROLLER_FRAME, self._goal_link, rospy.Time(0))
        torso_H_wrist = geom.trans_rot_to_hmat(trans, rot)

        # compute error between actual and desired wrist pose
        wrist_H_wrist_des = np.dot(linalg.inv(torso_H_wrist), torso_H_wrist_des)
        trans_err, rot_err = geom.hmat_to_trans_rot(wrist_H_wrist_des)

        curr_trans_err = np.abs(trans_err).max()
        curr_rpy_err = np.abs(transformations.euler_from_quaternion(rot_err)).max()
        if curr_trans_err < max_trans_err and curr_rpy_err < max_rpy_err:
            self.cancel_desired_pose()
            return True
        else:
            return False

    def cancel_desired_pose(self):
        '''
        Cancel the current desired pose.
        '''
        with self._state_lock:
            self._desired_pose_stamped = None

    def _pose_sending_func(self):
        '''
        Continually sends the desired pose to the controller.
        '''
        r = rospy.Rate(100)
        while not rospy.is_shutdown():
            try:
                r.sleep()
            except rospy.ROSInterruptException:
                break

            # get the desired pose as a transform matrix
            with self._state_lock:
                ps_des = copy.deepcopy(self._desired_pose_stamped)
                delta_dist = self._delta_dist

            if ps_des == None:
                # no desired pose to publish
                continue

            # convert desired pose to transform matrix from goal frame to CONTROLLER_FRAME
            self._transform_listener.waitForTransform(
                CONTROLLER_FRAME, ps_des.header.frame_id, rospy.Time(0), rospy.Duration(4.0))
            #trans, rot = self._transform_listener.lookupTransform(CONTROLLER_FRAME, ps_des.header.frame_id, rospy.Time(0))
            #torso_H_wrist_des = geom.trans_rot_to_hmat(trans, rot)
            
            ps_des.header.stamp = rospy.Time(0)
            ps_des = self._transform_listener.transformPose(CONTROLLER_FRAME, ps_des)
            torso_H_wrist_des = conversions.pose_to_mat(ps_des.pose)


            if self.br is not None:
                p = ps_des.pose.position
                q = ps_des.pose.orientation
                self.br.sendTransform(
                    (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                    rospy.Time.now(),
                    '/cci_goal',
                    CONTROLLER_FRAME)

            # get the current pose as a transform matrix
            goal_link = self._hand_description.hand_frame

            # really shouldn't need to do waitForTransform here, since rospy.Time(0) means "use the most
            # recent available transform". but without this, tf sometimes raises an ExtrapolationException. -jdb
            self._transform_listener.waitForTransform(
                CONTROLLER_FRAME, goal_link, rospy.Time(0), rospy.Duration(4.0))
            trans, rot = self._transform_listener.lookupTransform(CONTROLLER_FRAME, goal_link, rospy.Time(0))
            torso_H_wrist = geom.trans_rot_to_hmat(trans, rot)

            # compute difference between actual and desired wrist pose
            wrist_H_wrist_des = np.dot(linalg.inv(torso_H_wrist), torso_H_wrist_des)
            trans_err, rot_err = geom.hmat_to_trans_rot(wrist_H_wrist_des)

            # scale the relative transform such that the translation is equal
            # to delta_dist. we're assuming that the scaled angular error will
            # be reasonable
            scale_factor = delta_dist / linalg.norm(trans_err)
            if scale_factor > 1.0:
                scale_factor = 1.0 # don't overshoot
            scaled_trans_err = scale_factor * trans_err
            angle_err, direc, point = transformations.rotation_from_matrix(wrist_H_wrist_des)
            scaled_angle_err = scale_factor * angle_err
            wrist_H_wrist_control = np.dot(
                transformations.translation_matrix(scaled_trans_err),
                transformations.rotation_matrix(scaled_angle_err, direc, point))

            # find incrementally moved pose to send to the controller
            torso_H_wrist_control = np.dot(torso_H_wrist, wrist_H_wrist_control)
            desired_pose = conversions.mat_to_pose(torso_H_wrist_control)
            desired_ps = PoseStamped()
            desired_ps.header.stamp = rospy.Time(0)
            desired_ps.header.frame_id = CONTROLLER_FRAME
            desired_ps.pose = desired_pose

            if self.br is not None:
                p = desired_ps.pose.position
                q = desired_ps.pose.orientation
                self.br.sendTransform(
                    (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                    rospy.Time.now(),
                    '/cci_imm_goal',
                    CONTROLLER_FRAME)

            # send incremental pose to controller
            self._desired_pose_pub.publish(desired_ps)
