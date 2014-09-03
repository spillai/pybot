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
Wrappers for working with TF transform listeners.
'''

__docformat__ = "restructuredtext en"

import roslib; roslib.load_manifest('pr2_python')
import threading
import rospy
import tf
import geometry_msgs.msg

import copy

from pr2_python import geom

_transform_listener = None
_transform_listener_creation_lock = threading.Lock()

def get_transform_listener():
    '''
    Gets the transform listener for this process.

    This is needed because tf only allows one transform listener per process. Threadsafe, so
    that two threads could call this at the same time, at it will do the right thing.
    '''
    global _transform_listener
    with _transform_listener_creation_lock:
        if _transform_listener == None:
            _transform_listener = tf.TransformListener()
        return _transform_listener

def get_transform_as_tr(frame_tgt, frame_src, t=rospy.Time(0), timeout=5.0):
    '''
    Gets a transform from TF.

    Calls wait_for_transform before getting the tranfsform.

    **Args:**

        **frame_tgt (string):** Name of frame to transform to.
    
        **frame_src (string):** Name of frame to transform from.
        
        *t (rospy.Time):* Time for which the transform is desired.
        
        *timeout (float):* How long to wait for transform (in seconds) before giving up.

    **Returns:**
         Translation as (x, y, z) and orientation as a quaternion (x, y, z, w)

    **Raises:**

        All errors tf.Transformer.lookupTransform can raise
    '''
    transform_listener = get_transform_listener()    
    transform_listener.waitForTransform(frame_tgt, frame_src, t, rospy.Duration(timeout))
    return transform_listener.lookupTransform(frame_tgt, frame_src, t)

def get_transform(frame_src, frame_tgt, t=rospy.Time(0), timeout=5.0):
    '''
    Gets a transform as a homogenous transform matrix.

    Calls wait_for_transform before getting the tranfsform.

    **Args:**

        **frame_src (string):** Name of frame to transform from.

        **frame_tgt (string):** Name of frame to transform to.
    
        *t (rospy.Time):* Time for which the transform is desired.
        
        *timeout (float):* How long to wait for transform (in seconds) before giving up.

    **Returns:**
        trans, rot (np.array, np.array): translation, rotation as quaternion.

    **Raises:**

        All errors tf.Transformer.lookupTransform can raise
    '''


    trans, rot = get_transform_as_tr(frame_src, frame_tgt, t=t, timeout=timeout)
    return geom.rot_trans_to_matrix(rot, trans)

def transform_point(new_frame_id, old_frame_id, point):
    '''
    Uses TF to transform a point

    **Args:**
    
        **new_frame_id (string):** The frame ID the point should have after transformation

        **old_frame_id (string):** The current frame ID of the point

        **point (geometry_msgs.msg.Point):** Point to transform

    **Returns:**
        A geometry_msgs.msg.Point in new_frame_id
        
    **Raises:**
        
        All errors tf.Transformer.lookupTransform can raise
    '''
    point_stamped = geometry_msgs.msg.PointStamped()
    point_stamped.header.stamp = rospy.Time(0)
    point_stamped.header.frame_id = old_frame_id
    point_stamped.point = point
    new_point = transform_point_stamped(new_frame_id, point_stamped, use_most_recent=True)
    return new_point.point

def transform_quaternion(new_frame_id, old_frame_id, quat):
    '''
    Uses TF to transform a quaternion

    **Args:**
    
        **new_frame_id (string):** The frame ID the quaternion should have after transformation

        **old_frame_id (string):** The current frame ID of the quaternion

        **quat (geometry_msgs.msg.Quaternion):** Quaternion to transform

    **Returns:**
        A geometry_msgs.msg.Quaternion in new_frame_id
        
    **Raises:**
        
        All errors tf.Transformer.lookupTransform can raise
    '''

    quat_stamped = geometry_msgs.msg.QuaternionStamped()
    quat_stamped.header.stamp = rospy.Time(0)
    quat_stamped.header.frame_id = old_frame_id
    quat_stamped.quaternion = quat
    new_quat = transform_quaternion_stamped(new_frame_id, quat_stamped, use_most_recent=True)
    if not new_quat:
        return None
    return new_quat.quaternion

def transform_pose(new_frame_id, old_frame_id, pose):
    '''
    Uses TF to transform a pose

    **Args:**
    
        **new_frame_id (string):** The frame ID the pose should have after transformation

        **old_frame_id (string):** The current frame ID of the pose

        **pose (geometry_msgs.msg.Pose):** Pose to transform

    **Returns:**
        A geometry_msgs.msg.Pose in new_frame_id
        
    **Raises:**
        
        All errors tf.Transformer.lookupTransform can raise
    '''

    pose_stamped = geometry_msgs.msg.PoseStamped()
    pose_stamped.header.frame_id = old_frame_id
    pose_stamped.header.stamp = rospy.Time(0)
    pose_stamped.pose = pose
    new_pose = transform_pose_stamped(new_frame_id, pose_stamped, use_most_recent=True)
    return new_pose.pose

def transform_point_stamped(new_frame_id, point_stamped, use_most_recent=True):
    '''
    Uses TF to transform a point stamped

    **Args:**
    
        **new_frame_id (string):** The frame ID the point should have after transformation

        **point_stamped (geometry_msgs.msg.PointStamped):** Point to transform

        *use_most_recent (boolean):* If true, uses the most recent transform rather than the one at the time
        specified in the stamp.

    **Returns:**
        A geometry_msgs.msg.PointStamped in new_frame_id
        
    **Raises:**
        
        All errors tf.Transformer.lookupTransform can raise
    '''

    tf_listener = get_transform_listener()
    if use_most_recent: 
        point_stamped.header.stamp = rospy.Time(0)
    tf_listener.waitForTransform(point_stamped.header.frame_id, new_frame_id, point_stamped.header.stamp,
                                 rospy.Duration(5.0))
    return tf_listener.transformPoint(new_frame_id, point_stamped)

def transform_quaternion_stamped(new_frame_id, quat_stamped, use_most_recent=True):
    '''
    Uses TF to transform a quaternion stamped

    **Args:**
    
        **new_frame_id (string):** The frame ID the quaternion should have after transformation

        **quat_stamped (geometry_msgs.msg.QuaternionStamped):** Quaternion to transform

        *use_most_recent (boolean):* If true, uses the most recent transform rather than the one at the time
        specified in the stamp.

    **Returns:**
        A geometry_msgs.msg.QuaternionStamped in new_frame_id
        
    **Raises:**
        
        All errors tf.Transformer.lookupTransform can raise
    '''

    tf_listener = get_transform_listener()
    if use_most_recent: 
        quat_stamped.header.stamp = rospy.Time(0)
    tf_listener.waitForTransform(quat_stamped.header.frame_id, new_frame_id, quat_stamped.header.stamp,
                                 rospy.Duration(5.0))
    return tf_listener.transformQuaternion(new_frame_id, quat_stamped)

def transform_pose_stamped(new_frame_id, pose_stamped, use_most_recent=True):
    '''
    Uses TF to transform a pose stamped

    **Args:**
    
        **new_frame_id (string):** The frame ID the pose should have after transformation

        **pose_stamped (geometry_msgs.msg.PoseStamped):** Pose to transform

        *use_most_recent (boolean):* If true, uses the most recent transform rather than the one at the time
        specified in the stamp.

    **Returns:**
        A geometry_msgs.msg.PoseStamped in new_frame_id
        
    **Raises:**
        
        All errors tf.Transformer.lookupTransform can raise
    '''

    tf_listener = get_transform_listener()
    if use_most_recent: 
        pose_stamped.header.stamp = rospy.Time(0)
    tf_listener.waitForTransform(pose_stamped.header.frame_id, new_frame_id, pose_stamped.header.stamp,
                                 rospy.Duration(5.0))
    return tf_listener.transformPose(new_frame_id, pose_stamped)

def transform_constraints(frame_id, constraints):
    '''
    Uses TF to transform a set of constraints

    **Args:**
    
        **frame_id (string):** The frame ID the point should have after transformation

        **constraints (arm_navigation_msgs.msg.Constraints):** Constraints to transform

        *use_most_recent (boolean):* If true, uses the most recent transform rather than the one at the time
        specified in the stamp.

    **Returns:**
        arm_navigation_msgs.msg.Constraints in new_frame_id
        
    **Raises:**
        
        All errors tf.Transformer.lookupTransform can raise
    '''
    transformed_constraints = copy.deepcopy(constraints)
    for c in transformed_constraints.position_constraints:
        c.position = transform_point(frame_id, c.header.frame_id, c.position)
        c.constraint_region_orientation = transform_quaternion\
            (frame_id, c.header.frame_id, c.constraint_region_orientation)
        c.header.frame_id = frame_id
    
    for oc in transformed_constraints.orientation_constraints:
        oc.orientation = transform_quaternion\
            (frame_id, oc.header.frame_id, oc.orientation)
        oc.header.frame_id = frame_id
    for vc in transformed_constraints.visibility_constraints:
        vc.target = transform_point_stamped(frame_id, vc.target)
    return transformed_constraints
