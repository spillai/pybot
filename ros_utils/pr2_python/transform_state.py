#transform_state.py
'''
Defines the TransformState class which can transform without using TF.
'''

__docformat__ = "restructuredtext en"

import roslib; roslib.load_manifest('pr2_python')
from state_transformer.srv import GetTransform, GetTransformRequest
from pr2_python.exceptions import TransformStateError
from geometry_msgs.msg import PointStamped, QuaternionStamped, PoseStamped, TransformStamped
import pr2_python.geometry_tools as gt
import pr2_python.conversions as conv
import rospy

import copy

def relative_frame(frame):
    '''
    Removes the leading slash
    '''
    if frame[0] == '/':
        return frame[1:]
    return frame


class TransformState:
    '''
    Transforms using a robot state rather than TF.

    When working with a robot, we have many different frames.  The TF tool transforms between these frames 
    according to the current state of the world.  However, when planning, we need to know the transform between
    frames given a robot state that may not match the state in the current world.  This class uses the 
    get_state_transforms service from the state_transformer package to compute the transforms according to an input
    robot state.
    '''
    def __init__(self):
        self._state_trans = rospy.ServiceProxy('/get_state_transforms', GetTransform)
        rospy.loginfo('Waiting for state transformation service')
        self._state_trans.wait_for_service()
        rospy.loginfo('Ready to transform states!')

    def get_transform(self, to_frame, from_frame, robot_state):
        '''
        Returns a transfrom from from_frame to to_frame according to robot_state.

        This function follows the conventions of TF meaning that the transform returned will take the origin of 
        from_frame and return the origin of to_frame in from_frame.  For example, calling 
        get_transform(r_wrist_roll_link, base_link) will return the current position of the PR2's right wrist in the 
        base_link frame.

        Note that this is the INVERSE of the transform you would use to transform a point from from_frame to 
        to_frame.  Because this gets complicated to keep straight, we have also provided a number of 
        transfrom_DATATYPE functions.

        **Args:**
            
            **from_frame (string):** Frame to transform from

            **to_frame (string):** Frame to transform to
            
            **robot_state (arm_navigation_msgs.msg.RobotState):** Robot state to use for transformations

        **Returns:**
            A geometry_msgs.msg.TransformStamped (the timestamp is left at zero as it is irrelevant)
        
        **Raises:**

            **exceptions.TransformStateError:** if there is an error getting the transform
        '''
        if relative_frame(to_frame) == relative_frame(from_frame):
            #don't do a service call for this
            ts = TransformStamped()
            ts.header.frame_id = from_frame
            ts.child_frame_id = to_frame
            ts.transform.rotation.w = 1.0
            return ts

        req = GetTransformRequest()
        req.from_frame_id = from_frame
        req.to_frame_id = to_frame
        req.robot_state = robot_state
        res = self._state_trans(req)
        if res.val != res.SUCCESS:
            raise TransformStateError('Unable to transform from '+from_frame+' to '+to_frame, error_code=res.val)
        return res.transform_stamped
    
    def transform_point(self, new_frame_id, old_frame_id, point, robot_state):
        '''
        Transforms a point defined in old_frame_id into new_frame_id according to robot state.
        
        **Args:**

            **new_frame_id (string):** new frame id
            
            **old_frame_id (string):** old frame id

            **point (geometry_msgs.msg.Point):** point defined in old_frame_id
            
            **robot_state (arm_navigation_msg.msg.RobotState):** Robot state to use for transformation
          
        **Returns:**
            A geometry_msgs.msg.Point defined in new_frame_id

        **Raises:**

            **exceptions.TransformStateError:** if there is an error getting the transform
        '''
        trans = self.get_transform(old_frame_id, new_frame_id, robot_state)
        return gt.transform_point(point, conv.transform_to_pose(trans.transform))

    def transform_quaternion(self, new_frame_id, old_frame_id, quat, robot_state):
        '''
        Transforms a quaternion defined in old_frame_id into new_frame_id according to robot state.
        
        **Args:**

            **new_frame_id (string):** new frame id
            
            **old_frame_id (string):** old frame id

            **quat (geometry_msgs.msg.Quaternion):** quaternion defined in old_frame_id

            **robot_state (arm_navigation_msg.msg.RobotState):** Robot state to use for transformation
          
        **Returns:**
            A geometry_msgs.msg.Quaternion defined in new_frame_id

        **Raises:**

            **exceptions.TransformStateError:** if there is an error getting the transform
        '''
        trans = self.get_transform(old_frame_id, new_frame_id, robot_state)
        return gt.transform_quaternion(quat, conv.transform_to_pose(trans.transform))

    def transform_pose(self, new_frame_id, old_frame_id, pose, robot_state):
        '''
        Transforms a pose defined in old_frame_id into new_frame_id according to robot state.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **old_frame_id (string):** old frame id
            
            **pose (geometry_msgs.msg.Pose):** pose defined in old_frame_id

            **robot_state (arm_navigation_msg.msg.RobotState):** Robot state to use for transformation
          
        **Returns:**
            A geometry_msgs.msg.Pose defined in new_frame_id

        **Raises:**

            **exceptions.TransformStateError:** if there is an error getting the transform
        '''
        trans = self.get_transform(old_frame_id, new_frame_id, robot_state)
        return gt.transform_pose(pose, conv.transform_to_pose(trans.transform))
    
    def transform_point_stamped(self, new_frame, point_stamped, robot_state):
        '''
        Transforms a point into new_frame_id according to robot state.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **point_stamped (geometry_msgs.msg.PointStamped):** point stamped (current frame is defined by header)

            **robot_state (arm_navigation_msgs.msg.RobotState):** Robot state used for transformations
          
        **Returns:**
            A geometry_msgs.msg.PointStamped defined in new_frame_id

        **Raises:**

            **exceptions.TransformStateError:** if there is an error getting the transform
        '''
        ps = PointStamped()
        ps.header.frame_id = new_frame
        ps.point = self.transform_point(new_frame, point_stamped.header.frame_id, point_stamped.point, robot_state)
        return ps

    def transform_quaternion_stamped(self, new_frame, quat_stamped, robot_state):
        '''
        Transforms a quaternion into new_frame_id according to robot state.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **quat_stamped (geometry_msgs.msg.QuaternionStamped):** quaternion stamped (current frame is defined by 
            header)
            
            **robot_state (arm_navigation_msgs.msg.RobotState):** Robot state used for transformations
          
        **Returns:**
            A geometry_msgs.msg.QuaternionStamped defined in new_frame_id

        **Raises:**

            **exceptions.TransformStateError:** if there is an error getting the transform
        '''
        qs = QuaternionStamped()
        qs.header.frame_id = new_frame
        qs.quaternion = self.transform_quaternion(new_frame, quat_stamped.header.frame_id, 
                                                  quat_stamped.quaternion, robot_state)
        return qs

    def transform_pose_stamped(self, new_frame, pose_stamped, robot_state):
        '''
        Transforms a point into new_frame_id according to robot state.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **pose_stamped (geometry_msgs.msg.PoseStamped):** pose stamped (current frame is defined by header)

            **robot_state (arm_navigation_msgs.msg.RobotState):** Robot state used for transformations
          
        **Returns:**
            A geometry_msgs.msg.PoseStamped defined in new_frame_id

        **Raises:**

            **exceptions.TransformStateError:** if there is an error getting the transform
        '''
        ps = PoseStamped()
        ps.header.frame_id = new_frame
        ps.pose = self.transform_pose(new_frame, pose_stamped.header.frame_id, pose_stamped.pose, robot_state)
        return ps

    def transform_constraints(self, frame_id, constraints, robot_state):
        '''
        Transforms constraints into new_frame_id according to robot state.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **constraints (arm_navigation_msgs.msg.Constraints):** constraints

            **robot_state (arm_navigation_msgs.msg.RobotState):** Robot state used for transformations
          
        **Returns:**
            An arm_navigation_msgs.msg.Constraints defined in new_frame_id

        **Raises:**

            **exceptions.TransformStateError:** if there is an error getting the transform
        '''
        transformed_constraints = copy.deepcopy(constraints)
        for c in transformed_constraints.position_constraints:
            c.position = self.transform_point(frame_id, c.header.frame_id, c.position, robot_state)
            c.constraint_region_orientation = self.transform_quaternion\
                (frame_id, c.header.frame_id, c.constraint_region_orientation, robot_state)
            c.header.frame_id = frame_id
        for oc in transformed_constraints.orientation_constraints:
            oc.orientation = self.transform_quaternion\
                (frame_id, oc.header.frame_id, oc.orientation, robot_state)
            oc.header.frame_id = frame_id
        for vc in transformed_constraints.visibility_constraints:
            vc.target = self.transform_point_stamped(frame_id, vc.target, robot_state)
        return transformed_constraints

        
    
                                 
