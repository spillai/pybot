#hand_description.py
'''
Description of a robot end effector.
'''

__docformat__ = "restructuredtext en"

import roslib; roslib.load_manifest('pr2_python')
import rospy
import numpy as np

#change these as ros parameters rather than changing them here
DEFAULT_HAND_GROUP_SUFFIX = '_end_effector'
DEFAULT_ATTACH_LINK_SUFFIX = '_gripper_r_finger_tip_link'
DEFAULT_TOUCH_LINKS_SUFFIXES = ['_gripper_palm_link', 
                                '_gripper_r_finger_tip_link', 
                                '_gripper_l_finger_tip_link', 
                                '_gripper_l_finger_link', 
                                '_gripper_r_finger_link']

DEFAULT_FINGER_TIP_LINKS_SUFFIXES = ['_gripper_r_finger_tip_link', 
                                     '_gripper_l_finger_tip_link']
DEFAULT_FINGER_LINKS_SUFFIXES = ['_gripper_r_finger_tip_link', 
                                 '_gripper_l_finger_tip_link',
                                 '_gripper_r_finger_link',
                                 '_gripper_l_finger_link']
DEFAULT_HAND_FRAME_SUFFIX = '_wrist_roll_link'
DEFAULT_GRIPPER_JOINT_SUFFIX = '_gripper_joint'
DEFAULT_APPROACH_DIRECTION = [1.0, 0.0, 0.0]
DEFAULT_END_EFFECTOR_LENGTH = 0.18


class HandDescription:
    '''
    This class reads in the hand description file and stores useful information about a robot's arm and hand.
    
    A hand has fingertips, fingers, and a palm.

    **Attributes:**
    
        **arm_name (string):** The name of the arm to which this hand belongs

        **hand_group (string):** The name of the group describing all hand links

        **attach_link (string):** The link that remains stationary with respect to attached objects

        **hand_links ([string]):** The links that make up the hand

        **touch_links ([string]):** The links that touch the object when attached

        **finger_tip_links ([string]):** The names of the links corresponding to the fingertips

        **approach_direction ([x, y, z] normalized):** In the hand's frame, the long direction of the gripper.
  
        **hand_frame (string):** The name of the frame that controls the hand.  This is the frame for which planning
        is done.

        **gripper_joint (string):** The name of the joint controlling the gripper.

        **end_effector_length (double):** The length (in m) of the end effector from the hand_frame to the tip.
    '''
    def __init__(self, arm_name):
        self.arm_name = arm_name
        self.hand_group = rospy.get_param('/hand_description/'+arm_name+'/hand_group_name', 
                                          arm_name[0]+DEFAULT_HAND_GROUP_SUFFIX)
        self.attach_link = rospy.get_param('/hand_description/'+arm_name+'/attach_link',
                                           arm_name[0]+DEFAULT_ATTACH_LINK_SUFFIX)
        self.hand_links = rospy.get_param('/hand_description/'+arm_name+'/hand_links',
                                           [arm_name[0]+link for link in DEFAULT_TOUCH_LINKS_SUFFIXES])
        self.touch_links = rospy.get_param('/hand_description/'+arm_name+'/hand_touch_links',
                                           [arm_name[0]+link for link in DEFAULT_TOUCH_LINKS_SUFFIXES])
        self.finger_tip_links = rospy.get_param('/hand_description/'+arm_name+'/hand_fingertip_links',
                                                [arm_name[0]+link for link in DEFAULT_FINGER_TIP_LINKS_SUFFIXES])
        self.finger_links = rospy.get_param('/hand_description/'+arm_name+'/hand_finger_links',
                                            [arm_name[0]+link for link in DEFAULT_FINGER_LINKS_SUFFIXES])
        self.approach_direction = rospy.get_param('/hand_description/'+arm_name+'/hand_approach_direction',
                                                  DEFAULT_APPROACH_DIRECTION)
        #make sure this is normalized
        self.approach_direction = [d/np.sqrt(sum([v*v for v in self.approach_direction])) 
                                   for d in self.approach_direction]
        self.hand_frame = rospy.get_param('/hand_description/'+arm_name+'/hand_frame', 
                                          arm_name[0]+DEFAULT_HAND_FRAME_SUFFIX)
        self.gripper_joint = rospy.get_param('/hand_description/'+arm_name+'/gripper_joint', 
                                             arm_name[0]+DEFAULT_GRIPPER_JOINT_SUFFIX)
        self.end_effector_length = rospy.get_param('/hand_description/'+arm_name+'/end_effector_length',
                                                   DEFAULT_END_EFFECTOR_LENGTH)
