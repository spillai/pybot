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
Exception types for pr2_python
'''

__docformat__ = "restructuredtext en"


import roslib
roslib.load_manifest('pr2_python')

import arm_navigation_msgs.msg
import object_manipulation_msgs.msg

class ActionFailedError(Exception):
    """
    Raised when a ROS action fails for some reason.
    """
    pass

class AllYourBasePosesAreBelongToUs(Exception):
    """
    Raised when we can't find a base pose from which to manipulate an object
    """
    def __init__(self, pos):
        self.object_pos = pos
        


# build a mapping from arm navigation error codes to error names
arm_nav_error_dict = {}
for name in arm_navigation_msgs.msg.ArmNavigationErrorCodes.__dict__.keys():
    if not name[:1] == '_' and name != 'val':
        code = arm_navigation_msgs.msg.ArmNavigationErrorCodes.__dict__[name]
        arm_nav_error_dict[code] = name

manipulation_error_dict = {}
for name in object_manipulation_msgs.msg.ManipulationResult.__dict__.keys():
    if not name[:1] == '_' and name != 'value':
        code = object_manipulation_msgs.msg.ManipulationResult.__dict__[name]
        manipulation_error_dict[code] = name

class ArmNavError(Exception):
    '''
    Raised when arm planning or movement fails.
    
    **Attributes:**
        **msg (string):** User-readable description of what happened.
    
        **error_code (ArmNavigationErrorCodes):** Arm navigation error code message.
        
        **trajectory_error_codes ([ArmNavigationErrorCodes]):** List of ArmNavigationErrorCode numbers
        for each trajectory point.
            
        **trajectory (sensor_msgs.msg.JointTrajectory):** (Possibly partial) trajectory the planner found before 
        failing.
    '''

    def __init__(self, msg, error_code=None, trajectory_error_codes=None, trajectory=None):
        self.msg = msg
        self.error_code = error_code
        self.trajectory_error_codes = trajectory_error_codes
        self.trajectory = trajectory
        
    def __str__(self):
        if self.error_code == None:
            return self.msg
        else:
            try:
                error_msg = arm_nav_error_dict[self.error_code.val]
                return '%s: %d (%s)' % (self.msg, self.error_code.val, error_msg)
            except:
                return self.msg

    def __repr__(self):
        return self.__str__()
            
class ControllerManagerError(Exception):
    '''
    Raised when the controller manager is unable to change controllers in the way we want.
    
    **Attributes:**
    
        **msg (str):** User-readable description of what happened.
    '''

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class ManipulationError(Exception):
    '''
    Raised when there is an error with manipulation.
    
    **Attributes:**
    
         **msg (string):** User-readable description of what happened.
    
         **manipulation_error_code (object_manipulation_msgs.msg.ManipulationResult):** The manipulation error
         that occurred.
    
         **arm_navigation_error_code (arm_navigation_msgs.msg.ArmNavigationErrorCodes):** The arm navigation
         error that occurred.
    '''

    def __init__(self, msg, manipulation_error_code=None, arm_navigation_error_code=None, other_data=None):
        self.msg = msg
        self.manipulation_error_code = manipulation_error_code
        self.arm_navigation_error_code = arm_navigation_error_code
        self.other_data = other_data
        
    def __str__(self):
        ret = self.msg
        if self.manipulation_error_code == None:
            return ret
        try:
            ret = self.msg + ': ' + str(self.manipulation_error_code.value)+\
                '('+str(manipulation_error_dict[self.manipulation_error_code.value])+')'
        except: pass
        if self.arm_navigation_error_code:
            try:
                ret += ' with arm navigation error ' +str(self.arm_navigation_error_code.value)+'('+\
                    str(arm_nav_error_dict[self.arm_navigation_error_code.value])+')'
            except: pass
        return ret

class TransformStateError(Exception):
    '''
    Raised when there is an error transforming states not using TF.
    
    **Attributes:**
            
        **msg (string):** User-readable description of what happened

        **error_code (int):** The state_transformer.srv.GetTransformResult error code.
    '''

    def __init__(self, msg, error_code=None):
        self.msg = msg
        self.error_code = error_code

    def __str__(self):
        if self.error_code == None:
            return self.msg
        return self.msg + ': '+str(self.error_code)
