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
Defines the JointStateListener class for getting current joint states of the PR2.
'''

__docformat__ = "restructuredtext en"


import roslib; roslib.load_manifest('pr2_python')

import threading
import numpy as np
import rospy
from sensor_msgs.msg import JointState

class JointStateListener:
    '''
    Keeps track of robot joints states by listening to /joint_states.
    '''
    def __init__(self):
        self.lock = threading.Lock()
        self._joint_positions = {}
        self._joint_velocities = {}
        self._joint_efforts = {}
        self._last_joint_state_msg = None
        self._joint_state_sub = rospy.Subscriber('/joint_states', JointState, self._joint_state_cb)

    def _joint_state_cb(self, msg):
        with self.lock:
            self._last_joint_state_msg = msg
            for joint_i, joint_name in enumerate(msg.name):
                self._joint_positions[joint_name] = msg.position[joint_i]
                self._joint_velocities[joint_name] = msg.velocity[joint_i]
                self._joint_efforts[joint_name] = msg.effort[joint_i]

    def get_last_joint_state_msg(self):
        with self.lock:
            return self._last_joint_state_msg

    def get_joint_positions(self, joint_names):
        '''
        Get current joint positions.

        **Args:**

            **joint_names ([string]):** Names of joints for which we want positions.

        **Returns:**
            joint_positions (np.array): Last received position for each joint.

        **Raises:**

            **KeyError:** if no state yet received for a joint.
        '''
        with self.lock:
            return np.array([self._joint_positions[name] for name in joint_names])

    def get_joint_velocities(self, joint_names):
        '''
        Get current joint velocities.

        **Args:**

            **joint_names ([string]):** Names of joints for which we want velocities.

        **Returns:**

            **joint_positions (np.array):** Last received velocity for each joint.

        **Raises:**

            **KeyError:** if no state yet received for a joint.
        '''
        with self.lock:
            return np.array([self._joint_velocities[name] for name in joint_names])

    def get_joint_efforts(self, joint_names):
        '''
        Get current joint efforts.

        **Args:**

            **joint_names ([string]):** Names of joints for which we want efforts.

        **Returns:**
            joint_positions (np.array): Last received effort for each joint.

        **Raises:**

            **KeyError:** if no state yet received for a joint.
        '''
        with self.lock:
            return np.array([self._joint_efforts[name] for name in joint_names])

