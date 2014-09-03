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
Defines the ControllerManagerClient for working with controller switching.
'''

__docformat__ = "restructuredtext en"

import copy, os.path
import yaml
import roslib

import rospy
from pr2_mechanism_msgs.srv import LoadController, LoadControllerRequest, UnloadController, UnloadControllerRequest
from pr2_mechanism_msgs.srv import ListControllers, ListControllersRequest, SwitchController, SwitchControllerRequest
from pr2_python.exceptions import ControllerManagerError

class ControllerManagerClient:
    '''
    Class for starting/stopping/loading/unloading PR2 controllers.
    '''

    def __init__(self):
        load_controller_service_uri = 'pr2_controller_manager/load_controller'
        unload_controller_service_uri = 'pr2_controller_manager/unload_controller'
        list_controllers_service_uri = 'pr2_controller_manager/list_controllers'
        switch_controller_service_uri = 'pr2_controller_manager/switch_controller'

        rospy.wait_for_service(load_controller_service_uri)
        rospy.wait_for_service(unload_controller_service_uri)
        rospy.wait_for_service(list_controllers_service_uri)
        rospy.wait_for_service(switch_controller_service_uri)
        self._load_controller_service = rospy.ServiceProxy(load_controller_service_uri, LoadController)
        self._unload_controller_service = rospy.ServiceProxy(unload_controller_service_uri, UnloadController)
        self._list_controllers_service = rospy.ServiceProxy(list_controllers_service_uri, ListControllers)
        self._switch_controller_service = rospy.ServiceProxy(switch_controller_service_uri, SwitchController)

        # load list of controllers we know about, the joint groups that they control, and the default
        # controller for each joint group
        try:
            # load from ros param if it is there
            controllers_config = rospy.get_param('known_controllers')
        except KeyError:
            # no param, just load from local yaml file
            config_filename = os.path.join(roslib.packages.get_pkg_dir('pr2_python'), 'config/known_controllers.yaml')
            controllers_config = yaml.load(open(config_filename).read())
        self._known_controllers = controllers_config['controllers']
        self._default_controllers = controllers_config['default_controllers']

        # dictionary whose keys are the names of the loaded controllers, and whose values are
        # their respective states
        self._controllers = {}

        self._update()

    def _update(self):
        '''
        Update list of controllers and their state.
        '''
        res = self._list_controllers_service()
        self._controllers = {}
        for controller_name, controller_state in zip(res.controllers, res.state):
            self._controllers[controller_name] = controller_state

    def list_controllers(self):
        '''
        Get list of all loaded controllers, and their state.

        **Returns:**
            controllers ([string]): Dictionary whose keys are the names of the loaded controllers,
            and whose values are their respective states.
        '''
        self._update()
        return copy.copy(self._controllers)
        
    def load_controller(self, controller_name):
        '''
        Load the given controller.

        **Args:**

            **controller_name (string):** Name of controller to load.

        **Raises:**
        
            **exceptions.ControllerManagerError:** if unable to load controller.
        '''
        self._update()
        if controller_name in self._controllers:
            # controller already loaded
            return True
        res = self._load_controller_service(controller_name)
        if not res.ok:
            raise ControllerManagerError('Unable to load controller %s' % controller_name)

    def unload_controller(self, controller_name):
        '''
        Unload the given controller.

        Succeeds quietly if controller not loaded.

        **Raises:**

            **exceptions.ControllerManagerError:** if unable to unload controller.
        '''
        self._update()
        if not controller_name in self._loaded_controllers:
            return

        res = self._unload_controller_service(controller_name)
        if not res.ok:
            raise ControllerManagerError('Unable to load controller %s' % controller_name)

    def switch_controllers(self, start_controllers=[], stop_controllers=[]):
        '''
        Start some controllers, and stop others.

        **Args:**
            
            *start_controllers ([string]):* Names of controllers to start.
            
            *stop_controllers ([string]):* Names of controllers to stop.

        **Raises:**

            **exceptions.ControllerManagerError:** if unable to start/stop any of the requested controllers.
        '''
        self._update()

        # load any controllers which aren't loaded yet
        for controller_name in start_controllers:
            if not controller_name in self._controllers:
                self.load_controller(controller_name)
        
        # don't need to start a controller if it's already running
        controllers_to_start = set()
        for controller_name in start_controllers:
            if not(controller_name in self._controllers and self._controllers[controller_name] == 'running'):
                controllers_to_start.add(controller_name)

        # don't need to stop a controller if it isn't runnning
        controllers_to_stop = set()
        for controller_name in stop_controllers:
           if controller_name in self._controllers and self._controllers[controller_name] == 'running':
               controllers_to_stop.add(controller_name)

        # stop any other controllers which would coflict with the ones we're starting
        for controller_a, controller_a_state in self._controllers.items():
            if controller_a_state == 'running':
                for controller_b in controllers_to_start:
                    if self.check_conflict(controller_a, controller_b):
                        controllers_to_stop.add(controller_a)

        # if any joint group would end up without a controller, start its default controller
        # (for instance if we're stopping two_arm_controller to start r_arm_controller, we
        #  also need to start the default  controller for the left arm)
        joint_groups_losing_controllers = set()
        for c in controllers_to_stop:
            joint_groups_losing_controllers.update(set(self._known_controllers[c]['groups']))

        joint_groups_gaining_controllers = set()
        for c in controllers_to_start:
            joint_groups_gaining_controllers.update(set(self._known_controllers[c]['groups']))

        for joint_group in (joint_groups_losing_controllers - joint_groups_gaining_controllers):
            controllers_to_start.append(self._default_controllers[joint_group])

        # request controllers to switch
        if len(controllers_to_start) > 0 or len(controllers_to_stop) > 0:
            switch_req = SwitchControllerRequest()
            switch_req.start_controllers = list(start_controllers)
            switch_req.stop_controllers = list(controllers_to_stop)
            switch_req.strictness = SwitchControllerRequest.STRICT
            res = self._switch_controller_service(switch_req)
            if not res.ok:
                raise ControllerManagerError("Switch controllers failed!")

    def check_conflict(self, controller_a, controller_b):
        '''
        Check whether the two controllers conflict.

        If either controller is not in its dictionary of known controllers, assumes that
        they do not conflict.

        **Args:**

            **controller_a (string):** Name of first controller.

            **controller_b (string):** Name of second controller.

        **Returns:**
            conflict (boolean): True if there is a known conflict betweent the two; False otherwise.
        '''
        try:
            conflict_groups = set.intersection(
                set(self._known_controllers[controller_a]['groups']),
                set(self._known_controllers[controller_b]['groups']))
            if len(conflict_groups) > 0:
                return True
        except KeyError:
            return False
        return False
