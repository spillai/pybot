#planning_scene_interface.py
'''
Defines the planning scene interface class.
'''

__docformat__ = "restructuredtext en"

import roslib; roslib.load_manifest('pr2_python')
import rospy
from arm_navigation_msgs.msg import OrderedCollisionOperations, AttachedCollisionObject
from arm_navigation_msgs.srv import SetPlanningSceneDiff, SetPlanningSceneDiffRequest, GetRobotState
from pr2_python.hand_description import HandDescription
from pr2_python.transform_state import TransformState

import threading
import copy

_planning_scene_interface = None
_planning_scene_interface_creation_lock = threading.Lock()

def get_planning_scene_interface():
    '''
    Use this function to get a planning scene interface rather than creating one.  

    The planning scene is global so you want a global interface as well.  Please note that the planning scene 
    interface is not currently thread safe.
    '''
    global _planning_scene_interface
    with _planning_scene_interface_creation_lock:
        #does this release the lock?
        if _planning_scene_interface == None:
            _planning_scene_interface = PlanningSceneInterface()
        return _planning_scene_interface
    

class PlanningSceneInterface:
    '''
    This class is a wrapper for the ROS planning scene.  

    The planning scene is used to allow the robot to plan
    in a different version of the world than the one that currently exists.  For example, if we wish to plan
    a series of arm movements, the second arm movement starts from the end of the first one rather than from the
    robot's current pose in the world.  More importantly, we may wish to plan a sequence of manipulations, which not
    only change the state of the robot but also the state of objects in the world.  All planners in ROS plan in
    the planning scene rather than in the current state of the world.

    However, the ROS planning scene is not set up for planning sequences of actions.  The ROS service call to 
    get_planning_scene returns the current state of the world, NOT the scene planners are currently using.  Likewise
    the service call to set_planning_scene_diff sets the planning scene from the current state of the world, not
    current planning scene.  This makes it very difficult to plan consecutive sequences of actions as it is
    incumbent on the user to maintain the diff from the current state of the world throughout the entire set
    of plans.
    
    Instead, this class maintains a running diff, allowing the user to continually update the state of the planning
    scene instead of remembering all changes from the current state of the world.  It is easy to reset the diff
    to reflect exactly the state of the world, but for every function call we have also provided a function call
    that can exactly undo it.  Note that although this class keeps track of your running changes, it is, in the 
    end, still sending a diff from the current world.  Therefore, if you make a change to the world (by publishing
    on the appropriate topic or using a WorldInterface) the planning scene will reflect that.

    When planning we must also be careful not to call TF as that reflects only the current state of the world.
    This inteface has a state transformer (see transform_state.py) that transforms frames according to the current
    robot state in the planning scene rather than in the world.  When planning you should always use the transform
    functions in the planning scene rather than calling TF.

    In many ways, this class looks like world_interface.py.  This was intentional to highlight that the planning
    scene operates just like the world will.  When making calls like add_object or attach_object_to_gripper to the
    planning scene during planning, you will make a lot of those same calls to the world interface during execution.

    The ROS planning scene is global; whenever it is set all planners are immediately updated to the new planning
    scene.  For this reason, we recommend that your planning scene interface also be global and that you use
    the get_planning_scene_interface call whenever you need an interface.  However, if you are very comfortable with
    the planning scene mechanism, you can maintain multiple interfaces corresponding to different sets of plans.

    For an example of using the planning scene interface, see arm_planner.py.

    Please note that this class is **NOT CURRENTLY THREADSAFE!**
    
    **Attributes:**

        **world_frame (string):** The frame in which objects are added to the collision map and also the frame in 
        which planning is done.  Usually this is /map, but if not using a map it will be /odom_combined.  It is 
        found by looking at the parent of the multi DOF joint in the robot state.

        **robot_frame (string):** The root frame of the robot's link tree.  This has the same orientation as the 
        world frame but moves with the robot.  Usually it is /base_footprint.  It is found by looking at the child 
        of the multi DOF joint in the robot state.

        **hands (dictionary):** Hand descriptions for the robot indexed by 'left_arm' and 'right_arm'.  For more
        information about hand descriptions, see hand_description.py

        **current_diff (arm_navigation_msgs.srv.SetPlanningSceneDiffRequest):** The current diff between the actual 
        state of the world and the planning scene.

        **set_scene (rospy.ServiceProxy):** The set_planning_scene_diff service
        
        **transformer (transform_state.TransformState):** The state transformer used for transforming between frames
        because TF reflects only the current state of the world.
    '''
    def __init__(self):
        self.current_diff = SetPlanningSceneDiffRequest()
        self.set_scene = rospy.ServiceProxy('/environment_server/set_planning_scene_diff', SetPlanningSceneDiff)
        rospy.loginfo('Waiting for planning scene service')
        self.set_scene.wait_for_service()
        self._get_world_robot = rospy.ServiceProxy('/environment_server/get_robot_state', GetRobotState)
        rospy.loginfo('Waiting for get robot state service')
        self._get_world_robot.wait_for_service()
        
        self.transformer = TransformState()

        arms = ['left_arm', 'right_arm'] #could somehow get these off the parameter server I guess
        self.hands = {}
        #this is unfortunately necessary for dealing with attached objects
        for arm in arms:
            self.hands[arm] = HandDescription(arm)


        self.send_diff()
        self.world_frame = '/odom_combined'
        self.robot_frame = '/base_footprint'
        rs = self.get_robot_state()

        if rs.multi_dof_joint_state.frame_ids:
            self.world_frame = rs.multi_dof_joint_state.frame_ids[0]
            self.robot_frame = rs.multi_dof_joint_state.child_frame_ids[0]
        rospy.loginfo('Frame '+str(self.world_frame)+' is world frame and '+
                      str(self.robot_frame)+' is robot root frame')
        rospy.loginfo('Planning scene interface created')


    def send_diff(self):
        '''
        Sends the current diff.  
        
        All functions call this to set the diff.  If you set the diff yourself, call this to
        send it along to the planning scene.  In general use, you should call the helper functions rather than
        calling this directly.

        **Returns:**
            An arm_navigation_msgs.srv.SetPlanningSceneDiffResponse that is the current planning scene
        '''
        return self.set_scene(self.current_diff)

    def current_planning_scene(self):
        '''
        Returns the current planning scene.

        **Returns:**
            An arm_navigation_msgs.mgs.PlanningScene that is the current planning scene
        '''
        return self.send_diff().planning_scene
    
    def reset(self):
        '''
        Resets the planning scene to the current state of the world.  

        This does NOT empty the planning scene; it is
        unfortunately not possible to do that.  Calling this will wipe out your current diff and any changes you
        have applied to the planning scene.
        '''
        self.current_diff = SetPlanningSceneDiffRequest()
        self.send_diff()
        rospy.sleep(0.2)
        return True

    def set_collisions(self, ordered_collisions):
        '''
        Sets the ordered collisions in the scene, removing any ordered collisions you passed previously.  

        If you pass in  None, this removes all ordered collisions.
        
        **Args:**
        
            **ordered_collisions (arm_navigation_msgs.msg.OrderedCollisionOperations):** ordered collisions
        '''
        if not ordered_collisions:
            self.current_diff.operations = OrderedCollisionOperations()
        else:
            self.current_diff.operations = ordered_collisions
        self.send_diff()
        return True

    def add_ordered_collisions(self, ordered_collisions):
        '''
        Adds ordered collisions on top of whatever ordered collisions are already in the diff.  

        To exactly counter this effect, call remove_ordered collisions.

        **Args:**

            **ordered_collisions (arm_navigation_msgs.msg.OrderedCollisionOperations):** ordered collisions
        '''
        if not ordered_collisions or\
                not ordered_collisions.collision_operations:
            return
        self.current_diff.operations.collision_operations +=\
            ordered_collisions.collision_operations
        self.send_diff()
        return True

    def add_object(self, co):
        '''
        Adds an object to the planning scene.  

        This can also be used to move an object in the diff by passing in a new pose and the same id.  If you 
        previously removed an object with this ID, this function will undo that removal.
        
        **Args:**

            **co (arm_navigation_msgs.msg.CollisionObject):** The object to add
        '''
        for o in self.current_diff.planning_scene_diff.collision_objects:
            if o.id == co.id:
                self.current_diff.planning_scene_diff.collision_objects.remove(o)
                break
        co.operation.operation = co.operation.ADD
        self.current_diff.planning_scene_diff.collision_objects.append(co)
        self.send_diff()
        return True

    def remove_object(self, co):
        '''
        Removes an object from the planning scene.  

        If you previously added the object using add_object, this 
        function will undo that.  The full collision object is needed because the object may not actually exist
        in the world but only in the planning scene.

        **Args:**

            **co (arm_navigation_msgs.msg.CollisionObject):** Obbject to remove

        **TODO:**
        
           * Is it possible to do this by ID only?

           * Does this cause warnings if the object was initially added by the planning scene and not the world?
        '''
        for o in self.current_diff.planning_scene_diff.collision_objects:
            if o.id == co.id:
                self.current_diff.planning_scene_diff.collision_objects.remove(o)
                break
        co.operation.operation = co.operation.REMOVE
        self.current_diff.planning_scene_diff.collision_objects.append(co)
        self.send_diff()


    def attach_object_to_gripper(self, arm_name, object_id):
        '''
        Attaches an object to the robot's end effector.  

        The object must already exist in the planning scene 
        (although it does not need to exist in the current state of the world).  This does NOT "snap" the object to 
        the end effector. Rather, now when the robot moves, the object is assumed to remain stationary with respect 
        to the robot's end effector instead of the world.  Collisions will be checked between the object and the 
        world as the object moves, but collisions between the object and the end effector will be ignored.  
        The link the object is attached to and the links with which collisions are ignored are defined by the hand 
        description (see hand_description.py).

        The object will be attached to the robot according to the current state of the robot and object in the 
        planning scene as maintained by this class (i.e. with any prior changes you made to the diff).

        To undo this function, you can use detach_object and add_object but you must remember the position of the
        object where it was originally attached!  If you call detach_and_add_back_attached_object, it will add
        the object back at its current location in the planning scene, NOT at the location at which it was 
        originally attached.

        The planning scene doesn't support ATTACH_AND_REMOVE so we do those two operations simultaneously, passing 
        an attached object with operation ADD and a collision object with operation REMOVE.  

        **Args:**
        
            **arm_name (string):** The arm ('left_arm' or 'right_arm') to attach the object to

            **object_id (string):** The ID of the object to attach

        **Returns:**
            False if the object did not previously exist in the world or planning scene diff and therefore cannot be 
            attached; True otherwise.
        '''
        #Note: It is important to send the full collision object (not just the id) in the attached collision object.
        #This was quite complicated to figure out the first time.
        obj = AttachedCollisionObject()
        obj.link_name = self.hands[arm_name].attach_link
        obj.touch_links = self.hands[arm_name].touch_links
        diff_obj = None
        for co in self.current_diff.planning_scene_diff.collision_objects:
            if co.id == object_id:
                rospy.loginfo('Object was added to or modified in planning scene.')
                diff_obj = co
                break
        if not diff_obj:
            rospy.loginfo('Object was not previously added to or modified in planning scene.')
            diff_obj = self.collision_object(object_id)
            if not diff_obj:
                rospy.logerr('Cannot attach object '+object_id+'.  This object does not exist')
                return False
            self.current_diff.planning_scene_diff.collision_objects.append(diff_obj)
        self.current_diff.planning_scene_diff.attached_collision_objects.append(obj)

        #convert it into the frame of the hand - it remains stationary with respect to this
        for p in range(len(diff_obj.poses)):
            diff_obj.poses[p] = self.transform_pose(obj.link_name, diff_obj.header.frame_id, diff_obj.poses[p])
        diff_obj.header.frame_id = obj.link_name
        diff_obj.operation.operation = diff_obj.operation.REMOVE
        obj.object = copy.deepcopy(diff_obj)
        obj.object.operation.operation = obj.object.operation.ADD
        self.send_diff()
        return True

    def detach_object(self, arm_name, object_id):
        '''
        Detaches a single object from the arm and removes it from the collision space entirely.  

        The object must have been attached in the world or planning scene diff previously.  This removes the object 
        from the planning scene even if it currently exists in the world.

        **Args:**

            **arm_name (string):** The arm ('left_arm' or 'right_arm') from which to detach the object
            
            **object_id (string):** The ID of the object to detach

        **Returns:**
            False if the object was not previously attached in the world or the diff and therefore cannot be 
            detached; True otherwise.
        '''
        obj = AttachedCollisionObject()
        obj.link_name = self.hands[arm_name].attach_link
        #did the object ever exist in the world?
        #if so, we need to continually remove it after detaching
        #but it will always be in 
        #corresponding collision object if it exists
        
        for co in self.current_diff.planning_scene_diff.collision_objects:
           if co.id == object_id:
               #if the planning scene originally added this object, we could
               #just remove it from the list of collision objects but we have
               #no way of knowing that at this point
               co.operation.operation = co.operation.REMOVE
               for p in range(len(co.poses)):
                   co.poses[p] = self.transform_pose(self.world_frame, co.header.frame_id, co.poses[p])
               co.header.frame_id = self.world_frame
               break
        for ao in self.current_diff.planning_scene_diff.attached_collision_objects:
            if ao.object.id == object_id:
                rospy.loginfo('Object was attached by the planning scene interface')
                self.current_diff.planning_scene_diff.attached_collision_objects.remove(ao)
                self.send_diff()
                return True
        rospy.loginfo('Object was not attached by the planning scene interface')
        aos = self.attached_collision_objects()
        for ao in aos:
            if ao.object.id == object_id:
                obj.object = ao.object
                obj.object.operation.operation = obj.object.operation.REMOVE
                self.current_diff.planning_scene_diff.attached_collision_objects.append(obj)
                self.send_diff()
                return True
            
        rospy.logwarn('Object '+object_id+' not attached to gripper')
        return False

    
    def detach_and_add_back_attached_object(self, arm_name, object_id):
        '''
        Detaches a single object from the gripper and adds it back to the planning scene at its current location in 
        the diff.  

        From here on, it is assumed that the object remains stationary with respect to the world.  The
        object must have been attached to the robot in the world or in the diff previously.

        As always, this function is done with respect to the running diff.  Therefore, if you have used 
        set_robot_state to change the robot state in the diff, this will respect that change and add the object back
        at the location corresponding to the robot's new state.

        **Args:**

            **arm_name (string):** The arm ('right_arm' or 'left_arm') from which to detach the object
            
            **object_id (string):** The ID of the object to detach

        **Returns:**
            False if the object was not attached to the robot previously; True otherwise.
        '''
        #could save some work by doing this all in one step :)
        ao = self.attached_collision_object(object_id)
        if not self.detach_object(arm_name, object_id): return False
        #find its current pose in the world frame
        for p in range(len(ao.object.poses)):
            ao.object.poses[p] = self.transform_pose(self.world_frame, ao.object.header.frame_id, ao.object.poses[p])
        ao.object.header.frame_id = self.world_frame
        return self.add_object(ao.object)


    def set_robot_state(self, robot_state):
        '''
        Sets the robot state in the diff.  

        This will also update the position of all attached objects as their
        positions are defined relative to the position of the robot.  When planning, you should call this function
        rather than trying to set starting states for the planner.  All planners will plan assuming the state in
        the diff is the starting state.

        **Args:**

            **robot_state (arm_navigation_msgs.msg.RobotState):** New robot state
        '''
        self.current_diff.planning_scene_diff.robot_state = robot_state
        self.send_diff()
        return True
    
    def remove_ordered_collisions(self, ordered_collisions):
        '''
        Removes the ordered collisions from the current diff.  

        This will ONLY remove ordered collisions by removing 
        them from the diff.  It can be used to exactly undo the effects of add_ordered_collisions, but cannot be
        used to change any collisions that were not set using add_ordered_collisions.  To enable or disable 
        collisions in the scene, use add_ordered_collisions.

        For example, if you used add_ordered_collisions to add a collision operation that removed all collisions with
        the right gripper, passing the same collision operation to this function will reset the collisions to exactly
        what they were before.  It will NOT enable the collisions of the gripper with everything.

        **Args:**
        
            **ordered_collisions (arm_navigation_msgs.msg.OrderedCollisionOperations):** ordered collisions to remove
        '''
        if not ordered_collisions or\
                not ordered_collisions.collision_operations:
            return
        newops = OrderedCollisionOperations()
        for o in self.current_diff.operations.collision_operations:
            doadd = True
            for a in ordered_collisions.collision_operations:
                if o.object1 == a.object1 and o.object2 == a.object2 and\
                        o.operation == a.operation:
                    doadd = False
                    break
            if doadd:
                newops.collision_operations.append(o)
        self.set_collisions(newops)


    def get_robot_state(self):
        '''
        Returns the current robot state in the planning scene including the current diff

        **Returns:**
            The robot state as an arm_navigation_msgs.msg.RobotState in the current diff of the planning scene.  
            This is the state that all planners are assuming is the starting state.  If you have not set the robot
            state in the diff, this is the current robot state in the world.
        '''
        #note: done this way because it's much faster than sending the whole planning scene back and forth
        if self.current_diff.planning_scene_diff.robot_state.joint_state.name:
            return self.current_diff.planning_scene_diff.robot_state
        state = self._get_world_robot()
        return state.robot_state

    def collision_objects(self):
        '''
        Returns the list of collision objects in the planning scene including the current diff

        **Returns:**
            A list of arm_navigation_msgs.msg.CollisionObject in the planning scene the planners are using.  This
            includes any changes you have made using add_object, remove_object, etc.
        '''
        scene = self.current_planning_scene()
        return scene.collision_objects
    
    def attached_collision_objects(self):
        '''
        Returns the list of attached collision objects in the planning scene including the current diff

        **Returns:**
            A list of arm_navigation_msgs.msg.AttachedCollisionObject in the planning scene the planners are using.  
            This includes any changes you have made using attach_object_to_gripper, detach_object, etc.
        '''
        scene = self.current_planning_scene()
        return scene.attached_collision_objects
    
    def collision_object(self, object_id):
        '''
        Returns the collision object with this ID

        **Args:**

            **object_id (string):** The ID of a collision object

        **Returns:**
            The arm_navigation_msgs.msg.CollisionObject corresponding to object_id or None if no such object exists.
            This will reflect any changes in the collision objects (position or existence) you have made.
        '''
        cos = self.collision_objects()
        for co in cos:
            if co.id == object_id:
                return co
        return None

    def attached_collision_object(self, object_id):
        '''
        Returns the attached collision object with this ID

        **Args:**

            **object_id (string):** The ID of an attached collision object

        **Returns:**
            The arm_navigation_msgs.msg.AttachedCollisionObject corresponding to object_id or None if no such object 
            exists.  This will reflect any changes in the attached collision objects (position or existence) you 
            have made.
        '''
        aos = self.attached_collision_objects()
        for ao in aos:
            if ao.object.id == object_id:
                return ao
        return None
        
    def get_transform(self, to_frame, from_frame):
        '''
        All of the transform functions transform according to the robot state in the diff (or the current state of
        the world if you have not set a state in the diff).  

        This is because while planning, you cannot use TF as it
        reflects only the current state of the world.  Instead, you want to know what the transform will be given
        the state we expect the robot to be in.  You should always use these functions for transformation rather
        than TF while planning!
        
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

        **Returns:**
            A geometry_msgs.msg.TransformStamped (the timestamp is left at zero as it is irrelevant)
        
        '''
        return self.transformer.get_transform(to_frame, from_frame, self.get_robot_state())
    
    def transform_point(self, new_frame_id, old_frame_id, point):
        '''
        Transforms a point defined in old_frame_id into new_frame_id according to the current state in the diff.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **old_frame_id (string):** old frame id

            **point (geometry_msgs.msg.Point):** point defined in old_frame_id
          
        **Returns:**
            A geometry_msgs.msg.Point defined in new_frame_id
        '''
        return self.transformer.transform_point(new_frame_id, old_frame_id, point, self.get_robot_state())

    def transform_quaternion(self, new_frame_id, old_frame_id, quat):
        '''
        Transforms a quaternion defined in old_frame_id into new_frame_id according to the current state in the diff.
        
        **Args:**
        
            **new_frame_id (string):** new frame id

            **old_frame_id (string):** old frame id

            **quat (geometry_msgs.msg.Quaternion):** quaternion defined in old_frame_id
          
        **Returns:**
            A geometry_msgs.msg.Quaternion defined in new_frame_id
        '''
        return self.transformer.transform_quaternion(new_frame_id, old_frame_id, quat, self.get_robot_state())

    def transform_pose(self, new_frame_id, old_frame_id, pose):
        '''
        Transforms a pose defined in old_frame_id into new_frame_id according to the current state in the diff.
        
        **Args:**
            
            **new_frame_id (string):** new frame id

            **old_frame_id (string):** old frame id

            **pose (geometry_msgs.msg.Pose):** point defined in old_frame_id
          
        **Returns:**
            A geometry_msgs.msg.Pose defined in new_frame_id
        '''
        return self.transformer.transform_pose(new_frame_id, old_frame_id, pose, self.get_robot_state())
    
    def transform_point_stamped(self, new_frame_id, point_stamped):
        '''
        Transforms a point into new_frame_id according to the current state in the diff.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **point_stamped (geometry_msgs.msg.PointStamped):** point stamped (current frame is defined by header)
          
        **Returns:**
            A geometry_msgs.msg.PointStamped defined in new_frame_id
        '''
        return self.transformer.transform_point_stamped(new_frame_id, point_stamped, self.get_robot_state())

    def transform_quaternion_stamped(self, new_frame_id, quat_stamped):
        '''
        Transforms a quaternion into new_frame_id according to the current state in the diff.
        
        **Args:**

            **new_frame_id (string):** new frame id
            
            **quat_stamped (geometry_msgs.msg.QuaternionStamped):** quaternion stamped (current frame is defined by 
            header)
          
        **Returns:**
            A geometry_msgs.msg.QuaternionStamped defined in new_frame_id
        '''
        return self.transformer.transform_quaternion_stamped(new_frame_id, quat_stamped, self.get_robot_state())

    def transform_pose_stamped(self, new_frame_id, pose_stamped):
        '''
        Transforms a pose into new_frame_id according to the current state in the diff.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **pose_stamped (geometry_msgs.msg.PoseStamped):** pose stamped (current frame is defined by header)
          
        **Returns:**
            A geometry_msgs.msg.PoseStamped defined in new_frame_id
        '''
        return self.transformer.transform_pose_stamped(new_frame_id, pose_stamped, self.get_robot_state())

    def transform_constraints(self, new_frame_id, constraints):
        '''
        Transforms constraints into new_frame_id according to the current state in the diff.
        
        **Args:**

            **new_frame_id (string):** new frame id

            **constraints (arm_navigation_msgs.msg.Constraints):** constraints to transform
          
        **Returns:**
            An arm_navigation_msgs.msg.Constraints defined in new_frame_id
        '''
        return self.transformer.transform_constraints(new_frame_id, constraints, self.get_robot_state())
