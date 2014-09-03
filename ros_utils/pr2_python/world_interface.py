#world_interface.py
'''
Defines the WorldInterface class for working with the collision map.
'''

__docformat__ = "restructuredtext en"

import roslib; roslib.load_manifest('pr2_python')
import rospy
from arm_navigation_msgs.msg import CollisionObject, AttachedCollisionObject, Shape
from arm_navigation_msgs.srv import GetPlanningScene, GetRobotState
from geometry_msgs.msg import Pose, Quaternion
from std_srvs.srv import Empty
from pr2_python.hand_description import HandDescription
import random

PUB_WAIT = 0.5
'''
The time to wait after publishing a collision object before returning.
'''

class WorldInterface:
    '''
    An interface to the collision map.

    This class pre-dominantly provides an interface to the collision map right now although we encourage you to
    expand it to use a database.  This class instantiates all publishers and services necessary for dealing with
    the collision map so that they are persistent throughout your code.  We recommend using this class for
    any interaction with the collision map as there are complications with using transient publishers in Python.

    **Attributes:**

        **world_frame (string):** The frame in which objects are added to the collision map and also the frame in 
        which planning is done.  Usually this is /map, but if not using a map it will be /odom_combined.  It is 
        found by looking at the parent of the multi DOF joint in the robot state.
        
        **robot_frame (string):** The root frame of the robot's link tree.  This has the same orientation as the 
        world frame but moves with the robot.  Usually it is /base_footprint.  It is found by looking at the child 
        of the multi DOF joint in the robot state.
        
        **get_planning_scene (rospy.ServiceProxy):** Returns the current state of the world.  For more information 
        about the planning scene, see planning_scene_interface.py
        
        **hands (dictionary):** Hand descriptions for the robot indexed by 'left_arm' and 'right_arm'.  For more
        information about hand descriptions, see hand_description.py
    '''
    def __init__(self):

        #set up service clients, publishers, and action clients        
        self._collision_object_pub =\
            rospy.Publisher('/collision_object', CollisionObject)
        self._attached_object_pub =\
            rospy.Publisher('attached_collision_object',\
                                AttachedCollisionObject)
        self._reset_collider = rospy.ServiceProxy('/collider_node/reset', Empty)
        rospy.loginfo('Waiting for collider node reset service')
        self._reset_collider.wait_for_service()

        self._robot_state = rospy.ServiceProxy('/environment_server/get_robot_state', GetRobotState)
        rospy.loginfo('Waiting for get robot state service')
        self._robot_state.wait_for_service()
        #find the root frame... this is the parent id of the multi_dof_joint in the
        #robot's world state
        self.world_frame = '/odom_combined'
        self.robot_frame = '/base_footprint'
        rs = self.get_robot_state()
        if rs.multi_dof_joint_state.frame_ids:
            self.world_frame = rs.multi_dof_joint_state.frame_ids[0]
            self.robot_frame = rs.multi_dof_joint_state.child_frame_ids[0]
        rospy.loginfo('Frame '+str(self.world_frame)+' is world frame and '+
                      str(self.robot_frame)+' is robot root frame')

        rospy.loginfo('Robot state client created.')


        self.get_planning_scene = rospy.ServiceProxy('/environment_server/get_planning_scene',
                                                     GetPlanningScene())
        rospy.loginfo('Waiting for get planning scene service')
        self.get_planning_scene.wait_for_service()
        self.hands = {}
        self.hands['left_arm'] = HandDescription('left_arm')
        self.hands['right_arm'] = HandDescription('right_arm')
        rospy.loginfo('World interface initialized')

    def _publish(self, msg, publisher):
        rospy.sleep(PUB_WAIT)
        publisher.publish(msg)
        #it takes a second for things to be propagated to
        #the planning scene
        rospy.sleep(PUB_WAIT)

    def reset_collider_node(self, repopulate=True):
        '''
        Reset the collider node.

        The collision map fed by the collider node (usually just the data from the robot's tilting laser) takes
        a long time to decide voxels are no longer occupied.  If you notice the collision map is becoming crowded
        with ghost voxels, you can use this function to reset it.

        **Args:**
        
            *repopulate (boolean):* If true, this function will wait 5 seconds after resetting the collider node for
            the tilting laser to finish a new pass and repopulate the node with new data.  It is recommended that
            you always set repopulate to true.
        '''
        self._reset_collider()
        if not repopulate:
            rospy.loginfo('Reset collider, did not wait for repopulate')
            return
        rospy.loginfo('Reset collider and waiting 5 seconds for repopulate')
        rospy.sleep(rospy.Duration(5.0))



    def _get_reset_object(self):
        reset_object = CollisionObject()
        reset_object.operation.operation = reset_object.operation.REMOVE
        reset_object.header.frame_id = 'base_link'
        reset_object.header.stamp = rospy.Time.now()
        reset_object.id = 'all'
        return reset_object
    

    def reset_collision_objects(self):
        '''
        Removes all collision objects (but not attached objects!) from the collision map.  

        If they were replacing occupied voxels, those voxels will still be noted as occupied from the collider node 
        data.
        '''
        #remove all objects
        self._publish(self._get_reset_object(), self._collision_object_pub)
        rospy.logdebug('Collision object reset')
        
    def reset_attached_objects(self):
        '''
        Removes all collision objects that are currently attached to the robot.
        '''
        #and all attached objects
        reset_attached_objects = AttachedCollisionObject()
        reset_attached_objects.link_name = 'all'
        reset_attached_objects.object.header.frame_id = 'base_link'
        reset_attached_objects.object.header.stamp = rospy.Time.now()
        reset_attached_objects.object = self._get_reset_object()
        self._publish(reset_attached_objects, self._attached_object_pub)
        rospy.logdebug('Attached object reset')
        
    def reset_all_objects(self):
        '''
        Removes all collision objects and all attached collision objects.
        '''
        self.reset_collision_objects()
        self.reset_attached_objects()


    def reset(self, repopulate=True):
        '''
        Removes all collision objects and all attached collision objects and resets the collider node.

        **Args:**
        
            *repopulate (boolean):* If True, waits for the collider node to accumulate new data before returning.  
            You should almost always set this to true.
        '''
        self.reset_all_objects()
        self.reset_collider_node(repopulate=repopulate)


    def add_object(self, co):
        '''
        Adds a collision object to the map.
        
        **Args:**

            **co (arm_navigation_msgs.msg.CollisionObject):** The object to be added to the map.
        '''
        co.operation.operation = co.operation.ADD
        self._publish(co, self._collision_object_pub)

    def add_collision_box(self, box_pose_stamped, box_dims, object_id):
        '''
        Adds a box to the map as a collision object.

        **Args:**

            **box_pose_stamped (geometry_msgs.msg.PoseStamped):** The pose of the box.

            **box_dims (tuple of 3 doubles):** The dimensions of the box as (x_dimension, y_dimension, z_dimension)

            **object_id (string):** The ID the box should have in the collision map
        '''
        box = CollisionObject()
        box.operation.operation = box.operation.ADD
        box.header = box_pose_stamped.header
        shape = Shape()
        shape.type = Shape.BOX
        shape.dimensions = box_dims
        box.shapes.append(shape)
        box.poses.append(box_pose_stamped.pose)
        box.id = object_id
        self._publish(box, self._collision_object_pub)
        return box

    def add_collision_cluster(self, cluster, object_id):
        '''
        Adds a point cloud to the collision map as a single collision object composed of many small boxes.

        **Args:**

            **cluster (sensor_msg.msg.PointCloud):** The point cloud to add to the map
            
            **object_id (string):** The name the point cloud should have in the map
        '''
        many_boxes = CollisionObject()
        many_boxes.operation.operation = many_boxes.operation.ADD
        many_boxes.header = cluster.header
        many_boxes.header.stamp = rospy.Time.now()
        num_to_use = int(len(cluster.points)/100.0)
        random_indices = range(len(cluster.points))
        random.shuffle(random_indices)
        random_indices = random_indices[0:num_to_use]
        for i in range(num_to_use):
            shape = Shape()
            shape.type = Shape.BOX
            shape.dimensions = [.005]*3
            pose = Pose()
            pose.position.x = cluster.points[random_indices[i]].x
            pose.position.y = cluster.points[random_indices[i]].y
            pose.position.z = cluster.points[random_indices[i]].z
            pose.orientation = Quaternion(*[0,0,0,1])
            many_boxes.shapes.append(shape)
            many_boxes.poses.append(pose)
        
        many_boxes.id = object_id
        self._publish(many_boxes, self._collision_object_pub)

    def remove_collision_object(self, object_id):
        '''
        Removes a collision object from the map.
        
        **Args:**
        
            **object_id (string):** The ID of the object to remove
        '''
        reset_object = CollisionObject()
        reset_object.operation.operation = reset_object.operation.REMOVE
        reset_object.header.frame_id = 'base_link'
        reset_object.header.stamp = rospy.Time.now()
        reset_object.id = object_id
        self._publish(reset_object, self._collision_object_pub)


    def attach_object_to_gripper(self, arm_name, object_id):
        '''
        Attaches an object to the robot's end effector.  

        This does NOT "snap" the object to the end effector.
        Rather, now when the robot moves, the object is assumed to remain stationary with respect to the robot's
        end effector instead of the world.  Collisions will be checked between the object and the world as the 
        object moves, but collisions between the object and the end effector will be ignored.  The link the object
        is attached to and the links with which collisions are ignored are defined by the hand description
        (see hand_description.py).

        **Args:**
        
            **arm_name (string):** The arm ('left_arm' or 'right_arm') to attach the object ot

            **object_id (string):** The ID of the object to attach
        '''
        obj = AttachedCollisionObject()
        obj.link_name = self.hands[arm_name].attach_link
        obj.object.operation.operation = obj.object.operation.ATTACH_AND_REMOVE_AS_OBJECT
        obj.object.header.stamp = rospy.Time.now()
        obj.object.header.frame_id = 'base_link'
        obj.object.id = object_id
        obj.touch_links = self.hands[arm_name].touch_links
        self._publish(obj, self._attached_object_pub)


    def detach_all_objects_from_gripper(self, arm_name):
        '''
        Detaches all objects and removes them from the collision space entirely.

        **Args:**

            **arm_name (string):** The arm ('left_arm' or 'right_arm') from which to detach all objects.
        '''
        obj = AttachedCollisionObject()
        obj.object.header.stamp = rospy.Time.now()
        obj.object.header.frame_id = 'base_link'
        obj.link_name = self.hands[arm_name]
        obj.object.id = 'all'
        obj.object.operation.operation = obj.object.operation.REMOVE
        self._publish(obj, self._attached_object_pub)
        

    def detach_object(self, arm_name, object_id):
        '''
        Detaches a single object from the arm and removes it from the collision space entirely.

        **Args:**

            **arm_name (string):** The arm ('left_arm' or 'right_arm') from which to detach the object

            **object_id (string):** The ID of the object to detach
        '''
        obj = AttachedCollisionObject()
        obj.object.header.stamp = rospy.Time.now()
        obj.object.header.frame_id = 'base_link'
        obj.link_name = self.hands[arm_name].attach_link
        obj.object.id = object_id
        obj.object.operation.operation = obj.object.operation.REMOVE
        self._publish(obj, self._attached_object_pub)


    def detach_and_add_back_attached_object(self, arm_name, object_id):
        '''
        Detaches a single object from the gripper and adds it back to the world at its current location.  

        From here on, it is assumed that the object remains stationary in the world.

        **Args:**

            **arm_name (string):** The arm ('right_arm' or 'left_arm') from which to detach the object

            **object_id (string):** The ID of the object to detach
        '''
        obj = AttachedCollisionObject()
        obj.object.header.stamp = rospy.Time.now()
        obj.object.header.frame_id = 'base_link'
        obj.link_name = self.hands[arm_name].attach_link
        obj.object.id = object_id
        obj.object.operation.operation = obj.object.operation.DETACH_AND_ADD_AS_OBJECT
        self._publish(obj, self._attached_object_pub)

    def collision_objects(self):
        '''
        Returns the collision objects in the world.

        **Returns:**
            The list of arm_navigation_msgs.msg.CollisionObject of collision objects (not attached collision 
            objects!) currently in the world.
        '''
        world = self.get_planning_scene()
        return world.planning_scene.collision_objects

    def attached_collision_objects(self):
        '''
        Returns the attached collision objects in the world.

        **Returns:**
            The list of arm_navigation_msgs.msg.AttachedCollisionObject of collision objects currently attached to
            the robot.
        '''
        world = self.get_planning_scene()
        return world.planning_scene.attached_collision_objects
    
    def collision_object(self, object_id):
        '''
        Returns a collision object by ID.

        **Args:**

            **object_id (string):** The ID of a collision object in the world
        
        **Returns:**
            The arm_navigation_msgs.msg.CollisionObject corresponding to object_id or None if no such object exists
        '''
        cos = self.collision_objects()
        for c in cos:
            if c.id == object_id:
                return c
        return None

    def attached_collision_object(self, object_id):
        '''
        Returns an attached collision object by ID.

        **Args:**

            **object_id (string):** The ID of a collision object attached to the robot

        **Returns:**
            The arm_navigation_msgs.msg.AttachedCollisionObject corresponding to object_id or None if no such object
            exists
        '''
        aos = self.attached_collision_objects()
        for ao in aos:
            if ao.object.id == object_id:
                return ao
        return None

    def get_robot_state(self):
        '''
        Returns the current robot state.

        **Returns:**
            The arm_navigation_msgs.msg.RobotState representing the current robot state.
        '''
        res = self._robot_state()
        return res.robot_state
