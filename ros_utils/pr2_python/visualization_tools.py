#visualization_tools.py
'''
Functions for creating visualization_msgs.
'''

__docformat__ = "restructuredtext en"

import roslib; roslib.load_manifest('pr2_python')
from visualization_msgs.msg import Marker, MarkerArray
from arm_navigation_msgs.msg import RobotState
import copy
from state_transformer.srv import GetRobotMarker, GetRobotMarkerRequest
import rospy
import pr2_python.trajectory_tools as tt

def marker_at(pose_stamped, ns='', mid=0, mtype=Marker.SPHERE, sx=0.05, sy=0.05, sz=0.05, r=0.0, g=0.0, b=1.0, 
              a=0.8):
    '''
    Returns a single marker at a pose.

    See the visualization_msgs.msg.Marker documentation for more details on any of these fields.

    **Args:**
    
        **pose_stamped (geometry_msgs.msg.PoseStamped):** Pose for marker

        *ns (string):* namespace

        *mid (int):* ID
        
        *mtype (int):* Shape type

        *sx (double):* Scale in x

        *sy (double):* Scale in y

        *sz (double):* scale in z

        *r (double):* Red (scale 0 to 1)

        *g (double):* Green (scale 0 to 1)

        *b (double):* Blue (scale 0 to 1)

        *a (double):* Alpha (scale 0 to 1)

    **Returns:**
        visualization_msgs.msg.Marker at pose_stamped
    '''
    marker = Marker()
    marker.header = copy.deepcopy(pose_stamped.header)
    marker.ns = ns
    marker.id = mid
    marker.type = mtype
    marker.action = marker.ADD
    marker.pose = copy.deepcopy(pose_stamped.pose)
    marker.scale.x = sx
    marker.scale.y = sy
    marker.scale.z = sz
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = a
    return marker

def marker_at_point(point_stamped, ns='', mid=0, mtype=Marker.SPHERE, sx=0.05, sy=0.05, sz=0.05, r=0.0, g=0.0, 
                    b=1.0, a=0.8):
    '''
    Returns a single marker at a point.

    Orientation is always (0, 0, 0, 1).  See the visualization_msgs.msg.Marker documentation for more details on 
    any of these fields.

    **Args:**
    
        **point_stamped (geometry_msgs.msg.PointStamped):** Point for marker

        *ns (string):* namespace

        *mid (int):* ID
        
        *mtype (int):* Shape type

        *sx (double):* Scale in x

        *sy (double):* Scale in y

        *sz (double):* scale in z

        *r (double):* Red (scale 0 to 1)

        *g (double):* Green (scale 0 to 1)

        *b (double):* Blue (scale 0 to 1)

        *a (double):* Alpha (scale 0 to 1)

    **Returns:**
        visualization_msgs.msg.Marker at point_stamped
    '''

    marker = Marker()
    marker.header = copy.deepcopy(point_stamped.header)
    marker.ns = ns
    marker.id = mid
    marker.type = mtype
    marker.action = marker.ADD
    marker.pose.position = copy.deepcopy(point_stamped.point)
    marker.pose.orientation.w = 1.0
    marker.scale.x = sx
    marker.scale.y = sy
    marker.scale.z = sz
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = a
    return marker

def robot_marker(robot_state, link_names=None, ns='', r = 0.0, g = 0.0, b = 1.0, a=0.8, scale=1.0):
    '''
    Returns markers representing the robot.

    To show only the arm or the gripper, use the link_names field with the arm and gripper links filled in.

    **Args:**
    
        **robot_state (arm_navigation_msgs.msg.RobotState):** State at which you want the marker

        *link_names ([string]):* Will return markers for only these links.  If left at None, will return markers
        for the whole robot.

        *ns (string):* Marker namespace.

        *r (double):* Red (scale 0 to 1)
        
        *g (double):* Green (scale 0 to 1)

        *b (double):* Blue (scale 0 to 1)

        *a (double):* Alpha (scale 0 to 1)

        *scale (double):* Scaling for entire robot
        
    **Returns:**
        visualization_msgs.msg.Marker array representing the links or the whole robot at robot_state
    '''
    marker_srv = rospy.ServiceProxy("/get_robot_marker", GetRobotMarker)
    req = GetRobotMarkerRequest()
    req.robot_state = robot_state
    req.link_names = link_names
    if not req.link_names:
        req.link_names = []
    req.ns = ns
    req.color.r = r
    req.color.g = g
    req.color.b = b
    req.color.a = a
    req.scale = scale
    
    res = marker_srv(req)
    return res.marker_array

def hsv_to_rgb(h, s, v):
    '''
    Converts from HSV to RGB

    **Args:**
    
        **h (double):** Hue

        **s (double):** Saturation

        **v (double):** Value

    **Returns:**
       The (r, g, b) values corresponding to this HSV color.
    '''
    hp = h/60.0
    c = v*s
    x = c*(1 - abs(hp % 2 - 1))
    if hp < 1:
        return (c, x, 0)
    if hp < 2:
        return (x, c, 0)
    if hp < 3:
        return (0, c, x)
    if hp < 4:
        return (0, x, c)
    if hp < 5:
        return (x, 0, c)
    return (c, 0, x)


def trajectory_markers(robot_traj, ns='', link_names=None, color=None, a=0.8, scale=1.0, resolution=1):
    '''
    Returns markers representing the robot trajectory.
    
    The color of each point on the trajectory runs from -60 to 300 in hue with full saturation and value.  Note that
    -60 is red so this should approximately follow the rainbow.
    
    **Args:**
    
        **robot_traj (arm_navigation_msgs.msg.RobotTrajectory):** Trajectory for which you want markers
        
        *ns (string):* Marker namespace.
        
        *a (double):* Alpha (scale 0 to 1)
        
        *scale (double):* Scaling for entire robot at each point
        
        *resolution (int):* Draws a point only every resolution points on the trajectory.  Will always draw
        the first and last points.
    
    **Returns:**                                                                                                                            
        visualization_msgs.msg.MarkerArray that will draw the whole trajectory.  Each point on the trajectory is               
        made up of many markers so this will be substantially longer than the length of the trajectory.
    '''

    if resolution <= 0:
        resolution = 1
    limit = len(robot_traj.joint_trajectory.points)
    if not limit:
        limit = len(robot_traj.multi_dof_joint_trajectory.points)
    marray = MarkerArray()
    for i in range(0, limit, resolution):
        robot_state = RobotState()
        if len(robot_traj.joint_trajectory.points):
            robot_state.joint_state = tt.joint_trajectory_point_to_joint_state(robot_traj.joint_trajectory.points[i],
                                                                               robot_traj.joint_trajectory.joint_names)
        if len(robot_traj.multi_dof_joint_trajectory.points):
            robot_state.multi_dof_joint_state = tt.multi_dof_trajectory_point_to_multi_dof_state(
                robot_traj.multi_dof_joint_trajectory.points[i], robot_traj.multi_dof_joint_trajectory.joint_names,
                robot_traj.multi_dof_joint_trajectory.frame_ids, robot_traj.multi_dof_joint_trajectory.child_frame_ids,
                robot_traj.multi_dof_joint_trajectory.stamp)
        if not color:
            (r, g, b) = hsv_to_rgb((i/float(limit)*360 - 60)%360, 1, 1)
        else:
            (r, g, b) = (color.r, color.g, color.b)
        marray.markers += robot_marker(robot_state, link_names=link_names, ns=ns, r=r, g=g, b=b, a=a, scale=scale).markers
    if i != limit-1:
        if not color:
            (r, g, b) = hsv_to_rgb(300, 1, 1)
        else:
            (r, g, b) = (color.r, color.g, color.b)
        marray.markers += robot_marker(tt.last_state_on_robot_trajectory(robot_traj), link_names=link_names, ns=ns, r=r, g=g, 
                                       b=b, a=a, scale=scale).markers
    for (i, m) in enumerate(marray.markers):
        m.id = i
    return marray
