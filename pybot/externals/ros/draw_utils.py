""" ROS RViz drawing utils """

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

import numpy as np

# ROS libs, msgs
import roslib, rospy, tf; 
import visualization_msgs.msg as vis_msg
import geometry_msgs.msg as geom_msg
import std_msgs.msg as std_msg
import sensor_msgs.msg as sensor_msg

# Asynchronous decorator
from copy import deepcopy

# Utility imports
from pybot.externals.ros.pointclouds import xyz_array_to_pointcloud2, xyzrgb_array_to_pointcloud2
from pybot.externals.draw_helpers import reshape_arr, get_color_arr, \
    height_map, color_by_height_axis, copy_pointcloud_data
from pybot.geometry.rigid_transform import RigidTransform

global viz_pub_
viz_pub_ = None

def init(): 
    """
    Init ros modules for viz
    """
    try: 
        # Initialize node
        import rospy
        rospy.init_node('pybot_vision_draw_utils_node', anonymous=True, disable_signals=True)

        global viz_pub_
        viz_pub_ = VisualizationMsgsPub()
        print 'Inited ROS node'
    except: 
        pass


class VisualizationMsgsPub: 
    """
    Visualization publisher class
    """
    # Init publisher
    marker_pub_ = rospy.Publisher('viz_msgs_marker_publisher', 
                                  vis_msg.Marker, latch=True, queue_size=10)
    pose_pub_ = rospy.Publisher('viz_msgs_pose_publisher', 
                                geom_msg.PoseArray, latch=False, queue_size=10)
    geom_pose_pub_ = rospy.Publisher('viz_msgs_geom_pose_publisher', 
                                     geom_msg.PoseStamped, latch=False, queue_size=10)
    pc_pub_ = rospy.Publisher('viz_msgs_pc_publisher', 
                              sensor_msg.PointCloud2, latch=False, queue_size=10)
    octomap_pub_ = rospy.Publisher('octomap_publisher', 
                                   vis_msg.Marker, latch=True, queue_size=10)
    tf_pub_ = tf.TransformBroadcaster()

    def __init__(self): 
        self.pc_map = {}

    def pc_map_pub(self, ns): 
        if ns not in self.pc_map: 
            self.pc_map[ns] = rospy.Publisher(ns, 
                                              sensor_msg.PointCloud2, latch=False, queue_size=10)
        return self.pc_map[ns]

# Helper functions
def _publish_tf(*args): 
    global viz_pub_
    viz_pub_.tf_pub_.sendTransform(*args)

def _publish_marker(marker): 
    global viz_pub_
    viz_pub_.marker_pub_.publish(marker)

def _publish_poses(pose_arr): 
    global viz_pub_
    viz_pub_.pose_pub_.publish(pose_arr)

def _publish_pose(pose_arr): 
    global viz_pub_
    viz_pub_.geom_pose_pub_.publish(pose_arr)

def _publish_pc(pub_ns, pc): 
    global viz_pub_
    viz_pub_.pc_map_pub(pub_ns).publish(pc)

def _publish_octomap(marker): 
    global viz_pub_
    viz_pub_.octomap_pub_.publish(octomap)

def geom_pose_from_rt(rt): 
    msg = geom_msg.Pose()
    msg.position = geom_msg.Point(x=rt.tvec[0], y=rt.tvec[1], z=rt.tvec[2])
    xyzw = rt.quat.to_xyzw()
    msg.orientation = geom_msg.Quaternion(x=xyzw[0], y=xyzw[1], z=xyzw[2], w=xyzw[3])
    return msg

def rt_from_geom_pose(msg): 
    xyzw = msg.orientation
    xyz = msg.position
    return RigidTransform(tvec=np.array([xyz.x,xyz.y,xyz.z]), xyzw=[xyzw.x, xyzw.y, xyzw.z, xyzw.w])

def publish_tf(pose, stamp=None, frame_id='/camera', child_frame_id='map'): 
    x,y,z,w = pose.quat.to_xyzw()
    _publish_tf(tuple(pose.tvec), (x,y,z,w), rospy.Time.now(), frame_id, child_frame_id)

@run_async
def publish_cloud(pub_ns, _arr, c='b', stamp=None, flip_rb=False, frame_id='map', seq=None): 
    """
    Publish point cloud on:
    pub_ns: Namespace on which the cloud will be published
    arr: numpy array (N x 3) for point cloud data
    c: Option 1: 'c', 'b', 'r' etc colors accepted by matplotlibs color
       Option 2: float ranging from 0 to 1 via matplotlib's jet colormap
       Option 3: numpy array (N x 3) with r,g,b vals ranging from 0 to 1
    s: supported only by matplotlib plotting
    alpha: supported only by matplotlib plotting
    """
    arr, carr = copy_pointcloud_data(_arr, c, flip_rb=flip_rb)
    pc = xyzrgb_array_to_pointcloud2(arr, carr, stamp=stamp, frame_id=frame_id, seq=seq)
    _publish_pc(pub_ns, pc)

@run_async    
def publish_cloud_markers(pub_ns, _arr, c='b', stamp=None, flip_rb=False, frame_id='map'): 
    """
    Publish point cloud on:
    pub_ns: Namespace on which the cloud will be published
    arr: numpy array (N x 3) for point cloud data
    c: Option 1: 'c', 'b', 'r' etc colors accepted by matplotlibs color
       Option 2: float ranging from 0 to 1 via matplotlib's jet colormap
       Option 3: numpy array (N x 3) with r,g,b vals ranging from 0 to 1
    s: supported only by matplotlib plotting
    alpha: supported only by matplotlib plotting
    """
    arr, carr = copy_pointcloud_data(_arr, c, flip_rb=flip_rb)

    marker = vis_msg.Marker(type=vis_msg.Marker.SPHERE_LIST, ns=pub_ns, action=vis_msg.Marker.ADD)
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp if stamp is not None else rospy.Time.now()
    marker.scale.x = 0.01
    marker.scale.y = 0.01
   
    # XYZ
    inds, = np.where(~np.isnan(arr).any(axis=1))
    marker.points = [geom_msg.Point(arr[j,0], arr[j,1], arr[j,2]) for j in inds]
    
    # RGB (optionally alpha)
    N, D = arr.shape[:2]
    if D == 3: 
        marker.colors = [std_msg.ColorRGBA(carr[j,0], carr[j,1], carr[j,2], 1.0) 
                         for j in inds]
    elif D == 4: 
        marker.colors = [std_msg.ColorRGBA(carr[j,0], carr[j,1], carr[j,2], carr[j,3])
                         for j in inds]
         
    marker.lifetime = rospy.Duration()
    _publish_marker(marker)
    print 'Publishing marker', N


@run_async    
def publish_line_segments(pub_ns, _arr1, _arr2, c='b', stamp=None, frame_id='camera', flip_rb=False, size=0.002): 
    """
    Publish point cloud on:
    pub_ns: Namespace on which the cloud will be published
    arr: numpy array (N x 3) for point cloud data
    c: Option 1: 'c', 'b', 'r' etc colors accepted by matplotlibs color
       Option 2: float ranging from 0 to 1 via matplotlib's jet colormap
       Option 3: numpy array (N x 3) with r,g,b vals ranging from 0 to 1
    s: supported only by matplotlib plotting
    alpha: supported only by matplotlib plotting
    """
    arr1, carr = copy_pointcloud_data(_arr1, c, flip_rb=flip_rb)
    arr2 = (deepcopy(_arr2)).reshape(-1,3)

    marker = vis_msg.Marker(type=vis_msg.Marker.LINE_LIST, ns=pub_ns, action=vis_msg.Marker.ADD)
    if not arr1.shape == arr2.shape: raise AssertionError    

    marker.header.frame_id = frame_id
    marker.header.stamp = stamp if stamp is not None else rospy.Time.now()
    marker.scale.x = size
    marker.scale.y = size
    marker.color.b = 1.0
    marker.color.a = 1.0
    marker.pose.position = geom_msg.Point(0,0,0)
    marker.pose.orientation = geom_msg.Quaternion(0,0,0,1)

    # Handle 3D data: [ndarray or list of ndarrays]
    arr12 = np.hstack([arr1, arr2])
    inds, = np.where(~np.isnan(arr12).any(axis=1))
    
    marker.points = []
    for j in inds: 
        marker.points.extend([geom_msg.Point(arr1[j,0], arr1[j,1], arr1[j,2]), 
                              geom_msg.Point(arr2[j,0], arr2[j,1], arr2[j,2])])
                         
    # RGB (optionally alpha)
    marker.colors = [std_msg.ColorRGBA(carr[j,0], carr[j,1], carr[j,2], 1.0) 
                     for j in inds]
         
    marker.lifetime = rospy.Duration()
    _publish_marker(marker)

def publish_pose(pose, stamp=None, frame_id='camera'): 
    msg = geom_msg.PoseStamped();

    msg.header.frame_id = frame_id
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()    
    
    tvec = pose.tvec
    x,y,z,w = pose.quat.to_xyzw()
    msg.pose.position = geom_msg.Point(tvec[0],tvec[1],tvec[2])
    msg.pose.orientation = geom_msg.Quaternion(x,y,z,w)

    _publish_pose(msg)

@run_async
def publish_pose_list(pub_ns, _poses, texts=[], stamp=None, size=0.05, frame_id='camera', seq=1):
    """
    Publish Pose List on:
    pub_channel: Channel on which the cloud will be published
    """
    poses = deepcopy(_poses)
    if not len(poses): return 
    arrs = np.vstack([pose.matrix[:3,:3].T.reshape((1,9)) for pose in poses]) * size
    arrX = np.vstack([pose.tvec.reshape((1,3)) for pose in poses])
    arrx, arry, arrz = arrs[:,0:3], arrs[:,3:6], arrs[:,6:9]

    # Point width, and height
    N = len(poses)

    markers = vis_msg.Marker(type=vis_msg.Marker.LINE_LIST, ns=pub_ns, action=vis_msg.Marker.ADD)

    markers.header.frame_id = frame_id
    markers.header.stamp = stamp if stamp is not None else rospy.Time.now()
    markers.header.seq = seq
    markers.scale.x = size/20 # 0.01
    markers.scale.y = size/20 # 0.01
    
    markers.color.a = 1.0
    
    markers.pose.position = geom_msg.Point(0,0,0)
    markers.pose.orientation = geom_msg.Quaternion(0,0,0,1)
    
    markers.points = []
    markers.lifetime = rospy.Duration()        

    for j in range(N): 
        markers.points.extend([geom_msg.Point(arrX[j,0], arrX[j,1], arrX[j,2]), 
                                       geom_msg.Point(arrX[j,0] + arrx[j,0], 
                                                       arrX[j,1] + arrx[j,1], 
                                                       arrX[j,2] + arrx[j,2])])
        markers.colors.extend([std_msg.ColorRGBA(1.0, 0.0, 0.0, 1.0), std_msg.ColorRGBA(1.0, 0.0, 0.0, 1.0)])

        markers.points.extend([geom_msg.Point(arrX[j,0], arrX[j,1], arrX[j,2]), 
                               geom_msg.Point(arrX[j,0] + arry[j,0], 
                                               arrX[j,1] + arry[j,1], 
                                               arrX[j,2] + arry[j,2])])
        markers.colors.extend([std_msg.ColorRGBA(0.0, 1.0, 0.0, 1.0), std_msg.ColorRGBA(0.0, 1.0, 0.0, 1.0)])


        markers.points.extend([geom_msg.Point(arrX[j,0], arrX[j,1], arrX[j,2]), 
                               geom_msg.Point(arrX[j,0] + arrz[j,0], 
                                               arrX[j,1] + arrz[j,1], 
                                               arrX[j,2] + arrz[j,2])])
        markers.colors.extend([std_msg.ColorRGBA(0.0, 0.0, 1.0, 1.0), std_msg.ColorRGBA(0.0, 0.0, 1.0, 1.0)])

    _publish_marker(markers)

def publish_pose_list2(poses, stamp=None, frame_id='camera', child_frame_id='camera'):
    for idx, p in enumerate(poses): 
        publish_tf(p, stamp=None, frame_id='/camera-%i' % idx, child_frame_id=child_frame_id)

def publish_octomap(pub_ns, arr, carr, size=0.1, stamp=None, frame_id='map', flip_rb=False): 
    """
    Publish cubes list:
    """
    marker = vis_msg.Marker(type=vis_msg.Marker.CUBE_LIST, ns=pub_ns, action=vis_msg.Marker.ADD)

    marker.header.frame_id = frame_id
    marker.header.stamp = stamp if stamp is not None else rospy.Time.now()

    # Point width, and height
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
   
    N, D = arr.shape

    # XYZ
    inds, = np.where(~np.isnan(arr).any(axis=1))
    marker.points = [geom_msg.Point(arr[j,0], arr[j,1], arr[j,2]) for j in inds]
    
    # RGB (optionally alpha)
    rax, bax = 0, 2
    # carr = carr.astype(np.float32) * 1.0 / 255
    if flip_rb: rax, bax = 2, 0
    if D == 3: 
        marker.colors = [std_msg.ColorRGBA(carr[j,rax], carr[j,1], carr[j,bax], 1.0) 
                         for j in inds]
    elif D == 4: 
        marker.colors = [std_msg.ColorRGBA(carr[j,rax], carr[j,1], carr[j,bax], carr[j,3])
                         for j in inds]
         
    marker.lifetime = rospy.Duration()
    _publish_marker(marker)
         

def publish_height_map(ns, X, frame_id='map', height_axis=2): 
    carr = height_map(X[:,height_axis]) * 255
    publish_cloud(ns, X, carr, frame_id=frame_id)

def publish_voxels(ns, cells, carr, frame_id='map'): 
    publish_cloud(ns, cells, carr, frame_id=frame_id)


