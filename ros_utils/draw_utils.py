#!/usr/bin/env python
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import colorConverter

# Utility imports
from bot_geometry.rigid_transform import RigidTransform
from .pr2_python.pointclouds import xyz_array_to_pointcloud2, \
    xyzrgb_array_to_pointcloud2

# ROS libs
import roslib, rospy, rosbag, tf; 

# ROS Viz msgs
from visualization_msgs.msg import Marker, MarkerArray
import geometry_msgs.msg as geom_msgs

# ROS std msgs
from std_msgs.msg import ColorRGBA, Header
from sensor_msgs.msg import PointCloud2, PointField

# Octomap msgs
import octomap_msgs.msg as octo_msg

class VisualizationMsgsPub: 
    # Init publisher
    marker_pub_ = rospy.Publisher('viz_msgs_marker_publisher', Marker, latch=False)
    pose_pub_ = rospy.Publisher('viz_msgs_pose_publisher', geom_msgs.PoseArray, latch=False)
    geom_pose_pub_ = rospy.Publisher('viz_msgs_geom_pose_publisher', geom_msgs.PoseStamped, latch=False)
    pc_pub_ = rospy.Publisher('viz_msgs_pc_publisher', PointCloud2, latch=False)
    octomap_pub_ = rospy.Publisher('octomap_publisher', Marker, latch=True)
    tf_pub_ = tf.TransformBroadcaster()

    def __init__(self): 
        self.pc_map = {}

    def pc_map_pub(self, ns): 
        if ns not in self.pc_map: 
            self.pc_map[ns] = rospy.Publisher(ns, PointCloud2, latch=False)
        return self.pc_map[ns]

global viz_pub_
viz_pub_ = VisualizationMsgsPub()

def reshape_arr(arr):
    # if 3 dimensional (i.e. organized pt cloud), reshape to Nx3
    if arr.ndim == 3:
        return arr.reshape((-1,3))
    elif arr.ndim == 2: 
        assert(arr.shape[1] == 3)
        return arr
    else: 
        raise Exception('Invalid dimensions %s' % arr.shape)

def get_color_arr(c, n, color_func=plt.cm.gist_rainbow, 
                  color_by='value', palette_size=20, flip_rb=False):
    """ 
    Convert string c to carr array (N x 3) format
    """
    carr = None;

    if color_by == 'value': 
        if isinstance(c, str): # single color
            carr = np.tile(np.array(colorConverter.to_rgb(c)), [n,1])
        elif  isinstance(c, float):
            carr = np.tile(np.array(color_func(c)), [n,1])
        else:
            carr = reshape_arr(c.astype(float) * 1.0)

    elif color_by == 'label': 
        if c < 0: 
            carr = np.tile(np.array([0,0,0,0]), [n,1])
        else: 
            carr = np.tile(np.array(color_func( (c % palette_size) * 1. / palette_size)), [n,1])
    else: 
        raise Exception("unknown color_by argument")

    if flip_rb: 
        r, b = carr[:,0], carr[:,2]
        carr[:,0], carr[:,2] = b.copy(), r.copy()

    return carr        

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
    msg = geom_msgs.Pose()
    msg.position = geom_msgs.Point(x=rt.tvec[0], y=rt.tvec[1], z=rt.tvec[2])
    xyzw = rt.quat.to_xyzw()
    msg.orientation = geom_msgs.Quaternion(x=xyzw[0], y=xyzw[1], z=xyzw[2], w=xyzw[3])
    return msg

def rt_from_geom_pose(msg): 
    xyzw = msg.orientation
    xyz = msg.position
    return RigidTransform(tvec=np.array([xyz.x,xyz.y,xyz.z]), xyzw=[xyzw.x, xyzw.y, xyzw.z, xyzw.w])

def publish_tf(pose, stamp=None, frame_id='/camera', child_frame_id='map'): 
    x,y,z,w = pose.quat.to_xyzw()
    _publish_tf(tuple(pose.tvec), (x,y,z,w), rospy.Time.now(), frame_id, child_frame_id)

def publish_point_cloud2(pub_ns, arr, carr, 
                         stamp=None, flip_rb=False, frame_id='head_mount_kinect_rgb_optical_frame'): 
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
    pc = xyzrgb_array_to_pointcloud2(arr, carr, 
                                     stamp=stamp, 
                                     frame_id=frame_id)
    _publish_pc(pub_ns, pc)
    
def publish_point_cloud(pub_ns, arr, carr, stamp=None, flip_rb=False): 
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
    marker = Marker(type=Marker.POINTS, ns=pub_ns, action=Marker.ADD)

    marker.header.frame_id = 'sensor_link'
    marker.header.stamp = stamp if stamp is not None else rospy.Time.now()

    # Point width, and height
    marker.scale.x = 0.02
    marker.scale.y = 0.02
   
    N, D = arr.shape

    # XYZ
    inds, = np.where(~np.isnan(arr).any(axis=1))
    marker.points = [geom_msgs.Point(arr[j,0], arr[j,1], arr[j,2]) for j in inds]
    
    # RGB (optionally alpha)
    rax, bax = 0, 2
    carr = carr.astype(np.float32) * 1.0 / 255
    if flip_rb: rax, bax = 2, 0
    if D == 3: 
        marker.colors = [ColorRGBA(carr[j,rax], carr[j,1], carr[j,bax], 1.0) 
                         for j in inds]
    elif D == 4: 
        marker.colors = [ColorRGBA(carr[j,rax], carr[j,1], carr[j,bax], carr[j,3])
                         for j in inds]
         
    marker.lifetime = rospy.Duration()
    _publish_marker(marker)

def publish_line_segments(pub_ns, arr1, arr2, c='g'): 
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
    marker = Marker(type=Marker.LINE_LIST, ns=pub_ns, action=Marker.ADD)

    # check
    if not arr1.shape == arr2.shape: raise AssertionError    

    marker.header.frame_id = 'sensor_link'
    marker.header.stamp = stamp if stamp is not None else rospy.Time.now()

    # Point width, and height
    marker.scale.x = 0.005
    marker.scale.y = 0.001
    
    marker.color.b = 1.0
    marker.color.a = 1.0

    marker.pose.position = geom_msgs.Point(0,0,0)
    marker.pose.orientation = geom_msgs.Quaternion(0,0,0,1)
   
    N, D = arr1.shape
    carr = get_color_arr(c, N);

    # Handle 3D data: [ndarray or list of ndarrays]
    arr1, arr2 = reshape_arr(arr1), reshape_arr(arr2)
    arr12 = np.hstack([arr1, arr2])
    inds, = np.where(~np.isnan(arr12).any(axis=1))
    
    marker.points = []
    for j in inds: 
        marker.points.extend([geom_msgs.Point(arr1[j,0], arr1[j,1], arr1[j,2]), 
                              geom_msgs.Point(arr2[j,0], arr2[j,1], arr2[j,2])])
                         
    # RGB (optionally alpha)
    marker.colors = [ColorRGBA(carr[j,0], carr[j,1], carr[j,2], 1.0) 
                     for j in inds]
         
    marker.lifetime = rospy.Duration()
    _publish_marker(marker)

def publish_pose(pose, stamp=None, frame_id='camera'): 
    msg = geom_msgs.PoseStamped();

    msg.header.frame_id = frame_id
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()    
    
    tvec = pose.tvec
    x,y,z,w = pose.quat.to_xyzw()
    msg.pose.position = geom_msgs.Point(tvec[0],tvec[1],tvec[2])
    msg.pose.orientation = geom_msgs.Quaternion(x,y,z,w)

    _publish_pose(msg)

def publish_pose_list(pub_ns, poses, texts=[], stamp=None, size=0.05, frame_id='camera', seq=1):
    """
    Publish Pose List on:
    pub_channel: Channel on which the cloud will be published
    """
    if not len(poses): return 
    arrs = np.vstack([pose.to_homogeneous_matrix()[:3,:3].T.reshape((1,9)) for pose in poses]) * size
    arrX = np.vstack([pose.tvec.reshape((1,3)) for pose in poses])
    arrx, arry, arrz = arrs[:,0:3], arrs[:,3:6], arrs[:,6:9]

    # Point width, and height
    N = len(poses)

    markers = Marker(type=Marker.LINE_LIST, ns=pub_ns, action=Marker.ADD)

    markers.header.frame_id = frame_id
    markers.header.stamp = stamp if stamp is not None else rospy.Time.now()
    markers.header.seq = seq
    markers.scale.x = 0.01
    markers.scale.y = 0.01
    
    markers.color.a = 1.0
    
    markers.pose.position = geom_msgs.Point(0,0,0)
    markers.pose.orientation = geom_msgs.Quaternion(0,0,0,1)
    
    markers.points = []
    markers.lifetime = rospy.Duration()        

    for j in range(N): 
        markers.points.extend([geom_msgs.Point(arrX[j,0], arrX[j,1], arrX[j,2]), 
                                       geom_msgs.Point(arrX[j,0] + arrx[j,0], 
                                                       arrX[j,1] + arrx[j,1], 
                                                       arrX[j,2] + arrx[j,2])])
        markers.colors.extend([ColorRGBA(1.0, 0.0, 0.0, 1.0), ColorRGBA(1.0, 0.0, 0.0, 1.0)])

        markers.points.extend([geom_msgs.Point(arrX[j,0], arrX[j,1], arrX[j,2]), 
                               geom_msgs.Point(arrX[j,0] + arry[j,0], 
                                               arrX[j,1] + arry[j,1], 
                                               arrX[j,2] + arry[j,2])])
        markers.colors.extend([ColorRGBA(0.0, 1.0, 0.0, 1.0), ColorRGBA(0.0, 1.0, 0.0, 1.0)])


        markers.points.extend([geom_msgs.Point(arrX[j,0], arrX[j,1], arrX[j,2]), 
                               geom_msgs.Point(arrX[j,0] + arrz[j,0], 
                                               arrX[j,1] + arrz[j,1], 
                                               arrX[j,2] + arrz[j,2])])
        markers.colors.extend([ColorRGBA(0.0, 0.0, 1.0, 1.0), ColorRGBA(0.0, 0.0, 1.0, 1.0)])

    _publish_marker(markers)

def publish_pose_list2(poses, stamp=None, frame_id='camera', child_frame_id='camera'):
    for idx, p in enumerate(poses): 
        publish_tf(p, stamp=None, frame_id='/camera-%i' % idx, child_frame_id=child_frame_id)

def publish_octomap(pub_ns, arr, carr, size=0.1, stamp=None, frame_id='map', flip_rb=False): 
    """
    Publish cubes list:
    """
    marker = Marker(type=Marker.CUBE_LIST, ns=pub_ns, action=Marker.ADD)

    marker.header.frame_id = frame_id
    marker.header.stamp = stamp if stamp is not None else rospy.Time.now()

    # Point width, and height
    marker.scale.x = size
    marker.scale.y = size
    marker.scale.z = size
   
    N, D = arr.shape

    # XYZ
    inds, = np.where(~np.isnan(arr).any(axis=1))
    marker.points = [geom_msgs.Point(arr[j,0], arr[j,1], arr[j,2]) for j in inds]
    
    # RGB (optionally alpha)
    rax, bax = 0, 2
    # carr = carr.astype(np.float32) * 1.0 / 255
    if flip_rb: rax, bax = 2, 0
    if D == 3: 
        marker.colors = [ColorRGBA(carr[j,rax], carr[j,1], carr[j,bax], 1.0) 
                         for j in inds]
    elif D == 4: 
        marker.colors = [ColorRGBA(carr[j,rax], carr[j,1], carr[j,bax], carr[j,3])
                         for j in inds]
         
    marker.lifetime = rospy.Duration()
    _publish_marker(marker)
         

# ================================
# Helper functions for plotting

def height_map(hX, hmin=-0.20, hmax=5.0): 
    # print 'min: %s, max: %s' % (np.min(hX), np.max(hX))
    # hmin, hmax = min(hmin, np.min(hX)), max(hmax, np.max(hX))
    return np.array(plt.cm.hsv((hX-hmin)/(hmax-hmin)))[:,:3]

def color_by_height_axis(axis=2): 
    return height_map(X[:,axis]) * 255

def plot_cloud(ns, X, carr, frame_id='map'): 
    # st = time.time()
    publish_point_cloud2(ns, X, carr, 
                         stamp=None, frame_id=frame_id)
    # print 'Time taken to publish %s cloud %s' % (X.shape, time.time() - st)        

def plot_height_map(ns, X, frame_id='map', height_axis=2): 
    carr = height_map(X[:,height_axis]) * 255
    plot_cloud(ns, X, carr, frame_id=frame_id)


def plot_voxels(ns, cells, carr, frame_id='map'): 
    # st = time.time()
    publish_point_cloud2(ns, cells, carr, frame_id=frame_id)
    # print 'Time taken to publish octomap % s cloud %s'  % (cells.shape, time.time() - st)

