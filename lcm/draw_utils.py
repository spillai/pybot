#!/usr/bin/env python
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import colorConverter

# LCM libs
import lcm, vs

# Asynchronous decorator
from copy import deepcopy

# Utility imports
from bot_vision.draw_utils import reshape_arr, get_color_arr, height_map, color_by_height_axis
from bot_utils.async_utils import run_async
from bot_geometry.rigid_transform import RigidTransform
from .pointclouds import xyz_array_to_pointcloud2, xyzrgb_array_to_pointcloud2

global viz_pub_
viz_pub_ = None


class VisualizationMsgsPub: 
    """
    Visualization publisher class
    """
    # # Init publisher
    # marker_pub_ = rospy.Publisher('viz_msgs_marker_publisher', vis_msg.Marker, latch=False, queue_size=10)
    # pose_pub_ = rospy.Publisher('viz_msgs_pose_publisher', geom_msg.PoseArray, latch=False, queue_size=10)
    # geom_pose_pub_ = rospy.Publisher('viz_msgs_geom_pose_publisher', geom_msg.PoseStamped, latch=False, queue_size=10)
    # pc_pub_ = rospy.Publisher('viz_msgs_pc_publisher', sensor_msg.PointCloud2, latch=False, queue_size=10)
    # octomap_pub_ = rospy.Publisher('octomap_publisher', vis_msg.Marker, latch=True, queue_size=10)
    # tf_pub_ = tf.TransformBroadcaster()

    def __init__(self): 
        self.pc_map = {}
        self.lc = lcm.LCM()

    def pc_map_pub(self, ns): 
        if ns not in self.pc_map: 
            self.pc_map[ns] = rospy.Publisher(ns, sensor_msg.PointCloud2, latch=False, queue_size=10)
        return self.pc_map[ns]

@run_async
def publish_cloud(pub_ns, _arr, _carr, stamp=None, flip_rb=False, frame_id='map', seq=None): 
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
    arr, carr = deepcopy(_arr), deepcopy(_carr)
    N, D = arr.shape
    carr = get_color_arr(carr, N, flip_rb=flip_rb);

    # pc = xyzrgb_array_to_pointcloud2(arr, carr, stamp=stamp, frame_id=frame_id, seq=seq)
    # _publish_pc(pub_ns, pc)
