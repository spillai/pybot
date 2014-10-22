#!/usr/bin/env python
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import colorConverter
import time, logging

# LCM libs
import lcm, vs

# Asynchronous decorator
from copy import deepcopy

# Utility imports
from bot_vision.draw_utils import reshape_arr, get_color_arr, height_map, color_by_height_axis
from bot_utils.async_utils import run_async
from bot_geometry.rigid_transform import RigidTransform

# global viz_pub_
# viz_pub_ = None


# class VisualizationMsgsPub: 
#     """
#     Visualization publisher class
#     """
#     # # Init publisher
#     # marker_pub_ = rospy.Publisher('viz_msgs_marker_publisher', vis_msg.Marker, latch=False, queue_size=10)
#     # pose_pub_ = rospy.Publisher('viz_msgs_pose_publisher', geom_msg.PoseArray, latch=False, queue_size=10)
#     # geom_pose_pub_ = rospy.Publisher('viz_msgs_geom_pose_publisher', geom_msg.PoseStamped, latch=False, queue_size=10)
#     # pc_pub_ = rospy.Publisher('viz_msgs_pc_publisher', sensor_msg.PointCloud2, latch=False, queue_size=10)
#     # octomap_pub_ = rospy.Publisher('octomap_publisher', vis_msg.Marker, latch=True, queue_size=10)
#     # tf_pub_ = tf.TransformBroadcaster()

#     def __init__(self): 
#         self.pc_map = {}
#         self.lc = lcm.LCM()

#     def pc_map_pub(self, ns): 
#         if ns not in self.pc_map: 
#             self.pc_map[ns] = rospy.Publisher(ns, sensor_msg.PointCloud2, latch=False, queue_size=10)
#         return self.pc_map[ns]

# @run_async
# def publish_cloud(pub_ns, _arr, _carr, stamp=None, flip_rb=False, frame_id='map', seq=None): 
#     """
#     Publish point cloud on:
#     pub_ns: Namespace on which the cloud will be published
#     arr: numpy array (N x 3) for point cloud data
#     c: Option 1: 'c', 'b', 'r' etc colors accepted by matplotlibs color
#        Option 2: float ranging from 0 to 1 via matplotlib's jet colormap
#        Option 3: numpy array (N x 3) with r,g,b vals ranging from 0 to 1
#     s: supported only by matplotlib plotting
#     alpha: supported only by matplotlib plotting
#     """
#     arr, carr = deepcopy(_arr), deepcopy(_carr)
#     N, D = arr.shape
#     carr = get_color_arr(carr, N, flip_rb=flip_rb);

#     # pc = xyzrgb_array_to_pointcloud2(arr, carr, stamp=stamp, frame_id=frame_id, seq=seq)
#     # _publish_pc(pub_ns, pc)


# Globals ==============================================================
g_log = logging.getLogger(__name__)

# LCM pub
g_lc = lcm.LCM()
g_frames_pub, g_frames_id = dict(), dict()
# print 'KINECT FRAME to LOCAL\n', frame_pose # , frame_pose.to_homogeneous_matrix()


def get_frame_id(ch): 
    global g_frames_id
    if ch in g_frames_id: return g_frames_id[ch]
    g_frames_id[ch] = len(g_frames_id) + 12345
    return g_frames_id[ch]

def get_frame(frame): 
    global g_frames_pub
    frame_id = publish_sensor_frame(frame)
    return g_frames_pub[frame_id]

def reshape_arr(arr):
    # if 3 dimensional (i.e. organized pt cloud), reshape to Nx3
    if arr.ndim == 3:
        arr = np.hstack((np.reshape(arr[:,:,0], (-1, 1)), 
                         np.reshape(arr[:,:,1], (-1,1)), 
                         np.reshape(arr[:,:,2], (-1,1))))
    return arr    


def convert_image_to_carr(img):
    r,g,b = np.split(img.astype(float)*1.0/255.0, 3, axis=2)
    carr = np.hstack((np.reshape(r, (-1,1)), np.reshape(g ,(-1,1)), np.reshape(b, (-1,1))))
    return carr;

# ===== Color utils ====
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
    
# ===== Sensor frame drawing (viz) ====
def publish_sensor_frame(frame_ch):
    """ 
    Publish sensor frame in which the point clouds
    are drawn with reference to. sensor_frame_msg.id is hashed
    by its channel (may be collisions since its right shifted by 32)
    """

    global g_frames_pub
    frame_id = get_frame_id(frame_ch);
    if frame_id in g_frames_pub: return frame_id

    # Sensor frames msg
    msg = vs.obj_collection_t();
    msg.id = frame_id
    msg.name = frame_ch + '_BOTFRAME'
    msg.type = vs.obj_collection_t.AXIS3D;
    msg.reset = True;

    # Send sensor pose
    pose = vs.obj_t()

    # Get botframes sensor ref. frame
    # x, y, z, roll, pitch, yaw = 0, 0, 0, 0, 0, 0 # botframes_get_trans(frame_ch, 'local')
    
    x, y, z, roll, pitch, yaw = 0.15, 0.2, 1.48, -np.pi/2, 0, -np.pi/2
    # p = RigidTransform.from_roll_pitch_yaw_x_y_z(roll, pitch, yaw, 
    #                                          x, y, z, axes='sxyz')
    # publish_pose_list2(frame_ch+'_BOTFRAME', [p], texts=[frame_ch])

    pose.id, pose.x, pose.y, pose.z, \
        pose.roll, pose.pitch, pose.yaw  = 1, x, y, z, roll, pitch, yaw

    msg.objs = [pose];
    msg.nobjs = len(msg.objs);

    # Publish sensor frames as object collection
    g_lc.publish("OBJ_COLLECTION", msg.encode())

    g_frames_pub[frame_id] = RigidTransform.from_roll_pitch_yaw_x_y_z(roll, pitch, yaw, 
                                                                    x, y, z, axes='sxyz')

    # # Publish text
    # for frame_id, frame_pose in frames_pub.iteritems(): 
    #     frame_pose.tvec.reshape((-1,3))
    # # publish_text_list('BOT_FRAMES_TEXT', , texts, sensor_tf='KINECT', size=1, alpha=1):
    return frame_id

# ===== Point cloud drawing (matplotlib) ====
def draw_point_cloud(ax, arr, c='r', size=1):
    """ 
    Draw (N x 3) array in matplotlib axes ax
    """
    arr = reshape_arr(arr)
    ax.plot(arr[:,0], arr[:,1], arr[:,2],'.',markersize=size, c=c)

def arr_msg(arr, carr, frame_id): 
    # point3d collection msg
    msg = vs.point3d_list_t()
    msg.nnormals = 0;
    msg.normals = [];
    msg.npointids = 0;
    msg.pointids = [];
    msg.id = int(time.time() * 1e6);
    
    # comes from the sensor_frames_msg published earlier
    msg.collection = frame_id; 
    msg.element_id = 1;

    npoints = len(arr);
    msg.points = [vs.point3d_t() for j in range(0,npoints)];
    msg.npoints = len(msg.points);             
    inds = np.arange(0,npoints);

    for j in range(npoints):
        msg.points[j].x = arr[j,0]
        msg.points[j].y = arr[j,1]
        msg.points[j].z = arr[j,2];

    msg.colors = [vs.color_t() for j in range(0,npoints)];
    msg.ncolors = len(msg.colors);
    for j in range(npoints):
        msg.colors[j].r = carr[j,0]
        msg.colors[j].g = carr[j,1]
        msg.colors[j].b = carr[j,2];    


    return msg

# ===== Point cloud drawing (viz) ====
def publish_point_cloud(pub_channel, arr, c='r', point_type='POINT', flip_rb=False, sensor_tf='KINECT'):
    """
    Publish point cloud on:
    pub_channel: Channel on which the cloud will be published
    arr: numpy array (N x 3) for point cloud data
    c: Option 1: 'c', 'b', 'r' etc colors accepted by matplotlibs color
       Option 2: float ranging from 0 to 1 via matplotlib's jet colormap
       Option 3: numpy array (N x 3) with r,g,b vals ranging from 0 to 1
    s: supported only by matplotlib plotting
    alpha: supported only by matplotlib plotting
    """

    frame_id = publish_sensor_frame(sensor_tf)
    frame_pose = get_frame(sensor_tf)
    assert(frame_id is not None)

    # point3d list collection msg
    pc_list_msg = vs.point3d_list_collection_t();
    pc_list_msg.id = get_frame_id(pub_channel)
    pc_list_msg.name = pub_channel;
    pc_list_msg.type = getattr(vs.point3d_list_collection_t, point_type);
    pc_list_msg.reset = True;
    pc_list_msg.point_lists = []

    N, D = arr.shape[:2]
    carr = get_color_arr(c, N, flip_rb=flip_rb);
    print carr.shape, arr.shape


    # Create the point cloud msg
    pc_msg = arr_msg(arr, carr=carr, frame_id=frame_id)

    # add to point cloud list                
    pc_list_msg.point_lists.append(pc_msg)
    pc_list_msg.nlists = len(pc_list_msg.point_lists); 
    g_lc.publish("POINTS_COLLECTION", pc_list_msg.encode())
    # g_log.debug('Published %i points' % (tpoints))

# ===== Tangents drawing ====
def publish_line_segments(pub_channel, arr1, arr2, c='r', 
                          downsample=1, sensor_tf='KINECT'):
    """ 
    Publish point cloud tangents:
    note: draw line from p1 to p2
    pub_channel: Channel on which the cloud will be published
    arr1: numpy array (N x 3) for point cloud data (p1)
    arr2: numpy array (N x 3) for point cloud data (p2) aligned 
    c: Option 1: 'c', 'b', 'r' etc colors accepted by matplotlibs color
       Option 2: float ranging from 0 to 1 via matplotlib's jet colormap
       Option 3: numpy array (N x 3) with r,g,b vals ranging from 0 to 1
    s: size of tangent vector (assuming normalized tangent vector)
    alpha: supported only by matplotlib plotting
    """
    frame_id = publish_sensor_frame(sensor_tf)
    frame_pose = get_frame(sensor_tf)
    assert(frame_id is not None)

    # check
    if not arr1.shape == arr2.shape: raise AssertionError    

    # point3d list collection msg
    pc_list_msg = vs.point3d_list_collection_t();
    pc_list_msg.id = get_frame_id(pub_channel);
    pc_list_msg.name = pub_channel;
    pc_list_msg.type = vs.point3d_list_collection_t.LINES;
    pc_list_msg.reset = True;
    pc_list_msg.point_lists = []

    # Handle 3D data: [ndarray or list of ndarrays]
    arr1s, arr2s, carrs = [], [], []
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray): 
        # ensure arr is in (N x 3) format
        # carr in (N x 3) format
        arr1, arr2 = reshape_arr(arr1), reshape_arr(arr2)
        arr1s.append(arr1)
        arr2s.append(arr2)
    elif isinstance(arr1, list) and isinstance(arr2, list): 
        arr1s, arr2s = arr1, arr2
    else: 
        raise TypeError("publish_line_segments: Unknown pts3d array type")

    # Handle color: [string or ndarray]
    if isinstance(c,str):
        carrs = [c] * len(arr1s)
    elif isinstance(c, np.ndarray): 
        carrs.append(c)
    else: 
        raise TypeError("publish_line_segments: Unknown color array type")
        carrs = c

    # # Downsample array
    # if arr1.ndim == 3:     
    #     h,w,ch = arr1.shape;
    #     arr1 = arr1[np.ix_(np.arange(0,h,downsample), np.arange(0,w,downsample))]
    #     arr2 = arr2[np.ix_(np.arange(0,h,downsample), np.arange(0,w,downsample))]    

    tpoints = 0
    for arr1,arr2,c in zip(arr1s,arr2s,carrs): 
        # fill out points
        npoints = len(arr1)
        tpoints += npoints
    
        # Get the colors
        carr = get_color_arr(c, npoints);
        ch, cw = carr.shape
        carr = np.hstack([carr, carr]).reshape((-1,cw))

        # Interleaved arr1, and arr2
        arr = np.hstack([arr1, arr2]).reshape((-1,3))
        
        # Create the point cloud msg
        pc_msg = arr_msg(arr, carr=carr, frame_id=frame_id)

        pc_list_msg.point_lists.append(pc_msg)
    # pc_msg.normals = [vs.point3d_t() for j in range(0,npoints)];    
    # pc_msg.nnormals = len(pc_msg.normals);
    # for j in range(0,npoints):
    #     pc_msg.normals[j].x, pc_msg.normals[j].y, pc_msg.normals[j].z = tarr[j,0], tarr[j,1], tarr[j,2];

    # add to point cloud list    
    pc_list_msg.nlists = len(pc_list_msg.point_lists); 
    g_lc.publish("POINTS_COLLECTION", pc_list_msg.encode())
    # g_log.debug('Published %i normals' % (tpoints))


# # ===== Pose drawing (viz) ====
# def publish_pose_list(pub_channel, arr, sensor_tf=True, size=1, alpha=1, verbose=False):
#     """
#     Publish Pose List on:
#     pub_channel: Channel on which the cloud will be published
#     arr: numpy array (N x 7) for pose list data ([x y z qw qx qy qz])
#     c: Option 1: 'c', 'b', 'r' etc colors accepted by matplotlibs color
#        Option 2: float ranging from 0 to 1 via matplotlib's jet colormap
#        Option 3: numpy array (N x 3) with r,g,b vals ranging from 0 to 1
#     s: supported only by matplotlib plotting
#     alpha: supported only by matplotlib plotting
#     """
#     global frame_pose  
#     # pose list collection msg
#     pose_list_msg = vs.obj_collection_t();
#     pose_list_msg.id = hash(pub_channel) >> 32;
#     pose_list_msg.name = pub_channel;
#     pose_list_msg.type = vs.obj_collection_t.AXIS3D;
#     pose_list_msg.reset = True;

#     # # pose msg
#     # pose_msg = vs.obj_t()
#     # pose_msg.id = int(time.time() * 1e6);
#     # pose_msg.collection = hash('KINECT_FRAME') >> 32;  # comes from the sensor_frames_msg published earlier
#     # pose_msg.element_id = 1;

#     # ensure arr is in (N x 7) format
#     # carr in (N x 3) format
#     nposes = arr.shape[0]; 

#     # fill out points
#     pose_list_msg.objs = [vs.obj_t() for j in range(0,nposes)];
#     pose_list_msg.nobjs = nposes;             
#     inds = np.arange(0,nposes);

#     sensor_pose = tf.quaternion_from_euler(frame_pose.roll, frame_pose.pitch, frame_pose.yaw)
#     sensorT = tf.quaternion_matrix(sensor_pose)
#     sensorT[:3,3] = np.array([frame_pose.x, frame_pose.y, frame_pose.z])
#     # print sensorT

#     for j in range(nposes):
#         obsT = tf.quaternion_matrix(arr[j,-4:])
#         obsT[:3,3] = arr[j,:3]
#         if sensor_tf: 
#             obsTw = np.dot(sensorT, obsT)
#         else: 
#             obsTw = obsT

#         rpy = tf.euler_from_matrix(obsTw)

#         pose_list_msg.objs[j].id = j

#         pose_list_msg.objs[j].x = obsTw[0,3]
#         pose_list_msg.objs[j].y = obsTw[1,3]
#         pose_list_msg.objs[j].z = obsTw[2,3];

#         pose_list_msg.objs[j].roll = rpy[0]
#         pose_list_msg.objs[j].pitch = rpy[1]
#         pose_list_msg.objs[j].yaw = rpy[2]
        
#     g_lc.publish("OBJ_COLLECTION", pose_list_msg.encode())
#     if verbose: print 'Published %i poses' % (nposes)

#     carr = plt.cm.Spectral(np.arange(len(arr)))
#     publish_point_cloud(pub_channel+'_POINTS', arr[:,:3], c=carr[:,:3])


def publish_pose_list2(pub_channel, poses, 
                       texts=[],                 
                       pose_type='AXIS3D',
                       sensor_tf='KINECT'):
    """
    Publish Pose List on:
    pub_channel: Channel on which the cloud will be published
    """
    frame_id = publish_sensor_frame(sensor_tf)
    frame_pose = get_frame(sensor_tf)
    assert(frame_id is not None)

    # pose list collection msg
    pose_list_msg = vs.obj_collection_t();
    pose_list_msg.id = get_frame_id(pub_channel)
    pose_list_msg.name = pub_channel;
    pose_list_msg.type = getattr(vs.obj_collection_t, pose_type);
    pose_list_msg.reset = True;

    nposes = len(poses); 

    # fill out points
    pose_list_msg.objs = [vs.obj_t() for j in range(0,nposes)];
    pose_list_msg.nobjs = nposes;             
    inds = np.arange(0,nposes);

    arr = np.zeros((len(poses),3))
    for j,pose in enumerate(poses): 

        arr[j,0] = pose.tvec[0]
        arr[j,1] = pose.tvec[1]
        arr[j,2] = pose.tvec[2]

        p = frame_pose.oplus(RigidTransform(pose.quat, pose.tvec))
        roll, pitch, yaw, x, y, z = p.to_roll_pitch_yaw_x_y_z(axes='sxyz')

        pose_list_msg.objs[j].id = j

        pose_list_msg.objs[j].x = x
        pose_list_msg.objs[j].y = y
        pose_list_msg.objs[j].z = z

        pose_list_msg.objs[j].roll = roll
        pose_list_msg.objs[j].pitch = pitch
        pose_list_msg.objs[j].yaw = yaw
        
    g_lc.publish("OBJ_COLLECTION", pose_list_msg.encode())
    # g_log.debug('Published %i poses' % (nposes))

    carr = plt.cm.Spectral(np.arange(len(poses)))
    publish_point_cloud(pub_channel+'_POINTS', arr[:,:3], c='b', sensor_tf=sensor_tf)

    # Publish corresponding text
    if len(texts): 
        assert(len(arr) == len(texts))
        # publish_text_list(pub_channel+'-text', pub_channel, 
        #                   texts, sensor_tf=sensor_tf)
        publish_text_lcmgl(pub_channel+'-text', arr, texts, sensor_tf=sensor_tf)
        

# ===== Pose drawing (viz) ====
def publish_text_list(pub_channel, ref_channel, texts, 
                      sensor_tf='KINECT'):
    """
    Publish Text List on:
    pub_channel: Channel on which the cloud will be published
    """
    frame_id = publish_sensor_frame(sensor_tf)
    frame_pose = get_frame(sensor_tf)
    assert(frame_id is not None)

    # # pose list collection msg
    # pose_list_msg = vs.obj_collection_t();
    # pose_list_msg.id = get_frame_id(pub_channel+'-text')
    # pose_list_msg.name = pub_channel+'-text';
    # pose_list_msg.type = vs.obj_collection_t.AXIS3D;
    # pose_list_msg.reset = True;

    text_list_msg = vs.text_collection_t()
    text_list_msg.name = pub_channel;
    text_list_msg.id = frame_id
    text_list_msg.type = 1; # doesn't matter
    text_list_msg.reset = True;
    
    # nposes = len(arr); 

    # # fill out points
    # pose_list_msg.objs = [vs.obj_t() for j in range(0,nposes)];
    # pose_list_msg.nobjs = nposes;             
    
    nposes = len(texts)
    text_list_msg.texts = [vs.text_t() for j in range(0,nposes)]
    text_list_msg.n = nposes

    # sensorT = frame_pose.to_homogeneous_matrix()
    # print sensorT

    for j in range(nposes):
        # obsT = tf.quaternion_matrix(np.array([1,0,0,0]))
        # obsT[:3,3] = arr[j,:3]
        # obsTw = np.dot(sensorT, obsT)

        # rpy = tf.euler_from_matrix(obsTw)

        # tf_arr = frame_pose * arr[j,:3]

        # pose_list_msg.objs[j].id = j

        # pose_list_msg.objs[j].x = obsTw[0,3]
        # pose_list_msg.objs[j].y = obsTw[1,3]
        # pose_list_msg.objs[j].z = obsTw[2,3];

        # pose_list_msg.objs[j].roll = 0; # rpy[0]
        # pose_list_msg.objs[j].pitch = 0; # rpy[1]
        # pose_list_msg.objs[j].yaw = 0; # rpy[2]

        text_list_msg.texts[j].id = j
        text_list_msg.texts[j].collection_id = get_frame_id(ref_channel)
        text_list_msg.texts[j].object_id = j
        text_list_msg.texts[j].text = texts[j]
        
    # g_lc.publish("OBJ_COLLECTION", pose_list_msg.encode())
    g_lc.publish("TEXT_COLLECTION", text_list_msg.encode())
    # g_log.debug('Published %i poses' % (nposes))



    # carr = plt.cm.Spectral(np.arange(len(arr)))
    # publish_point_cloud(pub_channel+'_POINTS', arr[:,:3], c=carr[:,:3])


# LCMGL Calls ==============================================================
def publish_point_cloud_lcmgl(pub_channel, arr, c='r', size=1, alpha=1):
    """ 
    Publish point cloud array in lcmgl
    """
    arr = reshape_arr(arr)    
    g = lcmgl.lcmgl('posepairs', lc)
    g.glPointSize(size)
    g.glColor4f(c[0],c[1],c[2],alpha)
    g.glBegin(lcmgl.GL_POINTS)
    for j in range(0,len(arr)):
        g.glVertex3d(arr[j,0], arr[j,1], arr[j,2])
    g.glEnd()
    g.switch_buffer()

def publish_text_lcmgl(pub_channel, arr, texts, alpha=0.9, sensor_tf='KINECT'): 
    """ 
    Publish text with point cloud array in lcmgl
    """
    frame_id = publish_sensor_frame(sensor_tf)
    frame_pose = get_frame(sensor_tf)
    assert(frame_id is not None)

    # Setup lcmgl text
    g = lcmgl.lcmgl(pub_channel, lc)

    # w.r.t sensor frame
    arr = frame_pose * arr

    # Plot
    g.glColor4f(0.2,0.2,0.2,alpha)
    for idx, text in enumerate(texts):
        g.text(arr[idx,0],arr[idx,1],arr[idx,2], text)
    g.switch_buffer()


# Object Renderers ==============================================================
def draw_camera(pose): 
    
    depth = 0.25
    fov = np.pi * 60.0 / 180
    off = np.tan(fov / 2 * depth)

    p0, b0 = np.array([0,0,0]), np.array([0,0,depth])
    tl, tr, br, bl = b0 + np.array([-1, 1, 0]) * off, \
                     b0 + np.array([1, 1, 0]) * off, \
                     b0 + np.array([1, -1, 0]) * off, \
                     b0 + np.array([-1, -1, 0]) * off

    # Front Face
    faces = []
    faces.extend([tl, tr, br])
    faces.extend([br, tl, bl])

    # Walls 
    left, top, right, bottom = [bl, p0, tl], [tl, p0, tr], [tr, p0, br], [br, p0, bl]
    faces.extend(left) # left wall
    faces.extend(top) # top wall
    faces.extend(right) # right wall
    faces.extend(bottom) # bottom wall
    faces = pose * np.vstack(faces)

    # Face
    pts = []
    pts.extend([tl, tr, br, bl, tl])
    pts.extend([left, left[0]])
    pts.extend([top, top[0]])
    pts.extend([right, right[0]])
    pts.extend([bottom, bottom[0]])
    pts = pose * np.vstack(pts)
    
    return (faces, np.hstack([pts[:-1], pts[1:]]).reshape((-1,3)))

def draw_cameras(pub_channel, poses, c='y', texts=[], sensor_tf='KINECT'):

    cam_feats = [draw_camera(pose) for pose in poses]
    cam_faces = map(lambda x: x[0], cam_feats)
    cam_edges = map(lambda x: x[1], cam_feats)

    # Publish pose
    publish_pose_list2(pub_channel, poses, texts=texts, sensor_tf=sensor_tf)

    # Light faces
    publish_point_cloud(pub_channel+'-faces', cam_faces, point_type='TRIANGLES', c=c, 
                        sensor_tf=sensor_tf)

    # Darker yellow edge
    publish_point_cloud(pub_channel+'-edges', cam_edges, point_type='LINES', c='r', 
                        sensor_tf=sensor_tf)


    # # Publish corresponding text
    # if len(texts): 
    #     assert(len(poses) == len(texts))
    #     arr = np.vstack([pose.tvec for pose in poses])
    #     publish_text_lcmgl(pub_channel+'-text', arr, texts=texts, sensor_tf=sensor_tf)


