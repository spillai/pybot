""" LCM viewer drawing utils """

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT
# Dependencies: LCM, OpenCV, vs (Visualization renderer)
# botcore (libbot), and pybot.geometry (pybot_geometry)

import time
from itertools import izip
from copy import deepcopy
from collections import deque
import numpy as np

import cv2

from pybot.utils.timer import timeitmethod
from pybot.geometry.rigid_transform import RigidTransform
from pybot.externals import vs, serialize, publish, pose_t
from pybot.externals.draw_helpers import reshape_arr, get_color_arr, \
    height_map, color_by_height_axis, copy_pointcloud_data, Frustum

class VisualizationMsgsPub: 
    """
    Visualization publisher class
    """
    CAMERA_POSE = RigidTransform.from_rpyxyz(-np.pi/2, 0, -np.pi/2, 
                                             0, 0, 2, axes='sxyz')
    XZ_POSE = RigidTransform.from_rpyxyz(np.pi/2, 0, 0, 0, 0, 0, axes='sxyz')
    def __init__(self): 
        self._channel_uid = dict()
        self._sensor_pose = dict()

        self.reset_visualization()
        self.publish_sensor_frame('camera', pose=VisualizationMsgsPub.CAMERA_POSE)
        self.publish_sensor_frame('origin', pose=RigidTransform.identity())
        # self.publish_sensor_frame('origin_xz', pose=VisualizationMsgsPub.XZ_POSE)

    def publish(self, channel, data):
        publish(channel, data)
        
    def list_frames(self):
        return self._sensor_pose
        
    def channel_uid(self, channel): 
        uid = self._channel_uid.setdefault(channel, len(self._channel_uid))
        return uid + 1000

    # def sensor_uid(self, channel): 
    #     uid = self._sensor_uid.setdefault(channel, len(self._sensor_uid))
    #     return uid
    def has_sensor_pose(self, channel): 
        return channel in self._sensor_pose

    def get_sensor_pose(self, channel): 
        return self._sensor_pose[channel].copy()

    def set_sensor_pose(self, channel, pose): 
        self._sensor_pose[channel] = pose

    def reset_visualization(self): 
        print('{} :: Reseting Visualizations'.format(self.__class__.__name__))
        msg = vs.reset_collections_t()
        self.publish("RESET_COLLECTIONS", serialize(msg))

    def publish_sensor_frame(self, channel, pose=None): 
        """ 
        Publish sensor frame in which the point clouds
        are drawn with reference to. sensor_frame_msg.id is hashed
        by its channel (may be collisions since its right shifted by 32)
        """
        # Sensor frames msg
        msg = vs.obj_collection_t()
        msg.id = self.channel_uid(channel)
        msg.name = 'BOTFRAME_' + channel
        msg.type = vs.obj_collection_t.AXIS3D
        msg.reset = True
        
        # Send sensor pose
        pose_msg = vs.obj_t()
        roll, pitch, yaw, x, y, z = pose.to_rpyxyz(axes='sxyz')
        pose_msg.id = 0
        pose_msg.x, pose_msg.y, pose_msg.z, \
            pose_msg.roll, pose_msg.pitch, pose_msg.yaw = x, y, z, roll, pitch, yaw
        
        # Save pose
        self.set_sensor_pose(channel, pose)

        # msg.objs = [pose_msg]
        msg.objs.extend([pose_msg])
        msg.nobjs = len(msg.objs)
        self.publish("OBJ_COLLECTION", serialize(msg))

global g_viz_pub
g_viz_pub = VisualizationMsgsPub()

def reset():
    global g_viz_pub
    g_viz_pub = VisualizationMsgsPub()

def reset_visualization():
    global g_viz_pub
    g_viz_pub.reset_visualization()

def get_sensor_pose(frame_id='camera'): 
    global g_viz_pub
    return g_viz_pub.get_sensor_pose(frame_id)

def has_sensor_frame(frame_id): 
    global g_viz_pub
    return g_viz_pub.has_sensor_pose(frame_id)

def publish_sensor_frame(frame_id, pose): 
    global g_viz_pub
    g_viz_pub.publish_sensor_frame(frame_id, pose)

def publish_pose_t(channel, pose, frame_id='camera'): 
    global g_viz_pub
    frame_pose = g_viz_pub.get_sensor_pose(frame_id)
    out_pose = frame_pose.oplus(pose)

    p = pose_t(list(out_pose.quat.to_wxyz()),
               out_pose.tvec.tolist())
    g_viz_pub.publish(channel, serialize(p))

def publish_image_t(pub_channel, im, jpeg=False, flip_rb=True): 
    global g_viz_pub
    out = image_t()

    # Populate appropriate fields
    h,w = im.shape[:2]
    c = 3
    out.width, out.height = w, h
    out.row_stride = w*c
    out.utime = 1
        
    # Propagate encoded/raw data,
    color_code = cv2.COLOR_GRAY2RGB if flip_rb else cv2.COLOR_GRAY2BGR
    image = cv2.cvtColor(im, color_code) if im.ndim == 2 else im
    
    # Propagate appropriate encoding 
    out.pixelformat = image_t.PIXEL_FORMAT_MJPEG if jpeg \
                      else image_t.PIXEL_FORMAT_RGB
    out.data = cv2.imencode('.jpg', image)[1] if jpeg else image.tostring()
    out.size = len(out.data)
    out.nmetadata = 0

    # Pub
    g_viz_pub.publish(pub_channel, serialize(out))

def publish_botviewer_image_t(im, jpeg=False, flip_rb=True): 
    if not isinstance(im, np.ndarray): 
        raise TypeError('publish_botviewer_image_t type is not np.ndarray')
    publish_image_t('CAMERA_IMAGE', im, jpeg=jpeg, flip_rb=flip_rb)

def corners_to_edges(corners):
    """ Edges are represented in N x 6 form """
    return np.hstack([corners, np.roll(corners, 1, axis=0)])

def polygons_to_edges(polygons):
    """ Edges are represented in N x 6 form """
    return np.vstack([ corners_to_edges(corners) for corners in polygons])

def arr_msg(arr, carr, frame_uid, element_id): 
    # point3d collection msg
    msg = vs.point3d_list_t()
    msg.nnormals = 0
    # msg.normals = []
    msg.npointids = 0
    # msg.pointids = []
    msg.id = int(time.time() * 1e6)
    
    # comes from the sensor_frames_msg published earlier
    msg.collection = frame_uid 
    msg.element_id = element_id

    npoints = len(arr)
    msg.points.extend([vs.point3d_t() for _ in xrange(0,npoints)])
    msg.npoints = len(msg.points)             
    inds = np.arange(0,npoints)

    for j in xrange(npoints):
        msg.points[j].x = arr[j,0]
        msg.points[j].y = arr[j,1]
        msg.points[j].z = arr[j,2]

    msg.colors.extend([vs.color_t() for _ in xrange(0,npoints)])
    msg.ncolors = len(msg.colors)
    for j in xrange(npoints):
        msg.colors[j].r = carr[j,0]
        msg.colors[j].g = carr[j,1]
        msg.colors[j].b = carr[j,2]    

    return msg

@timeitmethod
def publish_point_type(pub_channel, _arr, c='r', point_type='POINT', 
                       flip_rb=False, frame_id='camera', element_id=0, reset=True):
    """
    Publish point cloud on:
    pub_channel: Channel on which the cloud will be published
    arr: numpy array (N x 3) for point cloud data
    c: Option 1: 'c', 'b', 'r' etc colors accepted by matplotlibs color
       Option 2: float ranging from 0 to 1 via matplotlib's jet colormap
       Option 3: numpy array (N x 3) with r,g,b vals ranging from 0 to 1
    s: supported only by matplotlib plotting
    alpha: supported only by matplotlib plotting

    Supports
    
    POINT=1, LINE_STRIP=2, LINE_LOOP=3, LINES=4,
    TRIANGLE_STRIP=5, TRIANGLE_FAN=6, TRIANGLES=7,
    QUAD_STRIP=8, QUADS=9, POLYGON=10

    """
    global g_viz_pub

    # point3d list collection msg
    pc_list_msg = vs.point3d_list_collection_t()
    pc_list_msg.id = g_viz_pub.channel_uid(pub_channel)
    pc_list_msg.name = pub_channel
    pc_list_msg.type = getattr(vs.point3d_list_collection_t, point_type)
    pc_list_msg.reset = reset
    # pc_list_msg.point_lists = []
    
    # Create the point cloud msg
    if isinstance(_arr, list) or isinstance(_arr, deque): 
        element_ids = element_id if isinstance(element_id, list) else [0] * len(_arr) 
        # print 'Multiple elements: ', element_ids
        assert(len(c) == len(_arr))
        for element_id, _arr_item, _carr_item in izip(element_ids, _arr, c): 
            arr, carr = copy_pointcloud_data(_arr_item, _carr_item, flip_rb=flip_rb)
            pc_msg = arr_msg(arr, carr=carr, 
                             frame_uid=g_viz_pub.channel_uid(frame_id), element_id=element_id)
            pc_list_msg.point_lists.extend([pc_msg])
    else: 
        # print 'Single element: ', element_id
        arr, carr = copy_pointcloud_data(_arr, c, flip_rb=flip_rb)
        pc_msg = arr_msg(arr, carr=carr, frame_uid=g_viz_pub.channel_uid(frame_id), element_id=element_id)
        pc_list_msg.point_lists.extend([pc_msg])

    # add to point cloud list                
    # print('published %i lists %s' % (len(_arr), reset))
    pc_list_msg.nlists = len(pc_list_msg.point_lists)
    g_viz_pub.publish("POINTS_COLLECTION", serialize(pc_list_msg))

# @run_async
@timeitmethod
def publish_cloud(pub_channel, arr, c='r', flip_rb=False, frame_id='camera', element_id=0, reset=True):
    publish_point_type(pub_channel, arr, c=c, point_type='POINT', 
                       flip_rb=flip_rb, frame_id=frame_id, element_id=element_id, reset=reset)

def publish_pose_list(pub_channel, poses, texts=[], covars=[], frame_id='camera', reset=True, object_type='AXIS3D'):
    """
    Publish Pose List on:
    pub_channel: Channel on which the cloud will be published
    element_id: None (defaults to np.arange(len(poses)), otherwise provide list of ids
    """
    global g_viz_pub
    # poses = deepcopy(_poses)
    frame_pose = g_viz_pub.get_sensor_pose(frame_id)
    
    # pose list collection msg
    pose_list_msg = vs.obj_collection_t()
    pose_list_msg.id = g_viz_pub.channel_uid(pub_channel)
    pose_list_msg.name = pub_channel
    pose_list_msg.type = getattr(vs.obj_collection_t, object_type)
    pose_list_msg.reset = reset
    
    nposes = len(poses)

    # fill out points
    pose_list_msg.objs.extend([vs.obj_t() for j in range(0,nposes)])
    pose_list_msg.nobjs = nposes
    inds = np.arange(0,nposes)

    arr = np.zeros((len(poses),3))
    for j,pose in enumerate(poses): 

        arr[j,0] = pose.tvec[0]
        arr[j,1] = pose.tvec[1]
        arr[j,2] = pose.tvec[2]

        # Pose compounding
        p = frame_pose.oplus(pose) 
        roll, pitch, yaw, x, y, z = p.to_rpyxyz(axes='sxyz')

        # Optionally get the id of the pose, 
        # for plotting clouds with corresponding pose
        # Note: defaults to index of pose
        pose_list_msg.objs[j].id = getattr(pose, 'id', j) 
        pose_list_msg.objs[j].x = x
        pose_list_msg.objs[j].y = y
        pose_list_msg.objs[j].z = z

        pose_list_msg.objs[j].roll = roll
        pose_list_msg.objs[j].pitch = pitch
        pose_list_msg.objs[j].yaw = yaw
        
    g_viz_pub.publish("OBJ_COLLECTION", serialize(pose_list_msg))

    # Publish corresponding text
    if len(texts): 
        assert(len(texts) == len(poses))
        publish_text_list(pub_channel, poses, texts, frame_id=frame_id, reset=reset)

    # Publish corresponding covariances
    if len(covars): 
        publish_covar_list(pub_channel, poses, covars, frame_id=frame_id, reset=reset)

def publish_text_list(pub_channel, poses, texts=[], frame_id='camera', reset=True):
    text_list_msg = vs.text_collection_t()
    text_list_msg.name = pub_channel+'-text'
    text_list_msg.id = g_viz_pub.channel_uid(text_list_msg.name)
    text_list_msg.type = 1 # doesn't matter
    text_list_msg.reset = reset

    assert(len(poses) == len(texts))
    nposes = len(texts)
    text_list_msg.texts = [vs.text_t() for j in range(nposes)]
    text_list_msg.n = nposes

    for j,pose in enumerate(poses):
        text_list_msg.texts[j].id = getattr(pose, 'id', j) 
        text_list_msg.texts[j].collection_id = g_viz_pub.channel_uid(pub_channel)
        text_list_msg.texts[j].object_id = getattr(pose, 'id', j) 
        text_list_msg.texts[j].text = texts[j]
       
    g_viz_pub.publish("TEXT_COLLECTION", serialize(text_list_msg))

def publish_covar_list(pub_channel, poses, covars=[], frame_id='camera', reset=True):
    covar_list_msg = vs.cov_collection_t()
    covar_list_msg.name = pub_channel+'-covars'
    covar_list_msg.id = g_viz_pub.channel_uid(covar_list_msg.name)
    covar_list_msg.type = vs.cov_collection_t.ELLIPSOID
    covar_list_msg.reset = reset

    assert(len(poses) == len(covars))
    nposes = len(covars)
    covar_list_msg.covs = [vs.cov_t() for j in range(nposes)]
    covar_list_msg.ncovs = nposes

    for j,pose in enumerate(poses):
        # assert(covars[j].ndim == 2 and covars[j].shape[1] == 6)
        
        covar_list_msg.covs[j].id = getattr(pose, 'id', j) 
        covar_list_msg.covs[j].collection = g_viz_pub.channel_uid(pub_channel)
        covar_list_msg.covs[j].element_id = getattr(pose, 'id', j) 
        covar_list_msg.covs[j].entries = covars[j]
        covar_list_msg.covs[j].n = len(covars[j])
        # print 'covar_list', nposes, covar_list_msg.covs[j].collection, covar_list_msg.covs[j].id, \
        #     covar_list_msg.covs[j].element_id, covars[j]

    g_viz_pub.publish("COV_COLLECTION", serialize(covar_list_msg))

def publish_line_segments(pub_channel, _arr1, _arr2, c='r', flip_rb=False, frame_id='camera', element_id=0, reset=True):
    publish_point_type(pub_channel, np.hstack([_arr1, _arr2]), c=c, point_type='LINES', 
                       flip_rb=flip_rb, frame_id=frame_id, element_id=element_id, reset=reset)
    
def publish_quads(pub_channel, quads, frame_id='camera', reset=True):
    publish_point_type(pub_channel, quads, point_type='QUADS', frame_id=frame_id, reset=reset)

# Object Renderers ==============================================================
def draw_camera(pose, zmin=0.0, zmax=0.1, fov=np.deg2rad(60)): 

    frustum = Frustum(pose, zmin=zmin, zmax=zmax, fov=fov)
    nul, nll, nlr, nur, ful, fll, flr, fur = frustum.vertices
    # nll, nlr, nur, nul, fll, flr, fur, ful = frustum.vertices

    faces = []

    # Triangles: Front Face
    # faces.extend([ful, fur, flr])
    # faces.extend([flr, ful, fll])

    # Triangles: Back Face
    # faces.extend([nul, nur, nlr])
    # faces.extend([nlr, nul, nll])

    # Triangles: Four walls (2-triangles per face)
    left, top, right, bottom = [fll, nll, ful, ful, nll, nul], \
                               [ful, nul, fur, fur, nul, nur], \
                               [fur, nur, flr, flr, nur, nlr], \
                               [flr, nlr, fll, fll, nlr, nll]
    faces.extend([left, top, right, bottom]) # left, top, right, bottom wall
    faces = np.vstack(faces)

    # Lines: zmin-zmax
    pts = []
    pts.extend([ful, fur, flr, fll, ful])
    pts.extend([ful, fll, nll, nul, ful])
    pts.extend([ful, nul, nur, fur, ful])
    pts.extend([fur, nur, nlr, flr, fur])
    pts.extend([flr, nlr, nll, fll, flr])
    pts.extend([flr, ful])
    pts.extend([fur, fll])    
    pts = np.vstack(pts)
    
    return (faces, np.hstack([pts[:-1], pts[1:]]).reshape((-1,3)))

def draw_laser_frustum(pose, zmin=0.0, zmax=10, fov=np.deg2rad(60)): 

    N = 30
    curve = np.vstack([(
        RigidTransform.from_rpyxyz(0, 0, rad, 0, 0, 0) * np.array([[zmax, 0, 0]])) 
             for rad in np.linspace(-fov/2, fov/2, N)])
    
    curve_w = pose * curve

    faces, edges = [], []
    for cpt1, cpt2 in zip(curve_w[:-1], curve_w[1:]): 
        faces.extend([pose.translation, cpt1, cpt2])
        edges.extend([cpt1, cpt2])

    # Connect the last pt in the curve w/ the current pose, 
    # then connect the the first pt in the curve w/ the curr. pose
    edges.extend([edges[-1], pose.translation])
    edges.extend([edges[0], pose.translation])

    faces = np.vstack(faces)
    edges = np.vstack(edges)
    return (faces, edges)

def draw_tag(pose=None, size=0.1):
    sz = size / 2.0
    corners = np.float32([[sz, sz, 0], [-sz, sz, 0], 
                          [-sz, -sz, 0], [sz, -sz, 0], 
                          [sz, sz, 0], [-sz, -sz, 0], 
                          [-sz, sz, 0], [sz, -sz, 0]])
    return pose * corners if pose is not None else corners

def draw_tag_edges(pose=None, size=0.1):
    return corners_to_edges(draw_tag(pose=pose, size=size))
    
def draw_tags_edges(poses, size=0.1): 
    return np.vstack([draw_tag_edges(p) for p in poses])

# Publish Objects ===============================================================
def publish_tags(pub_channel, poses, c='g', texts=[], covars=[], frame_id='camera', 
                 draw_edges=True, draw_nodes=False, element_id=0, size=1, reset=True):
 

    # Publish pose, and corresponding texts
    if draw_nodes: 
        publish_pose_list(pub_channel, poses, texts=texts, frame_id=frame_id, reset=reset)

    # Green tag edges
    tag_edges = np.vstack([draw_tag_edges(p) for p in poses])
    publish_line_segments(pub_channel + '-edges', tag_edges[:,:3], tag_edges[:,3:6], c=c, 
                          frame_id=frame_id, element_id=element_id, reset=reset)

def publish_cameras(pub_channel, poses, c='y', texts=[], covars=[], frame_id='camera', 
                    draw_faces=False, draw_edges=True, draw_nodes=False, size=1., zmin=0, zmax=0.25, reset=True):
    cam_feats = [draw_camera(pose, zmin=zmin * size, zmax=zmax * size) for pose in poses]
    cam_faces = map(lambda x: x[0], cam_feats)
    cam_edges = map(lambda x: x[1], cam_feats)

    # Publish pose, and corresponding texts
    publish_pose_list(pub_channel, poses, texts=texts, covars=covars, frame_id=frame_id, reset=reset)
    
    # Draw blue node at camera pose translation
    if draw_nodes: 
        publish_pose_list(pub_channel + '-nodes', poses, texts=texts, frame_id=frame_id, reset=reset, object_type='HEXAGON')

    # Darker yellow edge
    if draw_edges: 
        carr = ['r'] * len(cam_edges)
        publish_point_type(pub_channel+'-edges', cam_edges, point_type='LINES', c=carr, 
                           frame_id=frame_id, reset=reset)

    # Light faces
    if draw_faces: 
        carr = [c] * len(cam_faces)
        publish_point_type(pub_channel+'-faces', cam_faces, point_type='TRIANGLES', c=carr, frame_id=frame_id, reset=reset)


def publish_laser_frustums(pub_channel, poses, c='y', texts=[], frame_id='camera', 
                    draw_faces=True, draw_edges=True, size=1, zmin=0.01, zmax=5, reset=True):
    cam_feats = [draw_laser_frustum(pose, zmax=zmax * size, fov=np.deg2rad(80)) for pose in poses]
    cam_faces = map(lambda x: x[0], cam_feats)
    cam_edges = map(lambda x: x[1], cam_feats)

    # Publish pose, and corresponding texts
    publish_pose_list(pub_channel, poses, texts=texts, frame_id=frame_id, reset=reset)

    # Darker yellow edge
    if draw_edges: 
        carr = ['r'] * len(cam_edges)
        publish_point_type(pub_channel+'-edges', cam_edges, point_type='LINES', c=carr, 
                           frame_id=frame_id, reset=reset)

    # Light faces
    if draw_faces: 
        carr = [c] * len(cam_faces)
        publish_point_type(pub_channel+'-faces', cam_faces, point_type='TRIANGLES', c=carr, frame_id=frame_id, reset=reset)

# # @run_async
# def publish_point_type(pub_channel, arr, c='r', point_type='POINT', 
#                        flip_rb=False, frame_id='camera', element_id=0, reset=True):
#     _publish_point_type(pub_channel, arr, c=c, point_type=point_type, 
#                         flip_rb=flip_rb, frame_id=frame_id, element_id=element_id, reset=reset)

# # @run_async
# def publish_pose_list(pub_channel, poses, texts=[], covars=[], frame_id='camera', reset=True, object_type='AXIS3D'):
#     _publish_pose_list(pub_channel, deepcopy(poses), texts=texts, covars=covars, 
#                        frame_id=frame_id, reset=reset, object_type=object_type)

# # @run_async
# def publish_line_segments(pub_channel, arr1, arr2, c='r', flip_rb=False, frame_id='camera'):
#     _publish_line_segments(pub_channel, deepcopy(arr1), deepcopy(arr2), c=deepcopy(c), flip_rb=flip_rb, frame_id=frame_id)

# # LCMGL Calls ==============================================================
# def publish_point_cloud_lcmgl(pub_channel, arr, c='r', size=1, alpha=1):
#     """ 
#     Publish point cloud array in lcmgl
#     """
#     arr = reshape_arr(arr)    
#     g = lcmgl.lcmgl('posepairs', lc)
#     g.glPointSize(size)
#     g.glColor4f(c[0],c[1],c[2],alpha)
#     g.glBegin(lcmgl.GL_POINTS)
#     for j in range(0,len(arr)):
#         g.glVertex3d(arr[j,0], arr[j,1], arr[j,2])
#     g.glEnd()
#     g.switch_buffer()

# def publish_text_lcmgl(pub_channel, arr, texts, alpha=0.9, sensor_tf='camera'): 
#     """ 
#     Publish text with point cloud array in lcmgl
#     """
#     frame_id = publish_sensor_frame(sensor_tf)
#     frame_pose = get_frame(sensor_tf)
#     assert(frame_id is not None)

#     # Setup lcmgl text
#     g = lcmgl.lcmgl(pub_channel, lc)

#     # w.r.t sensor frame
#     arr = frame_pose * arr

#     # Plot
#     g.glColor4f(0.2,0.2,0.2,alpha)
#     for idx, text in enumerate(texts):
#         g.text(arr[idx,0],arr[idx,1],arr[idx,2], text)
#     g.switch_buffer()
