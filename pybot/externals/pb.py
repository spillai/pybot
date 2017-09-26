import time
import numpy as np
from pybot.externals.proto import vs_pb2 as vs

def pose_t(orientation, pos):
    p = vs.pose_t()
    p.utime = 0
    p.orientation.extend(orientation)
    p.pos.extend(pos)
    return p

def serialize(msg):
    return msg.SerializeToString()

def deserialize(msg):
    return ParseFromString(msg)

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
    
    msg.points = np.float32(arr).tostring()
    msg.colors = np.float32(carr[:,:3]).tostring()
    print 'arr: ', len(arr), arr[:3], carr[:3,:3], '\n' + '-' * 40
    msg.npoints = npoints
    msg.ncolors = npoints

    return msg
