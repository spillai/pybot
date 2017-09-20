import time
import numpy as np
from pybot.externals.proto import vs_pb2 as vs

def pose_t(orientation, pos):
    p = vs.pose_t()
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

    msg.points._values = np.float64(arr).flat
    msg.colors._values = np.float32(carr[:,:3]).flat
    
    msg.npoints = npoints
    msg.ncolors = npoints

    return msg
