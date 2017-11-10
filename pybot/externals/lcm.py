import vs
import lcm
import bot_core as bc
from bot_core import image_t

global g_lc
g_lc = None

def pose_t(orientation, pos):
    p = bc.pose_t()
    p.orientation = orientation
    p.pos = pos
    return p

def serialize(msg):
    return msg.encode()

def publish(channel, data): 
    global g_lc
    if g_lc is None:
        g_lc = lcm.LCM()
    g_lc.publish(channel, data)

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
    msg.points.extend([vs.point3d_t() for _ in range(0,npoints)])
    msg.npoints = len(msg.points)             
    inds = np.arange(0,npoints)

    for j in range(npoints):
        msg.points[j].x = arr[j,0]
        msg.points[j].y = arr[j,1]
        msg.points[j].z = arr[j,2]

    msg.colors.extend([vs.color_t() for _ in range(0,npoints)])
    msg.ncolors = len(msg.colors)
    for j in range(npoints):
        msg.colors[j].r = carr[j,0]
        msg.colors[j].g = carr[j,1]
        msg.colors[j].b = carr[j,2]    

    return msg
