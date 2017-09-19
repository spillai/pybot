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
