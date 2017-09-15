import vs
import lcm

global g_lc
g_lc = None

def serialize(msg):
    return msg.encode()

def publish(channel, data): 
    global g_lc
    if g_lc is None:
        g_lc = lcm.LCM()
    g_lc.publish(channel, data)
