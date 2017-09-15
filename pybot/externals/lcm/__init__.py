# from pybot import VIS_LCM_FLAG
# from pybot.externals import nop

# try:
#     import pybot.externals.draw_utils as draw_utils
# except:
#     from pybot.externals import nop
#     draw_utils = nop(''))

import vs
import lcm

global g_lc
g_lc = lcm.LCM()

def serialize(msg):
    return msg.encode()

def publish(channel, data): 
    global g_lc
    g_lc.publish(channel, data)
