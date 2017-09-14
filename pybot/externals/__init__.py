import sys
from collections import defaultdict
from pybot import get_environment

_PYBOT_VIS_BACKEND = get_environment('PYBOT_VIS_BACKEND',
                                      default='lcm',
                                      choices=['lcm', 'protobuf'])

def vis_backend():
    return _PYBOT_VIS_BACKEND

def set_vis_backend(value):
    _PYBOT_VIS_BACKEND = value

# ===============================================================================

if _PYBOT_VIS_BACKEND == 'lcm':
    sys.stderr.write('Using lcm-vis backend.\n')
    from pybot.externals.lcm import vs, serialize
    
elif _PYBOT_VIS_BACKEND == 'protobuf':
    sys.stderr.write('Using protobuf-vis backend.\n')
    from pybot.externals.pb import vs, serialize

else:
    raise Exception('Unknown backend: {}'.format(cfg.SLAM_BACKEND))
    

# ===============================================================================

class MayBeCalled(object):
    def __call__(self, *args, **kwargs):
        return None

class nop(object):
    """
    Nop class to handle misc optional imports
    Shamelessly ripped off 
    from http://stackoverflow.com/questions/24946321/how-do-i-write-a-no-op-or-dummy-class-in-python
    """
    def __init__(self, name=''): 
        self.name_ = name

    def __get__(self, *args):
        return MayBeCalled()

    def __hasattr__(self, attr):
        if len(self.name_): print('{}::{}'.format(self.name_, attr))
        return True

    def __getattr__(self, attr):
        if len(self.name_): print('{}::{}'.format(self.name_, attr))
        return MayBeCalled()

    
