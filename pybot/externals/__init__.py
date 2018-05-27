import sys
from collections import defaultdict
from pybot import get_environment
from pybot.utils.misc import color_green

_PYBOT_VIS_CHOICES = ['lcm', 'protobuf']
_PYBOT_VIS_BACKEND = get_environment('PYBOT_VIS_BACKEND',
                                      default='protobuf',
                                      choices=_PYBOT_VIS_CHOICES)

_PYBOT_MARSHALLING_CHOICES = ['lcm', 'zmq']
_PYBOT_MARSHALLING_BACKEND = get_environment('PYBOT_MARSHALLING_BACKEND',
                                      default='zmq',
                                      choices=_PYBOT_MARSHALLING_CHOICES)

def vis_backend():
    return _PYBOT_VIS_BACKEND

def set_vis_backend(value):
    assert(value in set(_PYBOT_VIS_CHOICES))
    _PYBOT_VIS_BACKEND = value

def marshalling_backend():
    return _PYBOT_MARSHALLING_BACKEND

def set_marshalling_backend(value):
    assert(value in set(_PYBOT_MARSHALLING_CHOICES))
    _PYBOT_MARSHALLING_BACKEND = value

# ===============================================================================

if _PYBOT_VIS_BACKEND == 'lcm':
    from pybot.externals.lcm import vs, serialize, image_t, pose_t, arr_msg

elif _PYBOT_VIS_BACKEND == 'protobuf':
    from pybot.externals.pb import vs, serialize, pose_t, arr_msg

else:
    raise Exception('''Unknown backend: {}, '''
                    '''choices are {}'''.format(_PYBOT_VIS_BACKEND,
                                                _PYBOT_VIS_CHOICES))
print('Using {} vis backend.'.format(color_green(_PYBOT_VIS_BACKEND)))

# ===============================================================================

if _PYBOT_MARSHALLING_BACKEND == 'lcm':
    from pybot.externals.lcm import publish

elif _PYBOT_MARSHALLING_BACKEND == 'zmq':
    from pybot.externals.zeromq import publish, pack, unpack

else:
    raise Exception('''Unknown backend: {}, '''
                    '''choices are {}'''.format(_PYBOT_MARSHALLING_BACKEND,
                                                _PYBOT_MARSHALLING_CHOICES))
print('Using {} marshalling backend.'.format(color_green(_PYBOT_MARSHALLING_BACKEND)))

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
