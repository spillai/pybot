import os

def get_environment(env_str, default, choices=None):
    var = os.getenv(env_str, default)
    if var is None: 
        raise RuntimeError('{} not in environment'.format(env_str))

    if choices is not None:
        if var not in set(choices):
            raise RuntimeError('{}: {} not in choices {}'.format(env_str, var, choices))
        
    return var

# _PYBOT_USER_DATA = get_environment('PYBOT_USER_DATA',
#                                    default=os.path.join(os.getenv('HOME'),
#                                                         'data'))

# _PYBOT_IMSHOW = bool(int(get_environment('PYBOT_IMSHOW', default=1)))

# def user_data():
#     return _PYBOT_USER_DATA

# IMSHOW_FLAG = _PYBOT_IMSHOW
