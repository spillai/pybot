import os

__version__ = '0.2'
__all__ = ['vision', 'utils', 'geometry', 'ml', 'externals', 'get_environment']

def get_environment(env_str, default, choices=None):
    var = os.getenv(env_str, default)
    if var is None: 
        raise RuntimeError('{} not in environment'.format(env_str))

    if choices is not None:
        if var not in set(choices):
            raise RuntimeError('{}: {} not in choices {}'.format(env_str, var, choices))
        
    return var
