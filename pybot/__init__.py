import os

_PYBOT_USER_DATA = os.getenv('PYBOT_USER_DATA', '/home/spillai/perceptual-learning/data')

def user_data():
    return _PYBOT_USER_DATA
