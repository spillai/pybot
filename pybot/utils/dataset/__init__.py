import os
_PYBOT_UTILS_DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
PYBOT_DATA_DIR = os.path.join(_PYBOT_UTILS_DATASET_DIR, 'data')

def data_file(*args):
    return os.path.join(PYBOT_DATA_DIR, *args)
