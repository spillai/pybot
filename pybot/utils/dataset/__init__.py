import os
_PYBOT_UTILS_DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_PYBOT_UTILS_DATASET_DIR, 'data')

def data_file(*args):
    return os.path.join(data_dir, *args)
