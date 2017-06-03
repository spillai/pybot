import sys
import numpy as np

from pybot.utils.db_utils import AttrDict
from pybot import get_environment

from pybot.mapping.mapper import Keyframe, Mapper, MultiViewMapper
from pybot.mapping.semi_dense_mapper import SemiDenseMapper, LSDMapper, ORBMapper


_PYBOT_SLAM_BACKEND = get_environment('PYBOT_SLAM_BACKEND',
                                      default='gtsam',
                                      choices=['gtsam', 'isam', 'toro'])

# Variables for SLAM
cfg = AttrDict(
    ODOM_NOISE = np.ones(6) * 0.1,
    PRIOR_POSE_NOISE = np.ones(6) * 0.001,
    PRIOR_POINT3D_NOISE = np.ones(3) * 0.2,
    MEASUREMENT_NOISE = np.ones(6) * 0.4,
    PX_MEASUREMENT_NOISE = [2.0, 2.0],
    VSLAM_MIN_LANDMARK_OBS = 2,
    SLAM_BACKEND = _PYBOT_SLAM_BACKEND
)    

def backend():
    return cfg.SLAM_BACKEND

def set_backend(name):
    cfg.SLAM_BACKEND = name

