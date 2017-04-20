import sys
import numpy as np

from .mapper import Keyframe, Mapper, MultiViewMapper
from .semi_dense_mapper import SemiDenseMapper, LSDMapper, ORBMapper

# Variables for SLAM
ODOM_NOISE = np.ones(6) * 0.01
PRIOR_NOISE = np.ones(6) * 0.001
MEASUREMENT_NOISE = np.ones(6) * 0.4
