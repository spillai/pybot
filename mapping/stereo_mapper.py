"""
=====================================================================
 Stereopsis-based Mapper 
=====================================================================

Map the environment using stereo data
 TODO
   1. Semi-dense stereo 

"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: TODO

import numpy as np
import cv2, os, time
import argparse
from itertools import izip
from collections import OrderedDict

from bot_utils.io_utils import read_config
from bot_vision.color_utils import get_color_by_label
from bot_vision.image_utils import to_color, to_gray, median_blur
from bot_vision.imshow_utils import imshow_cv
from bot_utils.db_utils import AttrDict
from bot_geometry.rigid_transform import Pose, Quaternion, \
    RigidTransform, Sim3, normalize_vec
