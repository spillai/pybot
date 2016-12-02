#!/usr/bin/env python

import time, cv2
import numpy as np

from pybot.utils.tests_utils import test_dataset
from pybot.vision.imshow_utils import imshow_cv

if __name__ == "__main__": 
    try: 
        import pybot.externals.ros.draw_utils as draw_utils
        draw_utils.init()
    except: 
        import pybot.vision.draw_utils as draw_utils

    for vel_pc in test_dataset().iter_velodyne_frames(): 
        X = vel_pc[:,:3]

        # Plot height map
        draw_utils.publish_height_map('velodyne_cloud', pose_bv * X, frame_id='body', height_axis=2)

