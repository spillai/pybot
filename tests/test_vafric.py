#!/usr/bin/env python
import numpy as np

from collections import deque

from pybot.utils.dataset.misc import VaFRICDatasetReader
from pybot.utils.test_utils import test_dataset
from pybot.vision.imshow_utils import imshow_cv
from pybot.geometry.rigid_transform import RigidTransform
import pybot.externals.draw_utils as draw_utils


if __name__ == "__main__": 
    # VaFRIC dataset
    dataset = VaFRICDatasetReader(directory='/home/spillai/HD1/data/VaFRIC', scene='200fps')

    poses = deque(maxlen=100)
    for idx, frame in enumerate(dataset.iteritems()): 
        imshow_cv('frame', frame.img)
        poses.append(frame.pose)

        # Publish ground truth poses
        draw_utils.publish_pose_list('ground_truth_poses', poses, frame_id='camera')
    
