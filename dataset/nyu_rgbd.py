"""
NYU-RGB-D Dataset reader
"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT 

import os, time
import numpy as np
import cv2

from itertools import izip, imap, islice
from collections import defaultdict

import h5py

from bot_utils.misc import setup_pbar
from bot_utils.db_utils import AttrDict
from bot_utils.dataset_readers import read_dir, read_files, natural_sort, \
    DatasetReader, ImageDatasetReader
from bot_vision.imshow_utils import annotate_bbox
from bot_vision.camera_utils import kinect_v1_params, \
    Camera, CameraIntrinsic, CameraExtrinsic, \
    check_visibility, get_object_bbox

from bot_geometry.rigid_transform import Quaternion, RigidTransform
from bot_externals.plyfile import PlyData

# =====================================================================
# Generic NYU-RGBD Dataset class
# ---------------------------------------------------------------------

class NYURGBDDataset(object): 
    def __init__(self, directory=None, ground_truth=None, version='v2'): # object_dir=None, scene_dir=None, targets=train_names, version='v1'): 
        
        # Setup Version
        self.version = version

        # Load main labeled dataset
        self._dataset = h5py.File(ground_truth, 'r')
        
        # # Get labels
        # labels = [ l for l in dataset['labels'] ]
        
        self._ims = self._dataset['images']
        self._depths = self._dataset['depths']
        self._instances = self._dataset['instances']
        self._labels = self._dataset['labels']

    def _process_items(self, index, rgb_im, depth_im, instance, label, bbox, pose): 
        # print 'Processing pose', pose, bbox
                

        # def _process_bbox(bbox): 
        #     return dict(category=bbox['category'], target=UWRGBDDataset.target_hash[str(bbox['category'])], 
        #                 left=bbox.coords['left'], right=bbox['right'], top=bbox['top'], bottom=bbox['bottom'])

        # # Compute bbox from pose and map (v2 support)
        # if self.version == 'v1': 
        #     if bbox is not None: 
        #         bbox = [_process_bbox(bb) for bb in bbox]
        #         bbox = filter(lambda bb: bb['target'] in UWRGBDDataset.train_ids_set, bbox)

        # if self.version == 'v2': 
        #     if bbox is None and hasattr(self, 'map_info'): 
        #         bbox = self.get_bboxes(pose)

        # print 'Processing pose', pose, bbox

        rgb_im = np.swapaxes(rgb_im, 0, 2)
        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)

        depth_im = np.swapaxes(depth_im, 0, 1) * 1000
        instance = np.swapaxes(instance, 0, 1)
        label = np.swapaxes(label, 0, 1)

        return AttrDict(index=index, img=rgb_im, depth=depth_im, instance=instance, 
                        label=label, bbox=bbox if bbox is not None else [], pose=pose)


    def iteritems(self, every_k_frames=1): 
        index = 0 
        # , bbox, pose
        bbox, pose = None, None
        for rgb_im, depth_im, instance, label in izip(islice(self._ims, 0, None, every_k_frames), 
                                                      islice(self._depths, 0, None, every_k_frames), 
                                                      islice(self._instances, 0, None, every_k_frames), 
                                                      islice(self._labels, 0, None, every_k_frames)
        ): 
            index += every_k_frames
            yield self._process_items(index, rgb_im, depth_im, instance, label, bbox, pose)

    # def iterinds(self, inds): 
    #     for index, rgb_im, depth_im, bbox, pose in izip(inds, 
    #                                                     self.rgb.iterinds(inds), 
    #                                                     self.depth.iterinds(inds), 
    #                                                     [self.bboxes[ind] for ind in inds], 
    #                                                     [self.poses[ind] for ind in inds]): 
    #         yield self._process_items(index, rgb_im, depth_im, bbox, pose)

def test_nyu_rgbd(version='v2'): 
    from bot_vision.image_utils import to_color
    from bot_vision.imshow_utils import imshow_cv

    v2_directory = '/media/spillai/MRG-HD1/data/nyu_rgbd/'
    v2_gt = os.path.join(v2_directory, 'nyu_depth_v2_labeled.mat')

    # if version == 'v1': 
    #     dataset = UWRGBDSceneDataset(version='v1', 
    #                                       directory=os.path.join(v1_directory, 'rgbd-scenes'), 
    #                                       aligned_directory=os.path.join(v1_directory, 'rgbd-scenes-aligned'))
    if version == 'v2': 
        dataset = NYURGBDDataset(version='v2', directory=v2_directory, ground_truth=v2_gt)
    else: 
        raise RuntimeError('''Version %s not supported. '''
                           '''Check dataset and choose v1/v2 scene dataset''' % version)


    for f in dataset.iteritems(every_k_frames=5): 
        # vis = rgbd_data_uw.annotate(f)
        imshow_cv('frame', f.img, text='Image')
        imshow_cv('depth', (f.depth / 16).astype(np.uint8), text='Depth')
        imshow_cv('instance', (f.instance).astype(np.uint8), text='Instance')
        imshow_cv('label', (f.label).astype(np.uint8), text='Label')
        cv2.waitKey(100)

    return dataset


if __name__ == "__main__": 
    test_nyu_rgbd()

