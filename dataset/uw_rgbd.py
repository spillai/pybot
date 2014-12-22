import os
import numpy as np
import cv2
from itertools import izip, imap
from collections import defaultdict

from scipy.io import loadmat

from bot_utils.db_utils import AttrDict
from bot_utils.dataset_readers import read_dir, read_files, natural_sort, \
    DatasetReader, ImageDatasetReader

import progressbar as pb

# __categories__ = ['flashlight', 'cap', 'cereal_box', 'coffee_mug', 'soda_can']

def setup_pbar(maxval): 
    widgets = ['Progress: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    pbar = pb.ProgressBar(widgets=widgets, maxval=maxval)
    pbar.start()
    return pbar

class UWRGBDDataset(object): 
    default_rgb_shape = (480,640,3)
    default_depth_shape = (480,640)

    class_names = ["apple", "ball", "banana", "bell_pepper", "binder", "bowl", "calculator", "camera", 
                   "cap", "cell_phone", "cereal_box", "coffee_mug", "comb", "dry_battery", "flashlight", 
                   "food_bag", "food_box", "food_can", "food_cup", "food_jar", "garlic", "glue_stick", 
                   "greens", "hand_towel", "instant_noodles", "keyboard", "kleenex", "lemon", "lightbulb", 
                   "lime", "marker", "mushroom", "notebook", "onion", "orange", "peach", "pear", "pitcher", 
                   "plate", "pliers", "potato", "rubber_eraser", "scissors", "shampoo", "soda_can", 
                   "sponge", "stapler", "tomato", "toothbrush", "toothpaste", "water_bottle", "BACKGROUND"]

    class_ids = np.arange(len(class_names), dtype=np.int)    
    target_hash = dict(zip(class_names, class_ids))
    target_unhash = dict(zip(class_ids, class_names))

    # train_names = ["cereal_box", "BACKGROUND"]
    train_names = ["bowl", "cap", "cereal_box", "coffee_mug", "soda_can", "BACKGROUND"]
    train_ids = [target_hash[name] for name in train_names]
    train_names_set, train_ids_set = set(train_names), set(train_ids)

    @classmethod
    def get_category_name(cls, target_id): 
        return cls.target_unhash[target_id] \
            if target_id in cls.train_ids_set else 'BACKGROUND'

    @classmethod
    def get_category_id(cls, target_name): 
        return cls.target_hash[target_name] \
            if target_name in cls.train_names_set else cls.target_hash['BACKGROUND']
    
class UWRGBDObjectDataset(UWRGBDDataset):
    """
    RGB-D Dataset readers
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """

    class _reader(object): 
        """
        RGB-D reader 
        Given mask, depth, and rgb files build an read iterator with appropriate process_cb
        """
        def __init__(self, target, instance, files): 
            self.target = target 
            self.instance = instance
            mask_files = natural_sort(filter(lambda fn: '_mask.png' in fn, files))
            depth_files = natural_sort(filter(lambda  fn: '_depth.png' in fn, files))
            rgb_files = natural_sort(list(set(files) - set(mask_files) - set(depth_files)))
            loc_files = natural_sort(map(lambda fn: fn.replace('.png', '_loc.txt'), rgb_files))
            assert(len(mask_files) == len(depth_files) == len(rgb_files) == len(loc_files))

            self.rgb = ImageDatasetReader.from_filenames(rgb_files)
            self.depth = ImageDatasetReader.from_filenames(depth_files)
            self.mask = ImageDatasetReader.from_filenames(mask_files)

        def iteritems(self, every_k_frames=1): 
            for rgb_im, depth_im, mask_im in izip(self.rgb.iteritems(every_k_frames=every_k_frames), 
                                                  self.depth.iteritems(every_k_frames=every_k_frames), 
                                                  self.mask.iteritems(every_k_frames=every_k_frames)): 
                yield AttrDict(target=self.target, instance=self.instance, 
                               img=rgb_im, depth=depth_im, mask=mask_im)

    class _cropped_reader(object): 
        """
        RGB-D reader 
        Given mask, depth, and rgb files build an read iterator with appropriate process_cb
        """
        def __init__(self, target, instance, files): 
            self.target = target 
            self.instance = instance
            mask_files = natural_sort(filter(lambda fn: '_maskcrop.png' in fn, files))
            depth_files = natural_sort(filter(lambda  fn: '_depthcrop.png' in fn, files))
            rgb_files = natural_sort(list(set(files) - set(mask_files) - set(depth_files)))
            loc_files = natural_sort(map(lambda fn: fn.replace('_crop.png', '_loc.txt'), rgb_files))

            # Ensure all have equal number of files (Hack! doesn't ensure filename consistency)
            nfiles = np.min([len(loc_files), len(mask_files), len(depth_files), len(rgb_files)])
            mask_files, depth_files, rgb_files, loc_files = mask_files[:nfiles], depth_files[:nfiles], \
                                                            rgb_files[:nfiles], loc_files[:nfiles]

            # print target, instance, len(loc_files), len(mask_files), len(depth_files), len(rgb_files)
            assert(len(mask_files) == len(depth_files) == len(rgb_files) == len(loc_files))

            # Read images
            self.rgb = ImageDatasetReader.from_filenames(rgb_files)
            self.depth = ImageDatasetReader.from_filenames(depth_files)
            self.mask = ImageDatasetReader.from_filenames(mask_files)

            # Read top-left locations of bounding box
            self.locations = np.vstack([np.loadtxt(loc, delimiter=',', dtype=np.int32) 
                                        for loc in loc_files])

        def iteritems(self, every_k_frames=1): 
            for rgb_im, depth_im, mask_im, loc in \
                izip(self.rgb.iteritems(every_k_frames=every_k_frames), 
                     self.depth.iteritems(every_k_frames=every_k_frames), 
                     self.mask.iteritems(every_k_frames=every_k_frames), 
                     self.locations[::every_k_frames]): 

                rgb = np.zeros(shape=UWRGBDObjectDataset.default_rgb_shape, dtype=np.uint8)
                depth = np.zeros(shape=UWRGBDObjectDataset.default_depth_shape, dtype=np.uint16)
                mask = np.zeros(shape=UWRGBDObjectDataset.default_depth_shape, dtype=np.uint8)

                rgb[loc[1]:loc[1]+rgb_im.shape[0], loc[0]:loc[0]+rgb_im.shape[1]] = rgb_im
                depth[loc[1]:loc[1]+depth_im.shape[0], loc[0]:loc[0]+depth_im.shape[1]] = depth_im
                mask[loc[1]:loc[1]+mask_im.shape[0], loc[0]:loc[0]+mask_im.shape[1]] = mask_im

                # Only a single bbox per image
                yield AttrDict(img=rgb, depth=depth, mask=mask, 
                               bbox={'left':loc[0], 'right':loc[0]+mask_im.shape[1], 
                                     'top':loc[1], 'bottom':loc[1]+mask_im.shape[0], 
                                     'category':self.target, 'instance':self.instance})

    def __init__(self, directory='', targets=UWRGBDDataset.train_names, blacklist=['']):         
        get_category = lambda name: '_'.join(name.split('_')[:-1])
        get_instance = lambda name: int(name.split('_')[-1])

        self._class_names = UWRGBDDataset.class_names
        self._class_ids = UWRGBDDataset.class_ids

        self.target_hash = UWRGBDDataset.target_hash
        self.target_unhash = UWRGBDDataset.target_unhash


        # Only randomly choose targets if not defined
        if targets is not None: 
            # If integer valued, retrieve targets
            if isinstance(targets, int) and targets <= len(self._class_names): 
                inds = np.random.randint(len(self._class_names), size=num_targets)
                targets = self._class_names[inds]
            # If targets are list of strings
            elif isinstance(targets, list) and len(targets) <= len(self._class_names): 
                pass
            else: 
                raise ValueError('targets are not list of strings or integer')
        else: 
            # Pick full/specified dataset
            targets = self._class_names

        # Fusing all object instances of a category into a single key
        print 'Train targets, ', targets
        self._dataset = read_dir(os.path.expanduser(directory), pattern='*.png', 
                                 recursive=False, expected=targets, verbose=False)
        print 'Classes: %i' % len(targets), self._dataset.keys()

        # Instantiate a reader for each of the objects
        self.data = {}
        for key, files in self._dataset.iteritems(): 
            if (targets is not None and get_category(key) not in targets) or key in blacklist: 
                continue
            target_id = self.target_hash[get_category(key)]
            # target_id = UWRGBDDataset.get_category_id(get_category(key))
            instance_id = get_instance(key)
            self.data[key] = UWRGBDObjectDataset._cropped_reader(target_id, instance_id, files)

        # Save target names for metrics
        self.target_names = targets

    def iteritems(self, every_k_frames=1, verbose=False): 
        pbar = setup_pbar(len(self.data)) if verbose else None
        for key, frames in self.data.iteritems(): 
            if verbose: 
                pbar.update(pbar.currval + 1)
                # print 'Processing: %s' % key

            for frame in frames.iteritems(every_k_frames=every_k_frames): 
                yield frame
        if verbose: pbar.finish()

class UWRGBDSceneDataset(UWRGBDDataset):
    """
    RGB-D Scene Dataset reader 
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """
    class _reader(object): 
        """
        RGB-D reader 
        Given mask, depth, and rgb files build an read iterator with appropriate process_cb
        """
        def __init__(self, files, meta_file, version): 
            if version == 'v2': 
                print '\n\n===> Version v1 and v2 have discrepancies in depth values, FIX!! <===\n\n'

            rgb_files, depth_files = UWRGBDSceneDataset._reader.scene_files(files, version)
            assert(len(depth_files) == len(rgb_files))

            self.rgb = ImageDatasetReader.from_filenames(rgb_files)

            # TODO: Check depth seems scaled by 256 not 16
            self.depth = ImageDatasetReader.from_filenames(depth_files)

            # Retrieve bounding boxes for each scene
            bboxes = loadmat(meta_file, squeeze_me=True, struct_as_record=True)['bboxes'] \
                     if meta_file is not None else [None] * len(rgb_files)
            self.bboxes = [ [bbox] 
                            if bbox is not None and bbox.size == 1 else bbox
                            for bbox in bboxes ]
            assert(len(self.bboxes) == len(rgb_files))

        @staticmethod
        def scene_files(files, version): 
            if version == 'v1': 
                depth_files = natural_sort(filter(lambda  fn: '_depth.png' in fn, files))
                rgb_files = natural_sort(list(set(files) - set(depth_files)))
            elif version == 'v2': 
                depth_files = natural_sort(filter(lambda  fn: '-depth.png' in fn, files))
                rgb_files = natural_sort(list(set(files) - set(depth_files)))    
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)
            return rgb_files, depth_files

        @staticmethod
        def meta_files(directory, version): 
            if version == 'v1': 
                mat_files = read_files(os.path.expanduser(directory), pattern='*.mat')
                return dict(((fn.split('/')[-1]).replace('.mat',''), fn) for fn in mat_files)
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose v1 scene dataset''' % version)

        def iteritems(self, every_k_frames=1): 
            for rgb_im, depth_im, bbox in izip(self.rgb.iteritems(every_k_frames=every_k_frames), 
                                               self.depth.iteritems(every_k_frames=every_k_frames), 
                                               self.bboxes[::every_k_frames]): 
                yield AttrDict(img=rgb_im, depth=depth_im, bbox=bbox if bbox is not None else [])

    def __init__(self, version, directory='', targets=None, num_targets=None, blacklist=['']):         
        if version not in ['v1', 'v2']: 
            raise ValueError('Version %s not supported. '''
                             '''Check dataset and choose either v1 or v2 scene dataset''' % version)

        self._dataset = read_dir(os.path.expanduser(directory), pattern='*.png', recursive=False)
        self._meta = UWRGBDSceneDataset._reader.meta_files(directory, version)

        # Instantiate a reader for each of the objects
        self.data = {}
        for key, files in self._dataset.iteritems(): 
            if (targets is not None and key not in targets) or key in blacklist: 
                continue

            # Get mat file for each scene
            meta_file = self._meta.get(key, None)
            # print key, meta_file
            # target_id = self.target_hash[key]
            self.data[key] = UWRGBDSceneDataset._reader(files, meta_file, version)


    def iteritems(self, every_k_frames=1, verbose=False): 
        pbar = setup_pbar(len(self.data)) if verbose else None
        for key, frames in self.data.iteritems(): 
            if verbose: 
                pbar.update(pbar.currval + 1)
                # print 'Processing: %s' % key

            for frame in frames.iteritems(every_k_frames): 
                yield frame
        if verbose: pbar.finish()

    @staticmethod
    def annotate(f): 
        vis = f.img.copy()
        for bbox in f.bbox: 
            cv2.rectangle(vis, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), 
                          (50, 50, 50), 2)
            category_name = str(bbox['category'])
            cv2.putText(vis, '[Category: [%i] %s]' % 
                        (UWRGBDDataset.get_category_id(category_name), category_name), 
                        (bbox['left'], bbox['top']-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), thickness = 1)
        return vis



def test_uw_rgbd_object(): 
    from bot_vision.image_utils import to_color
    from bot_vision.imshow_utils import imshow_cv

    object_directory = '~/data/rgbd_datasets/udub/rgbd-object-crop/rgbd-dataset'
    rgbd_data_uw = UWRGBDObjectDataset(directory=object_directory)

    for f in rgbd_data_uw.iteritems(every_k_frames=5): 
        bbox = f.bbox
        imshow_cv('frame', 
                  np.hstack([f.img, np.bitwise_and(f.img, to_color(f.mask))]), 
                  text='Image + Mask [Category: [%i] %s, Instance: %i]' % 
                  (bbox['category'], rgbd_data_uw.get_category_name(bbox['category']), bbox['instance']))
        imshow_cv('depth', (f.depth / 16).astype(np.uint8), text='Depth')

def test_uw_rgbd_scene(version='v1'): 
    from bot_vision.image_utils import to_color
    from bot_vision.imshow_utils import imshow_cv

    v1_directory = '/media/spillai/MRG-HD1/data/rgbd-scenes-v1/rgbd-scenes/'
    v2_directory = '/media/spillai/MRG-HD1/data/rgbd-scenes-v2/rgbd-scenes-v2/imgs/'

    if version == 'v1': 
        rgbd_data_uw = UWRGBDSceneDataset(version='v1', directory=v1_directory)
    elif version == 'v2': 
        rgbd_data_uw = UWRGBDSceneDataset(version='v2', directory=v2_directory)
    else: 
        raise RuntimeError('''Version %s not supported. '''
                           '''Check dataset and choose v1/v2 scene dataset''' % version)
        
    for f in rgbd_data_uw.iteritems(every_k_frames=5): 
        vis = rgbd_data_uw.annotate(f)
        imshow_cv('frame', np.hstack([f.img, vis]), text='Image')
        imshow_cv('depth', (f.depth / 16).astype(np.uint8), text='Depth')
        cv2.waitKey(100)


