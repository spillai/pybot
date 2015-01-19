import os
import numpy as np
import cv2
from itertools import izip, imap
from collections import defaultdict

from scipy.io import loadmat

from bot_utils.misc import setup_pbar
from bot_utils.db_utils import AttrDict
from bot_utils.dataset_readers import read_dir, read_files, natural_sort, \
    DatasetReader, ImageDatasetReader

from bot_vision.camera_utils import kinect_v1_params, Camera, CameraIntrinsic, CameraExtrinsic, \
    check_visibility, get_object_bbox
from bot_geometry.rigid_transform import Quaternion, RigidTransform
from bot_externals.plyfile import PlyData

# __categories__ = ['flashlight', 'cap', 'cereal_box', 'coffee_mug', 'soda_can']

class UWRGBDDataset(object): 
    default_rgb_shape = (480,640,3)
    default_depth_shape = (480,640)
    camera_params = kinect_v1_params
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

    # train_names = ["cereal_box", "cap", "BACKGROUND"]
    # train_names = ["bowl", "cap", "cereal_box", "BACKGROUND"]
    # train_names = ["cap", "cereal_box", "coffee_mug", "soda_can", "BACKGROUND"]
    train_names = ["bowl", "cap", "cereal_box", "coffee_mug", "soda_can", "BACKGROUND"]
    # train_names = ["bowl", "cap", "cereal_box", "coffee_mug", "flashlight", 
    #                 "keyboard", "kleenex", "scissors",  "soda_can", 
    #                 "stapler", "BACKGROUND"]
    # train_names = class_names

    train_ids = [target_hash[name] for name in train_names]
    train_names_set, train_ids_set = set(train_names), set(train_ids)

    @classmethod
    def get_category_name(cls, target_id): 
        tid = int(target_id)
        return cls.target_unhash[tid] \
            if tid in cls.train_ids_set else 'BACKGROUND'

    @classmethod
    def get_category_id(cls, target_name): 
        tname = str(target_name)
        return cls.target_hash[tname] \
            if tname in cls.train_names_set else cls.target_hash['BACKGROUND']

    @classmethod
    def setup_all_datasets(cls, object_dir=None, scene_dir=None, targets=train_names, version='v1'): 
        return AttrDict(

            # Main object dataset (single object instance per image)
            objects = UWRGBDObjectDataset(directory=object_dir, targets=targets) \
            if object_dir is not None else None, 

            # Scene dataset for evaluation
            scene = UWRGBDSceneDataset(version=version, directory=scene_dir) \
            if scene_dir is not None else None, 

            # Background dataset for hard-negative training
            background = UWRGBDSceneDataset(version=version, 
                                            directory=os.path.join(scene_dir, 'background')) \
            if scene_dir is not None else None
        )
    
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
                               bbox=[{'left':loc[0], 'right':loc[0]+mask_im.shape[1], 
                                      'top':loc[1], 'bottom':loc[1]+mask_im.shape[0], 
                                      'category':UWRGBDDataset.get_category_name(self.target), 
                                      'instance':self.instance}])

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
                pbar.increment()
                # print 'Processing: %s' % key

            for frame in frames.iteritems(every_k_frames=every_k_frames): 
                yield frame
        if verbose: pbar.finish()


class UWRGBDSceneDataset(UWRGBDDataset):
    """
    RGB-D Scene Dataset reader 
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """
    target_hash = dict(bowl=1, cap=2, cereal_box=3, coffee_mug=4, coffee_table=5, 
                       office_chair=6, soda_can=7, sofa=9, table=9, background=10)
    target_unhash = dict((v,k) for k,v in target_hash.iteritems())

    class _reader(object): 
        """
        RGB-D reader 
        Given mask, depth, and rgb files build an read iterator with appropriate process_cb
        """
        def __init__(self, files, meta_file, aligned_file, version): 
            if version == 'v2': 
                print '\n\n===> Version v1 and v2 have discrepancies in depth values, FIX!! <===\n\n'

            rgb_files, depth_files = UWRGBDSceneDataset._reader.scene_files(files, version)
            assert(len(depth_files) == len(rgb_files))

            # RGB, Depth
            # TODO: Check depth seems scaled by 256 not 16
            self.rgb = ImageDatasetReader.from_filenames(rgb_files)
            self.depth = ImageDatasetReader.from_filenames(depth_files)

            # BBOX
            self.bboxes = UWRGBDSceneDataset._reader.load_bboxes(meta_file, version) \
                         if meta_file is not None else [None] * len(rgb_files)
            assert(len(self.bboxes) == len(rgb_files))
            
            # POSE
            # Version 2 only supported! Version 1 support for rgbd scene (unclear)
            self.poses = UWRGBDSceneDataset._reader.load_poses(aligned_file.pose, version) \
                         if aligned_file is not None and version == 'v2' else [None] * len(rgb_files)
            assert(len(self.poses) == len(rgb_files))

            if aligned_file: 
                # ALIGNED POINT CLOUD
                # Version 2 only supported! 
                ply_xyz, ply_rgb = UWRGBDSceneDataset._reader.load_ply(aligned_file.ply, version)
                ply_label = UWRGBDSceneDataset._reader.load_plylabel(aligned_file.label, version)
                inds, = np.where(ply_label != 1)
                ply_xyz, ply_rgb = ply_xyz[inds], ply_rgb[inds]
                ply_label = ply_label[inds]

                # 1b. Plot centers with poses and text
                unique_labels = np.unique(ply_label)

                # 1c. Determine centroid of each cluster
                unique_centers = np.vstack([np.mean(ply_xyz[ply_label == l], axis=0) for l in unique_labels])

                # FIX: Needs to be re-worked to use arbitrary camera matrix
                intrinsic = CameraIntrinsic(K=UWRGBDSceneDataset.camera_params.K_rgb)
                camera = Camera.from_intrinsics_extrinsics(intrinsic, CameraExtrinsic.identity())
                self.map_info = AttrDict(
                    points=ply_xyz, color=ply_rgb, labels=ply_label, 
                    unique_labels=unique_labels, unique_centers=unique_centers, camera=camera
                ) if aligned_file is not None and version == 'v2' else None
                assert(len(ply_xyz) == len(ply_rgb))

        def get_bboxes(self, pose): 
            # 1. Get pose for a particular frame, 
            # and set camera extrinsic
            try: 
                self.map_info.camera.set_pose(pose.inverse())
            except: 
                # Otherwise break from detection loop
                print 'Failed to find pose'
                return None

            # 2. Get all cluster centers that are visible from camera's view
            visible_inds = check_visibility(self.map_info.camera, self.map_info.unique_centers)
            visible_labels = self.map_info.unique_labels[visible_inds]

            # 3. Determine bounding boxes for each of the visible clusters
            object_candidates = []
            for label in visible_labels:
                label_pts = self.map_info.points[label == self.map_info.labels]
                pts2d, bbox, depth = get_object_bbox(self.map_info.camera, label_pts, subsample=3)
                if bbox is not None: 
                    object_candidates.append(AttrDict(label=label, bbox=bbox, depth=depth))
            return object_candidates

        @classmethod
        def get_category_name(cls, target_id): 
            return cls.target_unhash[target_id] # \
                # if target_id in cls.train_ids_set else 'BACKGROUND'

        @classmethod
        def get_category_id(cls, target_name): 
            return cls.target_hash[target_name] # \
                # if target_name in cls.train_names_set else cls.target_hash['BACKGROUND']

        @staticmethod
        def load_bboxes(fn, version): 
            """ Retrieve bounding boxes for each scene """
            if version == 'v1': 
                bboxes = loadmat(fn, squeeze_me=True, struct_as_record=True)['bboxes']
                return [ [bbox] 
                         if bbox is not None and bbox.size == 1 else bbox
                         for bbox in bboxes ]

                # bboxes = loadmat(fn, squeeze_me=True, struct_as_record=True)['bboxes'] \
                #          if fn is not None else [None] * default_count
                # return [ [bbox] 
                #          if bbox is not None and bbox.size == 1 else bbox
                #          for bbox in bboxes ]

            elif version == 'v2': 
                return None
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)

        @staticmethod
        def load_poses(fn, version): 
            """ Retrieve poses for each scene """ 

            if version == 'v1': 
                P = np.loadtxt(os.path.expanduser(fn), usecols=(2,3,4,5,6,7,8), dtype=np.float64)
                return map(lambda p: RigidTransform(Quaternion.from_wxyz(p[:4]), p[4:]), P)
            elif version == 'v2': 
                P = np.loadtxt(os.path.expanduser(fn), dtype=np.float64)
                return map(lambda p: RigidTransform(Quaternion.from_wxyz(p[:4]), p[4:]), P)
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)


        @staticmethod
        def load_ply(fn, version): 
            """ Retrieve aligned point cloud for each scene """ 

            if version == 'v1': 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)
                # P = np.loadtxt(os.path.expanduser(fn), usecols=(2,3,4,5,6,7,8), dtype=np.float64)
                # return map(lambda p: RigidTransform(Quaternion.from_wxyz(p[:4]), p[4:]), P)
            elif version == 'v2': 
                ply = PlyData.read(os.path.expanduser(fn))
                xyz = np.vstack([ply['vertex'].data['x'], 
                                 ply['vertex'].data['y'], 
                                 ply['vertex'].data['z']]).T
                rgb = np.vstack([ply['vertex'].data['diffuse_red'], 
                                 ply['vertex'].data['diffuse_green'], 
                                 ply['vertex'].data['diffuse_blue']]).T
                return xyz, rgb

            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)


        @staticmethod
        def load_plylabel(fn, version): 
            """ Retrieve aligned point cloud labels for each scene """ 

            if version == 'v1': 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)
            elif version == 'v2': 
                return np.loadtxt(fn, dtype=np.int32)[1:]
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)


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
            elif version == 'v2': 
                label_files = read_files(os.path.expanduser(directory), pattern='*.label')
                return dict(((fn.split('/')[-1]).replace('.label',''), fn) for fn in label_files)
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose v1 scene dataset''' % version)

        @staticmethod
        def aligned_files(directory, version): 

            if version == 'v1': 
                pose_files = read_files(os.path.expanduser(directory), pattern='*.txt')
                return dict(((fn.split('/')[-1]).replace('.txt',''), fn) for fn in pose_files)
            elif version == 'v2': 

                aligned = defaultdict(AttrDict)
                pose_files = read_files(os.path.expanduser(directory), pattern='*.pose')
                label_files = read_files(os.path.expanduser(directory), pattern='*.label')
                ply_files = read_files(os.path.expanduser(directory), pattern='*.ply')

                pretty_name = lambda fn, ext: 'scene_' + (fn.split('/')[-1]).replace(ext,'')
                for fn in pose_files: 
                    aligned[pretty_name(fn, '.pose')].pose = fn
                for fn in label_files: 
                    aligned[pretty_name(fn, '.label')].label = fn
                for fn in ply_files: 
                    aligned[pretty_name(fn, '.ply')].ply = fn

                return aligned
                # return dict(('scene_' + (fn.split('/')[-1]).replace('.pose',''), fn) 
                #             for fn in pose_files)
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose v1 scene dataset''' % version)

        def iteritems(self, every_k_frames=1): 
            index = 0 
            for rgb_im, depth_im, bbox, pose in izip(self.rgb.iteritems(every_k_frames=every_k_frames), 
                                                     self.depth.iteritems(every_k_frames=every_k_frames), 
                                                     self.bboxes[::every_k_frames], 
                                                     self.poses[::every_k_frames]): 
                index += every_k_frames

                # Compute bbox from pose and map (v2 support)
                if bbox is None and hasattr(self, 'map_info'): 
                    bbox = self.get_bboxes(pose)

                yield AttrDict(index=index, img=rgb_im, depth=depth_im, 
                               bbox=bbox if bbox is not None else [], pose=pose)

    def __init__(self, version, directory='', targets=None, num_targets=None, 
                 blacklist=[''], aligned_directory=None):
        if version not in ['v1', 'v2']: 
            raise ValueError('Version %s not supported. '''
                             '''Check dataset and choose either v1 or v2 scene dataset''' % version)
        self.version = version
        self.blacklist = blacklist

        self._dataset = read_dir(os.path.expanduser(directory), pattern='*.png', recursive=False)
        self._meta = UWRGBDSceneDataset._reader.meta_files(directory, version)
        self._aligned = UWRGBDSceneDataset._reader.aligned_files(aligned_directory, version) \
                        if aligned_directory else None

        # # Instantiate a reader for each of the objects
        # self.data = {}
        # for key, files in self._dataset.iteritems(): 
        #     if (targets is not None and key not in targets) or key in blacklist: 
        #         continue

        #     # Get mat file for each scene
        #     meta_file = self._meta.get(key, None)
        #     aligned_file = self._aligned.get(key, None)
        #     print key, meta_file, aligned_file
        #     # target_id = self.target_hash[key]
        #     self.data[key] = UWRGBDSceneDataset._reader(files, meta_file, aligned_file, version)


    def iteritems(self, every_k_frames=1, verbose=False): 
        pbar = setup_pbar(len(self._dataset)) if verbose else None
        for key, frames in self.iterscenes(verbose=verbose): 
            if verbose: 
                pbar.increment()
                # print 'Processing: %s' % key

            for frame in frames.iteritems(every_k_frames=every_k_frames): 
                yield frame
        if verbose: pbar.finish()

    def scene(self, key): 
        if key in self.blacklist: 
            raise RuntimeError('Key %s is in blacklist, are you sure you want this!' % key)

        # Get mat file for each scene
        files = self._dataset[key]
        meta_file = self._meta.get(key, None)
        aligned_file = self._aligned.get(key, None) if self._aligned else None
        # print key, meta_file, aligned_file
        return UWRGBDSceneDataset._reader(files, meta_file, aligned_file, self.version)

    def scenes(self): 
        return self._dataset.keys()

    def iterscenes(self, targets=None, blacklist=None, verbose=False): 
        pbar = setup_pbar(len(self._dataset)) if verbose else None
        for key in self._dataset.iterkeys(): 
            # Optionally only iterate over targets, and avoid blacklist
            if (targets is not None and key not in targets) or \
               (blacklist is not None and key in blacklist): 
                continue
            if verbose: pbar.increment()
            yield key, self.scene(key)
        if verbose: pbar.finish()

    @staticmethod
    def annotate(f): 
        # TODO: Standardize
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

    v1_directory = '/media/spillai/MRG-HD1/data/rgbd-scenes-v1/'
    v2_directory = '/media/spillai/MRG-HD1/data/rgbd-scenes-v2/rgbd-scenes-v2/'

    if version == 'v1': 
        rgbd_data_uw = UWRGBDSceneDataset(version='v1', 
                                          directory=os.path.join(v1_directory, 'rgbd-scenes'), 
                                          aligned_directory=os.path.join(v1_directory, 'rgbd-scenes-aligned'))
    elif version == 'v2': 
        rgbd_data_uw = UWRGBDSceneDataset(version='v2', 
                                          directory=os.path.join(v2_directory, 'imgs'), 
                                          aligned_directory=os.path.join(v2_directory, 'pc'))
    else: 
        raise RuntimeError('''Version %s not supported. '''
                           '''Check dataset and choose v1/v2 scene dataset''' % version)
        
    for f in rgbd_data_uw.iteritems(every_k_frames=5): 
        vis = rgbd_data_uw.annotate(f)
        imshow_cv('frame', np.hstack([f.img, vis]), text='Image')
        imshow_cv('depth', (f.depth / 16).astype(np.uint8), text='Depth')
        cv2.waitKey(100)


