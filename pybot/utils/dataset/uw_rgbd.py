"""
UW-RGB-D Object/Scene Dataset reader

Annotations: 
see uw_annotation_conversion.py

"""

# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT 

import os, time
import numpy as np
import cv2
import fnmatch

from itertools import izip
from collections import defaultdict
from scipy.io import loadmat

from pybot.utils.timer import SimpleTimer
from pybot.utils.misc import progressbar
from pybot.utils.db_utils import AttrDict
from pybot.utils.dataset_readers import read_dir, read_files, natural_sort, \
    DatasetReader, ImageDatasetReader
from pybot.vision.draw_utils import annotate_bbox
from pybot.vision.geom_utils import intersection_over_union
from pybot.vision.image_utils import im_resize
from pybot.vision.camera_utils import Camera, CameraIntrinsic, CameraExtrinsic, \
    construct_K, check_visibility, get_object_bbox, get_discretized_projection
from pybot.geometry.rigid_transform import Quaternion, RigidTransform, Pose
from pybot.externals.plyfile import PlyData

# __categories__ = ['flashlight', 'cap', 'cereal_box', 'coffee_mug', 'soda_can']

def create_roidb_item(f): 
    try: 
        bboxes = np.vstack([bbox.coords for bbox in f.bbox])
        targets = np.int64([bbox.target for bbox in f.bbox])
    except: 
        bboxes = np.empty(shape=(0,4), dtype=np.int64)
        targets = []
    return f.img, bboxes, targets

class CustomInterpolator(object):
    """
    NearestNDInterpolator(points, values)

    Nearest-neighbour interpolation in N dimensions.

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``

    """

    def __init__(self, X, y, target, k=4): 
        from scipy.spatial import cKDTree
        # self.tree = cKDTree(X)
        # self.values = y
        self.K = k
        self.target = target

        # self.llut = {}
        # self.tree, self.values, self.target = {}, {}, {}
        # for lind in np.unique(target):
        #     inds, = np.where(target == lind)
        #     self.llut[lind] = inds
        #     self.tree[lind] = cKDTree(X[inds,:])
        #     self.values[lind] = y[inds]
        #     self.target[lind] = target[inds]
            
    def __call__(self, X, targets):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        xi : ndarray of float, shape (..., ndim)
            Points where to interpolate data at.

        Only validate with neighboring pixels
         : check distances (x,y) and depth disparities 
       """
        assert(X.ndim == 2 and X.shape[1] == 2)
        assert(len(X) == len(y))
        assert(len(X) == len(target))
            
        xs = np.arange(0, W, 5, dtype=np.int32)
        ys = np.arange(0, H, 5, dtype=np.int32)
        xys = np.dstack(np.meshgrid(xs, ys)).reshape(-1,2)

        D, L = len(self.llut), len(xys)

        zbuffer = np.ones((L, D)) * 100.0
        for lind in self.llut: 
            xids = self.tree[lind].query_ball_point(xys, r=20)

            for idx, xind in enumerate(xids): 
                d, l = self.values[lind][xind], self.target[lind][xind]
                dmin = np.argmin(d)

                if d[dmin] < zbuffer[idx,1]: 
                    zbuffer[idx,0] = l[dmin]
                    zbuffer[idx,1] = d[dmin]

            
        # Determine ordering of ids
        self.target[ids]
        
        # # Check if neighbors are good indicators
        # valid_nn = np.sum(dists > 10, axis=1) < self.K / 2
        # self.values[~valid_nn] = np.inf
        
        # # Check if disparities are within 3 px of neighbors
        # valid_disp = np.fabs(np.median(self.values[ids], axis=1) - self.values) < 0.3
        # self.values[~valid_disp] = np.inf

        return self.values

def finite_and_within_bounds(xys, shape): 
    H, W = shape[:2]
    if not len(xys): 
        return np.array([])
    return np.bitwise_and(np.isfinite(xys).all(axis=1), 
                          reduce(lambda x,y: np.bitwise_and(x,y), [xys[:,0] >= 0, xys[:,0] < W, 
                                                                   xys[:,1] >= 0, xys[:,1] < H]))


def dmap(X, d, targets, shape, size=10,
         dmin=0.1, interpolation=cv2.INTER_NEAREST, IoU=0.3):
    """
    Create depth map: z-buffer, z-buffered object index, and the
    corresponding bbox coordinates
    """
    visible_bboxes = {}
    
    assert(len(X) == len(d) == len(targets))
    H, W = shape[:2]
    BH, BW = shape[:2] / size

    zbuffer = np.ones(BH * BW, dtype=np.float32) * 100.
    zbufferl = np.ones(BH * BW, dtype=np.int32) * -1
    
    # Prune by minimum depth 
    mask = d > dmin
    if not mask.any():
        return im_resize(zbuffer.reshape(BH, BW), shape=(W,H),
                         interpolation=interpolation), \
            im_resize(zbufferl.reshape(BH, BW), shape=(W,H),
                      interpolation=interpolation), \
            visible_bboxes
    X, d, targets = X[mask], d[mask], targets[mask]

    # Z-buffer for each of the targets
    bboxes = {}
    assert(len(targets) == len(X))
    for lind in np.unique(targets):
        inds, = np.where(lind == targets)
        Xi, di = X[inds], d[inds]
        xs, ys = (Xi[:,0] / size).astype(np.int32), (Xi[:,1] / size).astype(np.int32)

        # Min-max bounds
        x0, x1 = int(max(0, Xi[:,0].min())), int(min(W-1, Xi[:,0].max()))
        y0, y1 = int(max(0, Xi[:,1].min())), int(min(H-1, Xi[:,1].max()))
        bboxes[lind] = np.int32([x0, y0, x1, y1])
        
        zinds = ys * BW + xs
        dinds = np.where(di < zbuffer[zinds])

        zinds = zinds[dinds]

        zbufferl[zinds] = lind
        zbuffer[zinds] = di[dinds]

    # Reshape depth buffer
    zbuffer = zbuffer.reshape(BH, BW)
    zbufferl = zbufferl.reshape(BH, BW)
    
    # Check visible bboxes
    for lind in np.unique(zbufferl):
        if lind < 0: continue
        xs, = np.where((zbufferl == lind).any(axis=0))
        ys, = np.where((zbufferl == lind).any(axis=1))

        # Min-max bounds
        assert(len(xs) and len(ys))
        x0, x1 = int(max(0, xs[0])), int(min(BW-1, xs[-1]+1))
        y0, y1 = int(max(0, ys[0])), int(min(BH-1, ys[-1]+1))
        vbbox = np.int32([x0, y0, x1, y1]) * size

        # Finally, check IoU w.r.t original bbox >= 0.3
        iou = intersection_over_union(vbbox, bboxes[lind])
        if iou >= IoU: 
            visible_bboxes[lind] = vbbox # bboxes[lind]
        
        
    return im_resize(zbuffer, shape=(W,H),
                     interpolation=interpolation), \
        im_resize(zbufferl, shape=(W,H),
                  interpolation=interpolation), \
        visible_bboxes
    
# =====================================================================
# Generic UW-RGBD Dataset class
# ---------------------------------------------------------------------

kinect_v1_params = AttrDict(
    K_depth = np.array([[576.09757860, 0, 319.5],
                        [0, 576.09757860, 239.5],
                        [0, 0, 1]], dtype=np.float64), 
    K_rgb = np.array([[528.49404721, 0, 319.5],
                      [0, 528.49404721, 239.5],
                      [0, 0, 1]], dtype=np.float64), 
    H = 480, W = 640, 
    shift_offset = 1079.4753, 
    projector_depth_baseline = 0.07214
)

class UWRGBDDataset(object): 
    default_rgb_shape = (480,640,3)
    default_depth_shape = (480,640)
    dfx = 570.3
    fx = 528.5 
    calib = CameraIntrinsic(
        K=construct_K(fx=fx, fy=fx,
                      cx=default_rgb_shape[1]/2-0.5, cy=default_rgb_shape[0]/2-0.5),
        shape=default_rgb_shape[:2])
    class_names = ["apple", "ball", "banana", "bell_pepper", "binder", "bowl", "calculator", "camera", 
                   "cap", "cell_phone", "cereal_box", "coffee_mug", "comb", "dry_battery", "flashlight", 
                   "food_bag", "food_box", "food_can", "food_cup", "food_jar", "garlic", "glue_stick", 
                   "greens", "hand_towel", "instant_noodles", "keyboard", "kleenex", "lemon", "lightbulb", 
                   "lime", "marker", "mushroom", "notebook", "onion", "orange", "peach", "pear", "pitcher", 
                   "plate", "pliers", "potato", "rubber_eraser", "scissors", "shampoo", "soda_can", 
                   "sponge", "stapler", "tomato", "toothbrush", "toothpaste", "water_bottle", "background", 
                   "sofa", "table", "office_chair", "coffee_table"]

    # Added more v2 objects into v1
    class_ids = np.arange(len(class_names), dtype=np.int)    
    target_hash = dict(zip(class_names, class_ids))
    target_unhash = dict(zip(class_ids, class_names))

    # train_names = ["cereal_box", "cap", "background"]
    # train_names = ["bowl", "cap", "cereal_box", "background"]
    # train_names = ["cap", "cereal_box", "coffee_mug", "soda_can", "background"]
    # train_names = ["bowl", "cap", "cereal_box", "soda_can", "background"]
    train_names = ["bowl", "cap", "cereal_box", "coffee_mug", "soda_can", "background"]
    # train_names = ["bowl", "cap", "cereal_box", "coffee_mug", "soda_can"]
    # train_names = ["bowl", "cap", "cereal_box", "coffee_mug", "flashlight", 
    #                "keyboard", "kleenex", "scissors",  "soda_can", 
    #                "stapler", "sofa", "table", "background"]
    # train_names = class_names

    train_ids = [target_hash[name] for name in train_names]
    train_names_set, train_ids_set = set(train_names), set(train_ids)

    train_hash = dict(zip(train_names, train_ids))
    train_unhash = dict(zip(train_ids, train_names))

    @classmethod
    def get_category_name(cls, target_id): 
        tid = int(target_id)
        return cls.target_unhash[tid] \
            if tid in cls.train_ids_set else 'background'

    @classmethod
    def get_category_id(cls, target_name): 
        tname = str(target_name)
        return cls.target_hash[tname] \
            if tname in cls.train_names_set else cls.target_hash['background']

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

    @classmethod
    def get_object_dataset(cls, object_dir, targets=train_names, verbose=False, version='v1'): 
        return UWRGBDObjectDataset(directory=object_dir, targets=targets, verbose=verbose)

    @classmethod
    def get_scene_dataset(cls, scene_dir, version='v1'): 
        return UWRGBDSceneDataset(version=version, directory=scene_dir)

    @classmethod
    def get_background(cls, scene_dir, version='v1'): 
        return  UWRGBDSceneDataset(version=version, 
                                   directory=os.path.join(scene_dir, 'background')) \

# =====================================================================
# UW-RGBD Object Dataset Reader
# ---------------------------------------------------------------------
    
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
                               bbox=[AttrDict(
                                   coords=np.float32([loc[0], loc[1], 
                                                      loc[0]+mask_im.shape[1], 
                                                      loc[1]+mask_im.shape[0]]), 
                                   target=self.target, 
                                   category=UWRGBDDataset.get_category_name(self.target), 
                                   instance=self.instance)])

    def __init__(self, directory='', targets=UWRGBDDataset.train_names, blacklist=[''], verbose=False):         
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
        st = time.time()
        self.dataset_ = read_dir(os.path.expanduser(directory), pattern='*.png', 
                                 recursive=False, expected_dirs=targets, verbose=verbose)
        print 'Time taken to read_dir %5.3f s' % (time.time() - st)
        print 'Classes: %i' % len(targets), self.dataset_.keys()

        # Instantiate a reader for each of the objects
        self.data = {}
        for key, files in self.dataset_.iteritems(): 
            if (targets is not None and get_category(key) not in targets) or key in blacklist: 
                continue
            target_id = self.target_hash[get_category(key)]
            # target_id = UWRGBDDataset.get_category_id(get_category(key))
            instance_id = get_instance(key)
            self.data[key] = UWRGBDObjectDataset._cropped_reader(target_id, instance_id, files)

        # Save target names for metrics
        self.target_names = targets

    def iteritems(self, every_k_frames=1, verbose=False):
        for key, frames in progressbar(self.data.iteritems(), size=len(self.data), verbose=verbose): 
            print 'Processing: %s' % key
            for frame in frames.iteritems(every_k_frames=every_k_frames): 
                yield frame
            # break
        
    def roidb(self, every_k_frames=1, verbose=True): 
        for item in self.iteritems(every_k_frames=every_k_frames, verbose=verbose):
            if len(item.bbox): 
                yield create_roidb_item(item)

# =====================================================================
# UW-RGBD Scene Dataset Reader
# ---------------------------------------------------------------------

class LazyDir(object):
    def __init__(self, path, pattern='*.png'): 
        self.path_ = path
        self.keys_ = os.listdir(os.path.expanduser(path))
        self.pattern_ = pattern
        print path, self.keys_
        
    def keys(self):
        return self.keys_
        
    def iterkeys(self):
        return (k for k in self.keys_)

    def __len__(self):
        return len(self.keys_)
    
    def __getitem__(self, key):
        assert(key in self.keys_)
        return read_files(os.path.join(self.path_, key), pattern=self.pattern_)

        
class UWRGBDSceneDataset(UWRGBDDataset):
    """
    RGB-D Scene Dataset reader 
    http://rgbd-dataset.cs.washington.edu/dataset.html
    """
    v2_target_hash = dict(bowl=1, cap=2, cereal_box=3, coffee_mug=4, coffee_table=5, 
                       office_chair=6, soda_can=7, sofa=8, table=9, background=10)
    # v2_target_unhash = dict((v,k) for k,v in v2_target_hash.iteritems())
    v2_to_v1 = dict((v2,UWRGBDDataset.target_hash[k2]) for k2,v2 in v2_target_hash.iteritems())

    def __init__(self, version, directory, targets=None, num_targets=None, blacklist=['']):
        if version not in ['v1', 'v2']: 
            raise ValueError('Version %s not supported. '''
                             '''Check dataset and choose either v1 or v2 scene dataset''' % version)
        self.version = version
        self.blacklist = blacklist

        # Recursively read, and categorize items based on folder
        scene_name = lambda idx: 'scene_{:02d}'.format(idx)
        self.dataset_ = LazyDir(os.path.join(os.path.expanduser(directory), 'imgs'), pattern='*.png')

        # Setup meta data
        if version == 'v1': 
            self.meta_ = UWRGBDSceneDataset._reader.meta_files(directory, version)
            self.aligned_ = None
        elif version == 'v2':
            self.meta_ = UWRGBDSceneDataset._reader.meta_files(os.path.join(directory, 'pc'), version)
            self.aligned_ = UWRGBDSceneDataset._reader.aligned_files(os.path.join(directory, 'pc'), version)
        else: 
            raise ValueError('Version %s not supported. '''
                             '''Check dataset and choose either v1 or v2 scene dataset''' % version)

    class _reader(object): 
        """
        RGB-D reader 
        Given mask, depth, and rgb files build an read iterator with appropriate process_cb
        """
        def __init__(self, files, meta_file, aligned_file, version, name=''): 
            self.name = name
            self.version = version

            self.rgb_files, self.depth_files = UWRGBDSceneDataset._reader.scene_files(files, version)
            assert(len(self.depth_files) == len(self.rgb_files))
            
            # RGB, Depth
            # TODO: Check depth seems scaled by 256 not 16
            self.rgb = ImageDatasetReader.from_filenames(self.rgb_files)
            self.depth = ImageDatasetReader.from_filenames(self.depth_files)

            # BBOX
            self.bboxes = UWRGBDSceneDataset._reader.load_bboxes(meta_file, version) \
                         if meta_file is not None else [None] * len(self.rgb_files)
            assert(len(self.bboxes) == len(self.rgb_files))

            # POSE
            # Version 2 only supported! Version 1 support for rgbd scene (unclear)
            self.poses = UWRGBDSceneDataset._reader.load_poses(aligned_file.pose, version) \
                         if aligned_file is not None and version == 'v2' else [None] * len(self.rgb_files)
            assert(len(self.poses) == len(self.rgb_files))

            # Camera
            intrinsic = UWRGBDSceneDataset.calib
            camera = Camera.from_intrinsics_extrinsics(intrinsic, CameraExtrinsic.identity())

            # TODO: Performance/Speedup: with SimpleTimer('aligned'): 
            # Aligned point cloud
            if aligned_file is not None: 
                if version != 'v2': 
                    raise RuntimeError('Version v2 is only supported')

                ply_xyz, ply_rgb = UWRGBDSceneDataset._reader.load_ply(aligned_file.ply, version)
                ply_label_ = UWRGBDSceneDataset._reader.load_plylabel(aligned_file.label, version)

                # Remapping to v1 index
                ply_label = ply_label_.copy()
                for l in np.unique(ply_label_):
                    lmask = ply_label_ == l
                    ply_label[lmask] = UWRGBDSceneDataset.v2_to_v1[l]

                # Get object info
                object_info = UWRGBDSceneDataset._reader.cluster_ply_labels(ply_xyz[::30], ply_rgb[::30], ply_label[::30])

                # Add camera info
                self.map_info = AttrDict(camera=camera, objects=object_info)
                assert(len(ply_xyz) == len(ply_rgb))

            print('*********************************')
            print('Scene {}, Images: {}, Poses: {}\nAligned: {}'
                  .format(self.scene_name, len(self.rgb_files), len(self.poses), aligned_file))


        @property
        def scene_name(self): 
            return self.name

        @property
        def calib(self): 
            return UWRGBDDataset.calib
            
        @staticmethod
        def cluster_ply_labels(ply_xyz, ply_rgb, ply_label, std=None): 
            """
            Separate multiple object instances cleanly, otherwise, 
            candidate projection becomes inconsistent
            """
            from pybot_pcl import euclidean_clustering            
            import pybot.externals.lcm.draw_utils as draw_utils

            object_info = []
            unique_labels = np.unique(ply_label)

            for lidx, l in enumerate(unique_labels): 

                # Only add clusters that are in target/train and not background
                if l not in UWRGBDDataset.train_ids or l == UWRGBDDataset.target_hash['background']: 
                    continue

                l_xyz = ply_xyz[ply_label == l]
                l_rgb = ply_rgb[ply_label == l]
                if std is None: 
                    std = np.std(l_xyz, axis=0).min() * 1.0
                # print 'Clustering: ', l_xyz.shape, l_rgb.shape, std, UWRGBDDataset.target_unhash[l], len(l_xyz)

                linds = euclidean_clustering(l_xyz.astype(np.float32), tolerance=std,
                                             scale=1.0, min_cluster_size=int(len(l_xyz)/3))
                unique_linds = np.unique(linds)

                ocounts = 0
                for lind in unique_linds:
                    if lind < 0: continue
                    
                    spoints = l_xyz[linds == lind]
                    scolors = l_rgb[linds == lind]

                    # At least 30% of the original point cloud
                    if len(spoints) * 1. / len(l_xyz) < 0.3:
                        continue

                    ocounts += 1
                    object_info.append(AttrDict(label=l, uid=len(object_info),  
                                                points=spoints, 
                                                colors=scolors, 
                                                center=np.mean(spoints, axis=0)))
                    # oidx = len(object_info)
                    # draw_utils.publish_cloud('aligned_cloud_' + str(oidx), object_info[-1].points,
                    #                          c={k:v for k,v in enumerate(['r','g','b','m','k','y'])}[oidx % 6], reset=True)
                    # draw_utils.publish_pose_list('aligned_poses_' + str(oidx),
                    #                              [RigidTransform(tvec=object_info[-1].center)], 
                    #                              texts=[UWRGBDDataset.target_unhash[object_info[-1].label] + str(oidx)],
                    #                              reset=True)

                    # print 'Pts: ', len(object_info[-1].points)

                # Ensure that at least one object is filtered from the clustering
                assert(ocounts)
                
            print 'Total unique objects in dataset: ', len(object_info), \
                [UWRGBDDataset.target_unhash[obj.label] for obj in object_info]
            
            return object_info


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
                return np.fromfile(fn, sep='\n', dtype=np.int32)[1:]
            else: 
                raise ValueError('''Version %s not supported. '''
                                 '''Check dataset and choose either v1 or v2 scene dataset''' % version)


        @staticmethod
        def scene_files(files, version): 
            if version == 'v1': 
                depth_files = natural_sort(filter(lambda  fn: '_depth.png' in fn, files))
                rgb_files = natural_sort(filter(lambda  fn: '-color.png' in fn, files))

            elif version == 'v2':
                depth_files = natural_sort(filter(lambda  fn: '-depth.png' in fn, files))
                rgb_files = natural_sort(filter(lambda  fn: '-color.png' in fn, files))
                
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

        def visualize_ground_truth(self): 

            from pybot.vision.color_utils import get_color_by_label
            import pybot.externals.lcm.draw_utils as draw_utils

            # Publish ground truth poses, and aligned point clouds
            draw_utils.publish_pose_list('ground_truth_poses', self.poses)

            
            # carr = [get_color_by_label(self.map_info.labels)]
            carr = np.hstack([np.ones(len(obj.points)) * obj.label for obj in self.map_info.objects])
            carr = get_color_by_label(carr)
            # carr = [obj.colors for obj in self.map_info.objects]
            draw_utils.publish_cloud('aligned_cloud', np.vstack([obj.points for obj in self.map_info.objects]), c=carr)
            draw_utils.publish_pose_list('aligned_poses', [RigidTransform(tvec=obj.center) for obj in self.map_info.objects], 
                                         texts=[str(obj.label) for obj in self.map_info.objects])

        def get_bboxes(self, pose): 
            """
            Support for occlusion handling is incomplete/bug-ridden
            """

            # 1. Get pose for a particular frame, 
            # and set camera extrinsic
            try: 
                self.map_info.camera.set_pose(pose.inverse())
            except: 
                # Otherwise break from detection loop
                print 'Failed to find pose'
                return None

            from pybot.vision.color_utils import colormap, im_normalize
            from pybot.vision.imshow_utils import imshow_cv
            from pybot.vision.draw_utils import draw_bboxes
            
            # 2. Recover bboxes for non-occluded objects in view
            # Uniquely ID objects (targets = [oidx1, oidx2, ..])
            # so that we can recover the visible objects and their categories
            ss = 2
            pts3d = np.vstack([obj.points[::ss] for obj in self.map_info.objects])
            targets = np.hstack([oidx * np.ones(len(obj.points[::ss]), dtype=np.int32)
                                 for oidx, obj in enumerate(self.map_info.objects)])
            pts2d, depth, valid = self.map_info.camera.project(
                pts3d, check_bounds=True, check_depth=True, min_depth=0.1,
                return_depth=True, return_valid=True)

            # 3. Create depth map and visible bounding boxes
            shape = self.map_info.camera.shape[:2]
            zbuffer, zbufferl, visible_bboxes = dmap(pts2d, depth, targets[valid], shape, size=10)

            # 4. Create object candidates
            object_candidates = []
            for oidx,bbox in visible_bboxes.iteritems():
                obj = self.map_info.objects[oidx]
                object_candidates.append(
                    AttrDict(target=obj.label,
                             category=UWRGBDDataset.target_unhash[obj.label],
                             coords=bbox, depth=0, uid=obj.uid))
                
            
            # vis = colormap(im_normalize(zbufferl))
            # if len(bbox): 
            #     vis = draw_bboxes(vis, np.vstack(bbox.values()), colored=False)
            # imshow_cv('dvis', vis)
            # imshow_cv('vis', colormap(zbuffer / 10.0))
            
            # # 4. Determine bounding boxes for visible clusters
            # object_centers = np.vstack([obj.center for obj in self.map_info.objects])
            # visible_inds, = np.where(check_visibility(self.map_info.camera, object_centers))

            # object_candidates = []
            # for ind in visible_inds:
            #     obj = self.map_info.objects[ind]
            #     label = obj.label
            #     pts2d, coords, depth = get_object_bbox(self.map_info.camera, obj.points, subsample=3, scale=1)

            #     if coords is not None: 
            #         object_candidates.append(
            #             AttrDict(
            #                 target=obj.label, 
            #                 category=UWRGBDDataset.target_unhash[obj.label], 
            #                 coords=coords, 
            #                 depth=depth, 
            #                 uid=obj.uid))

            # # 3. Ensure occlusions are handled, sort by increasing depth, and filter 
            # # based on overlapping threshold
            # sorted_object_candidates = sorted(object_candidates, key=lambda obj: obj.depth)
            
            # # Occlusion mask
            # im_sz = self.map_info.camera.shape[:2]
            # occ_mask = np.zeros(shape=im_sz, dtype=np.uint8)
            
            # # Non-occluded object_candidates
            # nonocc_object_candidates = []
            # for obj in sorted_object_candidates: 
            #     x0, y0, x1, y1 = obj.coords
            #     xc, yc = (x0 + x1) / 2, (y0 + y1) / 2

            #     # If the bbox center is previously occupied, skip
            #     if occ_mask[yc,xc]: 
            #         continue

            #     # Set as occupied
            #     occ_mask[y0:y1,x0:x1] = 1
            #     nonocc_object_candidates.append(obj)

            return object_candidates

        def _process_items(self, index, rgb_im, depth_im, bbox, pose): 
            def _process_bbox(bbox): 
                return AttrDict(category=bbox['category'], 
                                target=UWRGBDDataset.target_hash[str(bbox['category'])], 
                                coords=np.int64([bbox['left'], bbox['top'], bbox['right'], bbox['bottom']]))

            # Compute bbox from pose and map (v2 support)
            if self.version == 'v1': 
                if bbox is not None: 
                    bbox = [_process_bbox(bb) for bb in bbox]
                    bbox = filter(lambda bb: bb.target in UWRGBDDataset.train_ids_set, bbox)

            if self.version == 'v2': 
                if bbox is None and hasattr(self, 'map_info'): 
                    bbox = self.get_bboxes(pose)

            # print 'Processing pose', pose, bbox
            return AttrDict(index=index, img=rgb_im, depth=depth_im, 
                            bbox=bbox if bbox is not None else [], pose=pose)
            
        def iteritems(self, every_k_frames=1): 
            index = 0 
            for rgb_im, depth_im, bbox, pose in izip(self.rgb.iteritems(every_k_frames=every_k_frames), 
                                                     self.depth.iteritems(every_k_frames=every_k_frames), 
                                                     self.bboxes[::every_k_frames], 
                                                     self.poses[::every_k_frames]): 
                yield self._process_items(index, rgb_im, depth_im, bbox, pose)
                index += every_k_frames

                
        def roidb(self, every_k_frames=1, verbose=True, skip_empty=True): 
            for item in self.iteritems(every_k_frames=every_k_frames): 
                if not len(item.bbox) and skip_empty:
                    continue
                yield create_roidb_item(item)
                
        def iterinds(self, inds): 
            for index, rgb_im, depth_im, bbox, pose in izip(inds, 
                                                            self.rgb.iterinds(inds), 
                                                            self.depth.iterinds(inds), 
                                                            [self.bboxes[ind] for ind in inds], 
                                                            [self.poses[ind] for ind in inds]): 
                yield self._process_items(index, rgb_im, depth_im, bbox, pose)

    # @classmethod
    # def get_v2_category_name(cls, target_id): 
    #     return cls.v2_target_unhash[target_id] # \
    #         # if target_id in cls.train_ids_set else 'background'

    # @classmethod
    # def get_v2_category_id(cls, target_name): 
    #     return cls.v2_target_hash[target_name] # \
    #         # if target_name in cls.train_names_set else cls.target_hash['background']


    def iteritems(self, every_k_frames=1, verbose=False, with_ground_truth=False): 
        print 'Scenes ({}): {}, With GT: {}'.format(len(self.scenes), ','.join(self.scenes), with_ground_truth)
        for key, scene in progressbar(
                self.iterscenes(verbose=verbose, with_ground_truth=with_ground_truth), 
                size=len(self.scenes), verbose=verbose): 
            for frame in scene.iteritems(every_k_frames=every_k_frames): 
                yield frame
            # break
        
    def roidb(self, every_k_frames=1, verbose=True, skip_empty=True): 
        for item in self.iteritems(every_k_frames=every_k_frames, 
                                   with_ground_truth=True): 
            if not len(item.bbox) and skip_empty:
                continue
            yield create_roidb_item(item)

    def scene(self, key, with_ground_truth=False): 
        if key in self.blacklist: 
            raise RuntimeError('Key %s is in blacklist, are you sure you want this!' % key)

        # Get scene files
        files = self.dataset_[key]
        
        # Get meta data 
        meta_file = self.meta_.get(key, None)
        aligned_file = self.aligned_.get(key, None) if (self.aligned_ and with_ground_truth) else None

        print('Initializing scene {} WITH{} ground truth'.format(key, '' if with_ground_truth else 'OUT'))        
        return UWRGBDSceneDataset._reader(files, meta_file, aligned_file, self.version, key) 

    @property
    def scenes(self): 
        return self.dataset_.keys()

    def iterscenes(self, targets=None, blacklist=None, verbose=False, with_ground_truth=False): 
        for key in progressbar(self.dataset_.iterkeys(), size=len(self.dataset_), verbose=verbose): 
            # Optionally only iterate over targets, and avoid blacklist
            if (targets is not None and key not in targets) or \
               (blacklist is not None and key in blacklist): 
                continue
            yield key, self.scene(key, with_ground_truth=with_ground_truth)
        
    @staticmethod
    def annotate_bboxes(vis, bboxes, target_names): # , box_color=lambda target: (0, 200, 0) if UWRGBDDataset.get_category_name(target) != 'background' else (100, 100, 100)): 
        for bbox,target_name in izip(bboxes, target_names): 
            box_color = (0, 200, 0) # if UWRGBDDataset.get_category_name(target) != 'background' else (100, 100, 100)
            annotate_bbox(vis, bbox.coords, color=box_color, title=target_name.title().replace('_', ' '))

            # cv2.rectangle(vis, (bbox.coords[0], bbox.coords[1]), (bbox.coords[2], bbox.coords[3]), box_color, 2) 
            # cv2.rectangle(vis, (bbox.coords[0]-1, bbox.coords[1]-15), (bbox.coords[2]+1, bbox.coords[1]), box_color, -1)
            # cv2.putText(vis, '%s' % (), 
            #             (bbox[0], bbox[1]-5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), thickness=1, lineType=cv2.CV_AA)
        return vis

    @staticmethod
    def annotate(f): 
        # TODO: Standardize
        vis = f.img.copy()
        for bbox in f.bbox:
            coords = np.int32(bbox.coords)
            cv2.rectangle(vis, (coords[0], coords[1]), (coords[2], coords[3]), 
                          (50, 50, 50), 2)
            category_name = str(bbox.category)
            cv2.putText(vis, '[Category: [%i] %s]' % 
                        (UWRGBDDataset.get_category_id(category_name), category_name), 
                        (coords[0], coords[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), thickness = 1)
        return vis

def test_uw_rgbd_object(): 
    from pybot.vision.image_utils import to_color
    from pybot.vision.imshow_utils import imshow_cv

    object_directory = '~/data/rgbd_datasets/udub/rgbd-object-crop/rgbd-dataset'
    rgbd_data_uw = UWRGBDObjectDataset(directory=object_directory)

    for f in rgbd_data_uw.iteritems(every_k_frames=5): 
        bbox = f.bbox
        imshow_cv('frame', 
                  np.hstack([f.img, np.bitwise_and(f.img, to_color(f.mask))]), 
                  text='Image + Mask [Category: [%i] %s, Instance: %i]' % 
                  (bbox['category'], rgbd_data_uw.get_category_name(bbox['category']), bbox['instance']))
        imshow_cv('depth', (f.depth / 16).astype(np.uint8), text='Depth')

def test_uw_rgbd_scene(version='v1', return_dataset=False): 
    from pybot.vision.image_utils import to_color
    from pybot.vision.imshow_utils import imshow_cv
    from pybot.externals.lcm.draw_utils import publish_pose_list
    
    v1_directory = '/media/spillai/MRG-HD1/data/rgbd-scenes-v1/'
    v2_directory = '/media/spillai/MRG-HD1/data/rgbd-scenes-v2/rgbd-scenes-v2/'

    if version == 'v1': 
        rgbd_data_uw = UWRGBDSceneDataset(version='v1', 
                                          directory=os.path.join(v1_directory, 'rgbd-scenes'), 
                                          aligned_directory=os.path.join(v1_directory, 'rgbd-scenes-aligned'))
    elif version == 'v2': 
        rgbd_data_uw = UWRGBDSceneDataset(version='v2', directory=v2_directory)
    else: 
        raise RuntimeError('''Version %s not supported. '''
                           '''Check dataset and choose v1/v2 scene dataset''' % version)

    if return_dataset:
        return rgbd_data_uw
    
    for idx, f in enumerate(rgbd_data_uw.iteritems(every_k_frames=5, with_ground_truth=True)): 
        vis = rgbd_data_uw.annotate(f)
        imshow_cv('frame', np.hstack([f.img, vis]), text='Image')
        imshow_cv('depth', (f.depth / 16).astype(np.uint8), text='Depth')
        publish_pose_list('poses', [Pose.from_rigid_transform(idx, f.pose)], frame_id='camera', reset=False)
        cv2.waitKey(10)

def test_uw_rgbd_scene_iterscenes():
    dataset = test_uw_rgbd_scene(version='v2', return_dataset=True)
    for key, scene in dataset.iterscenes(with_ground_truth=True):
        for f in scene.iteritems(every_k_frames=10):
            print f.img.shape

def test_uw_rgbd_scene_roidb():
    dataset = test_uw_rgbd_scene(version='v2', return_dataset=True)
    for key, scene in dataset.iterscenes(with_ground_truth=True):
        for (im,bboxes,targets) in scene.roidb(every_k_frames=10, skip_empty=True): 
            print im.shape, bboxes.shape

if __name__ == "__main__":
    # test_uw_rgbd_object()
    # test_uw_rgbd_scene('v1')
    # test_uw_rgbd_scene('v2')
    # test_uw_rgbd_scene_iterscenes()
    # test_uw_rgbd_scene_roidb()
    print test_uw_rgbd_scene('v2', return_dataset=True).scenes
    
